# -*- coding: utf-8 -*-
"""
Churn Prediction + Uplift Segmentation
Freedom Hackathon 2026

Approach:
  1. Load feature_store from PostgreSQL (already pre-engineered)
  2. Build churn label: inactive 30+ days from May-12-2026
  3. Add derived features (trends, frustration signals)
  4. Train CatBoost classifier (random split — single snapshot data)
  5. SHAP explainability
  6. 4-quadrant Uplift Segmentation:
       Persuadables  = high churn prob + high pLTV  -> spend budget here
       Sure Things   = low churn prob  + high pLTV  -> upsell
       Sleeping Dogs = high error rate + churn risk -> do NOT contact (backfires)
       Lost Causes   = high churn prob + low pLTV   -> skip / exit survey only
  7. Save everything to outputs/
"""

import os
import sys
import warnings
import psycopg2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, f1_score, precision_score, recall_score
from catboost import CatBoostClassifier, Pool
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DB_PARAMS = dict(
    host='100.100.224.121', port=5433,
    dbname='freedom', user='postgres', password='admin',
    connect_timeout=15
)
REFERENCE_DATE = '2026-05-12'
CHURN_DAYS     = 30
OUTPUT_DIR     = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  Freedom Hackathon 2026 - Churn Intelligence Pipeline")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("\n[1/7] Loading data from PostgreSQL...")
conn = psycopg2.connect(**DB_PARAMS)

fs = pd.read_sql("SELECT * FROM feature_store", conn)
print(f"      feature_store: {fs.shape[0]:,} users, {fs.shape[1]} features")

# pltv: take max pltv per user (some users have multiple channels)
pltv = pd.read_sql(
    "SELECT user_id, MAX(pltv_predicted) as pltv_predicted FROM pltv_predictions GROUP BY user_id",
    conn
)
print(f"      pltv (deduped): {pltv.shape[0]:,} users")

# segments: one per user (take the highest-value segment)
segs = pd.read_sql(
    """SELECT user_id, segment FROM (
         SELECT user_id, segment,
                ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY
                    CASE segment WHEN 'VIP' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END) as rn
         FROM user_segments
       ) t WHERE rn = 1""",
    conn
)
print(f"      segments (deduped): {segs.shape[0]:,} users")

# acquisition: one channel per user (take most frequent paid, else organic)
acq = pd.read_sql(
    """SELECT user_id, channel FROM (
         SELECT user_id, channel,
                ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY id) as rn
         FROM acquisition
       ) t WHERE rn = 1""",
    conn
)
print(f"      acquisition (deduped): {acq.shape[0]:,} users")

# Churn label: no transactions AND no events for 30+ days before ref date
churn_sql = f"""
WITH last_event AS (
    SELECT user_id, MAX(event_timestamp)::date AS last_evt
    FROM events GROUP BY user_id
),
last_tx AS (
    SELECT user_id, MAX(date)::date AS last_tx
    FROM transactions GROUP BY user_id
),
combined AS (
    SELECT
        COALESCE(le.user_id, lt.user_id) AS user_id,
        le.last_evt,
        lt.last_tx,
        GREATEST(le.last_evt, lt.last_tx) AS last_active
    FROM last_event le
    FULL OUTER JOIN last_tx lt ON le.user_id = lt.user_id
    WHERE COALESCE(le.user_id, lt.user_id) IS NOT NULL
)
SELECT
    user_id,
    last_active,
    CASE
        WHEN last_active < DATE '{REFERENCE_DATE}' - INTERVAL '{CHURN_DAYS} days'
        THEN 1 ELSE 0
    END AS churn_label
FROM combined
"""
churn_df = pd.read_sql(churn_sql, conn)
# Safety dedup
churn_df = churn_df.drop_duplicates('user_id')
print(f"      churn labels: {churn_df.shape[0]:,} users | churn rate: {churn_df.churn_label.mean():.2%}")

conn.close()

# ─────────────────────────────────────────────
# 2. MERGE (careful — no row multiplication)
# ─────────────────────────────────────────────
print("\n[2/7] Merging datasets...")

df = fs.merge(churn_df[['user_id', 'churn_label', 'last_active']], on='user_id', how='inner')
before = len(df)
df = df.merge(pltv,  on='user_id', how='left')
df = df.merge(segs,  on='user_id', how='left')
df = df.merge(acq.rename(columns={'channel': 'acq_channel'}), on='user_id', how='left')

assert len(df) == before, f"Row count changed after merge! {before} -> {len(df)}"
print(f"      Final dataset: {len(df):,} rows (no duplicates)")
print(f"      Churn rate: {df.churn_label.mean():.2%}")
print(f"      Churn counts: 0={df.churn_label.eq(0).sum():,}  1={df.churn_label.eq(1).sum():,}")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[3/7] Feature engineering...")

eps = 0.5  # avoid division by zero

# Trend: is activity growing (>1) or declining (<1)?
df['tx_trend']   = df['tx_cnt_7d']  / (df['tx_cnt_14d']  - df['tx_cnt_7d']  + eps)
df['evt_trend']  = df['evt_cnt_7d'] / (df['evt_cnt_14d'] - df['evt_cnt_7d'] + eps)
df['vol_trend']  = df['tx_vol_7d']  / (df['tx_vol_14d']  - df['tx_vol_7d']  + eps)

# Frustration signals (primary churn triggers)
df['error_evt_rate']     = df['error_evt_7d']   / (df['evt_cnt_7d'] + 1)
df['failed_tx_rate']     = df['failed_tx_7d']   / (df['tx_cnt_7d'] + 1)
df['completed_evt_rate'] = df['completed_evt_7d'] / (df['evt_cnt_7d'] + 1)

# Digital engagement vs cash dependence
df['digital_ratio'] = df['purchase_7d'] / (df['p2p_7d'] + df['atm_7d'] + df['purchase_7d'] + 1)

# Cross-sell product depth
df['product_depth'] = (
    df['has_flights_7d'] + df['has_tickets_7d'] +
    df['has_groceries_7d'] + df['has_media_7d']
)

# Channel: paid partner vs organic/bank
PAID_CHANNELS = {'Ticketon', 'FTravel', 'Arbuz', 'FMobile',
                 'FMedia', 'FDrive', 'Insurance', 'DTP', 'Tours'}
df['acq_channel'] = df['acq_channel'].fillna('organic')
df['is_paid_channel'] = df['acq_channel'].isin(PAID_CHANNELS).astype(int)

# Gender fill
df['gender'] = df['gender'].fillna('Unknown')

# ─────────────────────────────────────────────
# 4. PREPARE ML FEATURES
# ─────────────────────────────────────────────
FEATURE_COLS = [
    # Rolling window transaction features
    'tx_cnt_1d', 'tx_vol_1d',
    'tx_cnt_3d', 'tx_vol_3d',
    'tx_cnt_7d', 'tx_vol_7d',
    'tx_cnt_14d', 'tx_vol_14d',
    'purchase_7d', 'p2p_7d', 'atm_7d', 'cashin_7d',
    'max_tx_7d', 'active_days_7d', 'failed_tx_7d',
    # Rolling window event features
    'evt_cnt_1d', 'evt_cnt_7d', 'uniq_evt_7d', 'evt_cnt_14d',
    'open_card_7d', 'transfer_evt_7d', 'completed_evt_7d', 'error_evt_7d',
    # Product features
    'products_7d', 'has_flights_7d', 'has_tickets_7d',
    'has_groceries_7d', 'has_media_7d',
    'large_p2p_7d', 'elite_p2p_7d',
    # LTV proxy
    'proxy_ltv_90d',
    # Demographic
    'age',
    # Derived
    'tx_trend', 'evt_trend', 'vol_trend',
    'error_evt_rate', 'failed_tx_rate', 'completed_evt_rate',
    'digital_ratio', 'product_depth', 'is_paid_channel',
]
CAT_COLS = ['gender']
ALL_FEATURES = FEATURE_COLS + CAT_COLS

X = df[ALL_FEATURES].copy()
y = df['churn_label'].copy()

X[FEATURE_COLS] = X[FEATURE_COLS].fillna(0)
X['gender'] = X['gender'].fillna('Unknown')

# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT (random — single snapshot)
# ─────────────────────────────────────────────
print("\n[4/7] Training CatBoost model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"      Train churn: {y_train.mean():.2%}  |  Test churn: {y_test.mean():.2%}")

cat_idx = [ALL_FEATURES.index(c) for c in CAT_COLS]
train_pool = Pool(X_train, y_train, cat_features=cat_idx)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_idx)

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50,
    class_weights={0: 1, 1: 2},   # upweight churn minority
)
model.fit(train_pool, eval_set=test_pool, use_best_model=True)

# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────
print("\n[5/7] Evaluation...")
y_proba = model.predict_proba(test_pool)[:, 1]
y_pred  = (y_proba >= 0.5).astype(int)

auc  = roc_auc_score(y_test, y_proba)
f1   = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)

print(f"\n{'='*50}")
print(f"  AUC-ROC   : {auc:.4f}")
print(f"  F1-score  : {f1:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Active','Churned'])}")
print('='*50)

pd.DataFrame([{'auc_roc': auc, 'f1': f1, 'precision': prec, 'recall': rec}])\
  .to_csv(f'{OUTPUT_DIR}/metrics.csv', index=False)

# ─────────────────────────────────────────────
# 7. SHAP
# ─────────────────────────────────────────────
print("\n[6/7] SHAP explainability...")
n_shap = min(3000, len(X_test))
X_shap = X_test.sample(n_shap, random_state=42)

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Pool(X_shap, cat_features=cat_idx))

mean_abs_shap = np.abs(shap_values).mean(axis=0)
fi_df = pd.DataFrame({'feature': ALL_FEATURES, 'mean_shap': mean_abs_shap})\
          .sort_values('mean_shap', ascending=False).reset_index(drop=True)
fi_df.to_csv(f'{OUTPUT_DIR}/shap_feature_importance.csv', index=False)

print("\n  Top 15 churn drivers (SHAP):")
print(fi_df.head(15).to_string(index=False))

# SHAP bar chart
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, feature_names=ALL_FEATURES,
                  plot_type='bar', max_display=15, show=False)
plt.title('Top Churn Drivers (SHAP mean |value|)', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()

# SHAP beeswarm
plt.figure(figsize=(12, 9))
shap.summary_plot(shap_values, X_shap, feature_names=ALL_FEATURES,
                  max_display=15, show=False)
plt.title('SHAP Beeswarm — Direction of Churn Impact', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved SHAP plots.")

# ─────────────────────────────────────────────
# 8. SCORE ALL + UPLIFT SEGMENTATION
# ─────────────────────────────────────────────
print("\n[7/7] Scoring all users + Uplift segmentation...")

full_pool = Pool(X, cat_features=cat_idx)
df['churn_prob'] = model.predict_proba(full_pool)[:, 1]
df['pltv_predicted'] = df['pltv_predicted'].fillna(0)

CHURN_THRESH = 0.40
pltv_median  = df['pltv_predicted'].median()

# Sleeping Dogs: frustrated users — contacting them accelerates churn
sleeping_mask = (
    (df['error_evt_rate']  > 0.30) &
    (df['failed_tx_rate']  > 0.30) &
    (df['churn_prob']      > CHURN_THRESH)
)

def segment(row):
    if sleeping_mask.loc[row.name]:
        return 'Sleeping Dogs'
    high_churn = row['churn_prob'] >= CHURN_THRESH
    high_value = row['pltv_predicted'] >= pltv_median
    if high_churn and high_value:
        return 'Persuadables'
    if high_churn and not high_value:
        return 'Lost Causes'
    if not high_churn and high_value:
        return 'Sure Things'
    return 'Low Value Stable'

df['uplift_segment'] = df.apply(segment, axis=1)

# Segment summary
seg_stats = df.groupby('uplift_segment').agg(
    users            = ('user_id',        'count'),
    avg_churn_prob   = ('churn_prob',     'mean'),
    avg_pltv         = ('pltv_predicted', 'mean'),
    total_pltv       = ('pltv_predicted', 'sum'),
    avg_error_rate   = ('error_evt_rate', 'mean'),
).sort_values('total_pltv', ascending=False).round(3)

print("\n" + "=" * 70)
print("  UPLIFT SEGMENTATION")
print("=" * 70)
print(seg_stats.to_string())
print("=" * 70)

# Business impact: budget needed only for Persuadables
persuadables = df[df['uplift_segment'] == 'Persuadables']
print(f"\n  Persuadables: {len(persuadables):,} users")
print(f"  Total pLTV at risk: {persuadables['pltv_predicted'].sum():,.0f}")
print(f"  Avg pLTV: {persuadables['pltv_predicted'].mean():,.1f}")

seg_stats.to_csv(f'{OUTPUT_DIR}/uplift_segments_summary.csv')

# Full scored user table
save_cols = [
    'user_id', 'age', 'gender', 'acq_channel',
    'churn_prob', 'pltv_predicted', 'uplift_segment',
    'error_evt_rate', 'failed_tx_rate', 'tx_trend',
    'product_depth', 'churn_label', 'proxy_ltv_90d',
    'tx_cnt_7d', 'evt_cnt_7d'
]
df[save_cols].to_csv(f'{OUTPUT_DIR}/users_scored.csv', index=False)

# Retention strategy table
strategy_df = pd.DataFrame({
    'segment':          ['Persuadables',   'Sure Things',         'Sleeping Dogs',          'Lost Causes'],
    'users':            [
        (df['uplift_segment'] == 'Persuadables').sum(),
        (df['uplift_segment'] == 'Sure Things').sum(),
        (df['uplift_segment'] == 'Sleeping Dogs').sum(),
        (df['uplift_segment'] == 'Lost Causes').sum(),
    ],
    'action':           [
        'Personal cashback 5-10% / bonus offer',
        'Upsell: flights, deposits, FMedia',
        'DO NOT CONTACT — risk of triggering deletion',
        'Exit survey via email only',
    ],
    'channel':          ['Push + In-App', 'In-App only', 'None', 'Email'],
    'timing':           ['Day 14 after last activity', 'Anytime', 'N/A', 'Day 45+'],
    'budget_priority':  ['HIGH', 'LOW', 'ZERO', 'MINIMAL'],
})
strategy_df.to_csv(f'{OUTPUT_DIR}/retention_strategies.csv', index=False)
print("\n  Retention strategies saved.")

# ─────────────────────────────────────────────
# 9. SAVE MODEL
# ─────────────────────────────────────────────
model.save_model(f'{OUTPUT_DIR}/catboost_churn.cbm')
joblib.dump({
    'feature_cols':    FEATURE_COLS,
    'cat_cols':        CAT_COLS,
    'all_features':    ALL_FEATURES,
    'cat_idx':         cat_idx,
    'churn_threshold': CHURN_THRESH,
    'pltv_median':     pltv_median,
}, f'{OUTPUT_DIR}/model_config.pkl')

print(f"\n{'='*60}")
print(f"  DONE. AUC-ROC = {auc:.4f}  |  F1 = {f1:.4f}")
print(f"  Output files in ./{OUTPUT_DIR}/:")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    size_kb = os.path.getsize(fpath) // 1024
    print(f"    {fname:<45} {size_kb:>6} KB")
print('='*60)
