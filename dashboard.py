# -*- coding: utf-8 -*-
"""
Streamlit Dashboard — Freedom Churn Intelligence
Run: streamlit run dashboard.py
"""

import os
import psycopg2
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Freedom Churn Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PARAMS = dict(
    host='100.100.224.121', port=5433,
    dbname='freedom', user='postgres', password='admin',
    connect_timeout=15
)

SEGMENT_COLORS = {
    'Persuadables':    '#E74C3C',
    'Sure Things':     '#27AE60',
    'Sleeping Dogs':   '#F39C12',
    'Lost Causes':     '#95A5A6',
    'Low Value Stable':'#BDC3C7',
}

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Freedom_Finance_logo.svg/320px-Freedom_Finance_logo.svg.png",
             width=160, use_container_width=False)
    st.title("Параметры")
    churn_thresh = st.slider("Порог P(churn)", 0.1, 0.9, 0.40, 0.05)
    pltv_pct     = st.slider("Мин. pLTV percentile", 10, 90, 50, 10)
    st.divider()
    page = st.radio("Раздел", [
        "Обзор сегментов",
        "Анализ данных (EDA)",
        "SHAP — факторы оттока",
        "Топ пользователи",
    ])

# ── Load model results ────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_scored():
    path = 'outputs/users_scored.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data(ttl=300)
def load_metrics():
    path = 'outputs/metrics.csv'
    if os.path.exists(path):
        return pd.read_csv(path).iloc[0]
    return None

@st.cache_data(ttl=300)
def build_live_segments(scored_df, churn_threshold, pltv_percentile):
    """
    Fast vectorized segmentation for interactive sliders.
    """
    df = scored_df.copy()
    pltv_thresh_val = df['pltv_predicted'].quantile(pltv_percentile / 100)

    sleeping = (
        (df.get('error_evt_rate', 0) > 0.30) &
        (df.get('failed_tx_rate', 0) > 0.30) &
        (df['churn_prob'] > churn_threshold)
    )
    high_churn = df['churn_prob'] >= churn_threshold
    high_value = df['pltv_predicted'] >= pltv_thresh_val

    df['segment_live'] = np.select(
        [
            sleeping,
            high_churn & high_value,
            high_churn & ~high_value,
            ~high_churn & high_value,
        ],
        [
            'Sleeping Dogs',
            'Persuadables',
            'Lost Causes',
            'Sure Things',
        ],
        default='Low Value Stable'
    )
    return df, pltv_thresh_val

@st.cache_data(ttl=300, show_spinner="Загрузка EDA из PostgreSQL...")
def load_eda():
    conn = psycopg2.connect(**DB_PARAMS)
    # Transactions by category
    tx_cat = pd.read_sql(
        "SELECT category, COUNT(*) as cnt, SUM(ABS(amount)) as volume "
        "FROM transactions WHERE category != '' GROUP BY category ORDER BY cnt DESC",
        conn
    )
    # Events by type (top 15)
    evt_type = pd.read_sql(
        "SELECT event_type, COUNT(*) as cnt FROM events GROUP BY event_type ORDER BY cnt DESC LIMIT 15",
        conn
    )
    # Users by gender/age
    users = pd.read_sql(
        "SELECT age, gender, city FROM users WHERE age BETWEEN 14 AND 75",
        conn
    )
    # Channel ROMI (already computed)
    romi = pd.read_sql("SELECT * FROM channel_romi ORDER BY romi_pct DESC", conn)
    # Products
    prods = pd.read_sql(
        "SELECT product_type, COUNT(*) as cnt FROM products GROUP BY product_type ORDER BY cnt DESC",
        conn
    )
    conn.close()
    return tx_cat, evt_type, users, romi, prods

scored  = load_scored()
metrics = load_metrics()

# ── Re-segment with slider values ────────────────────────────────────────
if scored is not None:
    scored, pltv_thresh_val = build_live_segments(scored, churn_thresh, pltv_pct)

# ══════════════════════════════════════════════════════════════════════════
# PAGE: ОБЗОР СЕГМЕНТОВ
# ══════════════════════════════════════════════════════════════════════════
if page == "Обзор сегментов":
    st.title("🎯 Freedom Churn Intelligence")
    st.caption("Uplift-based сегментация: кого удерживать, кого не трогать, а на кого не тратить бюджет")

    if scored is None:
        st.error("Модель не обучена. Запусти: `python churn_pipeline.py`")
        st.stop()

    # KPI row
    total     = len(scored)
    persua    = (scored['segment_live'] == 'Persuadables').sum()
    sure      = (scored['segment_live'] == 'Sure Things').sum()
    sleeping  = (scored['segment_live'] == 'Sleeping Dogs').sum()
    lost      = scored['segment_live'].isin(['Lost Causes', 'Low Value Stable']).sum()
    avg_churn = scored['churn_prob'].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Всего пользователей", f"{total:,}")
    c2.metric("🎯 Persuadables", f"{persua:,}")
    c2.caption(f"Доля сегмента: {persua/total:.1%}")
    c3.metric("✅ Sure Things", f"{sure:,}")
    c3.caption(f"Доля сегмента: {sure/total:.1%}")
    c4.metric("😴 Sleeping Dogs", f"{sleeping:,}")
    c4.caption(f"Доля сегмента: {sleeping/total:.1%}")
    c5.metric("Средний P(churn)", f"{avg_churn:.1%}")

    if metrics is not None:
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AUC-ROC",   f"{metrics['auc_roc']:.4f}")
        m2.metric("F1-score",  f"{metrics['f1']:.4f}")
        m3.metric("Precision", f"{metrics['precision']:.4f}")
        m4.metric("Recall",    f"{metrics['recall']:.4f}")

    st.divider()

    # 4-Quadrant scatter
    st.subheader("Uplift Matrix — 4 сегмента")
    col1, col2 = st.columns([3, 2])

    with col1:
        sample = scored.sample(min(5000, len(scored)), random_state=42).copy()

        fig = px.scatter(
            sample,
            x='churn_prob', y='pltv_predicted',
            color='segment_live',
            color_discrete_map=SEGMENT_COLORS,
            opacity=0.5,
            labels={'churn_prob': 'P(churn)', 'pltv_predicted': 'pLTV (predicted)'},
            title='Пользователи в пространстве Churn Prob × pLTV',
            hover_data=['user_id', 'age', 'gender'],
        )
        fig.add_vline(x=churn_thresh,    line_dash='dash', line_color='#555',
                      annotation_text=f"churn≥{churn_thresh}")
        fig.add_hline(y=pltv_thresh_val, line_dash='dash', line_color='#555',
                      annotation_text=f"pLTV≥{pltv_thresh_val:.1f}")
        fig.update_layout(height=500, legend_title="Сегмент")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Что делать с каждым сегментом")
        items = [
            ("🎯 Persuadables",   "E74C3C",
             "Уйдут без действия, но МОЖНО удержать.\n\n"
             "**Кешбэк 5–10%**, бонус или персональный оффер.\n\n"
             "Push + In-App. Запускать на Day 14 неактивности."),
            ("✅ Sure Things",    "27AE60",
             "Останутся сами. Не тратить бюджет на удержание.\n\n"
             "Подходящий момент для **upsell**: flights, FMedia, депозиты."),
            ("😴 Sleeping Dogs",  "F39C12",
             "**НЕ ТРОГАТЬ.** Раздражены (много ошибок/отказов).\n\n"
             "Пуш-уведомление может спровоцировать удаление приложения.\n\n"
             "Exit survey только при самом удалении."),
            ("❌ Lost Causes",    "95A5A6",
             "Уйдут в любом случае, низкий LTV.\n\n"
             "Email-опрос для продуктовой аналитики. Бюджет = 0."),
        ]
        for title, color, text in items:
            st.markdown(
                f'<div style="border-left:4px solid #{color};padding:8px 12px;'
                f'margin-bottom:10px;border-radius:4px;">'
                f'<b>{title}</b><br><small>{text}</small></div>',
                unsafe_allow_html=True
            )

    st.divider()

    # Segment breakdown charts
    col1, col2 = st.columns(2)
    with col1:
        seg_c = scored['segment_live'].value_counts().reset_index()
        seg_c.columns = ['segment', 'users']
        fig = px.pie(seg_c, values='users', names='segment',
                     color='segment', color_discrete_map=SEGMENT_COLORS,
                     title='Кол-во пользователей по сегментам')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_pltv = scored.groupby('segment_live')['pltv_predicted'].sum().reset_index()
        seg_pltv.columns = ['segment', 'total_pltv']
        seg_pltv = seg_pltv.sort_values('total_pltv', ascending=False)
        fig = px.bar(seg_pltv, x='segment', y='total_pltv',
                     color='segment', color_discrete_map=SEGMENT_COLORS,
                     title='Суммарный pLTV под риском по сегментам',
                     labels={'total_pltv': 'Суммарный pLTV', 'segment': ''})
        st.plotly_chart(fig, use_container_width=True)

    # Churn prob distribution
    st.subheader("Распределение P(churn) по сегментам")
    fig = px.histogram(
        scored.sample(min(20000, len(scored))),
        x='churn_prob', color='segment_live',
        color_discrete_map=SEGMENT_COLORS,
        nbins=60, barmode='overlay', opacity=0.65,
        labels={'churn_prob': 'P(churn)', 'segment_live': 'Сегмент'},
    )
    fig.add_vline(x=churn_thresh, line_dash='dash', line_color='red',
                  annotation_text=f"threshold={churn_thresh}")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════
elif page == "Анализ данных (EDA)":
    st.title("🔍 Exploratory Data Analysis")
    tx_cat, evt_type, users, romi, prods = load_eda()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(tx_cat, x='category', y='cnt',
                     title='Транзакции по категориям (кол-во)',
                     labels={'cnt': 'Количество', 'category': ''},
                     color_discrete_sequence=['#3498DB'])
        fig.update_xaxes(tickangle=40)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(tx_cat, x='category', y='volume',
                     title='Транзакции по категориям (объём KZT)',
                     labels={'volume': 'Объём', 'category': ''},
                     color_discrete_sequence=['#9B59B6'])
        fig.update_xaxes(tickangle=40)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(evt_type, x='event_type', y='cnt',
                     title='Топ-15 типов событий (Amplitude)',
                     labels={'cnt': 'Кол-во', 'event_type': ''},
                     color_discrete_sequence=['#E74C3C'])
        fig.update_xaxes(tickangle=50)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(users, x='age', nbins=30,
                           title='Возраст пользователей',
                           color_discrete_sequence=['#27AE60'],
                           labels={'age': 'Возраст', 'count': 'Кол-во'})
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(prods, x='product_type', y='cnt',
                     title='Продукты — кол-во активаций',
                     color_discrete_sequence=['#F39C12'],
                     labels={'cnt': 'Активаций', 'product_type': ''})
        fig.update_xaxes(tickangle=40)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        romi_clean = romi[romi['channel'].notna() & (romi['channel'] != '')]
        fig = px.bar(romi_clean, x='channel', y='romi_pct',
                     title='ROMI% по каналу привлечения',
                     color='romi_pct',
                     color_continuous_scale='RdYlGn',
                     labels={'romi_pct': 'ROMI %', 'channel': ''})
        fig.update_xaxes(tickangle=40)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Channel ROMI — детали")
    st.dataframe(
        romi_clean[['channel','users','avg_pltv','cac_kzt','romi_pct','net_value_per_user']]
        .sort_values('romi_pct', ascending=False)
        .style.format({'avg_pltv': '{:.1f}', 'romi_pct': '{:.0f}%',
                       'net_value_per_user': '{:.0f}', 'users': '{:,}'}),
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════
# PAGE: SHAP
# ══════════════════════════════════════════════════════════════════════════
elif page == "SHAP — факторы оттока":
    st.title("🧠 Что вызывает отток? — SHAP Analysis")

    if not os.path.exists('outputs/shap_bar.png'):
        st.warning("SHAP не рассчитан. Запусти `python churn_pipeline.py` сначала.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.image('outputs/shap_bar.png',
                 caption='Feature Importance (mean |SHAP|)',
                 use_container_width=True)
    with col2:
        st.image('outputs/shap_beeswarm.png',
                 caption='Beeswarm: направление влияния на отток',
                 use_container_width=True)

    st.subheader("Таблица важности признаков")
    fi = pd.read_csv('outputs/shap_feature_importance.csv')
    fi['rank'] = range(1, len(fi)+1)
    st.dataframe(
        fi[['rank','feature','mean_shap']].head(20)
        .style.format({'mean_shap': '{:.4f}'}),
        use_container_width=True
    )

    st.divider()
    st.markdown("""
    **Как читать SHAP:**
    - **Красные точки** (beeswarm) → высокое значение признака **увеличивает** P(churn)
    - **Синие точки** → низкое значение признака **уменьшает** P(churn)
    - **Длина полосы** (bar chart) → суммарное влияние признака на модель
    
    Ключевые инсайты из данных:
    - Снижение `evt_cnt_7d` → главный сигнал оттока
    - Высокий `error_evt_rate` → frustrated user → уходит
    - `proxy_ltv_90d = 0` → пользователь не монетизирован → риск потери
    - Падающий `tx_trend` (7d < 14d) → активность угасает
    """)

# ══════════════════════════════════════════════════════════════════════════
# PAGE: ТОП ПОЛЬЗОВАТЕЛИ
# ══════════════════════════════════════════════════════════════════════════
elif page == "Топ пользователи":
    st.title("👤 Приоритетные пользователи для удержания")

    if scored is None:
        st.error("Запусти `python churn_pipeline.py` сначала.")
        st.stop()

    segment_choice = st.selectbox("Сегмент", [
        'Persuadables', 'Sleeping Dogs', 'Lost Causes', 'Sure Things'
    ])

    filtered = scored[scored['segment_live'] == segment_choice]\
        .sort_values('pltv_predicted', ascending=False).head(200)

    st.metric("Пользователей в сегменте",
              f"{(scored['segment_live'] == segment_choice).sum():,}")

    st.dataframe(
        filtered[['user_id', 'age', 'gender', 'acq_channel',
                  'churn_prob', 'pltv_predicted', 'tx_cnt_7d',
                  'evt_cnt_7d', 'error_evt_rate', 'product_depth']]
        .style.format({
            'churn_prob': '{:.2%}', 'pltv_predicted': '{:.1f}',
            'error_evt_rate': '{:.2%}'
        }),
        use_container_width=True,
        height=600,
    )

    st.download_button(
        "Скачать CSV",
        data=filtered.to_csv(index=False).encode('utf-8'),
        file_name=f"{segment_choice.lower()}_users.csv",
        mime='text/csv'
    )

# Footer
st.divider()
st.markdown(
    "<center><small>Freedom Hackathon 2026 | Churn Intelligence | "
    "CatBoost + Uplift Segmentation + SHAP</small></center>",
    unsafe_allow_html=True
)
