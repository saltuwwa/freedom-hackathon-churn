import psycopg2
import pandas as pd

conn = psycopg2.connect(
    host='100.100.224.121', port=5433, dbname='freedom',
    user='postgres', password='admin', connect_timeout=10
)
cur = conn.cursor()

def q(sql):
    cur.execute(sql)
    return cur.fetchall()

print("=== DATE RANGES ===")
print("transactions:", q("SELECT MIN(date), MAX(date) FROM transactions"))
print("events:", q("SELECT MIN(event_timestamp), MAX(event_timestamp) FROM events"))
print("products:", q("SELECT MIN(purchase_date), MAX(purchase_date) FROM products"))
print("users:", q("SELECT MIN(registration_date), MAX(registration_date) FROM users"))

print("\n=== EVENT TYPES (top 20) ===")
rows = q("SELECT event_type, COUNT(*) as cnt FROM events GROUP BY event_type ORDER BY cnt DESC LIMIT 20")
for r in rows:
    print(f"  {r[0]}: {r[1]:,}")

print("\n=== EVENT STATUS values ===")
rows = q("SELECT status, COUNT(*) FROM events GROUP BY status ORDER BY COUNT(*) DESC")
for r in rows:
    print(f"  {r[0]}: {r[1]:,}")

print("\n=== TRANSACTION categories ===")
rows = q("SELECT category, COUNT(*) FROM transactions GROUP BY category ORDER BY COUNT(*) DESC")
for r in rows:
    print(f"  {r[0]}: {r[1]:,}")

print("\n=== TRANSACTION status values ===")
rows = q("SELECT status, COUNT(*) FROM transactions GROUP BY status ORDER BY COUNT(*) DESC LIMIT 20")
for r in rows:
    print(f"  {r[0]}: {r[1]:,}")

print("\n=== TRANSACTION terminal_type ===")
rows = q("SELECT terminal_type, COUNT(*) FROM transactions GROUP BY terminal_type ORDER BY COUNT(*) DESC")
for r in rows:
    print(f"  {r[0]}: {r[1]:,}")

print("\n=== USER SEGMENTS distribution ===")
rows = q("SELECT segment, COUNT(*) FROM user_segments GROUP BY segment ORDER BY COUNT(*) DESC")
for r in rows:
    print(f"  {r[0]}: {r[1]:,}")

print("\n=== PRODUCTS types ===")
rows = q("SELECT product_type, COUNT(*) FROM products GROUP BY product_type ORDER BY COUNT(*) DESC")
for r in rows:
    print(f"  {r[0]}: {r[1]:,}")

print("\n=== ACQUISITION channels ===")
rows = q("SELECT channel, COUNT(*) FROM acquisition GROUP BY channel ORDER BY COUNT(*) DESC")
for r in rows:
    print(f"  {r[0]}: {r[1]:,}")

print("\n=== FEATURE STORE sample stats ===")
rows = q("""
    SELECT 
        COUNT(*) as users,
        AVG(tx_cnt_7d) as avg_tx_7d,
        AVG(tx_vol_7d) as avg_vol_7d,
        SUM(CASE WHEN tx_cnt_7d = 0 AND evt_cnt_7d = 0 THEN 1 ELSE 0 END) as fully_inactive_7d,
        SUM(CASE WHEN tx_cnt_14d = 0 THEN 1 ELSE 0 END) as no_tx_14d,
        AVG(proxy_ltv_90d) as avg_pltv
    FROM feature_store
""")
for r in rows:
    print(f"  {r}")

print("\n=== CHURN LABEL ESTIMATION ===")
# Users with last event/tx > 30 days ago from max date
rows = q("""
    WITH last_activity AS (
        SELECT user_id, MAX(event_timestamp) as last_event
        FROM events
        GROUP BY user_id
    ),
    last_tx AS (
        SELECT user_id, MAX(date) as last_tx
        FROM transactions
        GROUP BY user_id
    ),
    combined AS (
        SELECT 
            COALESCE(la.user_id, lt.user_id) as user_id,
            la.last_event,
            lt.last_tx,
            GREATEST(la.last_event, lt.last_tx) as last_any
        FROM last_activity la
        FULL OUTER JOIN last_tx lt ON la.user_id = lt.user_id
    ),
    ref AS (SELECT MAX(last_any) as max_date FROM combined)
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN last_any < ref.max_date - INTERVAL '30 days' THEN 1 ELSE 0 END) as churned_30d,
        SUM(CASE WHEN last_any < ref.max_date - INTERVAL '14 days' THEN 1 ELSE 0 END) as churned_14d,
        MAX(last_any) as reference_date
    FROM combined, ref
""")
for r in rows:
    print(f"  total: {r[0]:,}, churned_30d: {r[1]:,}, churned_14d: {r[2]:,}, ref_date: {r[3]}")

conn.close()
print("\nDONE")
