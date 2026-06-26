import psycopg2
import pandas as pd

conn = psycopg2.connect(
    host='100.100.224.121',
    port=5433,
    dbname='freedom',
    user='postgres',
    password='admin',
    connect_timeout=10
)
cur = conn.cursor()

# List all tables
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name")
tables = [r[0] for r in cur.fetchall()]
print("=== TABLES ===")
for t in tables:
    print(t)

print()
# For each table: columns + row count + sample
for table in tables:
    print(f"\n=== TABLE: {table} ===")
    cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{table}' AND table_schema='public'")
    cols = cur.fetchall()
    for col in cols:
        print(f"  {col[0]}: {col[1]}")
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    count = cur.fetchone()[0]
    print(f"  ROW COUNT: {count}")
    cur.execute(f"SELECT * FROM {table} LIMIT 3")
    rows = cur.fetchall()
    print(f"  SAMPLE ROWS:")
    for row in rows:
        print(f"    {row}")

conn.close()
print("\nDONE")
