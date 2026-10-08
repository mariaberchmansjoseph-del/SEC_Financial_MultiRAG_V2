import sqlite3

conn = sqlite3.connect("data/processed/metrics.db")

tables = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()

print("Tables:", [t[0] for t in tables])

for t in [r[0] for r in tables]:
    n = conn.execute(
        f"SELECT COUNT(*) FROM {t}"
    ).fetchone()[0]
    print(f"  {t}: {n} rows")

tickers = conn.execute(
    "SELECT DISTINCT ticker FROM company_overview LIMIT 10"
).fetchall()
print("Sample tickers:", [t[0] for t in tickers])

conn.close()