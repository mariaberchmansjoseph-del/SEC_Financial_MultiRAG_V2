"""
Script 06: Build structured financial metrics database.
Reads yfinance JSON files and loads into SQLite.
Run: python scripts/06_build_metrics_db.py
"""

import os
import json
import sqlite3
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DB_PATH    = "data/processed/metrics.db"
PRICES_DIR = Path("data/raw/prices")

Path("data/processed").mkdir(parents=True, exist_ok=True)


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS company_overview (
            ticker          TEXT PRIMARY KEY,
            name            TEXT,
            sector          TEXT,
            industry        TEXT,
            description     TEXT,
            market_cap      REAL,
            pe_ratio        REAL,
            eps             REAL,
            revenue_ttm     REAL,
            profit_margin   REAL,
            week_52_high    REAL,
            week_52_low     REAL,
            analyst_target  REAL,
            beta            REAL,
            employees       INTEGER,
            updated_at      TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS financial_metrics (
            id              INTEGER PRIMARY KEY,
            ticker          TEXT,
            period_date     TEXT,
            total_revenue   REAL,
            net_income      REAL,
            gross_profit    REAL,
            operating_income REAL,
            ebitda          REAL,
            eps             REAL,
            UNIQUE(ticker, period_date)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id          INTEGER PRIMARY KEY,
            ticker      TEXT,
            date        TEXT,
            close_price REAL,
            volume      INTEGER,
            UNIQUE(ticker, date)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyst_summary (
            ticker          TEXT PRIMARY KEY,
            analyst_target  REAL,
            pe_ratio        REAL,
            revenue_ttm     REAL,
            profit_margin   REAL,
            market_cap      REAL,
            beta            REAL
        )
    """)

    conn.commit()
    return conn


def load_company(
    conn: sqlite3.Connection,
    json_path: Path
) -> dict:
    """Load one company JSON into database."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    ticker   = data.get("ticker", "")
    name     = data.get("name", "")
    overview = data.get("overview", {})
    prices   = data.get("prices", {})
    fin      = data.get("financials", {})

    if not ticker:
        return {"status": "skip", "reason": "no ticker"}

    # Company overview
    try:
        conn.execute("""
            INSERT OR REPLACE INTO company_overview
            (ticker, name, sector, industry,
             description, market_cap, pe_ratio, eps,
             revenue_ttm, profit_margin,
             week_52_high, week_52_low,
             analyst_target, beta, employees,
             updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                    datetime('now'))
        """, (
            ticker,
            overview.get("name", name),
            overview.get("sector", ""),
            overview.get("industry", ""),
            overview.get("description", "")[:300],
            overview.get("market_cap"),
            overview.get("pe_ratio"),
            overview.get("eps"),
            overview.get("revenue_ttm"),
            overview.get("profit_margin"),
            overview.get("52_week_high"),
            overview.get("52_week_low"),
            overview.get("analyst_target"),
            overview.get("beta"),
            overview.get("employees"),
        ))
    except Exception as e:
        pass

    # Financial metrics
    for period, metrics in fin.items():
        if not isinstance(metrics, dict):
            continue
        try:
            conn.execute("""
                INSERT OR IGNORE INTO financial_metrics
                (ticker, period_date, total_revenue,
                 net_income, gross_profit,
                 operating_income, ebitda, eps)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                ticker,
                period[:10],
                metrics.get("Total Revenue"),
                metrics.get("Net Income"),
                metrics.get("Gross Profit"),
                metrics.get("Operating Income"),
                metrics.get("EBITDA"),
                metrics.get("Basic EPS"),
            ))
        except Exception:
            pass

    # Price history
    for date, price_data in prices.items():
        try:
            conn.execute("""
                INSERT OR IGNORE INTO price_history
                (ticker, date, close_price, volume)
                VALUES (?,?,?,?)
            """, (
                ticker,
                date[:10],
                price_data.get("close"),
                price_data.get("volume"),
            ))
        except Exception:
            pass

    conn.commit()
    return {"status": "ok", "ticker": ticker}


def build_metrics_db():
    print("=" * 55)
    print("BUILDING FINANCIAL METRICS DATABASE")
    print("=" * 55)

    json_files = list(PRICES_DIR.glob("*_market.json"))
    print(f"Files to process: {len(json_files)}")

    conn    = init_db()
    success = 0
    failed  = 0

    for json_path in tqdm(json_files, desc="Loading"):
        result = load_company(conn, json_path)
        if result.get("status") == "ok":
            success += 1
        else:
            failed += 1

    # Print summary
    print(f"\n{'='*55}")
    print(f"DATABASE COMPLETE")
    print(f"{'='*55}")

    overview_count = conn.execute(
        "SELECT COUNT(*) FROM company_overview"
    ).fetchone()[0]
    metrics_count  = conn.execute(
        "SELECT COUNT(*) FROM financial_metrics"
    ).fetchone()[0]
    prices_count   = conn.execute(
        "SELECT COUNT(*) FROM price_history"
    ).fetchone()[0]

    print(f"Companies:         {overview_count}")
    print(f"Financial periods: {metrics_count}")
    print(f"Price records:     {prices_count:,}")

    # Show sample data
    print(f"\nSample — Top 5 by market cap:")
    rows = conn.execute("""
        SELECT ticker, name, market_cap, pe_ratio
        FROM company_overview
        WHERE market_cap IS NOT NULL
        ORDER BY market_cap DESC
        LIMIT 5
    """).fetchall()
    for r in rows:
        mc = f"${r[2]/1e12:.1f}T" \
             if r[2] and r[2] > 1e12 \
             else f"${r[2]/1e9:.0f}B" if r[2] else "N/A"
        print(f"  {r[0]:<6} {r[1]:<25} "
              f"MCap: {mc:<8} P/E: {r[3]}")

    conn.close()
    print(f"\nDatabase: {DB_PATH}")
    print(f"\nNext: python scripts/07_build_eval_set.py")


if __name__ == "__main__":
    build_metrics_db()