"""
Script 02: Collect structured financial data.
EdgarTools for XBRL financials + yfinance for market data.
Run: python scripts/02_collect_structured.py
"""

import os
import json
import time
import yaml
import yfinance as yf
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from edgar import Company, set_identity

set_identity("Research Project researcher@project.com")

Path("data/raw/structured").mkdir(parents=True, exist_ok=True)
Path("data/raw/prices").mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with open("configs/companies_100.yaml") as f:
        return yaml.safe_load(f)


def collect_yfinance(ticker: str, name: str) -> dict:
    """Collect market data via yfinance."""
    try:
        stock    = yf.Ticker(ticker)
        info     = stock.info

        overview = {
            "name":           info.get("longName", name),
            "sector":         info.get("sector", ""),
            "industry":       info.get("industry", ""),
            "description":    info.get(
                "longBusinessSummary", ""
            )[:500],
            "market_cap":     info.get("marketCap"),
            "pe_ratio":       info.get("trailingPE"),
            "eps":            info.get("trailingEps"),
            "revenue_ttm":    info.get("totalRevenue"),
            "profit_margin":  info.get("profitMargins"),
            "52_week_high":   info.get("fiftyTwoWeekHigh"),
            "52_week_low":    info.get("fiftyTwoWeekLow"),
            "analyst_target": info.get("targetMeanPrice"),
            "beta":           info.get("beta"),
            "employees":      info.get("fullTimeEmployees"),
        }

        # 5 years monthly prices
        hist   = stock.history(period="5y", interval="1mo")
        prices = {}
        for date, row in hist.iterrows():
            prices[str(date)[:10]] = {
                "close":  round(float(row.get("Close", 0)), 2),
                "volume": int(row.get("Volume", 0)),
            }

        # Annual financials
        inc      = stock.financials
        fin_data = {}
        if inc is not None and not inc.empty:
            for col in inc.columns[:4]:
                year = str(col)[:10]
                fin_data[year] = {}
                for row_name in inc.index:
                    val = inc.loc[row_name, col]
                    try:
                        fin_data[year][str(row_name)] = \
                            float(val) if val == val else None
                    except Exception:
                        pass

        return {
            "ticker":     ticker,
            "name":       name,
            "overview":   overview,
            "prices":     prices,
            "financials": fin_data,
            "collected":  datetime.utcnow().isoformat()
        }

    except Exception as e:
        print(f"  yfinance error {ticker}: {str(e)[:60]}")
        return {"ticker": ticker, "name": name,
                "error": str(e)[:100]}


def collect_edgartools(ticker: str, name: str) -> dict:
    """Collect structured financials via EdgarTools."""
    try:
        company  = Company(ticker)
        filings  = company.get_filings(form="10-K")
        if not filings:
            return {}

        filing_data = []
        for filing in filings[:4]:
            try:
                data = {
                    "period": str(
                        getattr(filing,
                                "period_of_report", "")
                    ),
                    "filed":  str(
                        getattr(filing, "filed", "")
                    ),
                    "form":   filing.form,
                }
                filing_data.append(data)
                time.sleep(0.3)
            except Exception:
                pass

        return {
            "ticker":   ticker,
            "name":     name,
            "filings":  filing_data,
            "collected": datetime.utcnow().isoformat()
        }

    except Exception as e:
        return {"ticker": ticker, "error": str(e)[:100]}


def run():
    config    = load_config()
    companies = config["companies"]

    print("="*60)
    print("STRUCTURED DATA COLLECTION")
    print("="*60)
    print(f"Companies: {len(companies)}")
    print()

    yf_success = 0
    ed_success = 0

    for company in tqdm(companies, desc="Collecting"):
        ticker = company["ticker"]
        name   = company["name"]

        # yfinance
        yf_data = collect_yfinance(ticker, name)
        yf_path = f"data/raw/prices/{ticker}_market.json"
        with open(yf_path, "w") as f:
            json.dump(yf_data, f, indent=2, default=str)
        if "error" not in yf_data:
            yf_success += 1

        # EdgarTools
        ed_data = collect_edgartools(ticker, name)
        ed_path = f"data/raw/structured/{ticker}_edgar.json"
        with open(ed_path, "w") as f:
            json.dump(ed_data, f, indent=2, default=str)
        if "error" not in ed_data and ed_data:
            ed_success += 1

        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"yfinance:   {yf_success}/{len(companies)}")
    print(f"EdgarTools: {ed_success}/{len(companies)}")
    print(f"\nNext: python scripts/03_parse_documents.py")


if __name__ == "__main__":
    run()