"""
Script 00: Get SEC CIK numbers for all 503 S&P 500 companies.
Downloads the SEC master company tickers file and matches
each S&P 500 ticker to its CIK number.
Run: python scripts/00_get_all_ciks.py
"""

import requests
import pandas as pd
from pathlib import Path

Path("configs").mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Research Project research@example.com",
    "Accept":     "application/json"
}


def get_sp500_tickers() -> pd.DataFrame:
    """Load S&P 500 tickers from SkData or Wikipedia."""

    # Try SkData first
    try:
        df    = pd.read_csv("../SkData_AI/configs/all_tickers.csv")
        sp500 = df[df["market"] == "SP500"].copy()
        if len(sp500) > 400:
            print(f"  Loaded {len(sp500)} tickers from SkData")
            return sp500
    except Exception:
        pass

    # Try local all_tickers if exists
    try:
        df    = pd.read_csv("configs/all_tickers.csv")
        sp500 = df[df["market"] == "SP500"].copy()
        if len(sp500) > 400:
            print(f"  Loaded {len(sp500)} tickers from local config")
            return sp500
    except Exception:
        pass

    # Fallback: Wikipedia
    print("  Fetching from Wikipedia...")
    headers = {
        "User-Agent": "Mozilla/5.0 Chrome/120.0.0.0"
    }
    url = (
        "https://en.wikipedia.org/wiki/"
        "List_of_S%26P_500_companies"
    )
    r      = requests.get(url, headers=headers, timeout=30)
    tables = pd.read_html(r.text)
    df     = tables[0][["Symbol", "Security",
                         "GICS Sector"]].copy()
    df.columns = ["ticker", "name", "sector"]
    df["ticker"] = df["ticker"].str.replace(
        ".", "-", regex=False
    )
    df["market"] = "SP500"
    print(f"  Loaded {len(df)} tickers from Wikipedia")
    return df


def download_sec_tickers() -> dict:
    """
    Download SEC master company tickers file.
    Maps ticker → CIK for all SEC-registered companies.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    print(f"  Downloading from {url}...")

    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()

    data = r.json()
    print(f"  SEC has {len(data):,} companies")

    # Build ticker → CIK lookup
    mapping = {}
    for _, company in data.items():
        ticker = company.get("ticker", "").upper()
        cik    = str(company.get("cik_str", "")).zfill(10)
        name   = company.get("title", "")
        if ticker:
            mapping[ticker] = {"cik": cik, "name": name}

    return mapping


def match_tickers_to_ciks(
    sp500_df:   pd.DataFrame,
    sec_mapping: dict
) -> pd.DataFrame:
    """Match S&P 500 tickers to SEC CIK numbers."""
    results = []

    for _, row in sp500_df.iterrows():
        ticker = str(row["ticker"]).upper()
        name   = row.get("name", "")
        sector = row.get("sector", "")

        # Try multiple ticker formats
        candidates = [
            ticker,
            ticker.replace("-", "."),
            ticker.replace(".", "-"),
            ticker.split("-")[0],
            ticker.split(".")[0],
        ]

        cik      = ""
        sec_name = ""

        for candidate in candidates:
            match = sec_mapping.get(candidate)
            if match:
                cik      = match["cik"]
                sec_name = match["name"]
                break

        results.append({
            "ticker":   row["ticker"],
            "name":     name or sec_name,
            "sector":   sector,
            "cik":      cik,
            "sec_name": sec_name,
            "found":    bool(cik),
        })

    return pd.DataFrame(results)


def main():
    print("="*60)
    print("GETTING CIK NUMBERS FOR ALL S&P 500 COMPANIES")
    print("="*60)
    print()

    # Step 1: Load S&P 500 tickers
    print("Step 1: Loading S&P 500 tickers...")
    sp500 = get_sp500_tickers()
    print(f"  Companies: {len(sp500)}")
    print()

    # Step 2: Download SEC master file
    print("Step 2: Downloading SEC company database...")
    sec_mapping = download_sec_tickers()
    print()

    # Step 3: Match tickers to CIKs
    print("Step 3: Matching tickers to CIK numbers...")
    result_df = match_tickers_to_ciks(sp500, sec_mapping)

    found   = result_df["found"].sum()
    missing = len(result_df) - found
    print(f"  Found:   {found}")
    print(f"  Missing: {missing}")
    print()

    # Save results
    result_df.to_csv(
        "configs/sp500_with_ciks.csv", index=False
    )
    print("Saved: configs/sp500_with_ciks.csv")
    print()

    # Show sample
    print("Sample matches:")
    for _, row in result_df[result_df["found"]].head(10).iterrows():
        print(
            f"  {row['ticker']:<8} "
            f"CIK:{row['cik']}  "
            f"{row['name'][:35]}"
        )

    if missing > 0:
        print(f"\nMissing CIKs ({missing} companies):")
        for _, row in result_df[
            ~result_df["found"]
        ].iterrows():
            print(f"  {row['ticker']:<8} {row['name']}")

    print()
    print("="*60)
    print("Next: python scripts/01_download_all_filings.py")


if __name__ == "__main__":
    main()