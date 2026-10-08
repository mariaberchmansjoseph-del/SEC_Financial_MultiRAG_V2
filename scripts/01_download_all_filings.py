"""
Script 01: Download SEC filings for all 502 S&P 500 companies.
Downloads 10-K and 10-Q filings from EDGAR.
Skips companies already downloaded.
Run: python scripts/01_download_all_filings.py
"""

import requests
import pandas as pd
import time
import json
import os
from pathlib import Path
from tqdm import tqdm

HEADERS = {
    "User-Agent": "Research Project research@example.com",
    "Accept":     "application/json, text/html"
}

FILINGS_DIR  = Path("data/raw/filings")
CIKS_PATH    = "configs/sp500_with_ciks.csv"
LOG_PATH     = "configs/download_log.csv"

FILINGS_DIR.mkdir(parents=True, exist_ok=True)

# Filing types to download
FORM_TYPES   = ["10-K", "10-Q"]
MAX_FILINGS  = 20   # per company (5 years of 10-K+10-Q)
START_YEAR   = "2020-01-01"
END_YEAR     = "2026-12-31"


def get_company_filings(cik: str) -> list:
    """
    Get list of filings for a company from EDGAR.
    Returns list of filing metadata dicts.
    """
    url = (
        f"https://data.sec.gov/submissions/"
        f"CIK{cik}.json"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return []

        data     = r.json()
        recent   = data.get("filings", {}).get("recent", {})
        forms    = recent.get("form", [])
        dates    = recent.get("filingDate", [])
        accnums  = recent.get("accessionNumber", [])
        docs     = recent.get("primaryDocument", [])

        filings = []
        for form, date, accnum, doc in zip(
            forms, dates, accnums, docs
        ):
            if form in FORM_TYPES and \
                    date >= START_YEAR[:10]:
                filings.append({
                    "form":       form,
                    "date":       date,
                    "accession":  accnum.replace("-", ""),
                    "document":   doc,
                    "cik":        cik,
                })

        return filings[:MAX_FILINGS]

    except Exception:
        return []


def download_filing(
    cik:       str,
    ticker:    str,
    accession: str,
    document:  str,
    form:      str,
    date:      str
) -> bool:
    """Download one filing HTML file."""
    # Create company directory
    company_dir = FILINGS_DIR / ticker
    company_dir.mkdir(exist_ok=True)

    # File name
    filename = (
        f"{ticker}_{form.replace('-','')}_{date}.html"
    )
    filepath = company_dir / filename

    # Skip if already downloaded
    if filepath.exists() and filepath.stat().st_size > 1000:
        return True

    # Build URL
    acc_formatted = (
        accession[:10] + "-" +
        accession[10:12] + "-" +
        accession[12:]
    )
    url = (
        f"https://www.sec.gov/Archives/edgar/full-index/"
        f"data/{int(cik)}/{accession}/{document}"
    )
    # Alternative URL format
    url2 = (
        f"https://www.sec.gov/Archives/edgar/full-index/"
        f"data/{int(cik)}/{accession}/"
        f"{accession}-index.htm"
    )

    for attempt_url in [url, url2]:
        try:
            r = requests.get(
                attempt_url,
                headers=HEADERS,
                timeout=30
            )
            if r.status_code == 200 and len(r.content) > 1000:
                filepath.write_bytes(r.content)
                return True
        except Exception:
            pass

    return False


def load_log() -> dict:
    """Load download log to track progress."""
    try:
        df = pd.read_csv(LOG_PATH)
        return dict(zip(df["ticker"], df["status"]))
    except Exception:
        return {}


def save_log(log: dict):
    """Save download log."""
    df = pd.DataFrame([
        {"ticker": k, "status": v}
        for k, v in log.items()
    ])
    df.to_csv(LOG_PATH, index=False)


def main():
    print("="*60)
    print("DOWNLOADING SEC FILINGS FOR ALL S&P 500")
    print("="*60)
    print()

    # Load companies with CIKs
    df = pd.read_csv(CIKS_PATH)
    df = df[df["cik"] != ""].copy()
    print(f"Companies with CIKs: {len(df)}")

    # Load existing log
    log = load_log()
    already_done = sum(
        1 for v in log.values() if v == "done"
    )
    print(f"Already downloaded: {already_done}")
    print()

    total_downloaded = 0
    total_skipped    = 0
    total_failed     = 0

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Companies"
    ):
        ticker = row["ticker"]
        cik    = str(row["cik"]).zfill(10)
        name   = row.get("name", ticker)

        # Skip if already fully downloaded
        if log.get(ticker) == "done":
            total_skipped += 1
            continue

        # Get filing list
        filings = get_company_filings(cik)
        time.sleep(0.1)

        if not filings:
            log[ticker] = "no_filings"
            total_failed += 1
            continue

        # Download each filing
        company_downloaded = 0
        for filing in filings:
            success = download_filing(
                cik       = cik,
                ticker    = ticker,
                accession = filing["accession"],
                document  = filing["document"],
                form      = filing["form"],
                date      = filing["date"],
            )
            if success:
                company_downloaded += 1
            time.sleep(0.05)

        if company_downloaded > 0:
            log[ticker] = "done"
            total_downloaded += company_downloaded
        else:
            log[ticker] = "failed"
            total_failed += 1

        # Save log every 50 companies
        if len(log) % 50 == 0:
            save_log(log)

    save_log(log)

    # Count files
    total_files = sum(
        len(list(d.glob("*.html")))
        for d in FILINGS_DIR.iterdir()
        if d.is_dir()
    )

    print()
    print("="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Files downloaded:   {total_downloaded:,}")
    print(f"Companies skipped:  {total_skipped:,}")
    print(f"Companies failed:   {total_failed:,}")
    print(f"Total files on disk:{total_files:,}")
    print(f"Location: {FILINGS_DIR}")
    print()
    print("Next: python scripts/02_parse_all_filings.py")


if __name__ == "__main__":
    main()