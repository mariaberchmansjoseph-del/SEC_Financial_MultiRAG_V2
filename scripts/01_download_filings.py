"""
Script 01: Download SEC filings for all 100 companies.
Uses EDGAR API — free, no key required.
Run: python scripts/01_download_filings.py
"""

import os
import time
import yaml
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ── SETUP ─────────────────────────────────────────────────────
Path("data/raw/filings").mkdir(parents=True, exist_ok=True)
Path("data/raw/metadata").mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Research Project researcher@project.com",
    "Accept-Encoding": "gzip, deflate",
}
BASE_URL  = "https://data.sec.gov"
EDGAR_URL = "https://www.sec.gov"


def load_config() -> dict:
    with open("configs/companies_100.yaml") as f:
        return yaml.safe_load(f)


def get_submissions(cik: str) -> dict:
    url = f"{BASE_URL}/submissions/CIK{cik.zfill(10)}.json"
    try:
        time.sleep(0.5)
        r = requests.get(url, headers=HEADERS, timeout=30)
        return r.json() if r.status_code == 200 else {}
    except Exception as e:
        print(f"    Submissions error: {e}")
        return {}


def get_filings_list(
    submissions: dict,
    form_types:  list,
    start_year:  int,
    end_year:    int
) -> list:
    recent = submissions.get(
        "filings", {}
    ).get("recent", {})
    if not recent:
        return []

    forms   = recent.get("form", [])
    dates   = recent.get("filingDate", [])
    accs    = recent.get("accessionNumber", [])
    pdocs   = recent.get("primaryDocument", [])

    filings = []
    for i, form in enumerate(forms):
        if form not in form_types:
            continue
        date = dates[i] if i < len(dates) else ""
        if not date:
            continue
        year = int(date[:4])
        if year < start_year or year > end_year:
            continue
        filings.append({
            "form_type":   form,
            "filing_date": date,
            "accession":   accs[i]  if i < len(accs)  else "",
            "primary_doc": pdocs[i] if i < len(pdocs) else "",
        })
    return filings


def download_document(
    cik: str, accession: str,
    primary_doc: str, save_path: Path
) -> bool:
    if save_path.exists() and \
       save_path.stat().st_size > 1000:
        return True

    acc_clean = accession.replace("-", "")
    cik_int   = int(cik)

    # Try primary document
    if primary_doc:
        url = (f"{EDGAR_URL}/Archives/edgar/data/"
               f"{cik_int}/{acc_clean}/{primary_doc}")
        try:
            time.sleep(0.5)
            r = requests.get(
                url, headers=HEADERS, timeout=30
            )
            if r.status_code == 200 and \
               len(r.content) > 1000:
                save_path.write_bytes(r.content)
                return True
        except Exception:
            pass

    # Fallback: fetch index
    idx_url = (f"{EDGAR_URL}/Archives/edgar/data/"
               f"{cik_int}/{acc_clean}/"
               f"{acc_clean}-index.json")
    try:
        time.sleep(0.5)
        r = requests.get(
            idx_url, headers=HEADERS, timeout=30
        )
        if r.status_code == 200:
            items = r.json().get(
                "directory", {}
            ).get("item", [])
            for item in items:
                name = item.get("name", "")
                if (name.lower().endswith(
                        (".htm", ".html")) and
                        "exhibit" not in name.lower() and
                        not name.lower().startswith("ex")):
                    doc_url = (
                        f"{EDGAR_URL}/Archives/edgar/data/"
                        f"{cik_int}/{acc_clean}/{name}"
                    )
                    time.sleep(0.5)
                    doc_r = requests.get(
                        doc_url, headers=HEADERS,
                        timeout=30
                    )
                    if doc_r.status_code == 200 and \
                       len(doc_r.content) > 1000:
                        save_path.write_bytes(doc_r.content)
                        return True
    except Exception:
        pass

    return False


def download_all(config: dict) -> pd.DataFrame:
    companies  = config["companies"]
    form_types = config["filing_types"]
    start_year = config["years"]["start"]
    end_year   = config["years"]["end"]

    print("=" * 60)
    print("SEC EDGAR FILING DOWNLOADER v2")
    print("=" * 60)
    print(f"Companies:    {len(companies)}")
    print(f"Filing types: {form_types}")
    print(f"Years:        {start_year} - {end_year}")
    print()

    all_meta = []
    failed   = []

    for company in tqdm(companies, desc="Companies"):
        name      = company["name"]
        cik       = company["cik"]
        ticker    = company["ticker"]
        sector    = company["sector"]
        sub_sec   = company.get("sub_sector", "")

        subs = get_submissions(cik)
        if not subs:
            failed.append(name)
            continue

        filings = get_filings_list(
            subs, form_types, start_year, end_year
        )

        count = 0
        for filing in filings:
            form_type   = filing["form_type"]
            filing_date = filing["filing_date"]
            accession   = filing["accession"]
            primary_doc = filing["primary_doc"]

            if not accession:
                continue

            safe_name = name.replace(
                " ", "_"
            ).replace("/", "_").replace("&", "and")
            save_dir  = (Path("data/raw/filings")
                         / sector / safe_name)
            save_dir.mkdir(parents=True, exist_ok=True)
            filename  = (f"{safe_name}_"
                         f"{form_type}_"
                         f"{filing_date}.htm")
            save_path = save_dir / filename

            ok = download_document(
                cik, accession, primary_doc, save_path
            )

            if ok:
                size_kb = save_path.stat().st_size / 1024
                count  += 1
                all_meta.append({
                    "company":      name,
                    "ticker":       ticker,
                    "sector":       sector,
                    "sub_sector":   sub_sec,
                    "cik":          cik,
                    "form_type":    form_type,
                    "filing_date":  filing_date,
                    "year":         filing_date[:4],
                    "accession":    accession,
                    "file_path":    str(save_path),
                    "file_size_kb": round(size_kb, 1),
                    "downloaded_at": datetime.utcnow()
                                     .isoformat()
                })

        tqdm.write(f"  {name}: {count} filings")
        if count == 0:
            failed.append(name)

        time.sleep(0.3)

    df = pd.DataFrame(all_meta)
    meta_path = "data/raw/metadata/filings_metadata.csv"
    df.to_csv(meta_path, index=False,
              encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Total downloaded: {len(df):,}")
    if len(df) > 0:
        print(f"Total size: "
              f"{df['file_size_kb'].sum()/1024:.1f} MB")
        print(f"\nBy sector:")
        print(df.groupby("sector")["ticker"]
              .nunique().to_string())
        print(f"\nBy form type:")
        print(df.groupby("form_type")["ticker"]
              .count().to_string())
    if failed:
        print(f"\nFailed ({len(failed)}): {failed}")
    print(f"\nMetadata: {meta_path}")
    print(f"\nNext: python scripts/02_collect_structured.py")

    return df


if __name__ == "__main__":
    config = load_config()
    download_all(config)