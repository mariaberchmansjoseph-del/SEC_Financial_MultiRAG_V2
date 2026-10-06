"""
Script 03: Parse SEC iXBRL filings into clean text.
Improved hierarchical section extraction.
Run: python scripts/03_parse_documents.py
"""

import re
import json
import warnings
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

Path("data/processed/parsed").mkdir(parents=True, exist_ok=True)


def extract_text(content: bytes) -> str:
    """Extract readable text from iXBRL filing."""
    try:
        text_content = content.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(text_content, "lxml")

        for tag in soup(["script", "style", "meta",
                         "link", "head"]):
            tag.decompose()
        for tag in soup.find_all("ix:header"):
            tag.decompose()
        for tag in soup.find_all("ix:hidden"):
            tag.decompose()

        full_text = soup.get_text(separator=" ", strip=True)

        # Clean XBRL artifacts
        full_text = re.sub(r'https?://\S+', ' ', full_text)
        full_text = re.sub(r'[a-z]+:[A-Za-z]+Member\b',
                           ' ', full_text)
        full_text = re.sub(r'iso4217:[A-Z]+', ' ', full_text)
        full_text = re.sub(r'xbrli?:[a-zA-Z]+', ' ', full_text)
        full_text = re.sub(r'us-gaap:[A-Za-z]+', ' ', full_text)
        full_text = re.sub(r'\s+', ' ', full_text).strip()

        return full_text
    except Exception:
        return ""


def score_quality(text: str) -> float:
    """Score text quality 0-1."""
    if not text or len(text) < 100:
        return 0.0
    words     = text.split()
    good      = sum(1 for w in words
                    if sum(c.isalpha() for c in w)
                    > len(w) * 0.5)
    return good / max(len(words), 1)


def find_section(text: str, section: str) -> str:
    """Find a named section in filing text."""
    patterns = {
        "item_1a": [
            r"item\s*1a\.?\s*risk\s*factor",
            r"risk\s*factors\s*(?:our|the)\s*business",
        ],
        "item_7": [
            r"item\s*7\.?\s*management",
            r"management.s\s*discussion\s*and\s*analysis",
            r"results\s*of\s*operations",
        ],
        "item_1": [
            r"item\s*1\.?\s*business\s*(?:overview|description)?",
        ],
        "item_8": [
            r"item\s*8\.?\s*financial\s*statement",
        ],
    }

    text_lower = text.lower()
    for pattern in patterns.get(section, []):
        matches = list(re.finditer(pattern, text_lower))
        for match in matches:
            start   = match.start()
            preview = text_lower[start:start + 80]
            # Skip table of contents
            if re.search(r'\d{1,3}\s*$', preview.strip()):
                continue
            content = text[start:start + 15000]
            content = re.sub(r'\s+', ' ', content).strip()
            if len(content) > 300:
                return content
    return ""


def parse_all():
    meta_path = "data/raw/metadata/filings_metadata.csv"
    if not Path(meta_path).exists():
        print(f"Not found: {meta_path}")
        return

    df = pd.read_csv(meta_path, encoding="utf-8-sig")
    print(f"Total filings: {len(df)}")

    all_docs    = []
    failed      = 0
    low_quality = 0

    for _, row in tqdm(df.iterrows(),
                       total=len(df),
                       desc="Parsing"):
        file_path = Path(row["file_path"])
        if not file_path.exists():
            failed += 1
            continue

        content = file_path.read_bytes()
        if len(content) < 1000:
            failed += 1
            continue

        full_text = extract_text(content)
        if not full_text:
            failed += 1
            continue

        quality = score_quality(full_text)
        if quality < 0.4:
            low_quality += 1

        item_1a = find_section(full_text, "item_1a")
        item_7  = find_section(full_text, "item_7")
        item_1  = find_section(full_text, "item_1")
        item_8  = find_section(full_text, "item_8")

        all_docs.append({
            "company":      row["company"],
            "ticker":       row["ticker"],
            "sector":       row["sector"],
            "sub_sector":   row.get("sub_sector", ""),
            "form_type":    row["form_type"],
            "filing_date":  row["filing_date"],
            "year":         row["year"],
            "file_path":    str(file_path),
            "full_text":    full_text[:500000],
            "word_count":   len(full_text.split()),
            "quality":      round(quality, 3),
            "item_1":       item_1,
            "item_1a":      item_1a,
            "item_7":       item_7,
            "item_8":       item_8,
            "has_item_1a":  len(item_1a) > 300,
            "has_item_7":   len(item_7) > 300,
        })

    result_df = pd.DataFrame(all_docs)
    out_path  = "data/processed/parsed/documents.csv"
    result_df.to_csv(out_path, index=False,
                     encoding="utf-8-sig")

    print(f"\n{'='*55}")
    print(f"PARSING COMPLETE")
    print(f"{'='*55}")
    print(f"Parsed:      {len(result_df):,}")
    print(f"Failed:      {failed}")
    print(f"Low quality: {low_quality}")
    print(f"\nWord count:")
    print(result_df["word_count"].describe()
          .round(0).to_string())
    print(f"\nSection coverage:")
    print(f"  Item 1A: {result_df['has_item_1a'].sum()}"
          f" / {len(result_df)}")
    print(f"  Item 7:  {result_df['has_item_7'].sum()}"
          f" / {len(result_df)}")
    print(f"\nBy sector:")
    print(result_df.groupby("sector")["company"]
          .nunique().to_string())
    print(f"\nSaved: {out_path}")
    print(f"\nNext: python scripts/04_chunk_documents.py")


if __name__ == "__main__":
    print("="*55)
    print("SEC FILING PARSER")
    print("="*55)
    parse_all()