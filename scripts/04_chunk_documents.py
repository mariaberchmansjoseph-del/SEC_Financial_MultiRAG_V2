"""
Script 04: Chunk parsed documents for RAG indexing.
Hierarchical chunking with section awareness.
Run: python scripts/04_chunk_documents.py
"""

import re
import json
import tiktoken
import pandas as pd
from pathlib import Path
from tqdm import tqdm

Path("data/processed/chunks").mkdir(parents=True, exist_ok=True)

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64
MIN_CHUNK     = 50

enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def chunk_text(text: str) -> list:
    if not text or len(text.strip()) < 50:
        return []

    sentences    = re.split(r'(?<=[.!?])\s+', text)
    sentences    = [s.strip() for s in sentences
                    if len(s.strip()) > 20]
    if not sentences:
        return []

    chunks       = []
    current      = []
    current_toks = 0

    for sentence in sentences:
        sent_toks = count_tokens(sentence)

        if sent_toks > CHUNK_SIZE:
            tokens = enc.encode(sentence)
            for i in range(0, len(tokens),
                           CHUNK_SIZE - CHUNK_OVERLAP):
                ct = enc.decode(tokens[i:i + CHUNK_SIZE])
                if count_tokens(ct) >= MIN_CHUNK:
                    chunks.append(ct)
            continue

        if current_toks + sent_toks > CHUNK_SIZE \
                and current:
            chunk = " ".join(current)
            if count_tokens(chunk) >= MIN_CHUNK:
                chunks.append(chunk)

            overlap_sents = []
            overlap_toks  = 0
            for s in reversed(current):
                st = count_tokens(s)
                if overlap_toks + st <= CHUNK_OVERLAP:
                    overlap_sents.insert(0, s)
                    overlap_toks += st
                else:
                    break
            current      = overlap_sents
            current_toks = overlap_toks

        current.append(sentence)
        current_toks += sent_toks

    if current:
        chunk = " ".join(current)
        if count_tokens(chunk) >= MIN_CHUNK:
            chunks.append(chunk)

    return chunks


def classify_section(text: str) -> str:
    t = text.lower()
    risk_words = ["risk", "risks", "uncertainty",
                  "adverse", "litigation", "failure",
                  "volatile", "competition", "regulatory",
                  "threat", "challenge", "exposure"]
    mda_words  = ["revenue", "income", "margin",
                  "operating", "growth", "increased",
                  "decreased", "million", "billion",
                  "compared", "results", "quarter"]
    fin_words  = ["balance sheet", "cash flow",
                  "earnings per share", "total assets",
                  "stockholders equity", "net income",
                  "gross profit", "operating income"]

    risk_score = sum(1 for w in risk_words if w in t)
    mda_score  = sum(1 for w in mda_words  if w in t)
    fin_score  = sum(1 for w in fin_words  if w in t)

    if fin_score >= 3:
        return "financials"
    if risk_score >= 3:
        return "risk_factors"
    if mda_score >= 4:
        return "mda"
    return "general"


def chunk_all():
    parsed_path = "data/processed/parsed/documents.csv"
    if not Path(parsed_path).exists():
        print(f"Not found: {parsed_path}")
        return

    df = pd.read_csv(parsed_path, encoding="utf-8-sig")
    print(f"Documents to chunk: {len(df):,}")

    all_chunks = []
    chunk_id   = 0
    skipped    = 0

    for _, row in tqdm(df.iterrows(),
                       total=len(df),
                       desc="Chunking"):

        full_text = str(row.get("full_text", ""))
        if len(full_text) < 100:
            skipped += 1
            continue

        # Chunk full text
        text_chunks = chunk_text(full_text)

        for i, chunk in enumerate(text_chunks):
            section = classify_section(chunk)
            all_chunks.append({
                "chunk_id":     f"chunk_{chunk_id:07d}",
                "company":      str(row.get("company", "")),
                "ticker":       str(row.get("ticker", "")),
                "sector":       str(row.get("sector", "")),
                "sub_sector":   str(row.get("sub_sector", "")),
                "form_type":    str(row.get("form_type", "")),
                "filing_date":  str(row.get("filing_date", "")),
                "year":         str(row.get("year", "")),
                "section":      section,
                "chunk_index":  i,
                "total_chunks": len(text_chunks),
                "token_count":  count_tokens(chunk),
                "text":         chunk,
            })
            chunk_id += 1

    # Save JSONL
    jsonl_path = "data/processed/chunks/all_chunks.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk,
                               ensure_ascii=False) + "\n")

    # Save summary CSV
    summary_df = pd.DataFrame([{
        k: v for k, v in c.items() if k != "text"
    } for c in all_chunks])
    summary_df.to_csv(
        "data/processed/chunks/chunks_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    print(f"\n{'='*55}")
    print(f"CHUNKING COMPLETE")
    print(f"{'='*55}")
    print(f"Total chunks:      {len(all_chunks):,}")
    print(f"Skipped docs:      {skipped}")
    print(f"Avg per document:  "
          f"{len(all_chunks)/max(len(df)-skipped,1):.0f}")
    print(f"\nChunks by section:")
    from collections import Counter
    sections = Counter(c["section"] for c in all_chunks)
    for sec, cnt in sections.most_common():
        print(f"  {sec:<15} {cnt:>8,}")
    print(f"\nChunks by sector:")
    sectors = Counter(c["sector"] for c in all_chunks)
    for sec, cnt in sectors.most_common():
        print(f"  {sec:<25} {cnt:>8,}")
    print(f"\nSaved: {jsonl_path}")
    print(f"\nNext: python scripts/05_index_documents.py")


if __name__ == "__main__":
    print("="*55)
    print("SEC FILING CHUNKER")
    print("="*55)
    chunk_all()