"""
Script 05: Index chunks into ChromaDB collections.
Uses separate collections per section type.
Run: python scripts/05_index_documents.py
"""

import json
import chromadb
from pathlib import Path
from tqdm import tqdm
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction
)

CHUNKS_PATH  = "data/processed/chunks/all_chunks.jsonl"
VECTORSTORE  = "data/vectorstore"

EMBED_MODEL  = "BAAI/bge-base-en-v1.5"
BATCH_SIZE   = 100

# Separate collections per section type
COLLECTIONS = {
    "risk_factors": "sec_risk_factors",
    "mda":          "sec_mda",
    "financials":   "sec_financials",
    "general":      "sec_general",
}

Path(VECTORSTORE).mkdir(parents=True, exist_ok=True)


def load_chunks() -> list:
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def is_prose(text: str) -> bool:
    if not text or len(text) < 30:
        return False
    digits = sum(1 for c in text if c.isdigit())
    return digits / max(len(text), 1) < 0.25


def build_indexes(chunks: list):
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    client = chromadb.PersistentClient(path=VECTORSTORE)

    # Create collections
    collections = {}
    for section, coll_name in COLLECTIONS.items():
        try:
            client.delete_collection(coll_name)
        except Exception:
            pass
        collections[section] = client.create_collection(
            name               = coll_name,
            embedding_function = embed_fn,
            metadata           = {"hnsw:space": "cosine"}
        )
        print(f"  Created: {coll_name}")

    # Split chunks by section
    section_chunks = {s: [] for s in COLLECTIONS}
    for chunk in chunks:
        section = chunk.get("section", "general")
        if section not in section_chunks:
            section = "general"
        if is_prose(chunk.get("text", "")):
            section_chunks[section].append(chunk)

    # Index each collection
    total_indexed = 0
    for section, coll_name in COLLECTIONS.items():
        coll        = collections[section]
        sec_chunks  = section_chunks[section]
        print(f"\nIndexing {coll_name}: "
              f"{len(sec_chunks):,} chunks")

        for i in tqdm(range(0, len(sec_chunks), BATCH_SIZE),
                      desc=f"  {section}"):
            batch = sec_chunks[i:i + BATCH_SIZE]
            ids   = []
            docs  = []
            metas = []

            for c in batch:
                text = c.get("text", "").strip()
                if not text or len(text) < 20:
                    continue
                ids.append(c["chunk_id"])
                docs.append(text)
                metas.append({
                    "company":    str(c.get("company", "")),
                    "ticker":     str(c.get("ticker", "")),
                    "sector":     str(c.get("sector", "")),
                    "sub_sector": str(c.get("sub_sector", "")),
                    "form_type":  str(c.get("form_type", "")),
                    "year":       str(c.get("year", "")),
                    "section":    str(c.get("section", "")),
                })

            if ids:
                try:
                    coll.add(ids=ids, documents=docs,
                             metadatas=metas)
                    total_indexed += len(ids)
                except Exception as e:
                    print(f"  Batch error: {str(e)[:60]}")

    return total_indexed


def verify(client):
    print("\nVerification:")
    for section, coll_name in COLLECTIONS.items():
        try:
            coll  = client.get_collection(coll_name)
            count = coll.count()
            print(f"  {coll_name:<25} {count:>8,} chunks")
        except Exception as e:
            print(f"  {coll_name}: error {e}")


if __name__ == "__main__":
    print("="*55)
    print("SEC FILING INDEXER")
    print("="*55)

    print("\nLoading chunks...")
    chunks = load_chunks()
    print(f"Loaded: {len(chunks):,}")

    print("\nBuilding indexes...")
    total = build_indexes(chunks)

    client = chromadb.PersistentClient(path=VECTORSTORE)
    verify(client)

    print(f"\n{'='*55}")
    print(f"INDEXING COMPLETE")
    print(f"{'='*55}")
    print(f"Total indexed: {total:,}")
    print(f"Location: {VECTORSTORE}")
    print(f"\nNext: python scripts/06_build_metrics_db.py")