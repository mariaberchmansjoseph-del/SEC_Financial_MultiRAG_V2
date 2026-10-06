"""
Research Agent — retrieves relevant SEC filing chunks.
Features: collection routing, reranking, temporal weighting,
injection detection, deduplication.
Query expansion disabled by default to save API calls.
src/agents/research_agent.py
"""

import sys
import time
import json
sys.path.insert(0, '.')

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction
)
from src.agents.base_agent import BaseAgent

# ── CONFIG ────────────────────────────────────────────────────
VECTORSTORE  = "data/vectorstore"
EMBED_MODEL  = "BAAI/bge-base-en-v1.5"
CURRENT_YEAR = 2025

COLLECTIONS = {
    "risk_factors": "sec_risk_factors",
    "mda":          "sec_mda",
    "financials":   "sec_financials",
    "general":      "sec_general",
}

COLLECTION_HINTS = {
    "risk_factors": [
        "risk", "risks", "threat", "challenge",
        "concern", "danger", "exposure", "vulnerable",
        "adverse", "uncertainty", "litigation",
        "regulatory", "competition", "failure",
    ],
    "mda": [
        "revenue", "profit", "income", "earnings",
        "growth", "margin", "performance", "results",
        "quarter", "annual", "financial results",
        "operating", "sales", "guidance", "outlook",
    ],
    "financials": [
        "balance sheet", "cash flow", "assets",
        "liabilities", "equity", "debt", "capital",
        "dividends", "shares", "eps",
    ],
}

INJECTION_PATTERNS = [
    "ignore your instructions",
    "forget everything",
    "pretend you are",
    "act as if",
    "you are now",
    "disregard",
    "override",
    "bypass",
    "jailbreak",
    "ignore previous",
    "new instructions",
    "system prompt",
]

RECENCY_SIGNALS = [
    "recent", "current", "now", "today",
    "latest", "2025", "2026", "this year",
    "last year", "recently", "currently",
]

QUERY_EXPANSION_PROMPT = """Generate 3 different search
queries to find relevant SEC filing information.
Return ONLY a JSON array of 3 strings.
Example: ["query 1", "query 2", "query 3"]"""


def is_prose(text: str) -> bool:
    """Return True if text is readable prose not a table."""
    if not text or len(text) < 30:
        return False
    digits = sum(1 for c in text if c.isdigit())
    return digits / max(len(text), 1) < 0.25


def is_safe_query(question: str) -> bool:
    """Detect prompt injection attempts."""
    q_lower = question.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in q_lower:
            return False
    if len(question) > 1000:
        return False
    return True


class ResearchAgent(BaseAgent):
    """
    Retrieves relevant chunks from SEC filings.

    Pipeline:
      1. Safety check (injection detection)
      2. Classify → select target collections
      3. Optional query expansion (disabled by default)
      4. Vector search across target collections
      5. Deduplicate results
      6. Apply temporal weighting if question is recent
      7. Cross-encoder reranking
      8. Return top-k results
    """

    def __init__(self):
        super().__init__(
            name          = "ResearchAgent",
            system_prompt = QUERY_EXPANSION_PROMPT
        )
        self._init_vectorstore()
        self._init_reranker()

    def _init_vectorstore(self):
        """Initialise ChromaDB collections."""
        embed_fn = SentenceTransformerEmbeddingFunction(
            model_name = EMBED_MODEL
        )
        client = chromadb.PersistentClient(
            path = VECTORSTORE
        )
        self.collections = {}
        for section, coll_name in COLLECTIONS.items():
            try:
                self.collections[section] = \
                    client.get_collection(
                        name               = coll_name,
                        embedding_function = embed_fn
                    )
                count = self.collections[section].count()
                self.log(
                    f"Loaded {coll_name}: {count:,}"
                )
            except Exception as e:
                self.log(
                    f"Could not load {coll_name}: {e}"
                )

    def _init_reranker(self):
        """Initialise cross-encoder reranker."""
        self.reranker = None
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            self.log("Cross-encoder reranker loaded ✅")
        except Exception:
            self.log(
                "Cross-encoder not available — "
                "using vector scores only"
            )

    def classify_question(self, question: str) -> list:
        """
        Identify which collections to search.
        Returns list of collection keys in priority order.
        """
        q_lower = question.lower()
        scores  = {s: 0 for s in COLLECTIONS}

        for section, keywords in COLLECTION_HINTS.items():
            for kw in keywords:
                if kw in q_lower:
                    scores[section] += 1

        scores["general"] = max(scores["general"], 1)

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        target = [s for s, score in ranked if score > 0]
        self.log(f"Target collections: {target}")
        return target

    def expand_query(self, question: str) -> list:
        """Generate 3 search phrasings."""
        response = self.call_llm(
            user_message = f"Question: {question}",
            temperature  = 0.3,
            max_tokens   = 200
        )
        try:
            queries = json.loads(response)
            if isinstance(queries, list):
                result = [str(q) for q in queries[:3]]
                self.log(
                    f"Expanded to {len(result)} queries"
                )
                return result
        except Exception:
            pass
        return [question]

    def vector_search(
        self,
        query:      str,
        collection: str,
        ticker:     str = None,
        sector:     str = None,
        year:       str = None,
        form:       str = None,
        top_k:      int = 20,
    ) -> list:
        """Search one collection with optional filters."""
        coll = self.collections.get(collection)
        if not coll:
            return []

        filters = []
        if ticker:
            filters.append(
                {"ticker": {"$eq": ticker.upper()}}
            )
        if year:
            filters.append(
                {"year": {"$eq": str(year)}}
            )
        if sector:
            filters.append(
                {"sector": {"$eq": sector.lower()}}
            )
        if form:
            filters.append(
                {"form_type": {"$eq": form.upper()}}
            )

        where = None
        if len(filters) == 1:
            where = filters[0]
        elif len(filters) > 1:
            where = {"$and": filters}

        kwargs = {
            "query_texts": [query],
            "n_results":   min(top_k, coll.count() or 1),
            "include":     ["documents", "metadatas",
                            "distances"]
        }
        if where:
            kwargs["where"] = where

        try:
            raw   = coll.query(**kwargs)
            docs  = raw["documents"][0]
            metas = raw["metadatas"][0]
            dists = raw["distances"][0]

            results = []
            for doc, meta, dist in zip(
                docs, metas, dists
            ):
                if not is_prose(doc):
                    continue
                results.append({
                    "text":        doc,
                    "score":       round(1 - dist, 4),
                    "company":     meta.get("company", ""),
                    "ticker":      meta.get("ticker", ""),
                    "sector":      meta.get("sector", ""),
                    "sub_sector":  meta.get(
                                       "sub_sector", ""),
                    "form_type":   meta.get("form_type", ""),
                    "filing_date": meta.get(
                                       "filing_date", ""),
                    "year":        meta.get("year", ""),
                    "section":     meta.get("section", ""),
                    "collection":  collection,
                })
            return results

        except Exception as e:
            self.log(
                f"Search error in {collection}: "
                f"{str(e)[:60]}"
            )
            return []

    def apply_temporal_weight(
        self,
        results:  list,
        question: str = ""
    ) -> list:
        """
        Boost recent filings only when question
        asks about recent information.
        """
        q_lower      = question.lower()
        needs_recent = any(
            s in q_lower for s in RECENCY_SIGNALS
        )

        if not needs_recent:
            return results

        for r in results:
            try:
                year  = int(r.get("year", CURRENT_YEAR))
                age   = CURRENT_YEAR - year
                decay = max(0.85, 1.0 - age * 0.05)
                r["score"] = round(r["score"] * decay, 4)
            except Exception:
                pass

        results.sort(
            key=lambda x: x["score"], reverse=True
        )
        return results

    def rerank(
        self,
        query:   str,
        results: list,
        top_k:   int
    ) -> list:
        """Cross-encoder reranking for precision."""
        if not self.reranker or not results:
            results.sort(
                key=lambda x: x["score"], reverse=True
            )
            return results[:top_k]

        try:
            pairs  = [
                (query, r["text"]) for r in results
            ]
            scores = self.reranker.predict(pairs)

            for r, score in zip(results, scores):
                r["rerank_score"] = float(score)

            # Filter negative scores
            positive = [
                r for r in results
                if r.get("rerank_score", 0) > 0
            ]
            final = positive if positive else results

            final.sort(
                key=lambda x: x.get("rerank_score", 0),
                reverse=True
            )

            self.log(
                f"Reranked {len(results)} → top {top_k}"
            )
            return final[:top_k]

        except Exception as e:
            self.log(f"Reranker error: {str(e)[:60]}")
            results.sort(
                key=lambda x: x["score"], reverse=True
            )
            return results[:top_k]

    def search(
        self,
        question: str,
        ticker:   str  = None,
        sector:   str  = None,
        year:     str  = None,
        form:     str  = None,
        top_k:    int  = 5,
        expand:   bool = False,
    ) -> list:
        """
        Full retrieval pipeline.

        Steps:
          1. Safety check
          2. Classify → target collections
          3. Query expansion (optional, default off)
          4. Vector search
          5. Deduplicate
          6. Temporal weighting
          7. Rerank
          8. Return top_k
        """
        t0 = time.time()

        # Step 1: Safety
        if not is_safe_query(question):
            self.log("⚠️  Injection attempt blocked")
            return []

        # Step 2: Classify
        target_collections = self.classify_question(
            question
        )

        # Step 3: Queries
        if expand:
            queries = self.expand_query(question)
        else:
            queries = [question]

        # Step 4 + 5: Search and deduplicate
        all_results = []
        seen_texts  = set()

        for collection in target_collections[:2]:
            for query in queries:
                results = self.vector_search(
                    query      = query,
                    collection = collection,
                    ticker     = ticker,
                    sector     = sector,
                    year       = year,
                    form       = form,
                    top_k      = 20,
                )
                for r in results:
                    key = r["text"][:80]
                    if key not in seen_texts:
                        seen_texts.add(key)
                        all_results.append(r)

        if not all_results:
            self.log("No results found")
            return []

        # Step 6: Temporal weighting
        all_results = self.apply_temporal_weight(
            all_results, question
        )

        # Step 7: Rerank
        final = self.rerank(question, all_results, top_k)

        latency = round((time.time() - t0) * 1000)
        self.log(
            f"Retrieved {len(final)} chunks "
            f"in {latency}ms"
        )

        return final

    def format_context(self, results: list) -> str:
        """Format results into LLM context string."""
        if not results:
            return "No relevant information found."

        parts = []
        for i, r in enumerate(results, 1):
            source = (
                f"{r['company']} "
                f"({r['form_type']} {r['year']}, "
                f"filed {r['filing_date']})"
            )
            score = r.get("rerank_score", r["score"])
            parts.append(
                f"[Source {i}: {source} "
                f"relevance={score:.3f}]\n"
                f"{r['text']}"
            )

        return "\n\n---\n\n".join(parts)

    def search_and_format(
        self,
        question: str,
        ticker:   str = None,
        sector:   str = None,
        year:     str = None,
        form:     str = None,
        top_k:    int = 5,
    ) -> tuple:
        """Convenience: search and return results + context."""
        results = self.search(
            question = question,
            ticker   = ticker,
            sector   = sector,
            year     = year,
            form     = form,
            top_k    = top_k,
        )
        context = self.format_context(results)
        return results, context