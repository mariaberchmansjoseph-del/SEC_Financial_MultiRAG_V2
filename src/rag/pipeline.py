"""
Pipeline — wrapper around multi-agent orchestrator.
src/rag/pipeline.py
"""

import sys
sys.path.insert(0, '.')
from src.agents.orchestrator import Orchestrator


class SECRagPipeline:

    def __init__(self):
        self.orchestrator = Orchestrator()

    def ask(
        self,
        question: str,
        ticker:   str = None,
        sector:   str = None,
        year:     str = None,
        top_k:    int = 5
    ) -> dict:
        return self.orchestrator.run(
            question = question,
            ticker   = ticker,
            sector   = sector,
            year     = year,
            top_k    = top_k,
        )

    def print_answer(self, result: dict):
        self.orchestrator.print_result(result)


if __name__ == "__main__":
    pipeline = SECRagPipeline()

    tests = [
        {
            "q":      "What are NVIDIA main supply chain risks?",
            "ticker": "NVDA"
        },
        {
            "q":      "How has Microsoft Azure revenue grown?",
            "ticker": "MSFT"
        },
        {
            "q":      "What are JPMorgan credit risk provisions?",
            "ticker": "JPM"
        },
        {
            "q":      "Compare ExxonMobil and Chevron oil strategy",
            "ticker": None
        },
    ]

    for t in tests:
        result = pipeline.ask(
            question = t["q"],
            ticker   = t.get("ticker"),
        )
        pipeline.print_answer(result)
        print()