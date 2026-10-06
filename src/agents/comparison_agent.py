"""
Comparison Agent — cross-company analysis.
src/agents/comparison_agent.py
"""

import sys
sys.path.insert(0, '.')
from src.agents.base_agent      import BaseAgent
from src.agents.financial_agent import FinancialAnalystAgent

SYSTEM_PROMPT = """You are a financial analyst comparing
multiple companies from their SEC filings.

Create clear side-by-side comparisons.
Use tables when comparing multiple metrics.
Always cite which filing each data point comes from.
Lead with a direct answer then supporting evidence.
End with a one-line conclusion.

SOURCES: [Company A] (Form Year), [Company B] (Form Year)"""


class ComparisonAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name          = "ComparisonAgent",
            system_prompt = SYSTEM_PROMPT
        )
        self.financial = FinancialAnalystAgent()

    def compare(
        self,
        question: str,
        contexts: dict,
        tickers:  list = None,
    ) -> str:
        if not contexts:
            return "No context provided for comparison."

        # Warn if more than 3 companies
        company_list = list(contexts.keys())
        if len(company_list) > 3:
            note = (
                f"Note: Comparing top 3 companies "
                f"({', '.join(company_list[:3])}). "
                f"For more specific comparison "
                f"ask about a specific pair.\n\n"
            )
        else:
            note = ""

        # Build structured data summary
        structured = ""
        for ticker in (tickers or company_list)[:3]:
            overview = self.financial.get_company_overview(
                ticker
            )
            if overview:
                mc  = self.financial.format_number(
                    overview.get("market_cap")
                )
                rev = self.financial.format_number(
                    overview.get("revenue_ttm")
                )
                structured += (
                    f"\n{ticker}: MCap={mc}, "
                    f"Revenue={rev}, "
                    f"P/E={overview.get('pe_ratio','N/A')}\n"
                )

        # Cap structured info
        structured = structured[:500]

        # Build context from filings (cap per company)
        combined_context = ""
        for ticker, context in list(
            contexts.items()
        )[:3]:
            combined_context += (
                f"\n=== {ticker} ===\n"
                f"{context[:1500]}\n"
            )

        prompt = (
            f"{note}"
            f"Question: {question}\n\n"
            f"Structured Data:{structured}\n"
            f"Filing Context:{combined_context}\n\n"
            f"Provide a clear comparison."
        )

        self.log(
            f"Comparing {company_list[:3]}..."
        )

        response = self.call_llm(
            user_message = prompt,
            temperature  = 0.1,
            max_tokens   = 600
        )

        if not response:
            self.log("LLM empty — using structured fallback")
            parts = [f"Comparison: {question}\n"]
            for ticker in company_list[:3]:
                overview = self.financial\
                               .get_company_overview(ticker)
                if overview:
                    mc  = self.financial.format_number(
                        overview.get("market_cap")
                    )
                    rev = self.financial.format_number(
                        overview.get("revenue_ttm")
                    )
                    parts.append(
                        f"{ticker}: "
                        f"Market Cap {mc}, "
                        f"Revenue {rev}, "
                        f"P/E {overview.get('pe_ratio','N/A')}"
                    )
            parts.append(
                "\nNote: Structured data shown. "
                "Text comparison unavailable."
            )
            return "\n".join(parts)

        return response