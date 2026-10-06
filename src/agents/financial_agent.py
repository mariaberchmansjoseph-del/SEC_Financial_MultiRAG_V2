"""
Financial Analyst Agent.
src/agents/financial_agent.py
"""

import sqlite3
import sys
sys.path.insert(0, '.')
from src.agents.base_agent import BaseAgent

DB_PATH = "data/processed/metrics.db"

SYSTEM_PROMPT = """You are a senior financial analyst
specialising in SEC filings analysis.

YOUR ROLE:
- Answer financial questions using provided context
- Cite sources precisely: company, filing type, year
- Include specific numbers when available
- Never hallucinate financial figures
- Maximum 5 sentences for simple questions

End every answer with:
SOURCES: [Company] ([Form] [Year])
CONFIDENCE: High / Medium / Low"""


class FinancialAnalystAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name          = "FinancialAnalyst",
            system_prompt = SYSTEM_PROMPT
        )

    def get_company_overview(self, ticker: str) -> dict:
        try:
            conn = sqlite3.connect(DB_PATH)
            row  = conn.execute("""
                SELECT ticker, name, sector, industry,
                       market_cap, pe_ratio, revenue_ttm,
                       profit_margin, analyst_target, beta
                FROM company_overview
                WHERE ticker = ?
            """, (ticker.upper(),)).fetchone()
            conn.close()
            if not row:
                return {}
            return {
                "ticker":        row[0],
                "name":          row[1],
                "sector":        row[2],
                "industry":      row[3],
                "market_cap":    row[4],
                "pe_ratio":      row[5],
                "revenue_ttm":   row[6],
                "profit_margin": row[7],
                "analyst_target": row[8],
                "beta":          row[9],
            }
        except Exception:
            return {}

    def get_metric_from_db(
        self, ticker: str,
        metric: str = "total_revenue",
        year: str = None
    ) -> dict:
        try:
            conn  = sqlite3.connect(DB_PATH)
            query = f"""
                SELECT period_date, {metric}
                FROM financial_metrics
                WHERE ticker = ?
                {"AND period_date LIKE ?" if year else ""}
                ORDER BY period_date DESC
                LIMIT 4
            """
            params = [ticker.upper()]
            if year:
                params.append(f"{year}%")
            rows = conn.execute(query, params).fetchall()
            conn.close()
            if not rows:
                return {}
            return {
                "ticker": ticker,
                "metric": metric,
                "data": [
                    {"period": r[0], "value": r[1]}
                    for r in rows if r[1] is not None
                ]
            }
        except Exception:
            return {}

    def format_number(self, value) -> str:
        if value is None:
            return "N/A"
        try:
            value = float(value)
            if abs(value) >= 1e12:
                return f"${value/1e12:.2f}T"
            if abs(value) >= 1e9:
                return f"${value/1e9:.1f}B"
            if abs(value) >= 1e6:
                return f"${value/1e6:.1f}M"
            return f"${value:,.0f}"
        except Exception:
            return "N/A"

    def analyze(
        self,
        question: str,
        context:  str,
        ticker:   str = None,
        question_type: str = "general"
    ) -> str:
        structured_info = ""
        if ticker:
            overview = self.get_company_overview(ticker)
            if overview:
                mc  = self.format_number(
                    overview.get("market_cap")
                )
                rev = self.format_number(
                    overview.get("revenue_ttm")
                )
                structured_info = (
                    f"\nStructured Data for {ticker}:\n"
                    f"Market Cap: {mc}\n"
                    f"Revenue TTM: {rev}\n"
                    f"P/E Ratio: {overview.get('pe_ratio','N/A')}\n"
                    f"Profit Margin: {overview.get('profit_margin','N/A')}\n"
                )
            rev_data = self.get_metric_from_db(
                ticker, "total_revenue"
            )
            if rev_data.get("data"):
                structured_info += "\nRevenue History:\n"
                for d in rev_data["data"]:
                    val = self.format_number(d["value"])
                    structured_info += (
                        f"  {d['period'][:7]}: {val}\n"
                    )

        prompt = (
            f"Filing Context:\n{context}\n"
            f"{structured_info}\n"
            f"Question: {question}\n\n"
            f"Provide precise financial analysis."
        )

        self.log(f"Analyzing: {question[:60]}...")
        response = self.call_llm(
            user_message = prompt,
            temperature  = 0.1,
            max_tokens   = 500
        )
        return response or \
               "Unable to generate analysis from context."