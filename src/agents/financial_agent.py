"""
Financial Analyst Agent.
Uses structured SQLite data for numbers.
Uses filing text for qualitative analysis only.
src/agents/financial_agent.py
"""

import sqlite3
import sys
sys.path.insert(0, '.')
from src.agents.base_agent import BaseAgent

DB_PATH = "data/processed/metrics.db"

SYSTEM_PROMPT = """You are a financial analyst assistant.

CRITICAL RULES — NEVER BREAK THESE:
1. ONLY use numbers from the VERIFIED DATA section
2. NEVER invent percentages, revenues, or growth rates
3. NEVER use numbers from your training knowledge
4. If VERIFIED DATA has no numbers, describe qualitatively
5. Always cite your source filing

If you cannot answer from the provided data, say:
"The specific figures are not available in the
retrieved context. Based on the filing text: [qualitative]"
"""


class FinancialAnalystAgent(BaseAgent):
    """
    Generates financial analysis combining:
    - Structured SQLite data (exact verified numbers)
    - Filing text context (qualitative analysis)

    Never hallucinate — uses structured data for numbers,
    filing text only for qualitative statements.
    """

    def __init__(self):
        super().__init__(
            name          = "FinancialAnalyst",
            system_prompt = SYSTEM_PROMPT
        )

    # ── DATABASE QUERIES ──────────────────────────────────

    def get_company_overview(self, ticker: str) -> dict:
        """Get company overview from SQLite."""
        try:
            conn = sqlite3.connect(DB_PATH)
            row  = conn.execute("""
                SELECT ticker, name, sector, industry,
                       market_cap, pe_ratio,
                       revenue_ttm, profit_margin,
                       analyst_target, beta, employees
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
                "employees":     row[10],
            }
        except Exception as e:
            self.log(f"DB overview error: {e}")
            return {}

    def get_revenue_history(self, ticker: str) -> list:
        """Get annual revenue history from SQLite."""
        try:
            conn = sqlite3.connect(DB_PATH)
            rows = conn.execute("""
                SELECT period_date, total_revenue,
                       net_income, gross_profit
                FROM financial_metrics
                WHERE ticker = ?
                AND total_revenue IS NOT NULL
                ORDER BY period_date DESC
                LIMIT 4
            """, (ticker.upper(),)).fetchall()
            conn.close()

            return [
                {
                    "period":      r[0][:7],
                    "revenue":     r[1],
                    "net_income":  r[2],
                    "gross_profit": r[3],
                }
                for r in rows
            ]
        except Exception:
            return []

    def get_price_summary(self, ticker: str) -> dict:
        """Get recent price data from SQLite."""
        try:
            conn  = sqlite3.connect(DB_PATH)
            rows  = conn.execute("""
                SELECT date, close_price
                FROM price_history
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT 12
            """, (ticker.upper(),)).fetchall()
            conn.close()

            if not rows:
                return {}

            prices = [r[1] for r in rows if r[1]]
            if not prices:
                return {}

            return {
                "latest_price": prices[0],
                "price_1y_ago": prices[-1] if len(prices) > 1 else None,
                "price_change_1y": (
                    round(
                        (prices[0] - prices[-1])
                        / prices[-1] * 100, 1
                    )
                    if len(prices) > 1 and prices[-1]
                    else None
                )
            }
        except Exception:
            return {}

    # ── NUMBER FORMATTING ─────────────────────────────────

    def format_number(self, value) -> str:
        """Format large numbers readably."""
        if value is None:
            return "N/A"
        try:
            v = float(value)
            if abs(v) >= 1e12:
                return f"${v/1e12:.2f}T"
            if abs(v) >= 1e9:
                return f"${v/1e9:.1f}B"
            if abs(v) >= 1e6:
                return f"${v/1e6:.1f}M"
            return f"${v:,.0f}"
        except Exception:
            return "N/A"

    def format_pct(self, value) -> str:
        """Format percentage."""
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.1%}"
        except Exception:
            return "N/A"

    # ── STRUCTURED DATA BUILDER ───────────────────────────

    def build_verified_data(self, ticker: str) -> str:
        """
        Build verified structured data section.
        These numbers come from SQLite — 100% reliable.
        LLM is allowed to cite these freely.
        """
        if not ticker:
            return "No ticker provided — no structured data."

        lines = [
            f"=== VERIFIED DATA FOR {ticker.upper()} ===",
            "(These numbers are from yfinance — cite freely)",
            ""
        ]

        # Overview
        overview = self.get_company_overview(ticker)
        if overview:
            lines.append("Company Snapshot:")
            lines.append(
                f"  Name: {overview.get('name','N/A')}"
            )
            lines.append(
                f"  Sector: {overview.get('sector','N/A')}"
            )
            lines.append(
                f"  Market Cap: "
                f"{self.format_number(overview.get('market_cap'))}"
            )
            lines.append(
                f"  P/E Ratio: "
                f"{overview.get('pe_ratio','N/A')}"
            )
            lines.append(
                f"  Revenue (TTM): "
                f"{self.format_number(overview.get('revenue_ttm'))}"
            )
            lines.append(
                f"  Profit Margin: "
                f"{self.format_pct(overview.get('profit_margin'))}"
            )
            lines.append(
                f"  Analyst Target Price: "
                f"{overview.get('analyst_target','N/A')}"
            )
            lines.append(
                f"  Beta: {overview.get('beta','N/A')}"
            )
            lines.append("")

        # Revenue history
        history = self.get_revenue_history(ticker)
        if history:
            lines.append("Annual Revenue History:")
            for h in history:
                rev = self.format_number(h["revenue"])
                ni  = self.format_number(h["net_income"])
                lines.append(
                    f"  {h['period']}: Revenue={rev}, "
                    f"Net Income={ni}"
                )
            lines.append("")

            # Calculate growth if we have 2+ years
            if len(history) >= 2:
                r0 = history[0]["revenue"]
                r1 = history[1]["revenue"]
                if r0 and r1 and r1 != 0:
                    growth = (r0 - r1) / abs(r1) * 100
                    lines.append(
                        f"Revenue Growth (YoY): "
                        f"{growth:.1f}%"
                    )
                    lines.append("")

        # Price summary
        prices = self.get_price_summary(ticker)
        if prices:
            lines.append("Price Data:")
            lines.append(
                f"  Latest Price: "
                f"${prices.get('latest_price','N/A')}"
            )
            chg = prices.get("price_change_1y")
            if chg is not None:
                lines.append(
                    f"  1-Year Price Change: {chg:+.1f}%"
                )
            lines.append("")

        if len(lines) <= 4:
            return (
                f"No structured data found "
                f"for ticker {ticker}."
            )

        return "\n".join(lines)

    # ── MAIN ANALYZE METHOD ───────────────────────────────

    def analyze(
        self,
        question:      str,
        context:       str,
        ticker:        str = None,
        question_type: str = "general"
    ) -> str:
        """
        Generate financial analysis.

        Priority:
        1. Use verified SQLite data for all numbers
        2. Use filing text for qualitative insights
        3. Never invent numbers
        """
        self.log(f"Analyzing: {question[:60]}...")

        # Build verified data section
        verified_data = self.build_verified_data(ticker) \
                        if ticker else ""

        prompt = f"""Answer this financial question about a company.

VERIFIED FINANCIAL DATA (100% accurate — use freely):
{verified_data if verified_data else "No verified data available for this query."}

SEC FILING TEXT (qualitative context only):
{context[:2000]}

QUESTION: {question}

INSTRUCTIONS:
- Use numbers ONLY from the VERIFIED FINANCIAL DATA section above
- Use the filing text for qualitative descriptions and strategy
- If the verified data does not contain a specific number, say
  "specific figures not available" rather than inventing them
- Write 3-5 sentences maximum
- Be precise and professional

End your answer with:
SOURCES: [list companies and filing types referenced]
CONFIDENCE: High (if using verified data) / Low (if text only)"""

        response = self.call_llm(
            user_message = prompt,
            temperature  = 0.0,
            max_tokens   = 450,
        )

        if not response:
            # Fallback: return structured data directly
            if verified_data and ticker:
                overview = self.get_company_overview(ticker)
                history  = self.get_revenue_history(ticker)

                if overview or history:
                    parts = [
                        f"Based on verified financial data:"
                    ]
                    if overview:
                        parts.append(
                            f"{overview.get('name',ticker)} "
                            f"has a market cap of "
                            f"{self.format_number(overview.get('market_cap'))} "
                            f"and revenue TTM of "
                            f"{self.format_number(overview.get('revenue_ttm'))}."
                        )
                    if history:
                        latest = history[0]
                        parts.append(
                            f"Most recent annual revenue: "
                            f"{self.format_number(latest['revenue'])} "
                            f"({latest['period']})."
                        )
                    return " ".join(parts)

            return (
                "Unable to generate analysis. "
                "The retrieved context may not contain "
                "sufficient financial data for this question."
            )

        return response

    # ── COMPARISON HELPER ─────────────────────────────────

    def compare_metrics(
        self,
        ticker_a: str,
        ticker_b: str
    ) -> str:
        """
        Build a structured metrics comparison table
        for two companies from verified SQLite data.
        """
        data_a   = self.get_company_overview(ticker_a)
        data_b   = self.get_company_overview(ticker_b)
        hist_a   = self.get_revenue_history(ticker_a)
        hist_b   = self.get_revenue_history(ticker_b)

        lines = [
            f"=== COMPARISON: {ticker_a} vs {ticker_b} ===",
            "",
            f"{'Metric':<20} {ticker_a:<15} {ticker_b:<15}",
            f"{'─'*50}",
        ]

        metrics = [
            ("Market Cap",
             self.format_number(data_a.get("market_cap")),
             self.format_number(data_b.get("market_cap"))),
            ("Revenue TTM",
             self.format_number(data_a.get("revenue_ttm")),
             self.format_number(data_b.get("revenue_ttm"))),
            ("P/E Ratio",
             str(data_a.get("pe_ratio","N/A")),
             str(data_b.get("pe_ratio","N/A"))),
            ("Profit Margin",
             self.format_pct(data_a.get("profit_margin")),
             self.format_pct(data_b.get("profit_margin"))),
            ("Beta",
             str(data_a.get("beta","N/A")),
             str(data_b.get("beta","N/A"))),
        ]

        for label, val_a, val_b in metrics:
            lines.append(
                f"{label:<20} {val_a:<15} {val_b:<15}"
            )

        if hist_a and hist_b:
            lines.append("")
            lines.append("Annual Revenue:")
            max_rows = min(len(hist_a), len(hist_b), 3)
            for i in range(max_rows):
                period = hist_a[i]["period"]
                lines.append(
                    f"  {period}: "
                    f"{ticker_a}="
                    f"{self.format_number(hist_a[i]['revenue'])}, "
                    f"{ticker_b}="
                    f"{self.format_number(hist_b[i]['revenue'])}"
                )

        return "\n".join(lines)