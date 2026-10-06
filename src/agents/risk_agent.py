"""
Risk Analyst Agent.
Extracts and categorises risk factors from SEC filings.
src/agents/risk_agent.py
"""

import re
import sys
sys.path.insert(0, '.')
from src.agents.base_agent import BaseAgent


SYSTEM_PROMPT = """You are a risk analyst specialising
in corporate risk assessment from SEC filings.

Analyse the provided SEC filing context and identify
all business risks mentioned.

Categorise each risk into one of these types:
- market_risk: commodity prices, currency, interest rates
- operational_risk: supply chain, manufacturing, technology
- regulatory_risk: compliance, legal, government policy
- financial_risk: liquidity, credit, capital structure
- strategic_risk: competition, innovation, market position
- geopolitical_risk: wars, sanctions, trade restrictions
- esg_risk: climate change, environmental, social
- cyber_risk: data security, IT systems, cyber attacks
- ai_risk: AI regulation, model risk, AI competition

Always respond with a structured analysis."""


class RiskAnalystAgent(BaseAgent):
    """
    Extracts structured risk factors from SEC filings.
    Handles both JSON and text responses from LLM.
    """

    def __init__(self):
        super().__init__(
            name          = "RiskAnalyst",
            system_prompt = SYSTEM_PROMPT
        )

    def extract_risks(
        self,
        context: str,
        company: str,
        filing:  str
    ) -> dict:
        """
        Extract and categorise risks from filing context.

        Returns structured dict with:
          company, filing, risk_level,
          risk_summary, top_3_risks, risks by category
        """
        self.log(f"Extracting risks for {company}...")

        # Truncate context to avoid token overflow
        context_truncated = context[:3000]

        # Build prompt
        prompt = f"""Analyse this SEC filing and identify risks.

Company: {company}
Filing: {filing}

SEC Filing Context:
{context_truncated}

Identify the main business risks and respond with:

RISK_LEVEL: High or Medium or Low
SUMMARY: One sentence overview of risk profile
TOP_RISK_1: [most important risk]
TOP_RISK_2: [second most important risk]
TOP_RISK_3: [third most important risk]
MARKET_RISK: [market related risks or NONE]
OPERATIONAL_RISK: [operational risks or NONE]
REGULATORY_RISK: [regulatory risks or NONE]
FINANCIAL_RISK: [financial risks or NONE]
STRATEGIC_RISK: [strategic risks or NONE]
GEOPOLITICAL_RISK: [geopolitical risks or NONE]
ESG_RISK: [environmental/social risks or NONE]
CYBER_RISK: [cybersecurity risks or NONE]
AI_RISK: [AI related risks or NONE]"""

        response = self.call_llm(
            user_message = prompt,
            temperature  = 0.0,
            max_tokens   = 1200,
        )

        if not response:
            self.log("Empty response — returning default")
            return self._default_result(company, filing)

        # Try to parse as JSON first
        result = self.parse_json(response)
        if result and result.get("top_3_risks"):
            result.setdefault("company", company)
            result.setdefault("filing", filing)
            return result

        # Parse structured text response
        return self._parse_text_response(
            response, company, filing
        )

    def _parse_text_response(
        self,
        text:    str,
        company: str,
        filing:  str
    ) -> dict:
        """
        Parse structured text response from LLM.
        Handles the FIELD: value format.
        """
        lines      = text.strip().split('\n')
        result     = {
            "company":      company,
            "filing":       filing,
            "risk_level":   "Medium",
            "risk_summary": "",
            "top_3_risks":  [],
            "risks": {
                "market_risk":      [],
                "operational_risk": [],
                "regulatory_risk":  [],
                "financial_risk":   [],
                "strategic_risk":   [],
                "geopolitical_risk":[],
                "esg_risk":         [],
                "cyber_risk":       [],
                "ai_risk":          [],
            }
        }

        field_map = {
            "RISK_LEVEL":       "risk_level",
            "SUMMARY":          "risk_summary",
            "TOP_RISK_1":       "top_1",
            "TOP_RISK_2":       "top_2",
            "TOP_RISK_3":       "top_3",
            "MARKET_RISK":      "market_risk",
            "OPERATIONAL_RISK": "operational_risk",
            "REGULATORY_RISK":  "regulatory_risk",
            "FINANCIAL_RISK":   "financial_risk",
            "STRATEGIC_RISK":   "strategic_risk",
            "GEOPOLITICAL_RISK":"geopolitical_risk",
            "ESG_RISK":         "esg_risk",
            "CYBER_RISK":       "cyber_risk",
            "AI_RISK":          "ai_risk",
        }

        top_risks = []

        for line in lines:
            if ':' not in line:
                continue

            parts = line.split(':', 1)
            if len(parts) != 2:
                continue

            key   = parts[0].strip().upper()
            value = parts[1].strip()

            if not value or value.upper() == "NONE":
                continue

            mapped = field_map.get(key)
            if not mapped:
                continue

            if mapped == "risk_level":
                for level in ["High", "Medium", "Low"]:
                    if level.lower() in value.lower():
                        result["risk_level"] = level
                        break

            elif mapped == "risk_summary":
                result["risk_summary"] = value[:200]

            elif mapped in ["top_1", "top_2", "top_3"]:
                top_risks.append(value[:150])

            elif mapped in result["risks"]:
                result["risks"][mapped].append(
                    value[:150]
                )

        result["top_3_risks"] = top_risks[:3]

        # If no summary extracted, build one
        if not result["risk_summary"] and top_risks:
            result["risk_summary"] = (
                f"{company} faces risks including: "
                f"{'; '.join(top_risks[:2])}"
            )

        # If nothing extracted, try to parse free text
        if not top_risks:
            result = self._extract_from_free_text(
                text, company, filing, result
            )

        return result

    def _extract_from_free_text(
        self,
        text:    str,
        company: str,
        filing:  str,
        result:  dict
    ) -> dict:
        """Last resort — extract key sentences as risks."""
        risk_keywords = [
            "risk", "threat", "challenge",
            "concern", "adverse", "uncertain",
            "volatile", "competition", "regulatory",
            "supply chain", "cyber", "climate"
        ]

        sentences = re.split(r'[.!?]', text)
        risk_sentences = []

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue
            sent_lower = sent.lower()
            if any(kw in sent_lower for kw in risk_keywords):
                risk_sentences.append(sent[:150])

        if risk_sentences:
            result["top_3_risks"] = risk_sentences[:3]
            result["risk_summary"] = (
                f"{company} faces multiple risks "
                f"as identified in {filing}."
            )
            result["risk_level"] = "Medium"

        return result

    def _default_result(
        self, company: str, filing: str
    ) -> dict:
        """Return a safe default when extraction fails."""
        return {
            "company":      company,
            "filing":       filing,
            "risk_level":   "Unknown",
            "risk_summary": (
                f"Risk data for {company} could not "
                f"be extracted from {filing}."
            ),
            "top_3_risks":  [],
            "risks": {
                "market_risk":       [],
                "operational_risk":  [],
                "regulatory_risk":   [],
                "financial_risk":    [],
                "strategic_risk":    [],
                "geopolitical_risk": [],
                "esg_risk":          [],
                "cyber_risk":        [],
                "ai_risk":           [],
            }
        }

    def format_as_text(self, risks: dict) -> str:
        """Format risk dict as readable answer."""
        if not risks:
            return "No risk data available."

        lines = []

        lines.append(
            f"Risk Level: {risks.get('risk_level','N/A')}"
        )
        lines.append(
            f"Summary: {risks.get('risk_summary','')}"
        )

        top3 = risks.get("top_3_risks", [])
        if top3:
            lines.append("\nTop Risks:")
            for i, r in enumerate(top3, 1):
                lines.append(f"  {i}. {r}")

        risk_cats = risks.get("risks", {})
        has_categories = any(
            bool(v) for v in risk_cats.values()
        )

        if has_categories:
            lines.append("\nBy Category:")
            for cat, items in risk_cats.items():
                if items:
                    label = (
                        cat.replace("_", " ").title()
                    )
                    lines.append(f"\n  {label}:")
                    for item in items[:2]:
                        lines.append(f"    • {item}")

        sources = risks.get("filing", "")
        if sources:
            lines.append(
                f"\nSOURCES: {risks.get('company','')} "
                f"({sources})"
            )

        return "\n".join(lines)