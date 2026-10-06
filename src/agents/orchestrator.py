"""
Orchestrator — routes questions to specialist agents.
Combines results, runs judge, retries if needed.
src/agents/orchestrator.py
"""

import re
import sys
import yaml
sys.path.insert(0, '.')

from src.agents.base_agent        import BaseAgent
from src.agents.research_agent    import ResearchAgent
from src.agents.financial_agent   import FinancialAnalystAgent
from src.agents.risk_agent        import RiskAnalystAgent
from src.agents.comparison_agent  import ComparisonAgent
from src.agents.judge_agent       import JudgeAgent

SYSTEM_PROMPT = """You are a financial question classifier.
Classify the question into one of these types:
RISK, FINANCIAL, COMPARISON, GENERAL
Return only one word."""

MULTI_COMPANY_PATTERN = re.compile(
    r'\b(vs|versus|compare|comparison|between|'
    r'both|against)\b',
    re.IGNORECASE
)

# Risk keywords
RISK_KEYWORDS = [
    "risk", "risks", "threat", "threats", "challenge",
    "danger", "exposure", "vulnerable", "adverse",
    "uncertainty", "litigation", "regulatory",
    "concern", "warning", "hazard", "obstacle"
]

# Financial keywords
FINANCIAL_KEYWORDS = [
    "revenue", "profit", "income", "earnings",
    "growth", "margin", "performance", "results",
    "quarter", "annual", "financial", "sales",
    "guidance", "outlook", "forecast", "eps",
    "market cap", "valuation", "price", "cost",
    "expense", "cash", "debt", "dividend", "return"
]


class Orchestrator(BaseAgent):
    """
    Routes questions to specialist agents.
    Uses keyword classification (no LLM needed).
    Runs judge evaluation on all answers.
    Retries if answer quality is poor.
    """

    MAX_RETRIES = 1

    def __init__(self):
        super().__init__(
            name          = "Orchestrator",
            system_prompt = SYSTEM_PROMPT
        )
        self.ticker_map = self._load_ticker_map()
        self.research   = ResearchAgent()
        self.financial  = FinancialAnalystAgent()
        self.risk       = RiskAnalystAgent()
        self.comparison = ComparisonAgent()
        self.judge      = JudgeAgent()

    def _load_ticker_map(self) -> dict:
        """Load company name to ticker mapping from YAML."""
        try:
            with open("configs/companies_100.yaml") as f:
                config = yaml.safe_load(f)

            mapping = {}
            for c in config["companies"]:
                name   = c["name"].lower()
                ticker = c["ticker"].upper()

                # Full company name
                mapping[name] = ticker

                # First word only
                first = name.split()[0]
                mapping[first] = ticker

                # Ticker itself
                mapping[ticker.lower()] = ticker

            self.log(
                f"Loaded {len(mapping)} name→ticker mappings"
            )
            return mapping

        except Exception as e:
            self.log(f"Could not load ticker map: {e}")
            return {}

    def classify(self, question: str) -> str:
        """
        Classify question type using keywords only.
        No LLM needed — fast and reliable.
        """
        q = question.lower()

        # Check comparison first
        if MULTI_COMPANY_PATTERN.search(question):
            return "COMPARISON"

        # Count keyword matches
        risk_score = sum(
            1 for w in RISK_KEYWORDS if w in q
        )
        fin_score  = sum(
            1 for w in FINANCIAL_KEYWORDS if w in q
        )

        if risk_score > fin_score and risk_score > 0:
            return "RISK"
        elif fin_score > 0:
            return "FINANCIAL"
        elif risk_score > 0:
            return "RISK"

        return "GENERAL"

    def extract_tickers(self, question: str) -> list:
        """
        Extract company tickers from question.
        Uses YAML mapping + uppercase pattern detection.
        """
        q_words = re.sub(
            r'[^\w\s]', ' ', question.lower()
        ).split()
        tickers = set()

        # Check single words
        for word in q_words:
            if word in self.ticker_map:
                tickers.add(self.ticker_map[word])

        # Check two-word combinations
        for i in range(len(q_words) - 1):
            pair = f"{q_words[i]} {q_words[i+1]}"
            if pair in self.ticker_map:
                tickers.add(self.ticker_map[pair])

        # Check three-word combinations
        for i in range(len(q_words) - 2):
            triple = (
                f"{q_words[i]} {q_words[i+1]} "
                f"{q_words[i+2]}"
            )
            if triple in self.ticker_map:
                tickers.add(self.ticker_map[triple])

        # Detect uppercase tickers in original question
        skip = {
            "SEC", "CEO", "CFO", "CTO", "COO",
            "US", "AI", "IT", "GDP", "IPO", "ETF",
            "RAG", "LLM", "AND", "THE", "FOR",
            "WITH", "FROM", "WHAT", "HOW", "WHY",
            "ARE", "DID", "HAS", "WAS", "ITS"
        }
        found = re.findall(r'\b[A-Z]{2,5}\b', question)
        for t in found:
            if t not in skip:
                tickers.add(t)

        return list(tickers)

    def run(
        self,
        question: str,
        ticker:   str = None,
        sector:   str = None,
        year:     str = None,
        top_k:    int = 5,
    ) -> dict:
        """
        Full multi-agent pipeline:
        1. Classify question type
        2. Extract tickers if not provided
        3. Research relevant chunks
        4. Route to specialist agent
        5. Judge evaluates answer
        6. Retry if needed
        """

        print(f"\n{'='*60}")
        print(f"Q: {question[:80]}")
        print(f"{'='*60}")

        # Step 1: Classify
        q_type = self.classify(question)
        self.log(f"Type: {q_type}")

        # Step 2: Handle comparison
        if q_type == "COMPARISON":
            return self._run_comparison(
                question, year, top_k
            )

        # Step 3: Extract ticker if not provided
        if not ticker:
            tickers = self.extract_tickers(question)
            if tickers:
                ticker = tickers[0]
                self.log(f"Detected ticker: {ticker}")

        # Step 4: Research
        results = self.research.search(
            question = question,
            ticker   = ticker,
            sector   = sector,
            year     = year,
            top_k    = top_k,
        )

        if not results:
            return self._empty_response(question)

        context = self.research.format_context(results)

        # Step 5: Generate + Judge with retry
        answer   = ""
        judgment = {}

        for attempt in range(self.MAX_RETRIES + 1):
            if attempt > 0:
                self.log(f"Retry {attempt}")

            # Route to specialist agent
            if q_type == "RISK":
                company = (
                    results[0]["company"]
                    if results else "Unknown"
                )
                filing  = (
                    f"{results[0]['form_type']} "
                    f"{results[0]['year']}"
                ) if results else ""

                risks  = self.risk.extract_risks(
                    context, company, filing
                )
                answer = self.risk.format_as_text(risks)

            elif q_type == "FINANCIAL":
                answer = self.financial.analyze(
                    question      = question,
                    context       = context,
                    ticker        = ticker,
                    question_type = q_type
                )

            else:  # GENERAL
                answer = self.financial.analyze(
                    question      = question,
                    context       = context,
                    ticker        = ticker,
                    question_type = "general"
                )

            if not answer:
                self.log("Empty answer — skipping judge")
                continue

            # Judge evaluation
            judgment = self.judge.evaluate(
                question = question,
                context  = context,
                answer   = answer
            )

            verdict = judgment.get("verdict", "REJECT")
            self.log(
                f"Judge: {verdict} "
                f"({judgment.get('overall', 0):.2f})"
            )

            if verdict == "PASS":
                break

        return self._format_response(
            question, answer, results,
            judgment, q_type
        )

    def _run_comparison(
        self,
        question: str,
        year:     str = None,
        top_k:    int = 5,
    ) -> dict:
        """Handle multi-company comparison questions."""
        tickers = self.extract_tickers(question)
        self.log(f"Comparing: {tickers}")

        if not tickers:
            # No tickers found — do general search
            results = self.research.search(
                question = question,
                top_k    = top_k
            )
            context  = self.research.format_context(
                results
            )
            answer   = self.financial.analyze(
                question, context
            )
            judgment = self.judge.evaluate(
                question, context, answer
            )
            return self._format_response(
                question, answer, results,
                judgment, "COMPARISON"
            )

        # Retrieve context per company (max 3)
        contexts    = {}
        all_results = []

        for t in tickers[:3]:
            res = self.research.search(
                question = question,
                ticker   = t,
                year     = year,
                top_k    = top_k,
            )
            if res:
                contexts[t] = (
                    self.research.format_context(res)
                )
                all_results += res

        if not contexts:
            return self._empty_response(question)

        answer   = self.comparison.compare(
            question, contexts, tickers
        )
        judgment = self.judge.evaluate(
            question,
            "\n\n".join(contexts.values()),
            answer
        )

        return self._format_response(
            question, answer, all_results,
            judgment, "COMPARISON"
        )

    def _format_response(
        self,
        question: str,
        answer:   str,
        results:  list,
        judgment: dict,
        q_type:   str,
    ) -> dict:
        """Format the final response dict."""
        sources = [
            {
                "company":   r["company"],
                "form_type": r["form_type"],
                "year":      r["year"],
                "score":     round(r.get(
                    "rerank_score", r["score"]
                ), 3),
                "preview":   r["text"][:150]
            }
            for r in results[:5]
        ]

        return {
            "question":      question,
            "answer":        answer or "No answer generated.",
            "sources":       sources,
            "question_type": q_type,
            "judgment": {
                "overall":        judgment.get("overall", 0),
                "groundedness":   judgment.get(
                    "groundedness", 0),
                "relevance":      judgment.get(
                    "relevance", 0),
                "completeness":   judgment.get(
                    "completeness", 0),
                "accuracy":       judgment.get(
                    "accuracy", 0),
                "verdict":        judgment.get(
                    "verdict", "UNKNOWN"),
                "issues":         judgment.get(
                    "issues", []),
                "hallucinations": judgment.get(
                    "hallucinations", []),
            },
            "agents_used": [
                "ResearchAgent",
                f"{q_type.title()}Agent",
                "JudgeAgent"
            ],
            "success": bool(answer),
        }

    def _empty_response(self, question: str) -> dict:
        """Return empty response when no results found."""
        return {
            "question": question,
            "answer":   (
                "No relevant SEC filing information "
                "found for this question."
            ),
            "sources":  [],
            "judgment": {
                "verdict": "REJECT",
                "overall": 0.0
            },
            "agents_used": [],
            "success":  False,
        }

    def print_result(self, result: dict):
        """Pretty print a pipeline result."""
        print(f"\nQ: {result['question']}")
        print(f"{'─'*60}")
        print(f"A: {result['answer']}")
        print(f"{'─'*60}")

        j = result.get("judgment", {})
        verdict = j.get("verdict", "?")
        overall = j.get("overall", 0)
        print(
            f"Judge: {verdict} "
            f"(overall={overall:.2f}, "
            f"ground={j.get('groundedness',0):.2f}, "
            f"rel={j.get('relevance',0):.2f})"
        )

        if j.get("issues"):
            print(f"Issues: {j['issues']}")

        if j.get("hallucinations"):
            print(
                f"⚠️  Hallucinations: "
                f"{j['hallucinations']}"
            )

        sources = result.get("sources", [])
        if sources:
            print(f"\nSources ({len(sources)}):")
            for s in sources[:3]:
                print(
                    f"  {s['company']} "
                    f"({s['form_type']} {s['year']}) "
                    f"[{s['score']}]"
                )

        print(
            f"Agents: {result.get('agents_used', [])}"
        )