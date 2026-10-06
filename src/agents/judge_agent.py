"""
Judge Agent — LLM-as-Judge quality evaluation.
Uses different model to avoid self-evaluation bias.
src/agents/judge_agent.py
"""

import sys
sys.path.insert(0, '.')
from src.agents.base_agent import BaseAgent

MODEL_OVERRIDE = "qwen/qwen3.8-27b"

SYSTEM_PROMPT = """You are a financial analysis quality
evaluator. Assess AI-generated answers about SEC filings.

EVALUATE ON FOUR DIMENSIONS (0.0 to 1.0):

GROUNDEDNESS: Every claim must be in the context.
Claims not in context = hallucination = low score.

RELEVANCE: Does the answer address the question?

COMPLETENESS: Are all parts of the question covered?

ACCURACY: Are numbers and facts correct?

Return ONLY valid JSON:
{
  "groundedness":   0.92,
  "relevance":      0.88,
  "completeness":   0.75,
  "accuracy":       0.95,
  "overall":        0.875,
  "verdict":        "PASS",
  "issues":         ["missed Q3 details"],
  "hallucinations": []
}

Verdicts:
  PASS:   overall >= 0.80
  REVISE: overall 0.60-0.79
  REJECT: overall < 0.60"""


class JudgeAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name          = "JudgeAgent",
            system_prompt = SYSTEM_PROMPT
        )

    def call_llm(self, user_message, **kwargs):
        """Override to use different model."""
        import os
        import time
        from groq import Groq
        from dotenv import load_dotenv
        load_dotenv()

        key = os.getenv("GROQ_API_KEY", "")
        if not key:
            return ""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                client   = Groq(api_key=key)
                response = client.chat.completions.create(
                    model    = MODEL_OVERRIDE,
                    messages = [
                        {"role": "system",
                         "content": self.system_prompt},
                        {"role": "user",
                         "content": user_message}
                    ],
                    temperature = 0.0,
                    max_tokens  = 400,
                )
                text = self._extract_text(response)
                if text:
                    return text
                time.sleep(2)

            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower():
                    wait = 30 * (attempt + 1)
                    self.log(f"Rate limit. Waiting {wait}s")
                    time.sleep(wait)
                else:
                    self.log(f"Judge LLM error: {err[:60]}")
                    return ""

        return ""

    def evaluate(
        self,
        question: str,
        context:  str,
        answer:   str
    ) -> dict:
        """Evaluate answer quality."""

        prompt = (
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{context[:3000]}\n\n"
            f"ANSWER:\n{answer}\n\n"
            f"Evaluate this answer strictly."
        )

        self.log("Evaluating answer quality...")

        response = self.call_llm(
            user_message = prompt,
        )

        result = self.parse_json(response)

        if not result:
            # Heuristic fallback when LLM fails
            self.log("LLM failed — using heuristic")
            has_sources = (
                "SOURCES:" in answer or
                "Source" in answer or
                "filing" in answer.lower() or
                "10-K" in answer or
                "10-Q" in answer
            )
            is_substantial = len(answer) > 100
            has_numbers    = any(
                c.isdigit() for c in answer
            )

            score = 0.50
            if has_sources:
                score += 0.15
            if is_substantial:
                score += 0.15
            if has_numbers:
                score += 0.10

            score   = round(min(score, 0.90), 2)
            verdict = (
                "PASS"   if score >= 0.80 else
                "REVISE" if score >= 0.60 else
                "REVISE"
            )

            return {
                "groundedness":   score,
                "relevance":      score,
                "completeness":   score,
                "accuracy":       score,
                "overall":        score,
                "verdict":        verdict,
                "issues":         [
                    "Heuristic evaluation — LLM unavailable"
                ],
                "hallucinations": []
            }

        # Compute overall if missing
        if "overall" not in result:
            scores = [
                result.get("groundedness", 0.5),
                result.get("relevance",    0.5),
                result.get("completeness", 0.5),
                result.get("accuracy",     0.5),
            ]
            result["overall"] = round(
                sum(scores) / len(scores), 3
            )

        # Set verdict if missing
        if "verdict" not in result:
            o = result["overall"]
            result["verdict"] = (
                "PASS"   if o >= 0.80 else
                "REVISE" if o >= 0.60 else
                "REJECT"
            )

        self.log(
            f"Verdict: {result['verdict']} "
            f"({result.get('overall',0):.2f})"
        )
        return result