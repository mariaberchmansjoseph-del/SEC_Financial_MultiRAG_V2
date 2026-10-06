"""
Judge Agent — LLM-as-Judge quality evaluation.
src/agents/judge_agent.py
"""

import sys
sys.path.insert(0, '.')
from src.agents.base_agent import BaseAgent

# Different model to avoid self-evaluation bias
#MODEL_OVERRIDE = "llama-3.1-8b-instant"
MODEL_OVERRIDE = "qwen/qwen3.8-27b"
SYSTEM_PROMPT = """You are a financial analysis quality
evaluator. Assess AI-generated answers about SEC filings.

EVALUATE ON FOUR DIMENSIONS (0.0 to 1.0):

GROUNDEDNESS: Every claim must be in the context.
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
        from groq import Groq
        from dotenv import load_dotenv
        load_dotenv()
        key = os.getenv("GROQ_API_KEY", "")
        if not key:
            return ""
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
            return response.choices[0].message.content
        except Exception as e:
            self.log(f"Judge LLM error: {str(e)[:60]}")
            return ""

    def evaluate(
        self, question: str,
        context: str, answer: str
    ) -> dict:
        prompt = (
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{context[:3000]}\n\n"
            f"ANSWER:\n{answer}\n\n"
            f"Evaluate this answer."
        )
        self.log("Evaluating answer quality...")
        response = self.call_llm(
            user_message = prompt,
            json_mode    = True
        )
        result = self.parse_json(response)
        if not result:
            return {
                "groundedness": 0.5,
                "relevance":    0.5,
                "completeness": 0.5,
                "accuracy":     0.5,
                "overall":      0.5,
                "verdict":      "REVISE",
                "issues":       ["evaluation failed"],
                "hallucinations": []
            }
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
        if "verdict" not in result:
            o = result["overall"]
            result["verdict"] = (
                "PASS"   if o >= 0.80 else
                "REVISE" if o >= 0.60 else
                "REJECT"
            )
        self.log(
            f"Verdict: {result['verdict']} "
            f"({result['overall']:.2f})"
        )
        return result