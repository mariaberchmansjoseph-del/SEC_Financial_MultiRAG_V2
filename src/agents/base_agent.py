"""
Base class for all SEC RAG agents.
Handles LLM calls with retry, rate limiting,
and reasoning model support.
src/agents/base_agent.py
"""

import os
import re
import json
import time
from dotenv import load_dotenv

load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY", "")
MODEL    = "qwen/qwen3.8-27b"


class BaseAgent:
    """
    Foundation class for all agents.
    Provides: LLM calls, JSON parsing, logging,
    rate limit retry, reasoning model support.
    """

    def __init__(self, name: str, system_prompt: str):
        self.name          = name
        self.system_prompt = system_prompt
        self.total_tokens  = 0
        self.total_calls   = 0
        self.client        = None

        if not GROQ_KEY:
            print(
                f"  [{name}] WARNING: "
                f"GROQ_API_KEY not found in .env"
            )
            return

        try:
            from groq import Groq
            self.client = Groq(api_key=GROQ_KEY)
        except Exception as e:
            print(f"  [{name}] Groq init error: {e}")

    # ── LLM CALL ─────────────────────────────────────────

    def call_llm(
        self,
        user_message: str,
        temperature:  float = 0.1,
        max_tokens:   int   = 600,
        json_mode:    bool  = False,
        model:        str   = None,
    ) -> str:
        """
        Call the LLM with automatic retry on rate limits.

        Args:
            user_message: The user prompt
            temperature:  Sampling temperature (0=deterministic)
            max_tokens:   Max tokens to generate
            json_mode:    Request JSON output format
            model:        Override default model

        Returns:
            Response text or empty string on failure
        """
        if not self.client:
            return ""

        self.total_calls += 1
        use_model = model or MODEL
        max_retries = 3

        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model":    use_model,
                    "messages": [
                        {
                            "role":    "system",
                            "content": self.system_prompt
                        },
                        {
                            "role":    "user",
                            "content": user_message
                        }
                    ],
                    "temperature": temperature,
                    "max_tokens":  max_tokens,
                }

                response = (
                    self.client.chat.completions.create(
                        **kwargs
                    )
                )

                self.total_tokens += (
                    response.usage.total_tokens
                )

                text = self._extract_text(response)

                if text:
                    return text

                # Empty response — log and retry
                self.log(
                    f"Empty response "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(2)

            except Exception as e:
                err = str(e)

                if "429" in err or \
                   "rate" in err.lower() or \
                   "limit" in err.lower():
                    wait = 30 * (attempt + 1)
                    self.log(
                        f"Rate limit hit. "
                        f"Waiting {wait}s... "
                        f"(attempt {attempt + 1})"
                    )
                    time.sleep(wait)
                    continue

                elif "404" in err:
                    self.log(
                        f"Model not found: {use_model}. "
                        f"Check available models."
                    )
                    return ""

                elif "401" in err or "invalid" in err.lower():
                    self.log(
                        "Invalid API key. "
                        "Check GROQ_API_KEY in .env"
                    )
                    return ""

                else:
                    self.log(
                        f"LLM error: {err[:100]}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(5)
                    continue

        self.log(
            f"All {max_retries} attempts failed"
        )
        return ""

    def _extract_text(self, response) -> str:
        """
        Extract text from LLM response.
        Handles both regular and reasoning models.

        Regular models: text in content field
        Reasoning models (gpt-oss-*): text in reasoning field
        """
        try:
            choice  = response.choices[0]
            message = choice.message

            # Standard content field
            content = getattr(message, "content", "") or ""
            if content and len(content.strip()) > 5:
                return content.strip()

            # Reasoning field (reasoning models)
            reasoning = (
                getattr(message, "reasoning", "") or ""
            )
            if reasoning and len(reasoning.strip()) > 5:
                # Try to find JSON in reasoning
                json_match = re.search(
                    r'\{.*\}', reasoning, re.DOTALL
                )
                if json_match:
                    return json_match.group()
                return reasoning.strip()

        except Exception as e:
            self.log(f"Text extraction error: {e}")

        return ""

    # ── JSON PARSING ──────────────────────────────────────

    def parse_json(self, text: str) -> dict:
        """
        Safely parse JSON from LLM response.
        Handles common formatting issues.
        """
        if not text:
            return {}

        # Try direct parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # Remove markdown code blocks
        cleaned = re.sub(
            r'```(?:json)?\s*', '', text
        ).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # Extract first JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass

        # Extract first JSON array
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return {"items": result}
            except Exception:
                pass

        return {}

    def parse_json_list(self, text: str) -> list:
        """Parse JSON array from LLM response."""
        if not text:
            return []

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except Exception:
            pass

        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return result
            except Exception:
                pass

        return []

    # ── LOGGING ───────────────────────────────────────────

    def log(self, message: str):
        """Print agent log message."""
        print(f"  [{self.name}] {message}")

    # ── STATS ─────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return agent usage statistics."""
        return {
            "agent":        self.name,
            "total_calls":  self.total_calls,
            "total_tokens": self.total_tokens,
            "model":        MODEL,
        }