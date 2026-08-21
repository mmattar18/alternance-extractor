"""Thin client for Groq's OpenAI-compatible chat completions API.

Used for: labelling the train set, the Groq baseline eval, and nothing else.
Retries on invalid JSON by feeding the validation error back to the model,
up to MAX_ATTEMPTS times, then gives up and reports the failure honestly
(callers must record failed extractions for the JSON-validity-rate metric,
not silently drop them).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from schema.posting import JobPosting, parse_llm_json  # noqa: E402
from label.prompt import build_messages  # noqa: E402

API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Llama 3.3 70B (the model originally specified) was deprecated by Groq for
# free/developer tier on 2026-06-17, before this project started. This is
# Groq's own recommended replacement for that exact migration path — not a
# preference swap. See console.groq.com/docs/deprecations.
MODEL = "openai/gpt-oss-120b"
MAX_ATTEMPTS = 3
MIN_SECONDS_BETWEEN_CALLS = 2.1  # conservative default; tightened further on 429


class GroqExtractionResult:
    def __init__(self, posting: Optional[JobPosting], raw_response: str,
                 error: Optional[str], attempts: int, latency_seconds: float):
        self.posting = posting
        self.raw_response = raw_response
        self.error = error
        self.attempts = attempts
        self.latency_seconds = latency_seconds

    @property
    def valid(self) -> bool:
        return self.posting is not None


class GroqClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ["GROQ_API_KEY"]
        self._last_call_at = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_at
        if elapsed < MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)

    def _chat(self, messages: list[dict]) -> str:
        for attempt in range(5):
            self._throttle()
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": messages,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
                timeout=60,
            )
            self._last_call_at = time.monotonic()
            if resp.status_code == 429:
                wait = float(resp.headers.get("retry-after", 5 * (attempt + 1)))
                # Visibility into *which* cap was hit (requests vs tokens, and
                # whether it's a per-minute or per-day window) -- without this,
                # a long wait here is indistinguishable from a hang.
                print(
                    f"    [429] retry-after={wait}s "
                    f"reqs={resp.headers.get('x-ratelimit-remaining-requests')}/{resp.headers.get('x-ratelimit-limit-requests')} "
                    f"tokens={resp.headers.get('x-ratelimit-remaining-tokens')}/{resp.headers.get('x-ratelimit-limit-tokens')} "
                    f"body={resp.text[:200]}",
                    flush=True,
                )
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                print(f"    [{resp.status_code}] {resp.text[:200]}", flush=True)
                time.sleep(3 * (attempt + 1))
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"Groq request failed ({resp.status_code}): {resp.text[:500]}")
            return resp.json()["choices"][0]["message"]["content"]
        raise RuntimeError("Gave up on Groq request after repeated 429/5xx")

    def extract(self, raw_text: str) -> GroqExtractionResult:
        """Runs the extraction prompt, retrying on schema-invalid output by
        showing the model its own mistake."""
        start = time.monotonic()
        messages = build_messages(raw_text)
        last_response = ""
        last_error = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            content = self._chat(messages)
            last_response = content
            posting, error = parse_llm_json(content)
            if posting is not None:
                return GroqExtractionResult(posting, content, None, attempt, time.monotonic() - start)
            last_error = error
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"That was not valid: {error}. Return ONLY a corrected JSON object matching the schema.",
            })
        return GroqExtractionResult(None, last_response, last_error, MAX_ATTEMPTS, time.monotonic() - start)
