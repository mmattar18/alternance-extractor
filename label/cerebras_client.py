"""Thin client for Cerebras's OpenAI-compatible chat completions API --
mirrors groq_client.py's interface (extract() -> result with .posting/
.valid/.error/.attempts/.latency_seconds) so run_labelling.py can use it
interchangeably with the other providers.

Exists to add labelling throughput on a separate quota from Groq's TPD cap
(see run_labelling.py's ACTIVE_PROVIDERS/partition()). Train-only: never
eligible for the test set (select_test_set.py only draws test candidates
from labeller values starting with "groq"), so mixing a different teacher
model's output can never corrupt the Groq-baseline eval.

MODEL default below was current as of this project's writing -- Cerebras
cycles model availability as they optimize/deprecate, so verify at
https://inference-docs.cerebras.ai before relying on it; override with
CEREBRAS_MODEL. Handles network faults and malformed 200 responses the same
way openrouter_client.py does, learned from that client crashing twice on a
free-tier provider's rough edges.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from schema.posting import JobPosting, parse_llm_json  # noqa: E402
from label.prompt import build_messages  # noqa: E402  -- same OpenAI-style message shape as Groq

API_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = os.environ.get("CEREBRAS_MODEL", "llama-3.3-70b")
MAX_ATTEMPTS = 3
MIN_SECONDS_BETWEEN_CALLS = 1.0


class CerebrasExtractionResult:
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


class CerebrasClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ["CEREBRAS_API_KEY"]
        self._last_call_at = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_at
        if elapsed < MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)

    def _post(self, messages: list[dict]):
        return requests.post(
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

    def _chat(self, messages: list[dict]) -> str:
        for attempt in range(5):
            self._throttle()
            try:
                resp = self._post(messages)
            except requests.exceptions.RequestException as e:
                print(f"    [cerebras network error] {e}", flush=True)
                time.sleep(3 * (attempt + 1))
                continue
            self._last_call_at = time.monotonic()
            if resp.status_code == 429:
                wait = float(resp.headers.get("retry-after", 5 * (attempt + 1)))
                print(f"    [cerebras 429] retry-after={wait}s body={resp.text[:300]}", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                print(f"    [cerebras {resp.status_code}] {resp.text[:300]}", flush=True)
                time.sleep(3 * (attempt + 1))
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"Cerebras request failed ({resp.status_code}): {resp.text[:500]}")
            try:
                return resp.json()["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError):
                print(f"    [cerebras malformed response] {resp.text[:300]}", flush=True)
                return json.dumps({"_cerebras_malformed_response": resp.text[:500]})
        raise RuntimeError("Gave up on Cerebras request after repeated 429/5xx/network errors")

    def extract(self, raw_text: str) -> CerebrasExtractionResult:
        start = time.monotonic()
        messages = build_messages(raw_text)
        last_response = ""
        last_error = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            content = self._chat(messages)
            last_response = content
            posting, error = parse_llm_json(content)
            if posting is not None:
                return CerebrasExtractionResult(posting, content, None, attempt, time.monotonic() - start)
            last_error = error
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"That was not valid: {error}. Return ONLY a corrected JSON object matching the schema.",
            })
        return CerebrasExtractionResult(None, last_response, last_error, MAX_ATTEMPTS, time.monotonic() - start)
