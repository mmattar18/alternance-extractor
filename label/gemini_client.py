"""Thin client for Gemini's generateContent API -- mirrors groq_client.py's
interface (extract() -> result with .posting/.valid/.error/.attempts/
.latency_seconds) so run_labelling.py can use either provider interchangeably.

Exists to add labelling throughput on a separate quota from Groq's 8000 TPM
cap. Only used for postings NOT eligible for the test set (see
run_labelling.py's partition() and select_test_set.py's groq-only filter) --
so the Groq-baseline eval in eval/score.py always scores genuine Groq output,
never a blend of two different teacher models.

MODEL default below was current as of this project's writing -- verify
against https://ai.google.dev/gemini-api/docs/models before relying on it;
override with the GEMINI_MODEL env var if it's been renamed/retired.
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
from label.prompt import SYSTEM_PROMPT, build_gemini_contents  # noqa: E402

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
MAX_ATTEMPTS = 3
MIN_SECONDS_BETWEEN_CALLS = 1.0


class GeminiExtractionResult:
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


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ["GEMINI_API_KEY"]
        self._last_call_at = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_at
        if elapsed < MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)

    def _generate(self, contents: list[dict]) -> str:
        for attempt in range(5):
            self._throttle()
            resp = requests.post(
                API_URL,
                params={"key": self.api_key},
                json={
                    "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    "contents": contents,
                    "generationConfig": {"temperature": 0, "response_mime_type": "application/json"},
                },
                timeout=60,
            )
            self._last_call_at = time.monotonic()
            if resp.status_code == 429:
                wait = float(resp.headers.get("retry-after", 5 * (attempt + 1)))
                print(f"    [gemini 429] retry-after={wait}s body={resp.text[:200]}", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                print(f"    [gemini {resp.status_code}] {resp.text[:200]}", flush=True)
                time.sleep(3 * (attempt + 1))
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"Gemini request failed ({resp.status_code}): {resp.text[:500]}")
            data = resp.json()
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                # Safety-filtered or otherwise empty response -- treat as a
                # schema violation so it goes through the same retry-with-
                # corrected-JSON path as any other invalid output.
                return json.dumps({"_gemini_empty_response": data})
        raise RuntimeError("Gave up on Gemini request after repeated 429/5xx")

    def extract(self, raw_text: str) -> GeminiExtractionResult:
        start = time.monotonic()
        contents = build_gemini_contents(raw_text)
        last_response = ""
        last_error = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            content = self._generate(contents)
            last_response = content
            posting, error = parse_llm_json(content)
            if posting is not None:
                return GeminiExtractionResult(posting, content, None, attempt, time.monotonic() - start)
            last_error = error
            contents.append({"role": "model", "parts": [{"text": content}]})
            contents.append({
                "role": "user",
                "parts": [{"text": f"That was not valid: {error}. Return ONLY a corrected JSON object matching the schema."}],
            })
        return GeminiExtractionResult(None, last_response, last_error, MAX_ATTEMPTS, time.monotonic() - start)
