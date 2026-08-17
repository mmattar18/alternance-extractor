"""Thin client for the France Travail "Offres d'emploi v2" API.

Auth: OAuth2 client-credentials against entreprise.francetravail.fr, then
Bearer-token requests against api.francetravail.io. Confirmed endpoints/flow
(no official machine-readable docs to link — francetravail.io's docs are a
JS app that doesn't render for automated fetching, so this was cross-checked
against two independent working integrations before being written down here):
  - token:  https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=/partenaire
  - search: https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search
  - referentiel: https://api.francetravail.io/partenaire/offresdemploi/v2/referentiel/{type}

Deliberately does NOT hardcode natureContrat/typeContrat code meanings —
call get_referentiel() and read the codes back from the API itself rather
than trusting a guessed table. See ingest/explore.py, which is meant to be
run once (after you have credentials) to look at real field names and real
per-keyword counts before we write the full fetch/clean logic.

Requires env vars FRANCE_TRAVAIL_CLIENT_ID and FRANCE_TRAVAIL_CLIENT_SECRET
(put them in a local .env file — see .env.example — never commit them).
"""
from __future__ import annotations

import os
import time
from typing import Any, Iterator, Optional

import requests

TOKEN_URL = "https://entreprise.francetravail.fr/connexion/oauth2/access_token"
API_BASE = "https://api.francetravail.io/partenaire/offresdemploi/v2"
SCOPE_TEMPLATE = "api_offresdemploiv2 o2dsoffre application_{client_id}"

PAGE_SIZE = 150          # max results per request, enforced by the API
MAX_RANGE_START = 1000   # API-enforced ceiling on the range start
MAX_RANGE_END = 1149     # API-enforced ceiling on the range end -> 1150 results/query max
MIN_SECONDS_BETWEEN_CALLS = 0.4  # ~2.5 req/s, under the documented 3 req/s limit


class FranceTravailError(RuntimeError):
    pass


class FranceTravailClient:
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        self.client_id = client_id or os.environ["FRANCE_TRAVAIL_CLIENT_ID"]
        self.client_secret = client_secret or os.environ["FRANCE_TRAVAIL_CLIENT_SECRET"]
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._last_call_at: float = 0.0

    def _get_token(self) -> str:
        if self._token and time.monotonic() < self._token_expires_at - 30:
            return self._token
        resp = requests.post(
            TOKEN_URL,
            params={"realm": "/partenaire"},
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": SCOPE_TEMPLATE.format(client_id=self.client_id),
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        if resp.status_code != 200:
            raise FranceTravailError(
                f"Token request failed ({resp.status_code}): {resp.text[:500]}"
            )
        payload = resp.json()
        self._token = payload["access_token"]
        self._token_expires_at = time.monotonic() + payload.get("expires_in", 1500)
        return self._token

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_at
        if elapsed < MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        for attempt in range(5):
            self._throttle()
            headers = kwargs.pop("headers", {})
            headers["Authorization"] = f"Bearer {self._get_token()}"
            resp = requests.request(method, url, headers=headers, timeout=30, **kwargs)
            self._last_call_at = time.monotonic()
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 2 * (attempt + 1)))
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            return resp
        raise FranceTravailError(f"Gave up after retries on {url}")

    def get_referentiel(self, name: str) -> list[dict]:
        """name examples to try: 'typesContrats', 'naturesContrats', 'niveauxFormations',
        'permis', 'themes'. Call this and print the result — don't guess."""
        resp = self._request("GET", f"{API_BASE}/referentiel/{name}")
        if resp.status_code != 200:
            raise FranceTravailError(f"referentiel/{name} failed ({resp.status_code}): {resp.text[:500]}")
        return resp.json()

    def search_page(self, params: dict[str, Any], range_start: int, range_end: int) -> tuple[dict, dict]:
        """One page of search results. Returns (body, content_range_dict)."""
        headers = {"Range": f"offres={range_start}-{range_end}"}
        resp = self._request("GET", f"{API_BASE}/offres/search", params=params, headers=headers)
        if resp.status_code not in (200, 206):
            raise FranceTravailError(f"search failed ({resp.status_code}): {resp.text[:500]}")
        body = resp.json() if resp.text else {"resultats": []}
        content_range = self._parse_content_range(resp.headers.get("Content-Range", ""))
        return body, content_range

    @staticmethod
    def _parse_content_range(header_value: str) -> dict:
        # Format observed: "offres 0-149/1234"
        if not header_value or "/" not in header_value:
            return {}
        range_part, total = header_value.rsplit("/", 1)
        nums = range_part.split()[-1]
        first, last = nums.split("-")
        return {"first_index": int(first), "last_index": int(last), "max_results": int(total)}

    def count(self, params: dict[str, Any]) -> int:
        """Cheap way to see how many results a query has: fetch page 0 and read the total."""
        _, content_range = self.search_page(params, 0, PAGE_SIZE - 1)
        return content_range.get("max_results", 0)

    def search_all(self, params: dict[str, Any], max_results: Optional[int] = None) -> Iterator[dict]:
        """Yields raw offer dicts, paginating until max_results, MAX_RANGE_END, or the
        query's real total is exhausted — whichever comes first."""
        start = 0
        yielded = 0
        while start <= MAX_RANGE_END:
            end = min(start + PAGE_SIZE - 1, MAX_RANGE_END)
            body, content_range = self.search_page(params, start, end)
            offers = body.get("resultats", [])
            for offer in offers:
                yield offer
                yielded += 1
                if max_results is not None and yielded >= max_results:
                    return
            total = content_range.get("max_results", 0)
            if not offers or end >= total - 1:
                return
            start = end + 1
