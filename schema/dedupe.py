"""Hashing helpers for de-duplicating postings and protecting the test set.

Two postings are treated as the same posting if either:
- they share a `posting_id` (assigned at ingestion, e.g. source + source-side
  ID), or
- their cleaned text hashes to the same value (catches reposts/mirrors that
  get different IDs across or within a source).

`assert_no_test_leakage` is meant to be called from ingest/ and from the
training notebook, right before anything gets written to the train set or
handed to the trainer, so a leak fails loudly instead of quietly inflating
eval numbers.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def assert_no_test_leakage(
    train_ids: Iterable[str],
    train_hashes: Iterable[str],
    test_ids: Iterable[str],
    test_hashes: Iterable[str],
) -> None:
    train_ids, test_ids = set(train_ids), set(test_ids)
    train_hashes, test_hashes = set(train_hashes), set(test_hashes)

    id_overlap = train_ids & test_ids
    hash_overlap = train_hashes & test_hashes
    if id_overlap or hash_overlap:
        raise ValueError(
            "Test-set leakage detected — refusing to continue.\n"
            f"  posting_id overlap ({len(id_overlap)}): {sorted(id_overlap)[:10]}\n"
            f"  text_hash overlap ({len(hash_overlap)}) — same posting text under a different id"
        )
