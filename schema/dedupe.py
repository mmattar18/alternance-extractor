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
import unicodedata
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


# --- near-duplicate detection -------------------------------------------------
# assert_no_test_leakage() above catches only EXACT collisions (same posting_id or
# same text_hash). France Travail re-lists the same job under a new reference with
# small edits, which slips straight past an exact hash. Measured on the real split:
# 10 of 100 test postings had a train posting at >= 0.5 five-gram Jaccard, four of
# them above 0.9 -- i.e. the fine-tuned model had effectively already seen 10% of its
# own test set. That inflates a FINE-TUNED model's score while leaving a
# prompted-only baseline untouched, so it silently biases exactly the comparison
# this project exists to make.

_WORD_RE = re.compile(r"[^a-z0-9 ]")


def shingles(text: str, k: int = 5) -> set[str]:
    """Case/accent/punctuation-insensitive word k-gram set."""
    s = unicodedata.normalize("NFKD", text)
    s = "".join(c for c in s if not unicodedata.combining(c))
    words = _WORD_RE.sub(" ", s.lower()).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b) if inter else 0.0


def find_near_duplicates(
    train_texts: dict[str, str],
    test_texts: dict[str, str],
    threshold: float = 0.5,
) -> list[tuple[str, str, float]]:
    """(test_id, train_id, jaccard) for every pair at or above `threshold`.

    Deliberately returns the pairs instead of raising: near-duplicates are a fact to
    measure and report (score the leak-free subset alongside the full set), not
    necessarily a hard error the way an exact-id collision is.
    """
    train_sh = {tid: shingles(t) for tid, t in train_texts.items()}
    out = []
    for te_id, te_text in test_texts.items():
        a = shingles(te_text)
        for tr_id, b in train_sh.items():
            j = jaccard(a, b)
            if j >= threshold:
                out.append((te_id, tr_id, round(j, 3)))
    return sorted(out, key=lambda r: -r[2])
