"""Fetch ADDITIONAL postings, targeted at the fields the fine-tuned model starves on.

Why this exists rather than just re-running fetch.py with a bigger cap: error analysis
(RESULTS.md) showed the remaining gap to the Groq baseline is concentrated in
LOW-FREQUENCY fields. Pearson r between per-field training-example count and gap-to-Groq
is -0.456, and fields with <150 training examples average a +0.133 gap versus +0.069 for
fields with >=150:

    remote_policy           32 training examples   gap +0.246
    alternance_rhythm       47                     gap +0.196
    salary_range            49                     gap +0.140
    language_requirements   57                     gap +0.296

The original KEYWORD_PLAN targets job *roles* (data engineer, developpeur...), which says
nothing about whether a posting happens to mention remote work, a language requirement, a
salary or a rhythm. The keywords below search for those attributes directly, so the new
postings should be enriched for exactly the fields that are starved.

Two hard safety rules, both enforced before anything is written:

1. Nothing already in data/raw/postings.jsonl (by offer id or by text_hash).
2. Nothing that is a NEAR-duplicate of a TEST posting. The existing pipeline only ever
   checked exact id/hash collisions, and that is precisely how 10% of the test set ended
   up with a near-identical twin in train (see schema/dedupe.find_near_duplicates and
   data/test/leaked_posting_ids.json). Growing the training set is the single easiest way
   to make that worse, so every candidate is checked against every test posting at a
   deliberately conservative Jaccard threshold before being kept.

Run: python ingest/fetch_more.py
Writes: data/raw/postings_v2.jsonl (new file; does not touch the original)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

from schema.dedupe import text_hash, shingles, jaccard  # noqa: E402
from client import FranceTravailClient  # noqa: E402
from fetch import build_record, MIN_DESCRIPTION_CHARS  # noqa: E402

EXISTING_PATH = REPO_ROOT / "data" / "raw" / "postings.jsonl"
TEST_PATH = REPO_ROOT / "data" / "test" / "test.jsonl"
OUTPUT_PATH = REPO_ROOT / "data" / "raw" / "postings_v2.jsonl"

# Conservative on purpose: cheaper to discard a usable posting than to poison the test set.
NEAR_DUP_THRESHOLD = 0.35

# (bucket, keyword, cap). Keywords chosen to surface the ATTRIBUTE, not the role.
KEYWORD_PLAN: list[tuple[str, str, int | None]] = [
    # remote_policy (32 examples today -- the most starved)
    ("remote_enriched", "alternance teletravail", 120),
    ("remote_enriched", "alternance hybride", 80),
    ("remote_enriched", "apprentissage teletravail partiel", 80),
    # language_requirements (57) -- biggest single gap at +0.296
    ("language_enriched", "alternance anglais courant", 120),
    ("language_enriched", "alternance bilingue", 80),
    ("language_enriched", "apprentissage anglais professionnel", 80),
    # alternance_rhythm (47)
    ("rhythm_enriched", "alternance rythme semaine entreprise", 100),
    ("rhythm_enriched", "apprentissage 3 semaines entreprise 1 semaine ecole", 80),
    # salary_range (49)
    ("salary_enriched", "alternance remuneration euros", 100),
    ("salary_enriched", "apprentissage salaire brut mensuel", 80),
]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    load_dotenv()

    existing = load_jsonl(EXISTING_PATH)
    existing_ids = {r["posting_id"] for r in existing}
    existing_hashes = {r["text_hash"] for r in existing}
    print(f"{len(existing)} postings already in {EXISTING_PATH.name}")

    test_rows = load_jsonl(TEST_PATH)
    test_shingles = [shingles(r["raw_text"]) for r in test_rows]
    print(f"{len(test_rows)} test postings loaded for near-duplicate screening "
          f"(threshold {NEAR_DUP_THRESHOLD})")

    client = FranceTravailClient()
    kept: list[dict] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    n_dup_existing = n_dup_batch = n_short = n_near_test = 0

    for bucket, keyword, cap in KEYWORD_PLAN:
        before = len(kept)
        for offer in client.search_all({"motsCles": keyword}, max_results=cap):
            pid = f"france_travail:{offer['id']}"
            if pid in existing_ids or pid in seen_ids:
                n_dup_existing += 1
                continue
            record = build_record(offer, bucket, keyword)
            if record["description_chars"] < MIN_DESCRIPTION_CHARS:
                n_short += 1
                continue
            if record["text_hash"] in existing_hashes:
                n_dup_existing += 1
                continue
            if record["text_hash"] in seen_hashes:
                n_dup_batch += 1
                continue
            cand = shingles(record["raw_text"])
            if any(jaccard(cand, t) >= NEAR_DUP_THRESHOLD for t in test_shingles):
                n_near_test += 1
                continue
            seen_ids.add(pid)
            seen_hashes.add(record["text_hash"])
            kept.append(record)
        print(f"[{bucket}] '{keyword}': +{len(kept) - before} kept (running total {len(kept)})")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nkept {len(kept)} new postings -> {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"  dropped: {n_dup_existing} already known, {n_dup_batch} dupes within batch, "
          f"{n_short} too short, {n_near_test} NEAR-DUPLICATE OF A TEST POSTING")
    from collections import Counter
    print("  by bucket:", dict(Counter(r["bucket"] for r in kept)))


if __name__ == "__main__":
    main()
