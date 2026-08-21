"""Split data/labelled/all_labelled.jsonl into train and test-candidates.

Test set: 100 postings, stratified proportionally by bucket (excluding
data_ai_stage — only 2 postings exist total, too few to hold any out and
still leave train with useful signal, so both stay in train; documented as a
limitation, not silently dropped). Selection is seeded for reproducibility.

Test-candidate pool is restricted to rows labelled by Groq specifically
(`labeller` starting with "groq"; rows from before that field existed are
treated as Groq too, since Groq was the only labeller at the time). If a
second provider (e.g. Gemini, see run_labelling.py) helped label some
postings, its rows can still be used for training but must never end up as
a "Groq baseline" prediction in eval/score.py -- that would silently corrupt
the baseline comparison the whole project is built around.

Train set: everything not selected for test, using each row's own
`prediction` as the label — but only where the labeller produced valid
JSON. Postings where extraction failed are dropped from train and counted,
not silently discarded. Train set is not restricted to Groq -- a second
provider's labels add noisy-but-usable training signal.

Enforces schema.dedupe.assert_no_test_leakage between the two splits before
writing anything, exactly at the point where the split is created.

Output:
  data/test/test_candidates.jsonl  — 100 postings + Groq's prediction as a
                                      prefill, ready for label/review_server.py
  data/train/train.jsonl           — Groq-labelled training data

Run: python label/select_test_set.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from schema.dedupe import assert_no_test_leakage  # noqa: E402

LABELLED_PATH = REPO_ROOT / "data" / "labelled" / "all_labelled.jsonl"
TEST_CANDIDATES_PATH = REPO_ROOT / "data" / "test" / "test_candidates.jsonl"
TRAIN_PATH = REPO_ROOT / "data" / "train" / "train.jsonl"

TEST_SIZE = 100
SEED = 42


def proportional_allocation(bucket_sizes: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder rounding so allocations sum to exactly `total`."""
    grand_total = sum(bucket_sizes.values())
    raw = {b: n * total / grand_total for b, n in bucket_sizes.items()}
    floors = {b: int(v) for b, v in raw.items()}
    remainder = total - sum(floors.values())
    order = sorted(raw, key=lambda b: raw[b] - floors[b], reverse=True)
    for b in order[:remainder]:
        floors[b] += 1
    return floors


def is_groq(record: dict) -> bool:
    return record.get("labeller", "groq_openai/gpt-oss-120b").startswith("groq")


def main() -> None:
    records = [json.loads(l) for l in LABELLED_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    groq_pool = [r for r in records if is_groq(r)]
    print(f"{len(records)} labelled postings loaded ({len(groq_pool)} Groq-labelled, "
          f"eligible for the test set)")

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in groq_pool:
        by_bucket[r["bucket"]].append(r)

    bucket_sizes = {b: len(rs) for b, rs in by_bucket.items()}
    print("bucket sizes:", bucket_sizes)

    allocation = proportional_allocation(bucket_sizes, TEST_SIZE)
    print("test allocation:", allocation)

    rng = random.Random(SEED)
    test_ids: set[str] = set()
    for bucket, n in allocation.items():
        pool = by_bucket[bucket]
        n = min(n, len(pool))
        chosen = rng.sample(pool, n)
        test_ids.update(r["posting_id"] for r in chosen)

    test_records = [r for r in records if r["posting_id"] in test_ids]
    train_pool = [r for r in records if r["posting_id"] not in test_ids]

    train_records = [r for r in train_pool if r["valid"]]
    dropped_invalid = len(train_pool) - len(train_records)

    assert_no_test_leakage(
        train_ids=[r["posting_id"] for r in train_records],
        train_hashes=[r["text_hash"] for r in train_records],
        test_ids=[r["posting_id"] for r in test_records],
        test_hashes=[r["text_hash"] for r in test_records],
    )
    print("Leakage check passed: no overlap between train and test.")

    TEST_CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TEST_CANDIDATES_PATH.open("w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_PATH.open("w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nTest candidates: {len(test_records)} -> {TEST_CANDIDATES_PATH.relative_to(REPO_ROOT)}")
    print(f"Train set: {len(train_records)} -> {TRAIN_PATH.relative_to(REPO_ROOT)} "
          f"({dropped_invalid} dropped: labeller produced invalid JSON)")
    print("Test bucket composition:", Counter(r["bucket"] for r in test_records))


if __name__ == "__main__":
    main()
