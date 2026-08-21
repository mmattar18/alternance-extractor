"""Label every posting in data/raw/postings.jsonl with the Groq teacher model.

Output: data/labelled/all_labelled.jsonl — one line per input posting, with
the original fields plus `prediction` (the schema dict, or null), `valid`,
`error`, `attempts`, `latency_seconds`. Failed extractions are recorded, not
dropped, so the JSON-validity rate is measurable later.

Resumable: already-labelled posting_ids (by id present in the output file)
are skipped on re-run, and each result is flushed immediately, so an
interrupted run can just be restarted.

Multiple providers can run at once, each in its own process, against
disjoint posting_ids (see partition() below) to get around any single
provider's rate/quota cap. Only Groq-labelled rows are eligible for the test
set (enforced in select_test_set.py) so the Groq-baseline eval never scores
a different model's output by accident.

IMPORTANT: ACTIVE_PROVIDERS below must list every provider you're actually
running concurrently, in the same order, on every invocation -- partition()
hashes posting_id against this exact list, so if two concurrently-running
processes disagree on it, they can double-label (wasted calls, harmless) or
worse, silently skip ids (if one process's list omits a provider that's
mid-run under a different list). Update it, then start all the provider
processes fresh -- don't mix an old-list process with a new-list one.

Run: python label/run_labelling.py [groq|gemini|openrouter]   (default: groq)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RAW_PATH = REPO_ROOT / "data" / "raw" / "postings.jsonl"
OUT_PATH = REPO_ROOT / "data" / "labelled" / "all_labelled.jsonl"

PROVIDERS = {
    "groq": "groq_openai/gpt-oss-120b",
    "gemini": None,       # resolved at runtime from GEMINI_MODEL
    "openrouter": None,   # resolved at runtime from OPENROUTER_MODEL
}

# Whichever providers are actually being run concurrently right now.
# Gemini is left out: its free-tier quota was exhausted almost immediately
# (see label/gemini_client.py's docstring) and it isn't currently running.
ACTIVE_PROVIDERS = ("groq", "openrouter")


def partition(posting_id: str) -> str:
    """Deterministic split across ACTIVE_PROVIDERS, by posting_id hash rather
    than list order (postings.jsonl is bucketed sequentially by fetch
    keyword -- an order-based split would starve one provider of certain
    buckets). Every concurrently-running process computes this the same way
    from the same ACTIVE_PROVIDERS, so there's no coordination needed."""
    digest = hashlib.sha256(posting_id.encode("utf-8")).hexdigest()
    return ACTIVE_PROVIDERS[int(digest, 16) % len(ACTIVE_PROVIDERS)]


def already_labelled() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    ids = set()
    for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line)["posting_id"])
    return ids


def main() -> None:
    provider = sys.argv[1] if len(sys.argv) > 1 else "groq"
    if provider not in PROVIDERS:
        raise SystemExit(f"unknown provider {provider!r}, expected one of {list(PROVIDERS)}")

    load_dotenv()
    import os
    if provider == "groq":
        from label.groq_client import GroqClient
        client = GroqClient()
        labeller_tag = PROVIDERS["groq"]
    elif provider == "gemini":
        from label.gemini_client import GeminiClient
        client = GeminiClient()
        labeller_tag = f"gemini_{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"
    else:
        from label.openrouter_client import OpenRouterClient
        client = OpenRouterClient()
        labeller_tag = f"openrouter_{os.environ.get('OPENROUTER_MODEL', 'openai/gpt-oss-20b:free')}"

    records = [json.loads(l) for l in RAW_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = already_labelled()
    todo = [r for r in records if r["posting_id"] not in done and partition(r["posting_id"]) == provider]

    print(f"provider={provider} labeller={labeller_tag}")
    print(f"{len(records)} total postings, {len(done)} already labelled, "
          f"{len(todo)} assigned to {provider} in this run")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_valid = 0
    n_failed = 0
    start = time.monotonic()

    with OUT_PATH.open("a", encoding="utf-8") as f:
        for i, record in enumerate(todo, 1):
            result = client.extract(record["raw_text"])
            out = dict(record)
            out["prediction"] = result.posting.model_dump() if result.valid else None
            out["valid"] = result.valid
            out["error"] = result.error
            out["attempts"] = result.attempts
            out["latency_seconds"] = result.latency_seconds
            out["labeller"] = labeller_tag
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

            if result.valid:
                n_valid += 1
            else:
                n_failed += 1
                print(f"  [{i}/{len(todo)}] FAILED {record['posting_id']}: {result.error}")

            if i % 25 == 0 or i == len(todo):
                elapsed = time.monotonic() - start
                rate = i / elapsed
                eta_min = (len(todo) - i) / rate / 60 if rate > 0 else 0
                print(f"[{i}/{len(todo)}] valid={n_valid} failed={n_failed} "
                      f"elapsed={elapsed/60:.1f}min eta={eta_min:.1f}min")

    print("\nDone.")
    print(f"This run: {n_valid} valid, {n_failed} failed out of {len(todo)}")
    print(f"Total in {OUT_PATH.relative_to(REPO_ROOT)}: {len(already_labelled())}")


if __name__ == "__main__":
    main()
