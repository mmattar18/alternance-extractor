"""Sanity check the Groq extraction prompt on a few real postings before
running the full labelling job. Run: python label/try_one.py [n]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from label.groq_client import GroqClient  # noqa: E402

RAW_PATH = REPO_ROOT / "data" / "raw" / "postings.jsonl"


def main() -> None:
    load_dotenv()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    lines = RAW_PATH.read_text(encoding="utf-8").splitlines()
    client = GroqClient()

    for i, line in enumerate(lines[:n]):
        record = json.loads(line)
        print("=" * 70)
        print(f"[{i}] {record['posting_id']}  ({record['bucket']})")
        print("-" * 70)
        print(record["raw_text"][:600])
        print("-" * 70)
        result = client.extract(record["raw_text"])
        if result.valid:
            print(f"OK in {result.attempts} attempt(s), {result.latency_seconds:.1f}s")
            print(json.dumps(result.posting.model_dump(), indent=2, ensure_ascii=False))
        else:
            print(f"FAILED after {result.attempts} attempts: {result.error}")
            print("Last raw response:", result.raw_response[:500])
        print()


if __name__ == "__main__":
    main()
