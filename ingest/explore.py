"""Run this once, after you have France Travail API credentials, before we
write the real fetch/clean pipeline. It doesn't save anything — it just
prints ground truth so we stop guessing:

  1. The real referentiel codes for typesContrats / naturesContrats (so we
     know how to filter for alternance instead of assuming a code).
  2. The raw JSON of one real offer, so cleaning/PII-stripping is written
     against real field names instead of half-remembered ones.
  3. Result counts for a candidate list of keywords, so we know before
     spending the whole quota whether "stage" postings even exist in this
     API in meaningful numbers.

Usage:
    pip install -r requirements.txt
    cp .env.example .env   # fill in your client id/secret
    python ingest/explore.py
"""
from __future__ import annotations

import json
import sys

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

from client import FranceTravailClient

CANDIDATE_KEYWORDS = [
    "data scientist stage",
    "stage intelligence artificielle",
    "stagiaire data",
    "stagiaire",
    "stage",
]


def main() -> None:
    load_dotenv()
    client = FranceTravailClient()

    print("=" * 60)
    print("REFERENTIEL: typesContrats")
    print("=" * 60)
    try:
        for row in client.get_referentiel("typesContrats"):
            print(row)
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n" + "=" * 60)
    print("REFERENTIEL: naturesContrats")
    print("=" * 60)
    try:
        for row in client.get_referentiel("naturesContrats"):
            print(row)
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n" + "=" * 60)
    print("SAMPLE RAW OFFER (motsCles='data scientist alternance')")
    print("=" * 60)
    try:
        sample = next(client.search_all({"motsCles": "data scientist alternance"}, max_results=1))
        print(json.dumps(sample, indent=2, ensure_ascii=False))
    except StopIteration:
        print("  No results for this keyword.")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n" + "=" * 60)
    print("KEYWORD COUNTS (real inventory, no filters beyond motsCles)")
    print("=" * 60)
    for kw in CANDIDATE_KEYWORDS:
        try:
            n = client.count({"motsCles": kw})
            print(f"  {n:>6}  {kw}")
        except Exception as e:
            print(f"  ERROR  {kw}: {e}")


if __name__ == "__main__":
    sys.exit(main() or 0)
