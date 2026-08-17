"""Fetch ~700 real postings from France Travail into data/raw/postings.jsonl.

Run: python ingest/fetch.py

Strategy: run a fixed list of motsCles queries across four buckets (see
KEYWORD_PLAN below), dedupe by offer id as we go, drop postings with a
near-empty description, then dedupe once more by text hash in case two
different ids carry identical text. Every drop is counted and printed —
nothing here is silent.

`raw_text` is deliberately just title + company + location + free-text
description — no synthetic structured header — so it matches what a human
would actually copy-paste from a job board, and so fields that only exist in
France Travail's separate structured metadata (never mentioned in the prose)
honestly come out as "not stated" rather than leaking in. That metadata is
kept in `reference_metadata` for the hand-correction UI (step 4) and sanity
checks only — it is never fed to Groq or the fine-tuned model.
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

from schema.dedupe import text_hash  # noqa: E402
from client import FranceTravailClient  # noqa: E402
from clean import clean_description  # noqa: E402
from pii import strip_pii  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "data" / "raw" / "postings.jsonl"
MIN_DESCRIPTION_CHARS = 200

# (keyword, max_results). max_results=None means "take everything available"
# — used for the data/AI stage bucket, where real inventory is already small.
KEYWORD_PLAN: dict[str, list[tuple[str, int | None]]] = {
    "data_ai_alternance": [
        ("data scientist alternance", 60),
        ("data analyst alternance", 60),
        ("intelligence artificielle alternance", 60),
        ("machine learning alternance", 60),
        ("data engineer alternance", 40),
        ("business intelligence alternance", 40),
        ("alternance data", 60),
        ("alternance ingénieur données", 40),
        ("alternance statistiques", 40),
        ("alternance décisionnel", 40),
        ("apprenti data", 40),
        ("alternance NLP", 40),
        ("alternance computer vision", 40),
        ("alternance dataviz", 40),
        ("alternance MLOps", 40),
        ("alternance chef de projet data", 40),
        ("alternance analytics", 40),
        ("alternance IA générative", 40),
        ("alternance data governance", 40),
        ("alternance ETL", 40),
        ("alternance cloud data", 40),
        ("alternance data engineering", 40),
        ("alternance big data", 40),
        ("alternance data warehouse", 40),
        ("alternance ingénieur IA", 40),
    ],
    "data_ai_stage": [
        ("stagiaire data", None),
        ("stage data scientist", None),
        ("stage intelligence artificielle", None),
        ("stage machine learning", None),
        ("stage data analyst", None),
        ("stage data engineer", None),
        ("stage big data", None),
        ("stage business intelligence", None),
        ("stagiaire intelligence artificielle", None),
        ("stagiaire machine learning", None),
    ],
    "other_alternance": [
        ("alternance comptabilité", 40),
        ("assistant marketing alternance", 40),
        ("alternance ressources humaines", 40),
        ("alternance commerce", 40),
        ("alternance communication", 40),
        ("alternance logistique", 30),
        ("alternance juridique", 30),
        ("alternance finance", 30),
    ],
    "general_other_roles": [
        ("vendeur", 30),
        ("chef de projet", 30),
        ("infirmier", 30),
        ("comptable", 30),
        ("assistant administratif", 30),
        ("technicien de maintenance", 30),
        ("cuisinier", 30),
        ("ingénieur mécanique", 30),
    ],
}


def build_record(offer: dict, bucket: str, keyword: str) -> dict:
    title = (offer.get("intitule") or "").strip()
    company = (offer.get("entreprise") or {}).get("nom")
    location = (offer.get("lieuTravail") or {}).get("libelle")
    description = strip_pii(clean_description(offer.get("description") or ""))

    header = [title]
    company_location = " — ".join(x for x in [company, location] if x)
    if company_location:
        header.append(company_location)
    raw_text = "\n".join(header) + "\n\n" + description

    return {
        "posting_id": f"france_travail:{offer['id']}",
        "source": "france_travail",
        "source_url": (offer.get("origineOffre") or {}).get("urlOrigine"),
        "bucket": bucket,
        "matched_keyword": keyword,
        "title": title,
        "raw_text": raw_text,
        "text_hash": text_hash(raw_text),
        "description_chars": len(description),
        # Reference only — never fed to a model. See module docstring.
        "reference_metadata": {
            "alternance": offer.get("alternance"),
            "typeContrat": offer.get("typeContrat"),
            "typeContratLibelle": offer.get("typeContratLibelle"),
            "natureContrat": offer.get("natureContrat"),
            "experienceLibelle": offer.get("experienceLibelle"),
            "salaireLibelle": (offer.get("salaire") or {}).get("libelle"),
            "dureeTravailLibelle": offer.get("dureeTravailLibelle"),
            "romeLibelle": offer.get("romeLibelle"),
            "dateCreation": offer.get("dateCreation"),
        },
    }


def main() -> None:
    load_dotenv()
    client = FranceTravailClient()

    seen_ids: set[str] = set()
    records: list[dict] = []
    bucket_counts: dict[str, int] = {}
    dropped_short = 0

    for bucket, queries in KEYWORD_PLAN.items():
        bucket_counts[bucket] = 0
        for keyword, cap in queries:
            fetched_this_query = 0
            for offer in client.search_all({"motsCles": keyword}, max_results=cap):
                fetched_this_query += 1
                if offer["id"] in seen_ids:
                    continue
                seen_ids.add(offer["id"])
                record = build_record(offer, bucket, keyword)
                if record["description_chars"] < MIN_DESCRIPTION_CHARS:
                    dropped_short += 1
                    continue
                records.append(record)
                bucket_counts[bucket] += 1
            print(f"[{bucket}] '{keyword}': {fetched_this_query} fetched, "
                  f"{bucket_counts[bucket]} kept so far in bucket")

    # Second dedupe pass: identical text under a different offer id.
    seen_hashes: set[str] = set()
    deduped: list[dict] = []
    dropped_hash_dupes = 0
    for r in records:
        if r["text_hash"] in seen_hashes:
            dropped_hash_dupes += 1
            continue
        seen_hashes.add(r["text_hash"])
        deduped.append(r)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for bucket, count in bucket_counts.items():
        print(f"  {bucket}: {count}")
    print(f"  dropped (description < {MIN_DESCRIPTION_CHARS} chars): {dropped_short}")
    print(f"  dropped (duplicate text, different id): {dropped_hash_dupes}")
    print(f"  TOTAL WRITTEN: {len(deduped)} -> {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
