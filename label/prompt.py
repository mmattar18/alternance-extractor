"""The extraction prompt, shared by every consumer that calls an LLM against
JobPosting: Groq labelling of the train set, the Groq baseline eval, and (for
comparability) the fine-tuned model's inference prompt in the training
notebook. One prompt, one place — if this drifts between labelling and eval,
the whole comparison is meaningless.

The field schema block is generated from schema/posting.py's own
model_json_schema(), not hand-duplicated, so the prompt can never fall out of
sync with the locked schema.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from schema.posting import JobPosting  # noqa: E402

_SCHEMA_JSON = json.dumps(JobPosting.model_json_schema(), indent=2, ensure_ascii=False)

SYSTEM_PROMPT = f"""You extract structured data from job postings (French or English) into a fixed JSON schema.

JSON schema (Pydantic-generated — field types double as the enum of valid values for enum-typed fields):
{_SCHEMA_JSON}

Non-negotiable rules:
1. Output ONLY a single JSON object matching this schema. No prose, no markdown fences, no explanation.
2. Every field is optional. Use `null` for anything the posting does not state. Never guess, infer from
   typical values, or fill in "reasonable defaults." If it isn't in the text, it's null.
3. `years_experience_min` is the one field where 0 and null mean different things:
   - 0 means the posting EXPLICITLY says no experience is required (e.g. "débutant accepté", "no experience necessary").
   - null means experience is simply not mentioned. Do not default a missing mention to 0.
4. `alternance_rhythm` must be null unless `contract_type` is "alternance" AND the posting states a rhythm.
5. For list fields (required_skills, nice_to_have_skills, language_requirements): only include items explicitly
   stated. If none are stated, use null — not an empty list.
6. Extract values as they are written (e.g. keep "Bac+5", don't normalize to "Master"). Do not translate
   French text to English or vice versa.
7. If the posting text is truncated or garbled, extract what you can and leave the rest null — don't refuse."""

_FEWSHOT_EXAMPLES = [
    (
        "Data Analyst en Alternance (H/F)\nAcme Corp — Lyon\n\n"
        "Nous recherchons un alternant Data Analyst pour une durée de 12 mois, à partir de septembre 2026. "
        "Rythme : 3 semaines en entreprise / 1 semaine en formation.\n"
        "Vous préparez un Bac+5 en école d'ingénieur ou université.\n"
        "Compétences requises : SQL, Python, Power BI.\n"
        "Un plus : connaissance de dbt.\n"
        "Poste en hybride, 2 jours de télétravail par semaine.",
        {
            "title": "Data Analyst en Alternance (H/F)",
            "company": "Acme Corp",
            "contract_type": "alternance",
            "duration_months": 12,
            "start_date": "septembre 2026",
            "location": "Lyon",
            "remote_policy": "hybrid",
            "required_skills": ["SQL", "Python", "Power BI"],
            "nice_to_have_skills": ["dbt"],
            "years_experience_min": None,
            "education_level": "Bac+5",
            "language_requirements": None,
            "salary_range": None,
            "alternance_rhythm": "3 semaines en entreprise / 1 semaine en formation",
        },
    ),
    (
        "Ingénieur Machine Learning - CDI\nDataWorks\n\n"
        "Poste basé à Paris. Salaire : 42-48K€ brut annuel selon profil. "
        "Débutant accepté, formation assurée en interne. Anglais courant exigé pour échanger avec l'équipe US.",
        {
            "title": "Ingénieur Machine Learning - CDI",
            "company": "DataWorks",
            "contract_type": "cdi",
            "duration_months": None,
            "start_date": None,
            "location": "Paris",
            "remote_policy": None,
            "required_skills": None,
            "nice_to_have_skills": None,
            "years_experience_min": 0,
            "education_level": None,
            "language_requirements": ["Anglais courant"],
            "salary_range": "42-48K€ brut annuel selon profil",
            "alternance_rhythm": None,
        },
    ),
    (
        "Assistant Marketing en Alternance\nStartupXYZ\n\n"
        "Rejoignez notre équipe marketing pour votre alternance ! Missions variées : réseaux sociaux, "
        "événementiel, reporting. Niveau Bac+3 minimum.",
        {
            "title": "Assistant Marketing en Alternance",
            "company": "StartupXYZ",
            "contract_type": "alternance",
            "duration_months": None,
            "start_date": None,
            "location": None,
            "remote_policy": None,
            "required_skills": None,
            "nice_to_have_skills": None,
            "years_experience_min": None,
            "education_level": "Bac+3",
            "language_requirements": None,
            "salary_range": None,
            # Contract type is alternance, but no rhythm is stated -> null, not a guess.
            "alternance_rhythm": None,
        },
    ),
]


def build_messages(raw_text: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_text, example_json in _FEWSHOT_EXAMPLES:
        messages.append({"role": "user", "content": example_text})
        messages.append({"role": "assistant", "content": json.dumps(example_json, ensure_ascii=False)})
    messages.append({"role": "user", "content": raw_text})
    return messages


def build_gemini_contents(raw_text: str) -> list[dict]:
    """Same few-shot examples as build_messages(), reshaped for Gemini's API:
    no system role inside `contents` (SYSTEM_PROMPT goes in the separate
    system_instruction field instead) and 'model' instead of 'assistant'."""
    contents = []
    for example_text, example_json in _FEWSHOT_EXAMPLES:
        contents.append({"role": "user", "parts": [{"text": example_text}]})
        contents.append({"role": "model", "parts": [{"text": json.dumps(example_json, ensure_ascii=False)}]})
    contents.append({"role": "user", "parts": [{"text": raw_text}]})
    return contents
