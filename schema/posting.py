"""Locked schema for structured job-posting extraction.

Core modelling rule: a field is `None` when the posting does not state it.
`None` NEVER means "zero" or "no requirement" — `years_experience_min=0` is a
real, distinct value (posting explicitly says "no experience required" /
"débutant accepté"), separate from `years_experience_min=None` (experience
not mentioned at all). Every optional field follows this convention.

Special case: `alternance_rhythm` (e.g. "3 semaines entreprise / 1 semaine
école") only makes sense when `contract_type == "alternance"`. For any other
contract type it is forced to `None`, even if the model extracted something,
because the field is *not applicable* rather than *not stated*. We fold both
cases into the same `None` bucket instead of adding a third state, to keep
scoring (field-level F1 on a binary present/absent + value match) simple.
This is a deliberate simplification — see README limitations.

This file has no dependency beyond pydantic. It is imported by ingest/,
label/, and eval/, so it must stay free of side effects and heavy imports.
"""
from __future__ import annotations

import json
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

ContractType = Literal["alternance", "stage", "cdi", "cdd", "freelance", "other"]
RemotePolicy = Literal["on_site", "hybrid", "remote"]

# Groq and the fine-tuned model will both produce casing/wording variants of
# the same value. Normalize here, once, instead of in every consumer.
_CONTRACT_SYNONYMS = {
    "alternance": "alternance",
    "apprenticeship": "alternance",
    "apprentissage": "alternance",
    "contrat d'apprentissage": "alternance",
    "contrat de professionnalisation": "alternance",
    "stage": "stage",
    "internship": "stage",
    "stagiaire": "stage",
    "cdi": "cdi",
    "permanent": "cdi",
    "full-time": "cdi",
    "full time": "cdi",
    "cdd": "cdd",
    "fixed-term": "cdd",
    "fixed term": "cdd",
    "temporary": "cdd",
    "intérim": "cdd",
    "interim": "cdd",
    "intérimaire": "cdd",
    "interimaire": "cdd",
    "mission d'intérim": "cdd",
    "mission d'interim": "cdd",
    "travail temporaire": "cdd",
    "temporaire": "cdd",
    "freelance": "freelance",
    "contractor": "freelance",
    "independent": "freelance",
    "indépendant": "freelance",
}

_REMOTE_SYNONYMS = {
    "on_site": "on_site",
    "onsite": "on_site",
    "on-site": "on_site",
    "on site": "on_site",
    "in office": "on_site",
    "in-office": "on_site",
    "presentiel": "on_site",
    "présentiel": "on_site",
    "hybrid": "hybrid",
    "hybride": "hybrid",
    "remote": "remote",
    "full remote": "remote",
    "full-remote": "remote",
    "télétravail": "remote",
    "teletravail": "remote",
    "distanciel": "remote",
}

# Strings a model emits when it means "not stated" instead of emitting null.
_SENTINEL_STRINGS = {
    "", "n/a", "na", "none", "null", "not stated", "not specified",
    "not mentioned", "unspecified", "unknown", "-", "tbd",
    "non précisé", "non precise", "non renseigné", "non renseigne",
    "non spécifié", "non specifie",
}


def _clean_scalar(v):
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in _SENTINEL_STRINGS:
            return None
        return s
    return v


def _clean_list(v):
    if v is None:
        return None
    if not isinstance(v, list):
        return v
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in v:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s or s.lower() in _SENTINEL_STRINGS:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)
    return cleaned or None


class JobPosting(BaseModel):
    """The extraction target. One instance = one posting's structured fields."""

    model_config = {"extra": "forbid"}

    title: str = Field(..., min_length=1, description="Job title exactly as written in the posting.")
    company: Optional[str] = Field(default=None, description="Employer name. null if anonymized or not stated.")
    contract_type: Optional[ContractType] = Field(default=None, description="One of the enum values, or null if not stated/unclear.")
    duration_months: Optional[int] = Field(default=None, ge=1, description="Contract/internship length in months. null if not stated. Never guess from typical durations.")
    start_date: Optional[str] = Field(default=None, description="Start date as written (e.g. 'septembre 2026', 'dès que possible'). null if not stated.")
    location: Optional[str] = Field(default=None, description="City/region as written. null if not stated.")
    remote_policy: Optional[RemotePolicy] = Field(default=None, description="on_site, hybrid, or remote. null if not stated.")
    skills: Optional[list[str]] = Field(default=None, description="All skills, tools, technologies and named competencies the posting states, whether framed as required or as a plus. null if none are stated.")
    years_experience_min: Optional[int] = Field(default=None, ge=0, description="Minimum years of experience. 0 ONLY if the posting explicitly says no experience is required (e.g. 'débutant accepté'). null if experience is not mentioned at all — do not default to 0.")
    education_level: Optional[str] = Field(default=None, description="Required education level as written (e.g. 'Bac+5', 'Master 2'). null if not stated.")
    language_requirements: Optional[list[str]] = Field(default=None, description="Languages required, with level if given (e.g. 'Anglais courant'). null if none stated.")
    salary_range: Optional[str] = Field(default=None, description="Salary/compensation as written, with currency/period if given. null if not stated in the text.")
    alternance_rhythm: Optional[str] = Field(default=None, description="Apprenticeship rhythm (e.g. '3 semaines entreprise / 1 semaine école'). Only ever non-null when contract_type is alternance. null otherwise, or if alternance but rhythm isn't stated.")

    @model_validator(mode="before")
    @classmethod
    def _normalize_raw(cls, data):
        """Clean every field the same way before type-specific validation runs."""
        if not isinstance(data, dict):
            return data
        list_fields = {"skills", "language_requirements"}
        out = dict(data)
        for key, val in out.items():
            out[key] = _clean_list(val) if key in list_fields else _clean_scalar(val)
        return out

    @field_validator("contract_type", mode="before")
    @classmethod
    def _normalize_contract_type(cls, v):
        if v is None:
            return None
        key = str(v).strip().lower()
        return _CONTRACT_SYNONYMS.get(key, "other")

    @field_validator("remote_policy", mode="before")
    @classmethod
    def _normalize_remote_policy(cls, v):
        if v is None:
            return None
        key = str(v).strip().lower()
        # Unmapped values fall back to None rather than a guessed enum member:
        # the set of remote-policy phrasings is small enough that anything
        # unrecognized is more likely noise than a genuine fourth category.
        return _REMOTE_SYNONYMS.get(key)

    @model_validator(mode="after")
    def _alternance_rhythm_only_for_alternance(self):
        if self.contract_type != "alternance" and self.alternance_rhythm is not None:
            self.alternance_rhythm = None
        return self


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_llm_json(raw: str) -> tuple[Optional[JobPosting], Optional[str]]:
    """Parse a raw LLM string into a JobPosting.

    Returns (posting, None) on success, (None, error_message) on failure.
    Used by both label/ (Groq) and eval/ (any model) so JSON-validity rate is
    measured identically everywhere. Strips ```json fences since Groq and the
    fine-tuned model both sometimes wrap output in markdown even when told
    not to.
    """
    text = _JSON_FENCE_RE.sub("", raw).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"invalid_json: {e}"
    try:
        return JobPosting.model_validate(payload), None
    except Exception as e:
        return None, f"schema_violation: {e}"
