"""PII stripping for posting text.

Scope, deliberately limited for a one-week project: emails and French phone
numbers typed directly into free text are scrubbed with regexes (high
precision, easy to verify). Recruiter contact details in the API's own
`contact` object (nom, courriel, telephone) are not regex-scrubbed — they're
dropped wholesale by the caller before this module ever sees them, since
that's a structured field, not text. Personal names embedded in prose (e.g.
"Contactez Marie Dupont") are NOT detected — that needs NER, which is out of
scope this week. Documented as a known limitation, not silently ignored.
"""
from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(
    r"(?:(?:\+33|0033)\s?[1-9]|0[1-9])(?:[\s.\-]?\d{2}){4}"
)


def strip_pii(text: str) -> str:
    text = _EMAIL_RE.sub("[email]", text)
    text = _PHONE_RE.sub("[téléphone]", text)
    return text
