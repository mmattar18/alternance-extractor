"""Text cleaning for France Travail offer fields.

France Travail's `description` field is usually already plain text, but some
employer-submitted postings embed stray HTML tags or HTML entities, so this
strips defensively rather than assuming. Stdlib only — no new dependency.
"""
from __future__ import annotations

import html
import re
from html.parser import HTMLParser

_WHITESPACE_RUN_RE = re.compile(r"[ \t]+")
_BLANK_LINE_RUN_RE = re.compile(r"\n{3,}")


class _TagStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.chunks: list[str] = []

    def handle_data(self, data):
        self.chunks.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in ("br", "p", "li", "div"):
            self.chunks.append("\n")

    def get_text(self) -> str:
        return "".join(self.chunks)


def strip_html(text: str) -> str:
    if "<" not in text:
        # Fast path: most descriptions have no markup at all.
        return html.unescape(text)
    stripper = _TagStripper()
    stripper.feed(text)
    return html.unescape(stripper.get_text())


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WHITESPACE_RUN_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return _BLANK_LINE_RUN_RE.sub("\n\n", text).strip()


def clean_description(raw: str) -> str:
    return normalize_whitespace(strip_html(raw))
