"""One-posting-per-screen review UI for hand-correcting the 100-posting test
set. Prefilled from Groq's prediction; you edit only what's wrong.

Stdlib only (http.server) — no new dependency, works identically everywhere,
reachable from your phone if it's on the same wifi as this machine.

Ground rule shown on every screen: correct based ONLY on what the text above
actually says. Don't use the France Travail reference panel to fill in a
field the text itself doesn't mention — that would make the test set an
unfair standard no model (Groq or the fine-tuned one) could ever meet, since
neither ever sees that metadata. The panel exists only to catch a field Groq
clearly missed or misread from the text itself.

Run: python label/review_server.py
Then open http://localhost:8765 (or http://<this-machine's-LAN-IP>:8765 from
your phone, printed on startup).
"""
from __future__ import annotations

import html
import json
import socket
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pydantic import ValidationError  # noqa: E402
from schema.posting import JobPosting  # noqa: E402

CANDIDATES_PATH = REPO_ROOT / "data" / "test" / "test_candidates.jsonl"
CORRECTED_PATH = REPO_ROOT / "data" / "test" / "test.jsonl"
PORT = 8765

TEXT_FIELDS = ["title", "company", "start_date", "location", "education_level",
               "salary_range", "alternance_rhythm"]
INT_FIELDS = ["duration_months", "years_experience_min"]
LIST_FIELDS = ["required_skills", "nice_to_have_skills", "language_requirements"]
ENUM_FIELDS = {
    "contract_type": ["", "alternance", "stage", "cdi", "cdd", "freelance", "other"],
    "remote_policy": ["", "on_site", "hybrid", "remote"],
}
FIELD_ORDER = ["title", "company", "contract_type", "duration_months", "start_date",
               "location", "remote_policy", "required_skills", "nice_to_have_skills",
               "years_experience_min", "education_level", "language_requirements",
               "salary_range", "alternance_rhythm"]


def load_candidates() -> list[dict]:
    return [json.loads(l) for l in CANDIDATES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_corrections() -> dict[str, dict]:
    if not CORRECTED_PATH.exists():
        return {}
    out = {}
    for line in CORRECTED_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["posting_id"]] = rec
    return out


def save_correction(record: dict) -> None:
    corrections = load_corrections()
    corrections[record["posting_id"]] = record
    CORRECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates()
    order = [c["posting_id"] for c in candidates]
    with CORRECTED_PATH.open("w", encoding="utf-8") as f:
        for pid in order:
            if pid in corrections:
                f.write(json.dumps(corrections[pid], ensure_ascii=False) + "\n")


def field_value(prefill: dict, field: str):
    return prefill.get(field)


def render_field_input(field: str, value) -> str:
    if field in ENUM_FIELDS:
        opts = []
        for opt in ENUM_FIELDS[field]:
            selected = " selected" if (value or "") == opt else ""
            label = opt if opt else "(not stated)"
            opts.append(f'<option value="{opt}"{selected}>{html.escape(label)}</option>')
        return f'<select name="{field}">{"".join(opts)}</select>'
    if field in LIST_FIELDS:
        text = ", ".join(value) if value else ""
        return f'<input type="text" name="{field}" value="{html.escape(text)}" placeholder="comma-separated, blank = not stated">'
    if field in INT_FIELDS:
        text = "" if value is None else str(value)
        return f'<input type="text" inputmode="numeric" name="{field}" value="{html.escape(text)}" placeholder="blank = not stated">'
    text = value or ""
    return f'<input type="text" name="{field}" value="{html.escape(text)}" placeholder="blank = not stated">'


def render_page(index: int, candidates: list[dict], corrections: dict[str, dict],
                 error: str | None = None, posted_values: dict | None = None) -> str:
    total = len(candidates)
    candidate = candidates[index]
    posting_id = candidate["posting_id"]
    existing = corrections.get(posting_id)
    prefill = posted_values if posted_values is not None else (
        existing["corrected"] if existing else (candidate.get("prediction") or {})
    )

    done_count = len(corrections)
    ref = candidate.get("reference_metadata", {})
    ref_html = "".join(
        f"<div><b>{html.escape(str(k))}:</b> {html.escape(str(v))}</div>"
        for k, v in ref.items() if v is not None
    )

    rows = []
    for field in FIELD_ORDER:
        rows.append(
            f'<div class="row"><label>{field}</label>{render_field_input(field, field_value(prefill, field))}</div>'
        )

    nav = []
    if index > 0:
        nav.append(f'<a href="/review/{index - 1}">&larr; prev</a>')
    if index < total - 1:
        nav.append(f'<a href="/review/{index + 1}">next &rarr;</a>')

    error_html = f'<div class="error">{html.escape(error)}</div>' if error else ""

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review {index + 1}/{total}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 700px; margin: 0 auto; padding: 12px; }}
.progress {{ color: #555; margin-bottom: 8px; }}
.rawtext {{ white-space: pre-wrap; background: #f5f5f5; padding: 10px; border-radius: 6px;
            max-height: 260px; overflow-y: auto; font-size: 14px; }}
.ref {{ background: #fff8e1; padding: 8px; border-radius: 6px; font-size: 13px; margin-top: 8px; }}
.row {{ margin: 10px 0; }}
label {{ display: block; font-weight: bold; font-size: 13px; margin-bottom: 3px; }}
input, select {{ width: 100%; box-sizing: border-box; padding: 8px; font-size: 15px; }}
.note {{ font-size: 12px; color: #777; margin: 10px 0; }}
.error {{ color: #b00020; font-weight: bold; margin: 8px 0; }}
nav {{ display: flex; justify-content: space-between; margin: 14px 0; }}
button {{ padding: 10px 16px; font-size: 15px; }}
</style></head><body>
<div class="progress">{done_count}/{total} corrected &middot; viewing {index + 1}/{total}
{"&middot; <b>already saved</b>" if existing else ""}</div>
<div class="rawtext">{html.escape(candidate["raw_text"])}</div>
<details class="ref"><summary>France Travail reference (hints only — don't invent from this)</summary>{ref_html}</details>
<p class="note">Correct only what the text above states. Leave a field blank/(not stated) if the text doesn't say it,
even if the reference panel knows more.</p>
{error_html}
<form method="post" action="/review/{index}">
{"".join(rows)}
<nav><span></span><button type="submit">Save &amp; next</button></nav>
</form>
<nav>{" | ".join(nav)}</nav>
</body></html>"""


def render_done_page(total: int) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Done</title></head>
<body style="font-family:system-ui,sans-serif;max-width:600px;margin:40px auto;">
<h2>All {total} postings corrected.</h2>
<p>Saved to data/test/test.jsonl. Go back and edit any of them via /review/0.</p>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep console quiet; errors still raise

    def do_GET(self):
        candidates = load_candidates()
        corrections = load_corrections()
        if self.path == "/" or self.path == "":
            first_uncorrected = next(
                (i for i, c in enumerate(candidates) if c["posting_id"] not in corrections),
                None,
            )
            target = f"/review/{first_uncorrected}" if first_uncorrected is not None else "/done"
            self.send_response(302)
            self.send_header("Location", target)
            self.end_headers()
            return
        if self.path == "/done":
            body = render_done_page(len(candidates)).encode("utf-8")
            self._send_html(body)
            return
        if self.path.startswith("/review/"):
            index = int(self.path.rsplit("/", 1)[-1])
            body = render_page(index, candidates, corrections).encode("utf-8")
            self._send_html(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if not self.path.startswith("/review/"):
            self.send_response(404)
            self.end_headers()
            return
        index = int(self.path.rsplit("/", 1)[-1])
        candidates = load_candidates()
        corrections = load_corrections()
        candidate = candidates[index]

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        form = {k: v[0] for k, v in parse_qs(body).items()}

        payload = {"title": form.get("title", "")}
        for field in TEXT_FIELDS + list(ENUM_FIELDS.keys()):
            if field == "title":
                continue
            val = form.get(field, "").strip()
            payload[field] = val or None
        for field in INT_FIELDS:
            val = form.get(field, "").strip()
            if val:
                try:
                    payload[field] = int(val)
                except ValueError:
                    payload[field] = val  # let pydantic raise a clear error
            else:
                payload[field] = None
        for field in LIST_FIELDS:
            val = form.get(field, "").strip()
            payload[field] = [s.strip() for s in val.split(",") if s.strip()] or None

        try:
            posting = JobPosting.model_validate(payload)
        except ValidationError as e:
            body = render_page(index, candidates, corrections, error=str(e), posted_values=payload).encode("utf-8")
            self._send_html(body)
            return

        record = {
            "posting_id": candidate["posting_id"],
            "source": candidate["source"],
            "bucket": candidate["bucket"],
            "title": candidate["title"],
            "raw_text": candidate["raw_text"],
            "text_hash": candidate["text_hash"],
            "groq_prediction": candidate.get("prediction"),
            "corrected": posting.model_dump(),
        }
        save_correction(record)

        next_index = index + 1 if index + 1 < len(candidates) else None
        target = f"/review/{next_index}" if next_index is not None else "/done"
        self.send_response(302)
        self.send_header("Location", target)
        self.end_headers()

    def _send_html(self, body: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    n = len(load_candidates())
    print(f"{n} postings to review.")
    print(f"Open http://localhost:{PORT} on this machine,")
    print(f"or http://{local_ip()}:{PORT} from your phone (same wifi).")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
