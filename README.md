# alternance-extractor

**Claim:** a small fine-tuned open model (Qwen2.5-1.5B-Instruct + QLoRA) can match a
large API model (Llama 3.3 70B on Groq) at extracting structured data from job
postings, at lower latency and lower cost per 1000 requests.

Status: **Data pipeline done, fine-tune not yet run.** Ingest, Groq labelling, the
100-posting hand-corrected test set, and eval harness are all built and committed.
`data/train/train.jsonl` (541 rows) has been cleaned up in three rounds — see
`LABELLING-NOTES.md` for the full history. Groq baseline against the hand-corrected test
set: `macro_f1 = 0.891`, `exact_match_rate = 34.0%` (see `eval/score.py`). Next and last
step: run `notebooks/kaggle_train.ipynb` then `notebooks/kaggle_benchmark.ipynb` on Kaggle
to get the fine-tuned model's numbers — this section gets replaced with the real writeup
(method, three charts, limitations) once that's done.

## Repo structure

- `schema/` — locked Pydantic schema + validator (`posting.py`), dedupe/leakage
  helpers (`dedupe.py`)
- `ingest/` — API clients / fetchers, cleaning, dedupe, PII stripping
- `data/` — raw postings, Groq-labelled train set (541 rows), hand-corrected test set
  (100 rows)
- `label/` — Groq/Gemini/OpenRouter/Cerebras labelling clients + baseline prompt, plus
  `review_server.py` (the hand-correction UI) and `select_test_set.py` (the train/test split)
- `eval/` — scoring harness: field-level F1, exact match, JSON-validity (`score.py`)
- `notebooks/` — `kaggle_train.ipynb` (QLoRA fine-tune, ready to run), `kaggle_benchmark.ipynb`
  (scores the fine-tuned adapter against the Groq baseline) — both run on Kaggle, not locally

## Schema

See `schema/posting.py` for the full model and docstring. The key design decision:
every optional field is `None` when the posting doesn't state it, and `None` never
means "zero" — e.g. `years_experience_min=0` (explicitly "no experience required") is
a distinct, valid value from `years_experience_min=None` (not mentioned).

Run the smoke tests:

```
cd schema && python test_posting.py
```
