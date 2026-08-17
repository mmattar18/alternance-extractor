# alternance-extractor

**Claim:** a small fine-tuned open model (Qwen2.5-1.5B-Instruct + QLoRA) can match a
large API model (Llama 3.3 70B on Groq) at extracting structured data from job
postings, at lower latency and lower cost per 1000 requests.

Status: **Step 1 — schema locked, nothing else built yet.** Numbers below are not
measured. This section will be replaced with the real writeup (method, three charts,
limitations) once the pipeline runs end to end.

## Repo structure

- `schema/` — locked Pydantic schema + validator (`posting.py`), dedupe/leakage
  helpers (`dedupe.py`)
- `ingest/` — API clients / fetchers, cleaning, dedupe, PII stripping (not built yet)
- `data/` — raw postings, Groq-labelled train set, hand-corrected test set (not built yet)
- `label/` — Groq labelling + baseline prompt (not built yet)
- `eval/` — scoring harness: field-level F1, exact match, JSON-validity (not built yet)
- `notebooks/` — `kaggle_train.ipynb`, `kaggle_benchmark.ipynb` (not built yet)

## Schema

See `schema/posting.py` for the full model and docstring. The key design decision:
every optional field is `None` when the posting doesn't state it, and `None` never
means "zero" — e.g. `years_experience_min=0` (explicitly "no experience required") is
a distinct, valid value from `years_experience_min=None` (not mentioned).

Run the smoke tests:

```
cd schema && python test_posting.py
```
