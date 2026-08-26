# alternance-extractor

**Question:** how much of the gap between a small open model and a large API model can
QLoRA fine-tuning close, on structured extraction from French job postings?

**Answer (measured, see [RESULTS.md](RESULTS.md)):** **73.7%** of it. A QLoRA fine-tune of
Qwen2.5-1.5B-Instruct on 541 Groq-labelled postings, trained on one free Kaggle T4, scores
**0.770 macro F1** against a hand-corrected 100-posting test set, versus **0.432** for the
same model few-shot-prompted and **0.891** for Llama-3.3-70B on Groq.

The remaining 0.121 gap to the 70B model is statistically significant (paired bootstrap
95% CI [+0.079, +0.173]), so **the fine-tuned model does not match it** -- an earlier
version of this README claimed it could, and the data does not support that.

On the 90 test postings free of near-duplicate leakage (10% of the test set had a
near-identical training posting -- see RESULTS.md) the fine-tuned model scores **0.755**,
and still closes ~72% of the gap.

It beats the 70B model on two fields: `contract_type` (0.986 vs 0.979) and
`duration_months` (0.850 vs 0.757), the latter because the training labels encode a
range-handling convention Groq itself does not follow.

Status: pipeline complete end to end. Ingest, labelling, a hand-corrected test set, the
eval harness, training and benchmark notebooks all run.

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
