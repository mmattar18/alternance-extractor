# alternance-extractor

**How small a model gets you most of the way to a 70B?**

Structured extraction from French job postings — turning free-text listings into a fixed
13-field schema (title, company, contract type, duration, skills, salary...). The question
is how far a small open model you run yourself can close the gap to a large rented one.

## The scaling curve

Every system scored by `eval/score.py` against the same 100-posting hand-corrected test set:

| system | params | prompt tokens | macro F1 | % of Groq |
|---|---|---|---|---|
| Qwen2.5-1.5B, few-shot (untrained) | 1.5B | 2718 | 0.486 | 53% |
| Qwen2.5-1.5B + QLoRA | 1.5B | 812 | 0.830 | 91% |
| **Qwen2.5-3B + QLoRA** | **3B** | **812** | **0.877 ± 0.003** | **96%** |
| Groq Llama-3.3-70B, few-shot | ~70B | ~2156 | 0.916 | 100% |

**A 3B model reaches 96% of a 70B's quality at 1/23rd the parameters and 1/2.6th the prompt
cost**, trained on 881 examples on a free Kaggle T4. The curve flattens sharply: fine-tuning
buys +0.344, the 1.5B→3B step buys +0.047, and the remaining 0.039 is concentrated in a
single field.

**It does not match the 70B.** The remaining gap is statistically significant against a
measured seed variance of σ = 0.003 (n=3), and roughly half of it is the `skills` field
alone (0.42 vs 0.635).

## Where the last 4% lives

`skills` resisted every intervention tried: two label-convention rewrites in opposite
directions, a 43% increase in its training examples, and inference-time filtering. It moved
only once — when the model doubled in size. That is the signature of a capacity limit rather
than a data or labelling problem, and it is why the honest answer to "how small can you go"
is *3B for 96%, and something larger for the rest*.

Qwen2.5-7B was attempted and does not fit: on a 16GB T4 it CPU-offloads and logs zero
training steps in 12 hours. See `TRAINING-NOTES.md`.

## What makes the numbers trustworthy

- **Hand-corrected gold set** — 100 postings corrected field by field, the yardstick everything
  is measured against
- **Near-duplicate test leakage found and disclosed** — 10% of the test set had a near-identical
  twin in training (France Travail re-lists jobs under new reference numbers); leak-free scores
  are reported alongside
- **Seed variance measured** (n=3, σ = 0.003) rather than assumed — and it corrected an earlier
  error in this repo where a noise floor inferred from one accidental comparison was ~10x too
  large, causing two valid findings to be wrongly withdrawn
- **Predictions pre-registered** before each run, with the misses left in (`PREDICTIONS.md`)
- **Prompt-length ablation** — the 1173-token JSON schema dump is dead weight after fine-tuning:
  0.815 vs 0.816 at 2.5x fewer tokens

Full results and limitations: [RESULTS.md](RESULTS.md). Plain-language walkthrough:
[EXPLAINER.md](EXPLAINER.md).

## Repo structure

- `schema/` — locked Pydantic schema + validator (`posting.py`), dedupe/leakage
  helpers (`dedupe.py`)
- `ingest/` — API clients / fetchers, cleaning, dedupe, PII stripping
- `data/` — raw postings, Groq-labelled train set (881 rows), hand-corrected test set
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
