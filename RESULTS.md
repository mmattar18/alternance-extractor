# Results

Groq baseline vs fine-tuned Qwen2.5-1.5B, scored by `eval/score.py` against the
100-posting hand-corrected test set (`data/test/test.jsonl`).

Reproduce: `python eval/score.py data/test/test.jsonl data/test/test_candidates.jsonl`
for the baseline; the fine-tuned side comes from `notebooks/kaggle_benchmark.ipynb`.

## Run 1 -- 1 epoch (pipeline-validation run, NOT the final model)

| metric | Groq (Llama-3.3-70B class) | Fine-tuned Qwen2.5-1.5B |
|---|---|---|
| macro F1 | **0.891** | 0.510 |
| exact-match rate | **34.0%** | 0.0% |
| JSON validity | 100% | 100% |

Per-field F1 (Groq / fine-tuned):

| field | Groq | Qwen 1ep | note |
|---|---|---|---|
| title | 1.000 | 0.969 | |
| company | 0.854 | 0.737 | |
| contract_type | 0.979 | 0.828 | |
| duration_months | 0.757 | 0.500 | |
| start_date | 1.000 | 0.439 | |
| location | 0.985 | 0.817 | |
| remote_policy | 0.952 | 0.100 | |
| required_skills | 0.565 | 0.248 | **over-generation, see below** |
| nice_to_have_skills | 0.800 | 0.000 | zero exact set matches |
| years_experience_min | 0.857 | 0.593 | |
| education_level | 0.919 | 0.643 | |
| language_requirements | 0.963 | 0.609 | |
| salary_range | 0.923 | 0.476 | |
| alternance_rhythm | 0.923 | 0.182 | |

### What the confusion counts say

The fine-tuned model **emits a value almost everywhere instead of predicting null**:

- `required_skills`: tp=14, **fp=85**, fn=0, **tn=1** -- it produced a non-null list for
  99/100 postings. Gold has one for 49/100. Recall is a perfect 1.000 precisely *because*
  it never abstains; precision collapses to 0.141.
- `nice_to_have_skills`: tp=0, fp=20, fn=20 -> F1 exactly 0.000.
- Same shape on `education_level` (recall 0.974, precision 0.481) and `company`
  (recall 0.966, precision 0.596).

**This is not a label-distribution problem.** The training labels it learned from had
`required_skills` non-null only 44% of the time (gold: 49%) -- the model is not copying a
skewed prior, it simply has not learned *when to abstain*. That points at undertraining,
which the run's own metrics corroborate: 514 rows / effective batch 16 = **33 optimizer
steps**, and the default linear scheduler had annealed `learning_rate` to **0.0 by step
30**, with `eval_loss` (0.4439) still *below* train loss (0.4885) and both still falling.
The 3-epoch rerun (~99 steps, with warmup) tests exactly this.

`exact_match_rate = 0.0%` follows mechanically: `required_skills` alone is wrong on ~85
postings, and exact-match requires all 14 fields correct simultaneously.

## Latency -- measured, but NOT a fair comparison

| | median | mean | p95 |
|---|---|---|---|
| Groq | 16.36s | 62.52s | 267.34s |
| Fine-tuned (T4) | 18.10s | 18.59s | 23.42s |

**Do not use this to claim the fine-tuned model is faster.** The Groq figures were
recorded during the original bulk labelling run against a free-tier API with rate limiting
and retry backoff -- the 267s p95 reflects queueing, not model inference. At the median,
Groq (16.4s) is in fact slightly *faster* than the fine-tuned model on a T4 (18.1s).

The one defensible reading: the fine-tuned model's latency is far more **consistent**
(p95 23.4s vs 267.3s), because it is not sharing a rate-limited endpoint. A real
latency/cost claim needs a controlled head-to-head -- same batching, same concurrency,
Groq on a paid tier -- which has not been run.

## Status

Run 1 is a pipeline-validation result, not the headline number: 1 epoch, and trained
before the 87-label cleanup. The 3-epoch run on cleaned labels is in flight.
