# Results

Three systems scored by `eval/score.py` against the 100-posting hand-corrected test set
(`data/test/test.jsonl`). Reproduce with `notebooks/kaggle_benchmark.ipynb`.

## Headline

| system | params | macro F1 (95% CI) | exact-match | JSON valid |
|---|---|---|---|---|
| Groq Llama-3.3-70B, few-shot | ~70B | **0.891** [0.860, 0.914] | 34% [25%, 43%] | 100% |
| Qwen2.5-1.5B **base**, few-shot | 1.5B | 0.432 [0.360, 0.480] | 1% [0%, 3%] | 99% |
| Qwen2.5-1.5B **fine-tuned** (QLoRA) | 1.5B | **0.770** [0.715, 0.807] | 14% [7%, 21%] | 100% |

**QLoRA fine-tuning closed 73.7% of the gap** between the 1.5B base model and a model
~47x larger, using 541 training examples on a single free T4.

Paired bootstrap on the same postings (2000 resamples):

| comparison | delta macro F1 | 95% CI | significant |
|---|---|---|---|
| fine-tuned - base | **+0.339** | [+0.284, +0.406] | **yes** |
| Groq - fine-tuned | **+0.121** | [+0.079, +0.173] | **yes** |

**The fine-tuned model does NOT match the 70B baseline.** The remaining 0.121 gap is
statistically significant, not noise. The README's original "can match" claim is not
supported by this evidence and has been corrected.

## Per-field F1

| field | Groq 70B | base 1.5B | fine-tuned 1.5B |
|---|---|---|---|
| title | 1.000 | 0.907 | 0.995 |
| company | 0.854 | 0.561 | 0.811 |
| contract_type | 0.979 | 0.621 | **0.986** |
| duration_months | 0.757 | 0.467 | **0.850** |
| start_date | 1.000 | 0.148 | 0.870 |
| location | 0.985 | 0.675 | 0.907 |
| remote_policy | 0.952 | 0.000 | 0.706 |
| required_skills | 0.565 | 0.077 | 0.475 |
| nice_to_have_skills | 0.800 | 0.000 | 0.390 |
| years_experience_min | 0.857 | 0.533 | 0.842 |
| education_level | 0.919 | 0.586 | 0.835 |
| language_requirements | 0.963 | 0.235 | 0.609 |
| salary_range | 0.923 | 0.632 | 0.783 |
| alternance_rhythm | 0.923 | 0.600 | 0.727 |

The fine-tuned model **beats the 70B model** on `contract_type` (0.986 vs 0.979) and
`duration_months` (0.850 vs 0.757). The second is attributable to label work rather than
raw model capability: Groq systematically leaves `duration_months` null when a posting
states a range ("12 a 24 mois"), and the training labels were corrected to a documented
minimum-of-range convention (see `LABELLING-NOTES.md`). The student learned a rule its
teacher does not follow.

Its weakest fields are the list fields, `required_skills` (0.475) and
`nice_to_have_skills` (0.390) -- the only fields where it loses badly to Groq.

## Known bias: near-duplicate test leakage

France Travail re-lists the same job under a new reference with small edits, which the
exact `text_hash` check in `schema/dedupe.py` cannot catch. **10 of 100 test postings have
a training posting at >= 0.5 five-gram Jaccard** (4 above 0.9). This inflates a
*fine-tuned* model score while leaving prompted-only baselines untouched -- it biases
exactly the comparison this project makes, in the flattering direction.

Scores on the 90 leak-free postings (ids pinned in `data/test/leaked_posting_ids.json`):

| system | full (n=100) | leak-free (n=90) | delta |
|---|---|---|---|
| Groq 70B | 0.891 | 0.889 | -0.002 |
| base 1.5B | 0.432 | 0.416 | -0.016 |
| fine-tuned 1.5B | 0.770 | **0.755** | **-0.015** |

Groq, which never trained on anything, is essentially unchanged (-0.002), confirming the
clean subset is of equivalent difficulty. The fine-tuned model drops -0.015, which is the
leakage effect itself. **0.755 is the more defensible headline number.** The conclusion is
unchanged: fine-tuning still closes ~72% of the gap on clean data.

## Alternative metrics (reported for transparency, not as headline)

| system | strict macro F1 | per-item partial | skills merged (union) |
|---|---|---|---|
| Groq 70B | 0.891 | 0.895 | 0.791 |
| base 1.5B | 0.432 | 0.443 | 0.314 |
| fine-tuned 1.5B | 0.770 | 0.775 | 0.610 |

- **Per-item partial credit** (list fields scored per skill rather than all-or-nothing set
  equality) moves every arm by less than 0.006. It is *not* simply a lenient metric: it
  improved `required_skills` but made `nice_to_have_skills` worse for Groq (0.800 ->
  0.709), because per-item scoring is more granular in both directions.
- **Skills union** merges `required_skills` + `nice_to_have_skills`, since a skill
  extracted correctly but filed in the wrong bucket is otherwise penalised twice.
  Misfiling accounts for 9.2% of gold skill items and runs entirely one way (22
  gold-required predicted as nice-to-have, 0 the reverse). Merging does not rescue the
  field: the dominant error is over-generation, not misfiling.

## The abstention problem

The fine-tuned model characteristic failure is emitting a value where the gold is null.
On `required_skills` it produces a non-null list for 76/100 postings against gold 49.

This was far worse before a training fix. The first run computed loss over the **entire
sequence**, because the dataset used trl `{"messages": [...]}` shape: `SYSTEM_PROMPT` is
1566 tokens and identical in every example, so **~94% of the gradient** went to
reproducing a constant prompt. Switching to prompt-completion shape scoped the loss to the
~129-token answer:

| | required_skills fp | required_skills tn | macro F1 |
|---|---|---|---|
| full-sequence loss (1 epoch) | 85 | 1 | 0.510 |
| completion-only loss (3 epochs) | **52** | **23** | **0.770** |

## Latency and cost

| system | median | mean | p95 | median prompt tokens |
|---|---|---|---|---|
| Groq (rate-limited API) | 16.36s | 62.52s | 267.34s | ~2156 |
| base 1.5B (T4) | 12.48s | 14.11s | 19.03s | 2718 |
| fine-tuned 1.5B (T4) | 17.72s | 18.45s | 24.95s | **2102** |

**The latency numbers do not support a speed claim and must not be quoted as one.** Groq
figures come from a rate-limited free-tier bulk labelling run; the 267s p95 is queueing,
not inference. At the median Groq is *faster* than the fine-tuned model on a T4.

Two honest observations that do hold:

- The fine-tuned model needs **2102 median prompt tokens vs 2718** for the few-shot base
  (-23%), because fine-tuning replaces the in-context examples. That is the real basis for
  a cost-per-request argument, and it compounds with the 47x parameter difference.
- It is *slower* than its own base model (17.72s vs 12.48s) despite the shorter prompt,
  because the LoRA adapter is unmerged and adds per-token overhead. `merge_and_unload()`
  before inference would likely recover most of that. Not yet measured.

## What is being compared

The arms do not receive identical prompts, by design:

| | prompt | how it learned the task |
|---|---|---|
| Groq 70B | `SYSTEM_PROMPT` + 3 few-shot examples + posting | in-context only |
| base 1.5B | `SYSTEM_PROMPT` + 3 few-shot examples + posting | in-context only |
| fine-tuned 1.5B | `SYSTEM_PROMPT` + posting | QLoRA on 541 postings |

Both baselines get the same few-shot construction (`label/prompt.py::build_messages`), so
the base arm is not a strawman. All three share the same `SYSTEM_PROMPT` object, so the
schema text cannot drift between them. This is a **system-level** comparison (each model
in its intended configuration), not a controlled prompt ablation.

**Distillation caveat**: the fine-tuned model training labels were generated by Groq. It
is a student trained on this teacher output, so it should not be expected to exceed the
teacher on the teacher own failure modes -- directly relevant to `required_skills`, where
Groq itself manages only 0.565.

## Limitations

1. **10% near-duplicate test leakage** (quantified above; leak-free numbers reported).
2. **n=100.** Exact-match carries an 18-point confidence interval. Differences below
   ~0.06 macro F1 are not distinguishable from sampling noise.
3. **Labels are Groq-generated**, hand-corrected only for the 100-posting test set.
   `train.jsonl` has had targeted cleanup passes but is not fully hand-verified.
4. **Single run per configuration.** No seed variance measured, so the reported
   confidence intervals cover test-set sampling only, not training stochasticity.
5. **Latency is not a controlled comparison** (see above).
6. The `data_ai_stage` bucket (2 postings) sits entirely in train and is never tested.
