# Results

Scored by `eval/score.py` against the 100-posting hand-corrected test set
(`data/test/test.jsonl`). Reproduce with `notebooks/kaggle_benchmark.ipynb`.

## Headline

| system | params | prompt tokens | macro F1 | exact-match |
|---|---|---|---|---|
| Groq Llama-3.3-70B, few-shot | ~70B | ~2156 | **0.904** | 36% |
| Qwen2.5-1.5B base, few-shot | 1.5B | 2718 | 0.471 | 1% |
| **Qwen2.5-1.5B fine-tuned (QLoRA)** | **1.5B** | **812** | **0.815** | 17% |

Fine-tuning closes **~75% of the gap** between the 1.5B base model and one ~47x larger,
using 654 training examples on a single free T4 -- while using **fewer prompt tokens than
either baseline**.

## Four training configurations

| arm | epochs | prompt | skills labels | macro F1 |
|---|---|---|---|---|
| A | 1 | short (812 tok) | recalibrated | 0.774 |
| B | 3 | short (812 tok) | recalibrated | **0.815** |
| C | 3 | **full (2022 tok)** | recalibrated | **0.816** |
| D | 3 | short (812 tok) | precision-fixed | 0.797 |

**B, C and D are one cluster, not a ranking.** See the noise floor below.

## The noise floor -- read this before comparing any two numbers

Arms B and D differ in **one field's training labels**. Every other field had byte-identical
data. Yet fields that were never touched moved substantially:

| field | Arm B | Arm D | delta | labels changed |
|---|---|---|---|---|
| `language_requirements` | 0.720 | 0.476 | **-0.244** | no |
| `remote_policy` | 0.842 | 0.900 | +0.058 | no |
| `skills` | 0.349 | 0.324 | -0.025 | **yes** |

Of the -0.018 macro difference, **-0.016 came from untouched fields** and -0.002 from the
one that changed. Across the 12 untouched fields: mean |delta| **0.032**, max **0.244**.

**Training stochasticity is therefore ~0.03 macro F1 on this setup, and it is larger than
most of the effects measured here.** Any single-run difference below ~0.03 is not
interpretable. The bootstrap CIs reported later cover *test-set sampling only* and do not
include this second source.

## What survives the noise floor

**1. The short prompt is free.** Arm B (812 tokens) vs Arm C (2022 tokens): **0.815 vs
0.816**, a difference of 0.001. A 2.5x prompt reduction costs nothing measurable and runs
18% faster (15.1s vs 18.4s median). Once the model has seen the output shape 654 times, the
1173-token JSON schema dump is dead weight. It is *not* dead weight for the prompted
baselines, which have never seen the schema -- this is a fine-tuning-specific saving.

**2. Targeted data fixes starved fields.** `remote_policy` went **0.706 -> 0.842 / 0.900**
after its training examples tripled (32 -> 101), reproduced across two independent runs and
far outside the noise band. This validates the starvation diagnosis: Pearson r = **-0.456**
between per-field training-example count and gap-to-Groq, with sub-150-example fields
averaging a +0.133 gap against +0.069 for the rest.

**3. Fine-tuning is worth ~0.34 macro F1.** Base 0.471 -> fine-tuned 0.815, paired bootstrap
on the earlier run **+0.339 [+0.284, +0.406]**, far outside both noise sources.

## What did NOT work

**Two skills-label interventions, in opposite directions, both failed.**

| attempt | train labels | skills F1 | precision |
|---|---|---|---|
| original cleanup | mean 3.31 items | 0.388 | -- |
| recalibrated (wider) | mean 4.20 | 0.349 | 0.218 |
| precision-fixed (tighter, cleaner) | mean 3.95 | 0.324 | 0.210 |

Widening the labels and then narrowing them at higher precision (0.754 -> 0.874 measured
against the gold set's 307 keep/drop decisions) both left precision at **~0.21**, with the
model emitting skills on **81-87 of 100 postings against gold's 60**.

When two opposite interventions produce the same failure, the variable being adjusted is not
the cause. The most likely reading is a **capability limit**: at 1.5B with ~650 examples the
model cannot reliably judge which skills belong, so it lists what it sees. Hand-corrected
training labels would test that; another regex would not.

`skills` remains the single largest per-field gap to Groq (0.33 vs 0.635).

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
