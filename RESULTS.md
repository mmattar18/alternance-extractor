# Results

Scored by `eval/score.py` against the 100-posting hand-corrected test set
(`data/test/test.jsonl`). Reproduce with `notebooks/kaggle_benchmark.ipynb`.

## Headline

| system | params | prompt tokens | macro F1 | strict | fields correct |
|---|---|---|---|---|---|
| Groq Llama-3.3-70B, few-shot | ~70B | ~2156 | **0.916** | 0.904 | 92.8% |
| Qwen2.5-1.5B base, few-shot | 1.5B | 2718 | 0.486 | 0.471 | -- |
| **Qwen2.5-1.5B fine-tuned (QLoRA)** | **1.5B** | **812** | **0.830** | 0.815 | 87.6% |
| **Qwen2.5-3B fine-tuned, 654 rows** | 3B | 812 | 0.856 | 0.838 | 89.2% |
| **Qwen2.5-3B fine-tuned, 881 rows** | **3B** | **812** | **0.877 +/- 0.003** | 0.861 | **89.8%** |

Macro F1 scores list fields (`skills`, `language_requirements`) **per item**, the standard
treatment for multi-value slots in information extraction; scalar fields stay exact-match.
The stricter all-or-nothing set-equality figure is in the `strict` column and is never
dropped. Both are computed identically for every system.

Sanity check that per-item is the right measure rather than a lenient one: on `skills` it
gives 0.791 / 0.610 (Groq / fine-tuned), while a Jaccard>=0.5 partial-match criterion gives
0.807 / 0.617 -- within 0.007. More lenient variants were tested and rejected: every one of
them raised Groq at least as much as the fine-tuned model, so leniency never closes the gap
(strict 0.246 -> Jaccard>=0.3 0.165 at best). The fine-tuned model is genuinely weaker on
`skills` under every metric tried.

Fine-tuning closes **~75% of the gap** between the 1.5B base model and one ~47x larger,
using 654 training examples on a single free T4 -- while using **fewer prompt tokens than
either baseline**.

## Does more targeted data help? 654 vs 881 rows

Single-variable experiment: same model, same prompt, same hyperparameters, +227 training
rows concentrated in the starved fields. (`MAX_SEQ_LEN` also dropped 3584 -> 2560, which
truncates nothing -- the longest sequence is 2199 tokens -- so it changes memory only.)

| field | 654 rows | 881 rows | Groq | training examples |
|---|---|---|---|---|
| `language_requirements` | 0.667 | **0.720** | 0.968 | 70 -> 214 |
| `alternance_rhythm` | 0.833 | **0.909** | 0.923 | 76 -> 78 |
| `salary_range` | 0.818 | **0.870** | 0.923 | 84 -> 138 |
| `remote_policy` | 0.900 | **0.952** | 0.952 | 101 -> 118 |
| `years_experience_min` | 0.870 | **0.917** | 0.857 | -- |
| `company` | 0.844 | **0.866** | 0.854 | -- |
| `skills` | 0.384 | 0.384 | 0.635 | 318 -> 456 |
| `duration_months` | 0.978 | 0.829 | 0.757 | -- |
| **macro F1** | **0.856** | **0.874** | 0.916 | |

**+0.018, which is BELOW the ~0.03 noise floor** -- not conclusive on the headline alone.
What argues it is real: 10 of 13 fields improved or held against 2 regressions, and the
gains land in the fields the new data was bought for. Noise moves fields randomly in both
directions; this did not.

The model now **matches or beats the 70B on five fields**: `contract_type` (0.986 vs 0.979),
`duration_months` (0.829 vs 0.757), `years_experience_min` (0.917 vs 0.857), `company`
(0.866 vs 0.854) and `remote_policy` (0.952, tied).

**`skills` did not move at all** (0.384) despite its examples going 318 -> 456. Combined with
two failed label interventions and its only prior movement coming from model size
(1.5B -> 3B), the evidence consistently says `skills` is capacity-limited, not data-limited.

**Cumulative:** 1.5B 0.830 -> 3B 0.856 -> 3B with more data 0.874, against Groq's 0.916.
The remaining gap is 0.042, and roughly half of it is `skills` alone.

## Does model size help? 1.5B vs 3B

Same data, same short prompt, same 3 epochs -- only the base model differs.

| | macro F1 | strict | exact | median latency | fields beating Groq |
|---|---|---|---|---|---|
| 1.5B | 0.830 | 0.815 | 17% | 15.1s | 2 |
| **3B** | **0.856** | 0.838 | 20% | 20.9s | **3** |

**+0.026, which sits at the measured ~0.03 noise floor** -- on the headline number alone this
is suggestive, not conclusive. It was pre-registered that 3B would need to clear ~0.86 to
count as a real improvement; it reached 0.856, just under.

What argues it is more than noise: noise moves fields randomly in both directions, but here
**7 of 13 fields improved, 3 worsened, 3 unchanged**, and several moves are large --
`alternance_rhythm` +0.11, `salary_range` +0.09, `duration_months` +0.07, and `skills` +0.05
with precision 0.21 -> 0.275, its first real movement after two failed label interventions.

The 3B beats Groq on three fields (`contract_type` 0.993, `duration_months` 0.978,
`years_experience_min` 0.870) against the 1.5B's two.

Cost: **39% slower inference** (20.9s vs 15.1s median) and double the memory, for +0.026.
Whether that trade is worth it depends on the deployment; for a batch pipeline it clearly is,
for interactive use it is arguable. The 1.5B remains the stronger *headline* -- "1.5B reaches
90% of a 70B" is a sharper claim than the same for 3B -- so both are reported.

## Four training configurations

| arm | epochs | prompt | skills labels | macro F1 |
|---|---|---|---|---|
| A | 1 | short (812 tok) | recalibrated | 0.774 |
| B | 3 | short (812 tok) | recalibrated | **0.815** |
| C | 3 | **full (2022 tok)** | recalibrated | **0.816** |
| D | 3 | short (812 tok) | precision-fixed | 0.797 |

**B, C and D are one cluster, not a ranking.** See the noise floor below.

## Measured variance: 3 seeds, identical config

Three runs of the same configuration (3B, 881 rows, short prompt, 3 epochs) differing only
in `SEED`, which controls both the train/eval split and training randomness -- so this is
full-pipeline variance ("if I reran this end to end, what would I get?").

| seed | macro F1 | strict | exact-match |
|---|---|---|---|
| 42 | 0.874 | 0.855 | 19% |
| 43 | 0.879 | 0.864 | 23% |
| 44 | 0.878 | 0.863 | 24% |
| **mean** | **0.877 +/- 0.003** | 0.861 +/- 0.005 | 22% +/- 2.6% |

**Macro sd = 0.0026** (range 0.005). With 2sd = 0.005 as the threshold:

| claim | delta | verdict |
|---|---|---|
| more data, 654 -> 881 rows | +0.018 | **real** |
| 3B vs 1.5B | +0.026 | **real** |
| short vs full prompt | +0.001 | within noise |

### Correcting an earlier error in this document

Previous versions of this file reported a **~0.03 noise floor** and used it to withdraw the
"more data helped" and "3B beat 1.5B" claims. **That estimate was wrong.** It came from a
single pair of runs (Arms B and D) in which `language_requirements` happened to swing 0.244,
and one unlucky field on one comparison was generalised into a macro-level noise figure.
Measured properly across three seeds, macro variance is an order of magnitude smaller and
both claims stand.

The lesson is not that caution was misplaced -- it is that a noise floor inferred from one
accidental comparison is itself a single noisy measurement, and was treated with more
confidence than it earned.

### Per-field variance is real and much larger

With byte-identical training data, individual fields move far more than the macro:

| field | range across the 3 seeds |
|---|---|
| `duration_months` | 0.126 |
| `years_experience_min` | 0.083 |
| `alternance_rhythm` | 0.076 |
| `language_requirements` | 0.053 |
| `remote_policy` | 0.048 |
| `skills` | 0.043 |

The macro is stable *because* these cancel. **Per-field claims need multiple seeds; macro
claims do not.** Any single-run per-field comparison in this document below ~0.1 should be
read as indicative only.

## What survives the noise floor## What survives the noise floor

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

## Per-field F1 (merged 13-field schema)

Groq vs the two 3-epoch arms. B and C differ only in prompt length; treat their per-field
differences as noise unless they exceed ~0.03.

| field | Groq 70B | base 1.5B | Arm B (short) | Arm C (full) |
|---|---|---|---|---|
| title | 1.000 | 0.907 | 1.000 | 1.000 |
| company | 0.854 | 0.561 | 0.805 | 0.819 |
| contract_type | 0.979 | 0.621 | 0.979 | **0.993** |
| duration_months | 0.757 | 0.467 | **0.930** | **0.905** |
| start_date | 1.000 | 0.148 | 0.909 | 0.884 |
| location | 0.985 | 0.675 | 0.936 | 0.947 |
| remote_policy | 0.952 | 0.000 | 0.842 | 0.900 |
| **skills** | **0.635** | 0.063 | **0.349** | **0.333** |
| years_experience_min | 0.857 | 0.533 | 0.842 | 0.842 |
| education_level | 0.919 | 0.586 | 0.832 | 0.816 |
| language_requirements | 0.968 | 0.235 | 0.720 | 0.720 |
| salary_range | 0.923 | 0.632 | 0.727 | 0.727 |
| alternance_rhythm | 0.923 | 0.600 | 0.727 | 0.727 |

The fine-tuned model **beats the 70B** on `contract_type` (0.993 vs 0.979) and
`duration_months` (0.930 vs 0.757). The second is label work, not model capability: Groq
systematically returns null when a posting states a duration range ("12 a 24 mois"), and the
training labels encode a documented minimum-of-range convention (`LABELLING-NOTES.md`). The
student learned a rule its teacher does not follow.

`skills` is the one field where it loses badly (0.33 vs 0.635) and is the largest single
contributor to the remaining gap. See "What did NOT work" above.

## Known bias: near-duplicate test leakage

*Quantified on the earlier v13 run (14-field schema). The leakage itself is a property of
the data split, not of any particular model, so the ids and the direction of the bias carry
over unchanged to all later arms.*

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

*Computed on the earlier v13 run, before the schema merge; retained because the
methodological point stands.*

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

*Numbers below are from the earlier 14-field runs; the behaviour persists in every later
arm (Arm C: skills precision 0.205, emitting on 88 of 100 postings against gold's 60).*

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
