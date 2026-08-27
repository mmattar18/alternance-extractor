# Pre-registered predictions

Written **before** any of the three runs below reported, so they can be checked rather than
retrofitted. Reference point: v13 = 3 epochs, full prompt, 541 rows, pre-recalibration
skills labels, scored under the merged schema with text folding = **macro F1 0.800**
(Groq 0.904, base Qwen 0.471).

## Arm A -- 1 epoch, short prompt, 654 rows, recalibrated labels

**Predicted macro F1: 0.72 - 0.78 (point estimate 0.75)** -- i.e. BELOW v13's 0.800.

Reasoning: it has 39 optimizer steps against v13's 99. The data and label improvements are
real but should not fully offset losing two thirds of the training. The earlier 1-vs-3
epoch comparison was 0.510 vs 0.770, though that gap is an overestimate here because it
predates the completion-only loss fix, which gave every step ~16x more task-relevant
gradient.

Sub-predictions, which matter more than the headline because they test the specific work:
- `skills` strict F1 **improves** over v13's 0.388 -> **0.42-0.50**, because the labels now
  carry ~30% more items matching the gold convention. If this does NOT improve, the
  recalibration did not work and that is the more interesting finding.
- `remote_policy` **improves** over v13's 0.706 -> **0.72-0.80** (training examples went
  32 -> 101).
- `title` stays ~1.000 and `contract_type` stays ~0.98; these are saturated.

## Arm B -- 3 epochs, short prompt, 654 rows, recalibrated labels

**Predicted macro F1: 0.80 - 0.86 (point estimate 0.83)** -- modestly ABOVE v13.

Reasoning: the field-gap arithmetic says closing the top six fields is worth +0.097
(0.800 -> 0.897). But the data enrichment only partially closed the starvation: three of
those fields are still under the ~150-example threshold where the measured gap halves
(`language_requirements` 70, `alternance_rhythm` 76, `salary_range` 84). Capturing perhaps
a third of that potential gives +0.03, plus whatever the skills recalibration is worth.

**I do not expect this to reach Groq's 0.904.** The remaining gap should stay
statistically significant.

## Arm C -- 3 epochs, FULL prompt, 654 rows, recalibrated labels

**Predicted macro F1: within +/-0.02 of Arm B (point estimate 0.84).**

This is the real question the A/B answers: does a fine-tuned model still need the
1173-token schema dump? My prediction is **no meaningful difference**, because the model
sees the output shape 654 times in training and should not need it respelled. If Arm C
beats Arm B by more than ~0.03, the schema dump is load-bearing at this data scale, and
the 2.3x speed win is not free.

## What would falsify the diagnosis

The claimed root causes are (1) rare-field starvation and (2) skills labels being 17%
terser than gold. Those predict specific movements:

- If `remote_policy` does NOT improve despite 3x the examples, the starvation explanation
  is wrong and the correlation (r=-0.456) was confounded.
- If `skills` does NOT improve despite 30% more label items, the recalibration missed the
  real problem -- likely meaning the issue is extraction capability, not label convention.
- If Arm B lands below 0.80, then more data plus better labels did not beat simply having
  more epochs on worse data, which would undercut the whole "data is the bottleneck"
  reading.

Any of those would be a more valuable finding than a number that merely goes up.


---

# OUTCOMES (recorded after the fact, predictions above unedited)

| arm | predicted | actual | |
|---|---|---|---|
| A (1ep, short) | 0.72-0.78 (pt 0.75) | **0.774** | within range |
| B (3ep, short) | 0.80-0.86 (pt 0.83) | **0.815** | within range |
| C (3ep, full) | within +/-0.02 of B | **0.816** | correct -- differs by 0.001 |

Headline predictions landed. The sub-predictions, which actually tested the diagnosis,
split -- one confirmed, one clearly wrong.

## CONFIRMED: rare-field starvation

Predicted `remote_policy` would improve from 0.706 given 3x the training examples
(32 -> 101).

- Arm A (1 epoch): **0.706** -- no movement, appeared to falsify the diagnosis
- Arm B (3 epochs): **0.842** -- large improvement

Arm A's flat result was an under-training artifact, not evidence against the diagnosis. Had
only Arm A been run, the correct conclusion would have been the wrong one. `language_requirements`
moved the same way (0.667 -> 0.720), as did `duration_months` (0.850 -> 0.930) and
`start_date` (0.870 -> 0.909). The r=-0.456 correlation between per-field example count and
gap-to-Groq holds up.

## WRONG: the skills recalibration overcorrected

Predicted `skills` would improve from 0.388 to 0.42-0.50 on labels carrying ~30% more items.

- Arm A: **0.327**
- Arm B: **0.349** -- still below the 0.388 baseline at full training

Confusion counts at 3 epochs: tp=19, **fp=68**, fn=3. The model emits a skills list for 87 of
100 postings against gold's 60, precision 0.218 against recall 0.864.

This is a self-inflicted error and the diagnosis behind it was half right. The original
problem was real -- training labels were 17% terser than gold and the model under-read
(recall 0.582). The fix over-shot: rebuilding from pre-cleanup Groq output through only a
duty-verb/vague filter took the mean to 4.20 items against gold's 3.98, and the model
amplified that into systematic over-emission. Under-generation was traded for worse
over-generation.

**What the evidence now says the fix is:** not "more items" but "the right items". Groq's raw
output has precision 0.704 on this field; the recalibrated labels inherit its verbosity
without its judgement. The next attempt should target gold's distribution directly
(60% non-null, mean 3.98) rather than reverting to raw Groq and filtering loosely -- or
accept that `skills` needs hand-corrected training labels, since it is the one field where
convention cannot be captured by a regex.

## Net

Arm B 0.815 vs v13 0.800 (+0.015) on a 21% larger, partially-recalibrated dataset, with the
prompt cut from 2102 to 812 median tokens. The gain came from the rare-field work; the
skills work actively cost points and would have shown a larger gain without it.


## Arm D and an accidental variance estimate -- this weakens claims made above

Arm D = Arm B with ONE change: skills labels rebuilt through a precision-focused filter
(precision 0.754 -> 0.874 measured against the gold set's 307 keep/drop decisions).
Result: **0.797** vs Arm B's 0.815, skills 0.324 vs 0.349.

Read naively that says the fix failed again. The per-field decomposition says otherwise:

| field | Arm B | Arm D | delta | labels changed |
|---|---|---|---|---|
| `language_requirements` | 0.720 | 0.476 | **-0.244** | **no** |
| `remote_policy` | 0.842 | 0.900 | **+0.058** | **no** |
| `skills` | 0.349 | 0.324 | -0.025 | yes |

Of the -0.018 macro difference, **-0.016 comes from fields whose training data was
byte-identical** and only -0.002 from the field actually modified. Across the 12 untouched
fields: mean |delta| 0.032, max |delta| 0.244.

**That is a seed-variance estimate obtained by accident, and it is large.** Consequences:

- **Arm B "beating" v13 (0.815 vs 0.800) is not supportable.** +0.015 sits inside the noise
  band. Stated too confidently above.
- **"The skills recalibration failed" is over-stated.** Each step (0.388 -> 0.349 -> 0.324)
  is ~0.025, comparable to noise. The direction is consistent across two independent runs,
  which is suggestive, not conclusive.
- **The bootstrap CIs in RESULTS.md cover test-set sampling only.** Training stochasticity
  is a second, apparently larger source of variance that none of the reported intervals
  include.
- **Any claim below ~0.03 macro on this setup needs multiple seeds.** One run per config
  cannot support it.

**What survives unambiguously:** `remote_policy` 0.706 -> 0.842/0.900 is a +0.14 to +0.19
move, far outside the noise band, reproduced in two independent runs. The rare-field
starvation finding stands. So does the prompt result: 2102 -> 812 median prompt tokens with
no macro cost detectable above noise.

**What the skills evidence now supports:** two label interventions in opposite directions
(more items, then fewer-but-cleaner) both left precision at ~0.21 with the model emitting
skills on 81-87 of 100 postings against gold's 60. Label convention does not appear to be
the lever. The more likely reading is that the model cannot reliably decide which skills
belong, and lists what it sees -- a capability limit at 1.5B with ~650 examples, not a
labelling artifact. Hand-corrected training labels would test that; another regex will not.


## Arm C: the prompt question, answered

| | prompt tokens (median) | macro F1 | median latency |
|---|---|---|---|
| Arm B, short prompt | **812** | 0.815 | 15.1s |
| Arm C, full prompt | 2022 | 0.816 | 18.4s |

**A 2.5x prompt reduction costs 0.001 macro F1** -- far inside the ~0.03 noise floor measured
from the B/D comparison. Predicted "within +/-0.02"; observed 0.001.

The 1173-token JSON schema dump is dead weight once the model has seen the output shape 654
times in training. It is NOT dead weight for the prompted baselines, which have never seen
the schema and genuinely need it -- so this is a fine-tuning-specific saving, and it is the
concrete basis for a cost-per-request argument: 812 tokens against Groq's ~2156, on top of
the 47x parameter difference.

Worth noting the shape of the win: the accuracy is identical, the saving is entirely in
serving cost and latency. That is a more defensible claim than a quality improvement,
because it sits well outside the noise band rather than inside it.
