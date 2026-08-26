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
