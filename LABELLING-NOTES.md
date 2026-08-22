# Labelling notes

Running log of problems found and decisions made while hand-correcting
`data/test/test.jsonl` via `label/review_server.py`, and while cleaning up
`data/train/train.jsonl`. Read this before resuming a labelling session —
it exists so decisions don't get re-litigated or made inconsistently across
sessions.

Last updated: 2026-08-22. Test-set progress: **100/100 corrected — test set is DONE.**

The first 27 were corrected by hand, screen by screen, via `label/review_server.py`.
The remaining 73 were corrected by a forked agent applying the same rules
documented in this file (all 100 re-validated against `schema.posting.JobPosting`
afterward — 0 failures). Groq baseline scored against the finished set:
`macro_f1 = 0.893`, `exact_match_rate = 35.0%` (run:
`python eval/score.py data/test/test.jsonl data/test/test_candidates.jsonl`).
Weakest fields by far: `required_skills` (F1 0.565, precision 0.453 — Groq
massively overgenerates) and `duration_months` (F1 0.757, recall 0.609 — the
range-to-null pattern below).

## Conventions established (apply these when reviewing)

- **`required_skills`/`nice_to_have_skills`**: only count a *concretely
  named* technology/tool/language (e.g. "Power BI", ".NET", "C#", "Chorus",
  "Apache Airflow"). Generic duty descriptions ("développer des
  applications", "assurer la veille technologique") or domain/product
  jargon with no transferable meaning (internal system names like
  "Centum", "PCB", "CU/RC") don't count, even when they sit in a bullet
  list that looks like a requirements section.
- Soft skills / aptitudes ("esprit d'analyse", "bon relationnel",
  "autonomie") never go in `required_skills` — schema field is for
  technical/professional skills.
- Split required vs nice-to-have strictly on the text's own framing: a
  skill/experience explicitly marked "serait un plus" / "est un plus" /
  "apprécié" is nice-to-have; everything else stated as needed is required.
- **`company`**: watch for postings where an intermediary is named first
  (a recruitment forum organizer, a training/apprenticeship broker like
  ISCOD) but the real employer is a *different* named company, or
  genuinely anonymized ("entreprise partenaire spécialisée dans...").
  Pick the real employer, or null if it's never named — don't default to
  whichever org name appears first in the text.
- **`years_experience_min`**: only fill when the text gives an actual
  number of years. Qualitative asks ("une première expérience
  significative") without a number stay null — don't infer a number.
- **`duration_months`**: when the text states a range ("12 à 24 mois"),
  gold value = **the minimum**. This is a deliberate convention, not
  discovered from the text — apply it consistently. (See eval note below —
  Groq/Llama systematically leaves this field null on ranges, so this
  convention matters for training-data consistency too.)
- **`education_level`**: if the text itself states conflicting levels in
  different places (e.g. one line says "M1", the closing line says "ouvert
  aux M2"), record both / note the conflict rather than silently picking
  one.
- Generic "pay follows the legal apprentice scale" boilerplate ("selon le
  barème légal", "en pourcentage du SMIC selon l'âge") is **not** a real
  salary_range — no actual figure was ever stated. As of the eval fix
  below this no longer needs manual blanking; see next section.

## `eval/score.py` normalizations added (so hand-correction doesn't need
to fight formatting noise that isn't a real extraction error)

- **`location`**: France Travail prefixes some postings with the
  department code ("62 - Vendin-le-Vieil") and not others, inconsistently,
  for the same city. Comparison now strips a leading `NN -` / `2A -` / `2B
  -` prefix before matching. Gold rows don't need the prefix manually
  stripped anymore.
- **`salary_range`**: any value with **no digit in it** is treated as
  equivalent to null on both sides (covers all the "selon le barème
  légal..." boilerplate variants — Herbal T, Inserm, Humantocomputer,
  ISCOD, Salvert all hit this pattern). A real figure ("1500€/mois", "80%
  du SMIC") still scores normally.
- Both changes are covered by smoke tests in `eval/score.py` — run
  `python eval/score.py` (no args) after touching the scoring logic.

## `data/train/train.jsonl` fix applied (commit `fd656ad`)

Groq's own labels (used directly as training targets) leave
`duration_months` null on every posting where the text states a duration
*range* rather than a single number — confirmed 10/10 range-postings in
train had this. Left uncorrected, Qwen would learn "range in text → leave
blank," which is exactly the behavior the hand-corrected test set now
penalizes (minimum-of-range convention above). Backfilled 9 of the 10 rows
with the minimum; skipped `france_travail:212DXSR` because its range
("18/24 mois") describes the BTS diploma length, not an explicitly stated
contract duration — didn't want to force an inference the text doesn't
actually make.

**If more training rows turn out to have this or a similar
gap-between-Groq's-habits-and-the-gold-convention problem, patch
`train.jsonl` the same way before fine-tuning, not after** — the whole
point is Qwen's training labels should reflect the same conventions the
test set is scored against.

## Judgment calls from the 73-posting batch worth a second look

- Staffing/ESN agencies (Adecco, Manpower, Randstad, Synergie, Proman,
  Davidson, Holenek Ingénierie, LIP) were kept as `company` — French
  intérim/ESN law makes the agency the legal employer — while
  training-brokers and recruitment cabinets (ISCOD-style, Le CabRH, ID
  Search Finance, Saint Junien Recrutement, Fulfillink, IFRIA) were nulled
  out per the broker-vs-employer rule above. The line between the two
  categories was inferred from firm-name patterns, not certain in every
  case.
- Ambiguous company identity, resolved by best judgment, not certainty:
  NGE vs GUINTOLI (kept NGE); FINAPOLLINE vs ACTEON (went with ACTEON);
  two Burger King postings named inconsistently (BKG vs "Burger King");
  FONDAT REGION OUEST LIGUE CANCER vs Oncobretagne (kept the header
  name); AKAMANMAN nulled after re-reading it as just the "site client"
  for an unnamed actual employer.
- `years_experience_min = 0` was applied to "n'est pas obligatoire"
  phrasing — looser than the schema docstring's literal "débutant
  accepté" example. Worth confirming this reading is what you want.
- `contract_type` for explicit "intérim" postings was standardized to
  `"cdd"` (there's no better fit in the locked enum) — see schema gap
  below, this is a workaround, not a schema fix.
- "Named skill" was extended beyond software to cover methodologies/
  certifications (Agile/Scrum, Cycle en V, DevOps, HACCP) and named trade
  specializations ("électrotechnicien", "maîtriser les cuissons des
  viandes") as `required_skills`.

## New Groq failure patterns found in the 73-posting batch

- **`required_skills` overgeneration is much worse than the single
  documented Enedis example** — confirmed on 10+ more postings (infirmier
  CDDs, Adecco, Le CabRH, Fulfillink, Caen TT, CAF Marne...): Groq dumps
  entire soft-skill/generic-competency bullet lists into
  `required_skills` even under a literal "Compétences requises"/
  "Savoir-faire" header. This is the single biggest driver of the low
  required_skills F1 above — worth being extra vigilant on this field in
  any future spot-checks.
- **Duration-range→null is more pervasive than the 9/10 `train.jsonl`
  cases already patched** — hit ~7 more instances in the test set alone
  ("X à Y mois", "X ou Y ans" phrasing). `train.jsonl` was only scanned
  for the exact "X à/-/et Y mois" pattern — **worth a broader re-scan**
  before fine-tuning (wider regex, and check year-based ranges too).
- **Company/location field swaps distinct from the broker pattern**: raw
  location string dumped into `company` (2 more instances beyond the
  Petit-Quevilly one already fixed by hand), plus a truncated header with
  a stray "— dept — city" suffix left inside `company` (CAF Marne
  posting).
- **Redundant language entries duplicated into skills fields** alongside
  a correct `language_requirements` entry — one more instance beyond the
  Inserm case already documented.
- **Salary boilerplate recurring far more than the 5 documented cases**,
  plus a real edge case for the eval fix: a posting stated payment
  "...sur 14 mois" (a payment-schedule detail, not a salary figure) —
  this contains a digit and would slip past `eval/score.py`'s
  `_HAS_DIGIT_RE` null-boilerplate heuristic despite not being a real
  salary_range. **The digit heuristic isn't foolproof** — keep an eye out
  for more of these if macro_f1 on salary_range looks off later.
- **Education-level hallucination confirmed a second time** (an "EO
  EVENTS" posting invented "Bac+3" with zero textual support, same
  pattern as the Enedis case already documented) — this is a recurring
  Groq behavior, not a one-off, budget for it in any future review pass.
- **Schema gap, not fixed**: `schema/posting.py`'s `_CONTRACT_SYNONYMS`
  has no French entry for "intérim"/"temporaire"/"mission d'intérim" —
  any raw "Intérim" string silently falls through to `"other"` via the
  `.get(key, "other")` default (hit 3+ times in this batch). The gold
  labels were hand-mapped to `"cdd"` as the closest fit, but the
  extraction pipeline itself (Groq prompt / synonym table) still doesn't
  know this — worth deciding whether to add an `_CONTRACT_SYNONYMS` entry
  or leave "other" as the honest answer for intérim, since the schema is
  documented as locked in the README and this wasn't changed without
  asking first.

## Open items / things to watch (test set is done — these are all about train.jsonl / next steps)

- **Test set is finished (100/100), not yet committed.** Next step is
  yours: commit `data/test/test.jsonl`, then decide on the two open
  items below before fine-tuning.
- Re-scan `train.jsonl` more broadly for the duration-range→null pattern
  — only the exact "X à/-/et Y mois" regex was checked (9/10 fixed,
  commit `fd656ad`); the test-set pass found ~7 more range-phrasing
  variants ("X ou Y ans" etc.) that a wider regex would likely also
  catch in train.
- `france_travail:212DXSR` in train.jsonl — `duration_months` still null,
  manual call needed if you decide the BTS-length framing should count
  after all.
- Decide on the `_CONTRACT_SYNONYMS` / "intérim" schema gap above before
  fine-tuning — it's a real, recurring French contract type the locked
  schema currently can't represent cleanly.
- One-off data artifact, not worth chasing: `france_travail:212NRDB`'s
  raw_text has a stray standalone "user" line (looks like a leaked
  chat-role marker from ingest) — confirmed it's the only occurrence
  across all 642 labelled postings, so not a systemic pipeline bug.
