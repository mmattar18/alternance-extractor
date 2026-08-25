# Labelling notes

Running log of problems found and decisions made while hand-correcting
`data/test/test.jsonl` via `label/review_server.py`, and while cleaning up
`data/train/train.jsonl`. Read this before resuming a labelling session —
it exists so decisions don't get re-litigated or made inconsistently across
sessions.

Last updated: 2026-08-25. Test-set progress: **100/100 corrected — test set is DONE**,
and a second cleanup pass has already gone over the ambiguous edge cases (see
"Second-pass cleanup" section below). `train.jsonl` has had three rounds of
targeted fixes (duration ranges, intérim contract_type, and now a full
required_skills/nice_to_have_skills cleanup across all 541 rows) — see the
`train.jsonl` sections below for the full history. Current baseline against
the test set, after all test-set fixes: `macro_f1 = 0.891`, `exact_match_rate
= 34.0%` (run: `python eval/score.py data/test/test.jsonl
data/test/test_candidates.jsonl`) — **this number describes Groq's baseline
and is unaffected by the train.jsonl cleanup below** (train.jsonl isn't read
by eval/score.py at all; it only matters once Qwen is actually fine-tuned on
it). Weakest fields by far: `required_skills` (F1 0.565, precision 0.453 —
Groq massively overgenerates) and `duration_months` (F1 0.757, recall 0.609
— the range-to-null pattern below). The small macro_f1 drop from the
original 0.893 is expected and *correct*, not a regression — it comes from
nulling out 4 `company`/`years_experience_min` values Groq had happened to
guess right without real textual support (hallucination corrections make
the baseline score more honest, not higher).

The first 27 test-set postings were corrected by hand, screen by screen, via
`label/review_server.py`. The remaining 73 were corrected by a forked agent
applying the same rules documented in this file (all 100 re-validated against
`schema.posting.JobPosting` — 0 failures, both times).

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

## `data/train/train.jsonl` fixes (two rounds)

**Round 1 (commit `fd656ad`)**: Groq's own labels (used directly as
training targets) leave `duration_months` null on every posting where the
text states a duration *range* rather than a single number — confirmed
10/10 range-postings matching the narrow pattern "X à/-/et Y mois" had
this. Backfilled 9 of the 10 with the minimum; skipped
`france_travail:212DXSR` because its range ("18/24 mois") describes the
BTS diploma length, not an explicitly stated contract duration.

**Round 2 (broader sweep, not yet committed as of this note)**: widened
the regex to also catch "ans" (years) and the connector "ou" — 27
additional candidate rows surfaced. Reading each in context: only **6
were real durations** and got patched (minimum-of-range, converting years
to months): `3941310`, `211XVYS`, `0990762`, `211XRYY`, `211ZCKW`,
`211PXYT`. The other 21 were false positives the wider regex
mis-triggered on — **13 were "years of experience required" ranges**
(e.g. "2 à 3 ans d'expérience"), **6 were age brackets** that have
nothing to do with contract length (a care facility's client ages, the
alternance 16–29 candidate-eligibility clause, a kids'-program age
range), and 2 more were diploma/formation-length framing (same category
as the already-skipped `212DXSR`). **Lesson: a wider regex on this
dataset finds far more false positives than true positives once "ans" is
included — always read context before patching, don't trust the match
alone.**

**Also fixed in round 2**: the "intérim" `contract_type` schema gap (see
below) — 11/28 candidate rows patched (7 null/other→cdd on explicit
intérim/temp-replacement signals, 4 cdi→null where the only intérim
mention was the staffing agency's own generic self-description rather
than a statement about that specific posting — nulled per the
no-hallucination rule, since "the employer happens to be a temp agency"
doesn't mean "this posting is a temp assignment"). 17 of the 28 left
unchanged (already correct, or insufficient evidence to override "no
explicit statement → null").

**If more training rows turn out to have this or a similar
gap-between-Groq's-habits-and-the-gold-convention problem, patch
`train.jsonl` the same way before fine-tuning, not after** — the whole
point is Qwen's training labels should reflect the same conventions the
test set is scored against.

## `required_skills`/`nice_to_have_skills` full training-set cleanup (round 3)

Rationale: `required_skills` is Groq's worst field by far (test-set F1 =
0.565), and since `train.jsonl` trains Qwen by distillation from Groq's
own raw predictions, that overgeneration habit would otherwise get baked
straight into the fine-tune, capping Qwen at "as bad as Groq" on this
field instead of giving it a real shot at beating it. Split all 541 rows
into 4 batches, ran 4 parallel forks, each independently re-reading full
`raw_text` and re-deciding both fields from scratch (not lightly editing
Groq's output) using the rules above. Patches were written to separate
files and merged/validated centrally rather than letting 4 forks write
`train.jsonl` concurrently.

**Result: 242/541 rows changed** (content-diff, case/order-insensitive).
Re-validated against `JobPosting`: 0 failures.

**New sub-patterns confirmed at this larger scale** (beyond what the
73-posting test-set batch already found):
- **"ROME savoir-faire/savoir-être boilerplate" is a specifically
  identifiable, 100%-noise sub-pattern** — France Travail postings
  frequently carry a verbatim generic-competency block (adaptability,
  listening, rigor, etc., pulled from the ROME job-classification
  taxonomy) that Groq reliably dumps wholesale into `required_skills`.
  Always safe to strip entirely.
- **Language duplication into skills fields is far more common than the
  single Inserm instance** — hit repeatedly across all 4 batches
  (Anglais/Espagnol appearing in `required_skills` on top of a proper
  `language_requirements` entry). Always stripped from the skills field.
- **Undergeneration also exists, not just overgeneration** — a handful of
  rows had `required_skills: null` despite the text being packed with
  named tools (e.g. ChatGPT/Claude/SEO/SEA, or SuccessFactors/Power
  BI/VBA/Power Query) that Groq simply missed. Worth remembering the
  training-set problem isn't purely "too much noise," a few rows were
  also "too little signal."
- **Training-outcome credentials aren't required_skills** — postings
  describing what you'll *earn during* the apprenticeship ("vous
  apprendrez... passage du CACES") got miscoded as prerequisites; these
  belong to neither field.

**Genuinely fuzzy judgment lines the 4 forks converged on, worth knowing
about rather than treating as settled fact:**
- Generic trade-domain nouns with no product name attached (électricité,
  mécanique, hydraulique, pneumatique, CVC, "conception mécanique",
  "droit social") were kept as legitimate named-domain skills for
  trade/technical roles — looser than "concrete product name," applied
  for consistency, but the line here is genuinely softer than for named
  software.
- Category acronyms with no vendor (CRM, ERP, WMS, CAO, SIRH) were kept
  in some batches and excluded in others depending on how directly they
  were framed — batch 2 explicitly leaned stricter (excluded WMS even
  under "un plus" framing) while batch 3/4 kept several. **Not fully
  consistent across the 4 batches** — if this matters for your writeup,
  worth a second consistency pass specifically on category-acronym rows.
- "Idéalement"/"intérêt fort pour"/"OU êtes motivé pour acquérir" framing
  was treated as equivalent to "un plus" (→ nice-to-have) in several
  rows — softer signal than the established "est un plus" rule, applied
  by extension/analogy rather than being a literal match to the rule.
- `Permis B` (driving license) accepted as a required_skill on ~6 rows —
  it's a credential, not a technical skill, arguably belongs in a
  different bucket the schema doesn't have.
- A few single-mention tools inferred from a duty bullet rather than a
  formal "Compétences" section (e.g. PEP, WINFRET, one CRM mention
  mid-paragraph) — reasonable but shakier evidence than an explicit
  skills-list entry.

None of these are "wrong," they're judgment calls at the edge of the
established rule, made independently by 4 parallel workers without
cross-checking each other. **If required_skills eval numbers look
inconsistent or surprising later, this is the first place to look.**

## "Intérim" contract_type schema gap — RESOLVED

`schema/posting.py`'s `_CONTRACT_SYNONYMS` had no French entry for
intérim/temp work, so any such raw string silently fell through to
`"other"`. Added `intérim`/`interim`/`intérimaire`/`interimaire`/`mission
d'intérim`/`mission d'interim`/`travail temporaire`/`temporaire` → `"cdd"`
(closest legal fit: intérim is a fixed-term arrangement via a staffing
agency). This is a schema.py change, not just a data patch — future Groq
runs and the fine-tuned model both benefit from it automatically. The
already-stored `train.jsonl` predictions needed a separate manual patch
(round 2 above) since they'd already been normalized through the old,
gapped mapping and the original raw string was lost.

## Second-pass cleanup on `data/test/test.jsonl` (6 rows changed)

Revisited the judgment calls flagged as uncertain after the 73-posting
batch, re-reading each against `test_candidates.jsonl`'s raw_text:

- `france_travail:5201009` and `france_travail:5684960`: `company:
  "ISCOD"` → null. These were the same broker-not-employer bug as the
  Sélestat CRM posting fixed by hand earlier in the session — turned out
  only that one had actually been corrected; two more slipped through in
  the 73-posting batch.
- `france_travail:212BPNY`: `company: "NGE"` → `"GUINTOLI"`. The raw_text
  header says NGE (the parent group) but the body explicitly says "Notre
  entité GUINTOLI... est à la recherche" — GUINTOLI is the actual hiring
  subsidiary.
- `france_travail:5780538`: `company: "Burger King"` → null. Text says
  "sous la supervision du franchisé" — Burger King is the brand, the
  actual franchisee/legal employer is never named. (The *other* Burger
  King posting in the test set keeps its `BKG` company value as-is —
  that one names a specific header entity, a genuinely different
  situation, not the same bug.)
- `france_travail:211WRFK` and `france_travail:211SCDM`:
  `years_experience_min: 0` → null. Text said a specific type of
  experience "n'est pas obligatoire" (not that no experience at all is
  required) — softer than "débutant accepté", and the nuance is already
  captured in `nice_to_have_skills`, so forcing `0` overstated it.

Spot-checked and confirmed already correct, no changes needed: the
agency-vs-broker `company` distinction (Adecco/Manpower/Randstad/
Synergie/Proman/Davidson/Holenek/LIP all recruit under explicit "pour
notre client"/temp-contract framing, confirming agency-as-legal-employer
was the right call), AKAMANMAN (correctly null), FINAPOLLINE/ACTEON (kept
ACTEON — body speaks in first person as ACTEON), CAF Marne (no
truncation bug, already clean), redundant-language-in-skills pattern
(searched broadly, none found beyond the Inserm case already fixed
during manual review), company/location field-swap bugs (searched
broadly, none found beyond the Petit-Quevilly one already fixed by
hand).

## Judgment calls from the 73-posting batch worth a second look

- Staffing/ESN agencies (Adecco, Manpower, Randstad, Synergie, Proman,
  Davidson, Holenek Ingénierie, LIP) were kept as `company` — French
  intérim/ESN law makes the agency the legal employer — while
  training-brokers and recruitment cabinets (ISCOD-style, Le CabRH, ID
  Search Finance, Saint Junien Recrutement, Fulfillink, IFRIA) were nulled
  out per the broker-vs-employer rule above. The line between the two
  categories was inferred from firm-name patterns, not certain in every
  case.
- ~~Ambiguous company identity~~ — **[RESOLVED, see "Second-pass cleanup"
  below]** NGE was corrected to GUINTOLI and the ambiguous Burger King
  row was nulled; FINAPOLLINE/ACTEON, FONDAT REGION OUEST LIGUE CANCER,
  and AKAMANMAN were re-checked and confirmed already correct.
- ~~`years_experience_min = 0` applied to "n'est pas obligatoire"~~ —
  **[RESOLVED]** re-read as too soft a signal to quantify as 0; the two
  affected rows were corrected to null (see "Second-pass cleanup").
- ~~`contract_type` for "intérim" standardized to `"cdd"` as a
  workaround~~ — **[RESOLVED]** this is now an actual schema fix, not a
  workaround — see "Intérim contract_type schema gap — RESOLVED" below.
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
- ~~Duration-range→null more pervasive than the 9/10 already patched~~ —
  **[RESOLVED]** broader re-scan done, see "train.jsonl fixes" round 2
  above (found the wider regex mostly over-triggers on unrelated "years
  of X" and age-bracket phrasing — only 6 of 27 candidates were real).
- ~~Company/location field swaps distinct from the broker pattern~~ —
  **[CHECKED, not found]** re-searched broadly in the second-pass
  cleanup; no additional instances beyond the Petit-Quevilly one already
  fixed by hand, and the CAF Marne posting turned out to be clean (not a
  truncation bug after all — matches the raw text as-is).
- ~~Redundant language entries duplicated into skills fields~~ —
  **[CHECKED, not found]** re-searched broadly; no instances beyond the
  Inserm case already fixed during manual review.
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

## Open items / things to watch

- **Test set is finished and cleaned up (100/100, two review passes).
  `data/test/test.jsonl`, `data/train/train.jsonl`, and
  `schema/posting.py` all have uncommitted/committed fixes as of this
  note — check `git log`/`git status` for current state before assuming
  anything below is still pending.**
- `required_skills` overgeneration by Groq is the single biggest known
  weakness in the baseline (F1 0.565) — this is a real finding about the
  model being benchmarked, not a labelling problem, but worth remembering
  when interpreting the fine-tuned model's score on this field too (a
  fine-tuned model beating this specific weakness would be a genuinely
  meaningful result to call out in the writeup).
- `france_travail:212DXSR` in train.jsonl — `duration_months` still null,
  manual call needed if you decide the BTS-length framing ("18/24 mois -
  Bac+2") should count as a stated contract duration after all. Same
  open question applies to the 2 similar diploma/formation-length rows
  found and skipped in the round-2 duration sweep.
- `eval/score.py`'s `_HAS_DIGIT_RE` salary_range heuristic isn't
  foolproof — a posting stating payment terms like "...sur 14 mois" (a
  payment-schedule detail) would slip past it and be treated as a real
  salary figure since it contains a digit. Not yet hit in practice as a
  scoring error, just a known blind spot.
- One-off data artifact, not worth chasing: `france_travail:212NRDB`'s
  raw_text has a stray standalone "user" line (looks like a leaked
  chat-role marker from ingest) — confirmed it's the only occurrence
  across all 642 labelled postings, so not a systemic pipeline bug.
- Next real milestone per the README: `notebooks/kaggle_train.ipynb` (the
  QLoRA fine-tune itself) — everything so far exists to feed that step.
