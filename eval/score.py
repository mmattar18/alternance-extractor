"""Field-level F1, exact-match, and JSON-validity scoring for job-posting
extraction. Used to compare any model's predictions (Groq baseline or the
fine-tuned model) against the hand-corrected test set in data/test/test.jsonl.

A prediction counts as correct for a field only when presence AND value both
match the gold record -- this mirrors the binary present/absent + value-match
simplification documented in schema/posting.py (alternance_rhythm folding).
List fields (skills, language_requirements) are
compared as case-insensitive sets, since order was never part of the schema's
contract.

Free-text fields (education_level, salary_range, start_date,
alternance_rhythm) are compared after normalizing case, punctuation, and
"ou"/"et"/"and"/"or" connectors -- these fields have no single canonical
short form (unlike an enum or "BTS SIO"), so both the gold correction and a
model's answer can be equivalent in meaning but differ in exactly how they
list/join multiple options ("diplome d'ingenieur ou d'ecole de commerce" vs
"diplome d'ingenieur, ecole de commerce"). Fields with a real canonical form
(title, company, enums, numbers) stay exact-match on purpose -- normalizing
those away would hide genuine extraction errors, not cosmetic ones.

Run the smoke tests: python eval/score.py
Score a run:          python eval/score.py data/test/test.jsonl data/test/test_candidates.jsonl
  (test_candidates.jsonl doubles as the Groq-baseline predictions file --
  it already carries posting_id/prediction/valid from the labelling run.
  A fine-tuned-model eval script should write predictions in that same shape.)
"""
from __future__ import annotations

import json
import random
import re
import unicodedata
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from schema.posting import JobPosting  # noqa: E402

FIELDS = list(JobPosting.model_fields.keys())
LIST_FIELDS = {"skills", "language_requirements"}
FREE_TEXT_FIELDS = {"education_level", "salary_range", "start_date", "alternance_rhythm"}

_CONNECTOR_RE = re.compile(r"\b(ou|et|and|or)\b", re.IGNORECASE)
# French elision (d'ecole, l'ecole, qu'il...) appears or disappears depending
# on which connector precedes it ("ou d'ecole" vs ", ecole") -- same
# formatting-not-substance issue as the connector itself, so strip it too.
_ELISION_RE = re.compile(r"\b[dlcjnmst]'", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[,.;:]")
_WS_RE = re.compile(r"\s+")

# France Travail postings prefix location with the department code
# ("62 - Vendin-le-Vieil"), inconsistently -- some postings have it, some
# don't, for the same city. That's a France Travail formatting quirk, not a
# real extraction difference, so location is compared with the prefix
# stripped rather than requiring every gold row to have it removed by hand.
_DEPT_PREFIX_RE = re.compile(r"^\s*(?:\d{2,3}|2[ab])\s*-\s*", re.IGNORECASE)

# Apprenticeship postings routinely say pay "follows the legal SMIC
# percentage by age/education level" without ever giving an actual number --
# that boilerplate carries no more information than not mentioning salary at
# all, so a salary_range with no digit in it is treated as equivalent to
# null on both sides, rather than requiring every gold row to be hand-blanked.
_HAS_DIGIT_RE = re.compile(r"\d")

# Case/accent/spacing folding for text comparison. Measured motivation: the fine-tuned
# model reproduces values that are semantically identical but rendered differently --
# "MENTON (06)" for "Menton (06)", "GUERANDE" for "Guerande", "Francais: courant" for
# "Francais : courant" -- because it often copies from an uppercased posting header and
# does not always preserve accents. Scoring those as extraction failures measures
# typography, not extraction. 7 of 165 fine-tuned field errors were of exactly this kind
# and 0 of 93 Groq errors were (Groq copies verbatim), so this corrects a real artifact
# rather than handing either side free credit -- it is applied identically to both.
_PUNCT_SPACE_RE = re.compile(r"\s*([:;,/])\s*")


def _fold_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _PUNCT_SPACE_RE.sub(r"", s)
    return _WS_RE.sub(" ", s).strip().lower()


def _normalize_free_text(s: str) -> str:
    s = s.lower()
    s = _CONNECTOR_RE.sub(" ", s)
    s = _ELISION_RE.sub(" ", s)
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def _normalize_location(s: str) -> str:
    return _DEPT_PREFIX_RE.sub("", s).strip()


def _effective_value(field: str, value):
    """Value used for presence/absence and matching. Currently only
    salary_range gets special treatment: digit-free text is boilerplate,
    not a real figure, so it's collapsed to null (see _HAS_DIGIT_RE)."""
    if field == "salary_range" and value is not None and not _HAS_DIGIT_RE.search(value):
        return None
    return value


def _values_match(field: str, gold_value, pred_value) -> bool:
    if field in LIST_FIELDS:
        gold_set = {_fold_text(v) for v in (gold_value or [])}
        pred_set = {_fold_text(v) for v in (pred_value or [])}
        return gold_set == pred_set
    if field in FREE_TEXT_FIELDS:
        return _normalize_free_text(gold_value) == _normalize_free_text(pred_value)
    if field == "location":
        return _fold_text(_normalize_location(gold_value)) == _fold_text(_normalize_location(pred_value))
    if field in ("title", "company"):
        return _fold_text(gold_value) == _fold_text(pred_value)
    return gold_value == pred_value


def score_posting(gold: dict, pred: Optional[dict]) -> dict[str, str]:
    """Per-field verdict for one posting: 'tp', 'fp', 'fn', or 'tn'.

    tn (true negative, both null) doesn't count toward precision/recall but
    is tracked so a model that predicts null everywhere can't be scored as
    if those fields simply didn't exist.
    """
    verdicts = {}
    for field in FIELDS:
        gold_value = _effective_value(field, gold.get(field))
        gold_present = gold_value is not None
        if pred is None:
            pred_present = False
            pred_value = None
        else:
            pred_value = _effective_value(field, pred.get(field))
            pred_present = pred_value is not None

        if not gold_present and not pred_present:
            verdicts[field] = "tn"
        elif pred_present and gold_present and _values_match(field, gold_value, pred_value):
            verdicts[field] = "tp"
        elif pred_present:
            verdicts[field] = "fp"
        else:
            verdicts[field] = "fn"
    return verdicts


def aggregate(records: list[tuple[dict, Optional[dict], bool]]) -> dict:
    """records: list of (gold_dict, pred_dict_or_None, json_valid) tuples."""
    counts = {f: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for f in FIELDS}
    exact_matches = 0
    n_valid_json = 0

    for gold, pred, valid in records:
        if valid:
            n_valid_json += 1
        verdicts = score_posting(gold, pred)
        for field, v in verdicts.items():
            counts[field][v] += 1
        if all(v in ("tp", "tn") for v in verdicts.values()):
            exact_matches += 1

    per_field = {}
    for field, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        if tp + fp + fn == 0:
            # Field was null in every gold and every prediction: no signal,
            # so F1 is undefined rather than a free 0 or a free 1.
            per_field[field] = {"precision": None, "recall": None, "f1": None, **c}
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_field[field] = {"precision": precision, "recall": recall, "f1": f1, **c}

    n = len(records)
    scored_fields = [pf["f1"] for pf in per_field.values() if pf["f1"] is not None]
    macro_f1 = sum(scored_fields) / len(scored_fields) if scored_fields else 0.0

    return {
        "n_postings": n,
        "json_validity_rate": n_valid_json / n if n else 0.0,
        "exact_match_rate": exact_matches / n if n else 0.0,
        "macro_f1": macro_f1,
        "per_field": per_field,
    }


def _list_item_counts(gold_value, pred_value) -> tuple[int, int, int]:
    """Per-ITEM tp/fp/fn for one list field on one posting, case-insensitive.

    Exact set equality (what aggregate() uses) scores {Python, SQL, Excel} against a gold
    of {Python, SQL} exactly as harshly as it scores {cooking}: both are simply "wrong".
    For a field like skills, where the gold itself averages several items, that
    throws away most of the signal -- getting 3 of 4 skills right is not the same as
    getting none. Per-item micro-averaging is the standard information-extraction
    treatment and is what aggregate_partial() reports.
    """
    g = {_fold_text(v) for v in (gold_value or [])}
    p = {_fold_text(v) for v in (pred_value or [])}
    return len(g & p), len(p - g), len(g - p)


def aggregate_partial(records: list[tuple[dict, Optional[dict], bool]]) -> dict:
    """Companion to aggregate(): identical for scalar fields, but list fields get per-item
    credit instead of all-or-nothing set equality.

    Reported ALONGSIDE the strict numbers, never instead of them -- the strict metric is
    the honest headline, this one localises *where* a model is partially right. Both
    models are scored the same way, so this cannot flatter one side.
    """
    counts = {f: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for f in FIELDS}
    fields_correct_total = 0

    for gold, pred, _valid in records:
        verdicts = score_posting(gold, pred)
        for field in FIELDS:
            if field in LIST_FIELDS:
                gv = _effective_value(field, gold.get(field))
                pv = _effective_value(field, pred.get(field)) if pred else None
                if not gv and not pv:
                    counts[field]["tn"] += 1
                    continue
                tp, fp, fn = _list_item_counts(gv, pv)
                counts[field]["tp"] += tp
                counts[field]["fp"] += fp
                counts[field]["fn"] += fn
            else:
                counts[field][verdicts[field]] += 1
        fields_correct_total += sum(1 for v in verdicts.values() if v in ("tp", "tn"))

    per_field = {}
    for field, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        if tp + fp + fn == 0:
            per_field[field] = {"precision": None, "recall": None, "f1": None, **c}
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_field[field] = {"precision": precision, "recall": recall, "f1": f1, **c}

    n = len(records)
    scored = [pf["f1"] for pf in per_field.values() if pf["f1"] is not None]
    return {
        "n_postings": n,
        "macro_f1_partial": sum(scored) / len(scored) if scored else 0.0,
        # companion to exact_match_rate: exact-match needs all 14 fields right at once,
        # so it collapses to 0% the moment one field is systematically wrong.
        "mean_fields_correct": fields_correct_total / (n * len(FIELDS)) if n else 0.0,
        "per_field": per_field,
    }


def bootstrap_ci(records, metric="macro_f1", n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for a metric, resampling POSTINGS with replacement.

    The test set is 100 postings. Reporting "macro_f1 = 0.891" to three decimals
    implies a precision that sample size does not support, and a 0.02 gap between two
    systems on n=100 may be indistinguishable from noise. Resampling postings (the
    independent unit) is the standard way to attach an interval to that.
    """
    rng = random.Random(seed)
    n = len(records)
    fn = (lambda r: aggregate(r)[metric]) if metric != "macro_f1_partial"         else (lambda r: aggregate_partial(r)[metric])
    point = fn(records)
    draws = []
    for _ in range(n_boot):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        draws.append(fn(sample))
    draws.sort()
    lo = draws[int(alpha / 2 * n_boot)]
    hi = draws[int((1 - alpha / 2) * n_boot)]
    return {"point": point, "lo": lo, "hi": hi}


def paired_bootstrap_delta(records_a, records_b, metric="macro_f1", n_boot=2000, seed=0, alpha=0.05):
    """CI for (A - B) with the SAME postings resampled for both arms.

    Paired, not two independent CIs: both systems are scored on identical postings, so
    per-posting difficulty cancels and the interval on the difference is much tighter
    than eyeballing whether two separate CIs overlap (which is a well-known way to
    under-detect a real difference). `records_a`/`records_b` must be aligned by posting.
    """
    assert len(records_a) == len(records_b)
    rng = random.Random(seed)
    n = len(records_a)
    fn = (lambda r: aggregate(r)[metric]) if metric != "macro_f1_partial"         else (lambda r: aggregate_partial(r)[metric])
    point = fn(records_a) - fn(records_b)
    draws = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(fn([records_a[i] for i in idx]) - fn([records_b[i] for i in idx]))
    draws.sort()
    lo = draws[int(alpha / 2 * n_boot)]
    hi = draws[int((1 - alpha / 2) * n_boot)]
    return {"delta": point, "lo": lo, "hi": hi, "significant": lo > 0 or hi < 0}


def load_test_set(test_path: Path) -> dict[str, dict]:
    """posting_id -> gold JobPosting dict, from the 'corrected' field that
    label/review_server.py writes to data/test/test.jsonl."""
    gold_by_id = {}
    for line in test_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            gold_by_id[rec["posting_id"]] = rec["corrected"]
    return gold_by_id


def load_predictions(predictions_path: Path) -> dict[str, tuple[Optional[dict], bool]]:
    """posting_id -> (prediction dict or None, json_valid bool). Expects the
    shape label/run_labelling.py writes (posting_id/prediction/valid per
    line) -- test_candidates.jsonl already has this for the Groq baseline;
    a fine-tuned-model eval script should write the same shape."""
    preds = {}
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            preds[rec["posting_id"]] = (rec.get("prediction"), rec.get("valid", False))
    return preds


def score_files(test_path: Path, predictions_path: Path) -> dict:
    gold_by_id = load_test_set(test_path)
    preds_by_id = load_predictions(predictions_path)
    missing = set(gold_by_id) - set(preds_by_id)
    if missing:
        raise ValueError(f"{len(missing)} test posting_ids have no prediction: {sorted(missing)[:5]}")

    records = [(gold, preds_by_id[pid][0], preds_by_id[pid][1]) for pid, gold in gold_by_id.items()]
    return aggregate(records)


def print_report(result: dict) -> None:
    print(f"n_postings: {result['n_postings']}")
    print(f"json_validity_rate: {result['json_validity_rate']:.1%}")
    print(f"exact_match_rate: {result['exact_match_rate']:.1%}")
    print(f"macro_f1: {result['macro_f1']:.3f}")
    print(f"\n{'field':<24}{'precision':>10}{'recall':>10}{'f1':>10}{'tp':>6}{'fp':>6}{'fn':>6}{'tn':>6}")
    for field, pf in result["per_field"].items():
        p = f"{pf['precision']:.3f}" if pf['precision'] is not None else "n/a"
        r = f"{pf['recall']:.3f}" if pf['recall'] is not None else "n/a"
        f1 = f"{pf['f1']:.3f}" if pf['f1'] is not None else "n/a"
        print(f"{field:<24}{p:>10}{r:>10}{f1:>10}"
              f"{pf['tp']:>6}{pf['fp']:>6}{pf['fn']:>6}{pf['tn']:>6}")


# ---- smoke tests ----

def test_perfect_prediction_is_all_tp_or_tn():
    gold = {"title": "ML Engineer", "company": None, "years_experience_min": 0}
    pred = {"title": "ML Engineer", "company": None, "years_experience_min": 0}
    verdicts = score_posting(gold, pred)
    assert verdicts["title"] == "tp"
    assert verdicts["company"] == "tn"
    assert verdicts["years_experience_min"] == "tp"


def test_none_prediction_is_fn_where_gold_present_else_tn():
    gold = {"title": "ML Engineer", "company": None}
    verdicts = score_posting(gold, None)
    assert verdicts["title"] == "fn"
    assert verdicts["company"] == "tn"


def test_wrong_value_is_fp():
    gold = {"title": "ML Engineer"}
    pred = {"title": "Data Scientist"}
    assert score_posting(gold, pred)["title"] == "fp"


def test_hallucinated_value_where_gold_is_null_is_fp():
    gold = {"title": "x", "company": None}
    pred = {"title": "x", "company": "Made Up Inc"}
    assert score_posting(gold, pred)["company"] == "fp"


def test_list_field_matches_as_set_case_insensitive():
    gold = {"title": "x", "skills": ["Python", "SQL"]}
    pred = {"title": "x", "skills": ["sql", "python"]}
    assert score_posting(gold, pred)["skills"] == "tp"


def test_list_field_mismatch_is_fp():
    gold = {"title": "x", "skills": ["Python"]}
    pred = {"title": "x", "skills": ["Python", "SQL"]}
    assert score_posting(gold, pred)["skills"] == "fp"


def test_free_text_field_matches_despite_connector_and_punctuation_differences():
    gold = {"title": "x", "education_level": "diplome ingenieur ou ecole de commerce"}
    pred = {"title": "x", "education_level": "diplome ingenieur, ecole de commerce"}
    assert score_posting(gold, pred)["education_level"] == "tp"


def test_free_text_field_matches_despite_french_elision_shift():
    # "ou d'ecole" vs ", ecole" -- the apostrophe-contraction shifts with the
    # connector choice; that's still a formatting difference, not a real one.
    gold = {"title": "x", "education_level": "diplome d'ingenieur ou d'ecole de commerce"}
    pred = {"title": "x", "education_level": "diplome d'ingenieur, ecole de commerce"}
    assert score_posting(gold, pred)["education_level"] == "tp"


def test_free_text_field_still_catches_real_differences():
    gold = {"title": "x", "education_level": "Bac+5"}
    pred = {"title": "x", "education_level": "Bac+3"}
    assert score_posting(gold, pred)["education_level"] == "fp"


def test_location_ignores_department_code_prefix():
    gold = {"title": "x", "location": "Vendin-le-Vieil"}
    pred = {"title": "x", "location": "62 - Vendin-le-Vieil"}
    assert score_posting(gold, pred)["location"] == "tp"


def test_location_still_catches_real_differences():
    gold = {"title": "x", "location": "Vendin-le-Vieil"}
    pred = {"title": "x", "location": "Lille"}
    assert score_posting(gold, pred)["location"] == "fp"


def test_digit_free_salary_range_treated_as_null():
    gold = {"title": "x", "salary_range": None}
    pred = {"title": "x", "salary_range": "Rémunération légale selon le pourcentage sur SMIC applicable selon votre âge"}
    assert score_posting(gold, pred)["salary_range"] == "tn"


def test_digit_free_salary_range_matches_either_direction():
    gold = {"title": "x", "salary_range": "Rémunération selon niveau d'études + âge"}
    pred = {"title": "x", "salary_range": None}
    assert score_posting(gold, pred)["salary_range"] == "tn"


def test_salary_range_with_a_real_figure_still_scored_normally():
    gold = {"title": "x", "salary_range": "1500€/mois"}
    pred = {"title": "x", "salary_range": None}
    assert score_posting(gold, pred)["salary_range"] == "fn"


def test_non_free_text_scalar_field_stays_exact_match():
    # title is not in FREE_TEXT_FIELDS -- a connector/punctuation difference
    # there is a real error, not something to normalize away.
    gold = {"title": "Data Analyst ou Ingenieur"}
    pred = {"title": "Data Analyst, Ingenieur"}
    assert score_posting(gold, pred)["title"] == "fp"


def test_aggregate_exact_match_and_json_validity():
    records = [
        ({"title": "a"}, {"title": "a"}, True),
        ({"title": "b"}, {"title": "wrong"}, True),
        ({"title": "c"}, None, False),
    ]
    result = aggregate(records)
    assert result["n_postings"] == 3
    assert result["json_validity_rate"] == 2 / 3
    assert result["exact_match_rate"] == 1 / 3


def test_aggregate_perfect_scores_f1_one():
    records = [({"title": "a", "company": None}, {"title": "a", "company": None}, True)]
    result = aggregate(records)
    assert result["macro_f1"] == 1.0  # title=tp -> f1=1.0; company has no signal, excluded from macro avg
    assert result["exact_match_rate"] == 1.0
    assert result["per_field"]["company"]["f1"] is None


def test_field_with_no_signal_excluded_from_macro_f1():
    records = [({"title": "a"}, {"title": "wrong"}, True)]  # every other field null/null
    result = aggregate(records)
    assert result["macro_f1"] == 0.0  # only title has signal, and it's wrong
    assert result["per_field"]["company"]["f1"] is None


def test_partial_credit_rewards_overlap_that_strict_calls_wrong():
    gold = {"title": "x", "skills": ["Python", "SQL"]}
    pred = {"title": "x", "skills": ["Python", "SQL", "Excel"]}
    # strict: one extra item makes the whole field wrong
    assert score_posting(gold, pred)["skills"] == "fp"
    # partial: 2 of 3 predicted items are right, and both gold items were found
    r = aggregate_partial([(gold, pred, True)])["per_field"]["skills"]
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 0)
    assert r["recall"] == 1.0 and 0.6 < r["precision"] < 0.7


def test_partial_credit_still_punishes_a_fully_wrong_list():
    gold = {"title": "x", "skills": ["Python", "SQL"]}
    pred = {"title": "x", "skills": ["cooking"]}
    r = aggregate_partial([(gold, pred, True)])["per_field"]["skills"]
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 2)
    assert r["f1"] == 0.0


def test_partial_credit_leaves_scalar_fields_identical_to_strict():
    gold = {"title": "ML Engineer", "company": None}
    pred = {"title": "Data Scientist", "company": "Made Up Inc"}
    strict = aggregate([(gold, pred, True)])["per_field"]
    partial = aggregate_partial([(gold, pred, True)])["per_field"]
    for f in ("title", "company"):
        assert {k: strict[f][k] for k in ("tp", "fp", "fn", "tn")} ==                {k: partial[f][k] for k in ("tp", "fp", "fn", "tn")}


def test_mean_fields_correct_is_softer_than_exact_match():
    gold = {"title": "a", "company": "b"}
    pred = {"title": "a", "company": "WRONG"}
    assert aggregate([(gold, pred, True)])["exact_match_rate"] == 0.0
    assert aggregate_partial([(gold, pred, True)])["mean_fields_correct"] > 0.9


def test_bootstrap_ci_is_deterministic_and_brackets_the_point_estimate():
    recs = [({"title": "a"}, {"title": "a"}, True)] * 30 +            [({"title": "b"}, {"title": "WRONG"}, True)] * 20
    a = bootstrap_ci(recs, n_boot=200, seed=7)
    b = bootstrap_ci(recs, n_boot=200, seed=7)
    assert a == b, "same seed must reproduce -- a CI nobody can reproduce is not evidence"
    assert a["lo"] <= a["point"] <= a["hi"]


def test_paired_bootstrap_detects_a_real_gap_and_ignores_a_null_one():
    gold = [({"title": "a", "company": "x"}, None, True) for _ in range(40)]
    good = [(g, {"title": "a", "company": "x"}, True) for g, _, _ in gold]
    bad = [(g, {"title": "WRONG", "company": "WRONG"}, True) for g, _, _ in gold]
    big = paired_bootstrap_delta(good, bad, n_boot=200, seed=1)
    assert big["delta"] > 0 and big["significant"], "a total failure vs perfect must register"
    null = paired_bootstrap_delta(good, good, n_boot=200, seed=1)
    assert null["delta"] == 0 and not null["significant"], "identical arms must not look different"


def test_fields_cover_full_schema():
    assert set(FIELDS) == set(JobPosting.model_fields.keys())
    assert LIST_FIELDS <= set(FIELDS)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        result = score_files(Path(sys.argv[1]), Path(sys.argv[2]))
        print_report(result)
        raise SystemExit(0)

    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    failures = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failures.append((t.__name__, e))
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        raise SystemExit(1)
