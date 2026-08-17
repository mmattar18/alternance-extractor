"""Smoke tests for schema/posting.py and schema/dedupe.py.

Plain asserts, no pytest dependency. Run with: python schema/test_posting.py
"""
from pydantic import ValidationError

from posting import JobPosting, parse_llm_json
from dedupe import assert_no_test_leakage, text_hash


def test_zero_is_not_none_for_experience():
    # Explicit "no experience required" -> 0, a real value.
    stated_zero = JobPosting(title="Data Analyst Alternance", years_experience_min=0)
    assert stated_zero.years_experience_min == 0

    # Field never mentioned -> None, not 0.
    not_mentioned = JobPosting(title="Data Analyst Alternance")
    assert not_mentioned.years_experience_min is None


def test_sentinel_strings_become_none():
    p = JobPosting(
        title="ML Engineer",
        company="N/A",
        location="  ",
        start_date="not specified",
        salary_range="non précisé",
    )
    assert p.company is None
    assert p.location is None
    assert p.start_date is None
    assert p.salary_range is None


def test_contract_type_synonyms_normalize():
    assert JobPosting(title="x", contract_type="Apprenticeship").contract_type == "alternance"
    assert JobPosting(title="x", contract_type="Internship").contract_type == "stage"
    assert JobPosting(title="x", contract_type="Permanent").contract_type == "cdi"
    assert JobPosting(title="x", contract_type="quantum consultant").contract_type == "other"


def test_remote_policy_unmapped_falls_back_to_none():
    assert JobPosting(title="x", remote_policy="télétravail").remote_policy == "remote"
    assert JobPosting(title="x", remote_policy="on the moon").remote_policy is None


def test_alternance_rhythm_forced_none_when_not_alternance():
    p = JobPosting(
        title="Stage Data Science",
        contract_type="stage",
        alternance_rhythm="3 weeks company / 1 week school",
    )
    assert p.alternance_rhythm is None

    p2 = JobPosting(
        title="Alternance Data Science",
        contract_type="alternance",
        alternance_rhythm="3 weeks company / 1 week school",
    )
    assert p2.alternance_rhythm == "3 weeks company / 1 week school"


def test_empty_list_becomes_none_not_empty_list():
    p = JobPosting(title="x", required_skills=[], nice_to_have_skills=["", "  "])
    assert p.required_skills is None
    assert p.nice_to_have_skills is None


def test_skill_list_dedupes_case_insensitively():
    p = JobPosting(title="x", required_skills=["Python", "python", " SQL "])
    assert p.required_skills == ["Python", "SQL"]


def test_title_is_required():
    try:
        JobPosting()
        raise AssertionError("expected ValidationError for missing title")
    except ValidationError:
        pass


def test_extra_fields_rejected():
    try:
        JobPosting(title="x", made_up_field="oops")
        raise AssertionError("expected ValidationError for extra field")
    except ValidationError:
        pass


def test_parse_llm_json_valid():
    raw = '```json\n{"title": "Alternance IA", "years_experience_min": 0}\n```'
    posting, err = parse_llm_json(raw)
    assert err is None
    assert posting.title == "Alternance IA"
    assert posting.years_experience_min == 0


def test_parse_llm_json_invalid():
    posting, err = parse_llm_json("this is not json")
    assert posting is None
    assert err.startswith("invalid_json")


def test_parse_llm_json_schema_violation():
    posting, err = parse_llm_json('{"title": "x", "years_experience_min": -1}')
    assert posting is None
    assert err.startswith("schema_violation")


def test_dedupe_hash_is_whitespace_and_case_insensitive():
    a = text_hash("Data   Scientist  Alternance")
    b = text_hash("data scientist alternance")
    assert a == b


def test_leakage_detection_raises_on_id_overlap():
    try:
        assert_no_test_leakage(
            train_ids=["a", "b"], train_hashes=["h1"],
            test_ids=["b"], test_hashes=["h2"],
        )
        raise AssertionError("expected ValueError for id overlap")
    except ValueError:
        pass


def test_leakage_detection_raises_on_hash_overlap():
    try:
        assert_no_test_leakage(
            train_ids=["a"], train_hashes=["h1"],
            test_ids=["z"], test_hashes=["h1"],
        )
        raise AssertionError("expected ValueError for hash overlap")
    except ValueError:
        pass


def test_leakage_detection_passes_when_clean():
    assert_no_test_leakage(
        train_ids=["a", "b"], train_hashes=["h1", "h2"],
        test_ids=["c"], test_hashes=["h3"],
    )


if __name__ == "__main__":
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
