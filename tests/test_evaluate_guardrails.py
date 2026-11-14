import json

import pytest

from scripts.evaluation.evaluate_guardrails import (
    build_summary,
    default_cases,
    evaluate_cases,
    main,
    validate_cases,
)


def test_default_guardrail_scenarios_all_pass_offline():
    rows = evaluate_cases(default_cases())
    summary = build_summary(rows)

    assert summary["total_cases"] == 9
    assert summary["overall"]["passed"] == 9
    assert summary["false_positives"]["count"] == 0
    assert summary["false_negatives"]["count"] == 0
    assert summary["execution_errors"]["count"] == 0


def test_duplicate_case_ids_are_rejected():
    cases = default_cases()
    cases[1]["case_id"] = cases[0]["case_id"]

    with pytest.raises(ValueError, match="Duplicate case_id"):
        validate_cases(cases)


def test_main_writes_jsonl_and_summary(tmp_path):
    exit_code = main(["--output-dir", str(tmp_path), "--tag", "test"])

    assert exit_code == 0
    detail_path = tmp_path / "guardrail_details_test.jsonl"
    summary_path = tmp_path / "guardrail_summary_test.json"
    assert len(detail_path.read_text(encoding="utf-8").splitlines()) == 9
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["overall"]["pass_rate"] == 1.0
    assert summary["meta"]["cases_source"] == "built-in"


def test_custom_jsonl_cases_are_supported(tmp_path):
    case = default_cases()[0]
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")

    exit_code = main(
        [
            "--cases",
            str(cases_path),
            "--output-dir",
            str(tmp_path / "reports"),
        ]
    )

    assert exit_code == 0
    summary = json.loads(
        (tmp_path / "reports" / "guardrail_summary_default.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["total_cases"] == 1
    assert summary["meta"]["cases_source"] == str(cases_path)
