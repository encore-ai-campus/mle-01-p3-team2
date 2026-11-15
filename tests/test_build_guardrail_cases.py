import json
from pathlib import Path

from scripts.evaluation.build_guardrail_cases import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_GOLDEN_PATH,
    generate_cases,
    main,
    read_jsonl,
)
from scripts.evaluation.evaluate_guardrails import build_summary, evaluate_cases


def _source_rows():
    return read_jsonl(DEFAULT_GOLDEN_PATH), read_jsonl(DEFAULT_CHUNKS_PATH)


def test_default_generation_is_reproducible_and_has_expected_mix():
    golden_rows, chunk_rows = _source_rows()

    first = generate_cases(golden_rows, chunk_rows, seed=42)
    second = generate_cases(golden_rows, chunk_rows, seed=42)

    assert first == second
    assert len(first) == 30
    assert len({case["case_id"] for case in first}) == 30
    assert {case["scenario_type"] for case in first} == {
        "normal",
        "pesticide_warning",
        "irrelevant",
        "missing_evidence",
        "pesticide_unsafe",
        "unsupported_answer",
        "malformed_evidence",
    }


def test_generated_cases_match_the_current_guardrail_contract_offline():
    golden_rows, chunk_rows = _source_rows()
    cases = generate_cases(golden_rows, chunk_rows)

    summary = build_summary(evaluate_cases(cases))

    assert summary["total_cases"] == 30
    assert summary["overall"]["passed"] == 30
    assert summary["execution_errors"]["count"] == 0
    assert summary["false_positives"]["count"] == 0
    assert summary["false_negatives"]["count"] == 0


def test_main_writes_final_set_and_manifest(tmp_path: Path):
    output_path = tmp_path / "guardrail_cases_final.jsonl"
    manifest_path = tmp_path / "guardrail_cases_final_summary.json"

    exit_code = main(
        [
            "--output",
            str(output_path),
            "--manifest",
            str(manifest_path),
            "--seed",
            "42",
        ]
    )

    assert exit_code == 0
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 30
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["total_cases"] == 30
    assert manifest["seed"] == 42
    assert manifest["review_required"] is True
    assert len(manifest["output_sha256"]) == 64
