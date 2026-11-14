"""Tests for the multi-target guardrail scenario runner.

The NeMo rails target is not exercised here because booting ``LLMRails``
loads heavy optional dependencies; ``docs/guardrail-eval-set-v1.md`` records
the recorded before/after runs for that tier instead.
"""

from __future__ import annotations

from pathlib import Path

from scripts.evaluation.evaluate_guardrail_scenarios import (
    _case_input,
    _run_scope_gate,
    build_summary,
    evaluate_cases,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DISEASE_EVIDENCE = {
    "source": "vector",
    "text": "마늘 잎마름병은 4~5월 다습한 환경에서 심해진다.",
    "evidence_ids": ["EVD_A1"],
    "source_ids": ["SRC_A1"],
    "chunk_ids": ["CHK_A1"],
    "metadata": {"evidence_type": "disease_record"},
}


def _gate_case(case_id: str, question: str, decision: str) -> dict:
    return {
        "case_id": case_id,
        "tier": "G_input_rail",
        "target": "router_scope_gate",
        "request_class": "normal" if decision == "ALLOW" else "irrelevant",
        "branch_id": "IR-TEST",
        "scenario_type": "test",
        "description": "test",
        "expected_allowed": decision == "ALLOW",
        "expected_decision": decision,
        "discriminating": True,
        "request": {"question": question},
    }


def _pipeline_case(case_id: str, *, evidence: list, decision: str) -> dict:
    return {
        "case_id": case_id,
        "tier": "A_structure",
        "target": "guardrail_pipeline",
        "request_class": "normal" if decision == "ALLOW" else "quality_risk",
        "branch_id": "EV-TEST",
        "scenario_type": "test",
        "description": "test",
        "expected_allowed": decision == "ALLOW",
        "expected_decision": decision,
        "discriminating": True,
        "request": {
            "question": "마늘 잎마름병은 언제 심해지나요?",
            "answer": DISEASE_EVIDENCE["text"],
            "evidence": evidence,
            "expected_facts": ["4~5월", "다습한 환경"],
        },
    }


def test_scope_gate_maps_reject_to_block() -> None:
    outcome = _run_scope_gate({"question": "비트코인 가격을 예측해 주세요."})

    assert outcome.allowed is False
    assert outcome.decision == "BLOCK"
    assert outcome.reason
    assert outcome.details["route"] == "REJECT"


def test_scope_gate_maps_in_scope_to_allow() -> None:
    outcome = _run_scope_gate({"question": "마늘 재배에서 잎마름병 방제 방법은?"})

    assert outcome.allowed is True
    assert outcome.decision == "ALLOW"
    assert outcome.details["route"] is None


def test_scope_gate_tolerates_a_non_string_question() -> None:
    outcome = _run_scope_gate({"question": None})

    assert outcome.decision == "BLOCK"


def test_evaluate_cases_dispatches_per_target() -> None:
    cases = [
        _gate_case("T1", "비트코인 가격을 예측해 주세요.", "BLOCK"),
        _gate_case("T2", "마늘 재배에서 잎마름병 방제 방법은?", "ALLOW"),
        _pipeline_case("T3", evidence=[DISEASE_EVIDENCE], decision="ALLOW"),
        _pipeline_case("T4", evidence=[], decision="BLOCK"),
    ]

    rows = evaluate_cases(cases)

    assert [row["case_id"] for row in rows] == ["T1", "T2", "T3", "T4"]
    assert all(row["passed"] for row in rows)
    assert all(row["error_type"] is None for row in rows)
    assert {row["target"] for row in rows} == {
        "router_scope_gate",
        "guardrail_pipeline",
    }
    # Every row carries the four columns the report requires.
    for row in rows:
        assert row["input"]
        assert row["expected_decision"]
        assert row["actual_decision"]
        assert isinstance(row["passed"], bool)


def test_evaluate_cases_records_a_mismatch_without_raising() -> None:
    wrong = _pipeline_case("T5", evidence=[], decision="ALLOW")

    rows = evaluate_cases([wrong])

    assert rows[0]["passed"] is False
    assert rows[0]["actual_decision"] == "BLOCK"
    assert rows[0]["error_type"] is None


def test_summary_separates_false_positives_from_false_negatives() -> None:
    # Expected ALLOW but blocked -> false positive (정상을 막음).
    over_block = _pipeline_case("FP1", evidence=[], decision="ALLOW")
    # Expected BLOCK but allowed -> false negative (위험을 통과).
    under_block = _gate_case("FN1", "마늘 재배에서 잎마름병 방제 방법은?", "BLOCK")

    rows = evaluate_cases([over_block, under_block])
    summary = build_summary(
        rows, cases_path=Path("test.jsonl"), rails_runner=None
    )

    assert summary["false_positives"]["case_ids"] == ["FP1"]
    assert summary["false_negatives"]["case_ids"] == ["FN1"]
    assert summary["overall"]["passed"] == 0
    assert "오탐" in summary["reporting_note"]
    assert "nemo_rails" not in summary


def test_summary_groups_by_request_class_and_distinct_reasons() -> None:
    rows = evaluate_cases(
        [
            _pipeline_case("A1", evidence=[], decision="ALLOW"),
            _pipeline_case("A2", evidence=[], decision="ALLOW"),
            _gate_case("B1", "비트코인 가격을 예측해 주세요.", "BLOCK"),
        ]
    )
    summary = build_summary(rows, cases_path=Path("t.jsonl"), rails_runner=None)

    assert summary["by_request_class"]["normal"]["total"] == 2
    assert summary["by_request_class"]["irrelevant"]["passed"] == 1
    assert summary["by_target"]["guardrail_pipeline"]["failed"] == 2

    # Two cases, one shared cause: the report must show one distinct reason.
    reasons = summary["distinct_failure_reasons"]
    assert len(reasons) == 1
    assert reasons[0]["count"] == 2
    assert sorted(reasons[0]["case_ids"]) == ["A1", "A2"]


def test_summary_excludes_non_discriminating_rows_from_defect_grouping() -> None:
    marked = _pipeline_case("ND1", evidence=[], decision="ALLOW")
    marked["discriminating"] = False

    rows = evaluate_cases([marked])
    summary = build_summary(rows, cases_path=Path("t.jsonl"), rails_runner=None)

    assert summary["non_discriminating_case_ids"] == ["ND1"]
    assert summary["distinct_failure_reasons"] == []
    assert summary["overall"]["failed"] == 1
    assert summary["overall_discriminating_only"]["total"] == 0


def test_case_input_renders_the_gate_question_and_pipeline_payload() -> None:
    gate = _case_input(_gate_case("T1", "비트코인 가격", "BLOCK"))
    assert "질문=" in gate and "근거" not in gate

    pipeline = _case_input(
        _pipeline_case("T2", evidence=[DISEASE_EVIDENCE], decision="ALLOW")
    )
    assert "근거=1건" in pipeline

    unsafe = _pipeline_case("T3", evidence=[DISEASE_EVIDENCE], decision="BLOCK")
    unsafe["request"]["pesticide_use_requested"] = True
    assert "농약사용요청=True" in _case_input(unsafe)
