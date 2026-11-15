"""Tests for the discriminating guardrail evaluation set builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation.build_guardrail_eval_set import (
    BRANCHES,
    TARGETS,
    build_manifest,
    coverage,
    generate_cases,
    judge_cases,
    main,
    read_jsonl,
)
from scripts.evaluation.evaluate_guardrails import validate_cases


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = PROJECT_ROOT / "data" / "evaluation" / "golden_set_final.jsonl"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

ALLOWED_DECISIONS = {"ALLOW", "WARN"}
BLOCKED_DECISIONS = {"BLOCK", "FALLBACK"}


@pytest.fixture(scope="module")
def sources() -> tuple[list[dict], list[dict]]:
    if not GOLDEN_PATH.exists() or not CHUNKS_PATH.exists():
        pytest.skip("golden set or chunks are not available")
    return read_jsonl(GOLDEN_PATH), read_jsonl(CHUNKS_PATH)


@pytest.fixture(scope="module")
def cases(sources: tuple[list[dict], list[dict]]) -> list[dict]:
    golden_rows, chunk_rows = sources
    return generate_cases(golden_rows, chunk_rows)


def test_cases_match_the_public_report_shape(cases: list[dict]) -> None:
    validate_cases(cases)
    assert len(cases) > 60


def test_generation_is_deterministic(sources: tuple[list[dict], list[dict]]) -> None:
    golden_rows, chunk_rows = sources
    first = generate_cases(golden_rows, chunk_rows)
    second = generate_cases(golden_rows, chunk_rows)
    assert json.dumps(first, ensure_ascii=False, sort_keys=True) == json.dumps(
        second, ensure_ascii=False, sort_keys=True
    )


def test_every_registered_branch_has_a_case(cases: list[dict]) -> None:
    report = coverage(cases, include_judge_tier=False)
    assert report["uncovered"] == []
    assert report["covered"] == report["expected_in_this_build"]


def test_judge_branches_are_covered_when_the_tier_is_included(
    sources: tuple[list[dict], list[dict]],
) -> None:
    golden_rows, chunk_rows = sources
    with_judge = generate_cases(golden_rows, chunk_rows, include_judge_tier=True)
    report = coverage(with_judge, include_judge_tier=True)
    assert report["uncovered"] == []
    assert report["expected_in_this_build"] == len(BRANCHES)


def test_judge_tier_is_excluded_by_default_and_flagged(cases: list[dict]) -> None:
    assert all(case["tier"] != "E_judge" for case in cases)
    assert all(case["requires_api_key"] is False for case in cases)
    judge = judge_cases()
    assert judge, "judge tier must not be empty"
    assert all(case["requires_api_key"] is True for case in judge)
    # The judge path is only reached when expected_facts is absent or empty.
    assert all(not case["request"].get("expected_facts") for case in judge)


def test_decision_and_allowed_flag_agree(cases: list[dict]) -> None:
    for case in cases:
        decision = case["expected_decision"]
        if decision in ALLOWED_DECISIONS:
            assert case["expected_allowed"] is True, case["case_id"]
        else:
            assert decision in BLOCKED_DECISIONS
            assert case["expected_allowed"] is False, case["case_id"]


def test_every_case_declares_a_known_branch_and_a_rationale(cases: list[dict]) -> None:
    for case in cases:
        assert case["branch_id"] in BRANCHES, case["case_id"]
        assert case["branch_description"] == BRANCHES[case["branch_id"]]
        rationale = case["policy_rationale"].strip()
        # Korean is dense, so this is a floor against empty or one-word
        # placeholders rather than a style rule.
        assert len(rationale) >= 15, case["case_id"]
        assert rationale != case["branch_description"], case["case_id"]


def test_expected_decisions_cover_all_four_outcomes(cases: list[dict]) -> None:
    decisions = {case["expected_decision"] for case in cases}
    assert decisions == {"ALLOW", "WARN", "FALLBACK", "BLOCK"}


def test_both_over_and_under_blocking_probes_exist(cases: list[dict]) -> None:
    scenarios = {case["scenario_type"] for case in cases}
    assert "grounding_over_block_probe" in scenarios
    assert "grounding_under_block_probe" in scenarios

    over = [c for c in cases if c["scenario_type"] == "grounding_over_block_probe"]
    under = [c for c in cases if c["scenario_type"] == "grounding_under_block_probe"]
    # Over-blocking probes must be labelled allowed, under-blocking rejected,
    # otherwise they cannot detect what they are named for.
    assert all(c["expected_allowed"] is True for c in over)
    assert all(c["expected_allowed"] is False for c in under)


def test_golden_tier_uses_the_golden_answer_and_facts_unmodified(
    sources: tuple[list[dict], list[dict]], cases: list[dict]
) -> None:
    golden_rows, _ = sources
    by_id = {str(row["case_id"]): row for row in golden_rows}
    golden_tier = [c for c in cases if c["tier"] == "D_golden" and c["discriminating"]]
    assert golden_tier

    for case in golden_tier:
        source = by_id[case["source_case_id"]]
        request = case["request"]
        assert request["question"] == source["question"]
        assert request["answer"] == source["reference"]
        assert request["expected_facts"] == list(source["expected_facts"])
        assert request["evidence"], case["case_id"]


def test_pesticide_tier_holds_grounding_constant(cases: list[dict]) -> None:
    """The documented design invariant that isolates the pesticide rule."""

    pesticide = [c for c in cases if c["tier"] == "B_pesticide"]
    assert pesticide
    for case in pesticide:
        request = case["request"]
        evidence_text = "\n".join(str(item["text"]) for item in request["evidence"])
        assert request["answer"] == evidence_text, case["case_id"]


def test_non_discriminating_cases_are_marked_and_explained(cases: list[dict]) -> None:
    marked = [c for c in cases if not c["discriminating"]]
    assert marked, "coverage gaps must stay visible in the data"
    branches = {c["branch_id"] for c in marked}
    assert branches <= {"GD-REJECT-NOT-COVERED", "GD-WARNING-NOT-COVERED"}
    for case in marked:
        assert "discriminating=false" in case["policy_rationale"]


def test_golden_warning_cases_without_pesticide_evidence_are_not_discriminating(
    cases: list[dict],
) -> None:
    """A WARN can only come from the pesticide rule, so the rest cannot pass."""

    warning = [
        c
        for c in cases
        if c["branch_id"] in {"GD-WARNING", "GD-WARNING-NOT-COVERED"}
    ]
    assert warning
    for case in warning:
        has_pesticide = any(
            "pesticide"
            in str((item.get("metadata") or {}).get("evidence_type", "")).lower()
            or "검증상태" in str(item.get("text", ""))
            for item in case["request"]["evidence"]
        )
        assert case["discriminating"] is has_pesticide, case["case_id"]
        assert case["expected_decision"] == "WARN"


def test_labels_are_marked_unreviewed_and_contestable_ones_explained(
    cases: list[dict],
) -> None:
    assert all(c["label_review_status"] == "unreviewed" for c in cases)
    contestable = [c for c in cases if c["label_contestable"]]
    assert contestable, "judgement-call labels must be flagged for review"
    for case in contestable:
        assert case["label_contest_note"]
        assert len(case["label_contest_note"].strip()) >= 15, case["case_id"]
    assert all(
        c["label_contest_note"] is None for c in cases if not c["label_contestable"]
    )


def test_every_case_declares_a_known_target(cases: list[dict]) -> None:
    assert all(case["target"] in TARGETS for case in cases)
    targets = {case["target"] for case in cases}
    # The offline build covers every target that needs no paid call; the
    # NeMo input rail only appears with --include-judge-tier.
    assert targets == {
        "guardrail_pipeline",
        "router_scope_gate",
        "supervisor_adapter",
    }


def test_rubric_request_classes_are_all_populated(cases: list[dict]) -> None:
    """정상·무관·위험 are the three scenario kinds the requirements name."""

    counts: dict[str, int] = {}
    for case in cases:
        counts[case["request_class"]] = counts.get(case["request_class"], 0) + 1
    for required in ("normal", "irrelevant", "unsafe_request"):
        assert counts.get(required, 0) > 0, required


def test_request_class_follows_the_documented_rules(cases: list[dict]) -> None:
    for case in cases:
        if case["target"] == "router_scope_gate":
            expected = "normal" if case["expected_decision"] == "ALLOW" else "irrelevant"
            assert case["request_class"] == expected, case["case_id"]
        elif case["request"].get("pesticide_use_requested") is True:
            assert case["request_class"] == "unsafe_request", case["case_id"]


def test_input_rail_tier_measures_the_scope_gate(cases: list[dict]) -> None:
    gate = [c for c in cases if c["target"] == "router_scope_gate"]
    assert len(gate) >= 10

    # The gate only ever sees the question, so cases must not smuggle evidence.
    for case in gate:
        assert set(case["request"]) == {"question"}, case["case_id"]

    assert any(c["request_class"] == "irrelevant" for c in gate)
    # A positive control is required, otherwise over-blocking is invisible.
    assert any(c["request_class"] == "normal" for c in gate)
    scenarios = {c["scenario_type"] for c in gate}
    assert "input_over_block_probe" in scenarios
    assert "input_under_block_probe" in scenarios


def test_supervisor_adapter_tier_uses_langgraph_state(cases: list[dict]) -> None:
    """guardrails_node takes a state dict, not a GuardrailRequest payload."""

    adapter = [c for c in cases if c["target"] == "supervisor_adapter"]
    assert len(adapter) >= 4
    for case in adapter:
        request = case["request"]
        # State keys the node reads; "evidence"/"expected_facts" would be
        # silently ignored, so they must not appear.
        assert "results" in request
        assert "evidence" not in request
        assert "expected_facts" not in request
        assert "question" in request
        # The node never passes expected_facts, so production uses the judge.
        assert case["validation_path"] == "llm_judge", case["case_id"]

    assert any(c["request_class"] == "unsafe_request" for c in adapter)


def test_offline_build_needs_no_api_key(cases: list[dict]) -> None:
    assert all(case["requires_api_key"] is False for case in cases)


def test_input_rail_and_paid_adapter_cases_are_gated(
    sources: tuple[list[dict], list[dict]],
) -> None:
    golden_rows, chunk_rows = sources
    with_judge = generate_cases(golden_rows, chunk_rows, include_judge_tier=True)

    input_rail = [c for c in with_judge if c["target"] == "nemo_input_rail"]
    assert input_rail
    assert all(c["requires_api_key"] is True for c in input_rail)
    assert any(c["scenario_type"] == "prompt_injection" for c in input_rail)
    assert any(c["expected_decision"] == "ALLOW" for c in input_rail)

    paid_adapter = [
        c
        for c in with_judge
        if c["target"] == "supervisor_adapter" and c["requires_api_key"]
    ]
    assert paid_adapter, "운영 경로 양성 대조군은 판사 호출이 필요하다"


def test_legacy_runner_refuses_a_multi_target_set(
    tmp_path: Path, cases: list[dict]
) -> None:
    """evaluate_guardrails.py only calls validate_request; it must not score
    router or rails cases against the wrong component."""

    from scripts.evaluation import evaluate_guardrails

    path = tmp_path / "multi.jsonl"
    path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as excinfo:
        evaluate_guardrails.main(["--cases", str(path)])
    assert "evaluate_guardrail_scenarios" in str(excinfo.value)


def test_contract_tier_covers_the_integration_entry_point(cases: list[dict]) -> None:
    contract = [c for c in cases if c["tier"] == "F_contract"]
    assert len(contract) >= 4
    requests = [c["request"] for c in contract]
    assert any("bot_message" in request for request in requests)
    assert any(request.get("evidence") is None for request in requests)
    assert any(not isinstance(request.get("answer"), str) for request in requests)
    assert any(not isinstance(request.get("question"), str) for request in requests)


def test_structure_tier_blocks_only_on_structure(cases: list[dict]) -> None:
    structure = [c for c in cases if c["tier"] == "A_structure"]
    blocking = [c for c in structure if c["expected_decision"] == "BLOCK"]
    allowing = [c for c in structure if c["expected_decision"] == "ALLOW"]
    assert len(blocking) >= 15
    assert allowing, "structure tier needs a positive control"


def test_manifest_records_coverage_limits_and_reporting_rules(
    tmp_path: Path, cases: list[dict]
) -> None:
    output = tmp_path / "set.jsonl"
    output.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases),
        encoding="utf-8",
    )
    manifest = build_manifest(
        cases,
        output_path=output,
        golden_path=GOLDEN_PATH,
        chunks_path=CHUNKS_PATH,
        seed=42,
        include_judge_tier=False,
    )
    assert manifest["total_cases"] == len(cases)
    assert manifest["coverage"]["uncovered"] == []
    assert manifest["non_discriminating_case_ids"]
    assert manifest["reporting_rules"]
    assert manifest["known_corpus_limits"]
    assert manifest["review_required"] is True
    assert manifest["review_note"].strip()
    assert manifest["contestable_case_ids"]
    assert len(manifest["output_sha256"]) == 64
    assert "정책" in manifest["labeling_policy"]


def test_cli_writes_a_reproducible_set(tmp_path: Path) -> None:
    if not GOLDEN_PATH.exists() or not CHUNKS_PATH.exists():
        pytest.skip("golden set or chunks are not available")

    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    assert main(["--output", str(first), "--manifest", str(tmp_path / "a.json")]) == 0
    assert main(["--output", str(second), "--manifest", str(tmp_path / "b.json")]) == 0
    assert first.read_bytes() == second.read_bytes()

    manifest = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert manifest["total_cases"] == len(rows)
