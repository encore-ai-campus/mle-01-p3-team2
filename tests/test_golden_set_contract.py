"""`docs/golden-set-contract.md` 의 규칙이 실제로 강제되는지 확인합니다.

확정본이 계약을 지키는지와, 검증기가 각 위반을 실제로 잡아내는지를 함께 봅니다.
후자가 없으면 검증기가 조용히 무력해져도 알 수 없습니다.
"""

from pathlib import Path

import pytest

from scripts.evaluation.validate_golden_set import (
    DEFAULT_GOLDEN_PATH,
    Violation,
    load_rows,
    main,
    summarize,
    validate,
    warning_triggers,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _answer_case(**overrides):
    row = {
        "case_id": "T001",
        "question": "배추 노균병은 어떤 환경에서 발생하나요?",
        "reference": "다습한 환경에서 발생한다.",
        "question_type": "fact",
        "difficulty": "easy",
        "query_strategy": "vector",
        "expected_behavior": "ANSWER",
        "expected_facts": ["다습한 환경에서 발생한다"],
        "expected_entity_ids": [],
        "expected_relation_ids": [],
        "expected_evidence_ids": ["EVD_T1"],
        "expected_source_chunk_ids": ["CHK_T1"],
        "safety_assertions": [],
        "status": "READY",
    }
    row.update(overrides)
    return row


def _reject_case(**overrides):
    row = {
        "case_id": "T900",
        "question": "파이썬으로 퀵정렬을 구현해 주세요.",
        "reference": "",
        "question_type": "out_of_scope",
        "difficulty": "easy",
        "query_strategy": "reject",
        "expected_behavior": "REJECT",
        "expected_facts": [],
        "expected_entity_ids": [],
        "expected_relation_ids": [],
        "expected_evidence_ids": [],
        "expected_source_chunk_ids": [],
        "safety_assertions": [],
        "status": "READY",
    }
    row.update(overrides)
    return row


def _rules(violations):
    return {v.rule for v in violations}


def _validate(rows):
    return validate(rows, check_corpus=False)


# --- 확정본 자체 -------------------------------------------------------------


def test_final_golden_set_satisfies_contract():
    rows = load_rows(DEFAULT_GOLDEN_PATH)
    violations = validate(rows, processed_dir=PROCESSED_DIR, check_corpus=True)
    assert violations == [], "\n".join(str(v) for v in violations)


def test_final_golden_set_has_expected_shape():
    stats = summarize(load_rows(DEFAULT_GOLDEN_PATH))
    assert stats["total"] == 67
    assert stats["by_strategy"] == {"graph": 21, "vector": 19, "hybrid": 17, "reject": 10}
    assert stats["by_behavior"]["REJECT"] == 10


def test_removed_duplicate_cases_stay_removed():
    ids = {row["case_id"] for row in load_rows(DEFAULT_GOLDEN_PATH)}
    assert "GS014" not in ids and "GS046" not in ids


def test_contract_document_exists():
    assert (PROJECT_ROOT / "docs" / "golden-set-contract.md").exists()


# --- 통과해야 하는 최소 사례 -------------------------------------------------


def test_clean_rows_produce_no_violations():
    assert _validate([_answer_case(), _reject_case()]) == []


# --- 스키마 ------------------------------------------------------------------


def test_missing_field_is_caught():
    row = _answer_case()
    del row["difficulty"]
    assert "schema" in _rules(_validate([row]))


def test_unknown_field_is_caught():
    assert "schema" in _rules(_validate([_answer_case(extra_note="x")]))


def test_wrong_type_is_caught():
    assert "schema" in _rules(_validate([_answer_case(expected_facts="문자열")]))


def test_non_ready_status_is_caught():
    assert "status" in _rules(_validate([_answer_case(status="DRAFT")]))


def test_unknown_strategy_is_caught():
    assert "strategy" in _rules(_validate([_answer_case(query_strategy="graphrag")]))


def test_duplicate_case_id_is_caught():
    rows = [_answer_case(), _answer_case(expected_evidence_ids=["EVD_T2"])]
    assert "case_id" in _rules(_validate(rows))


# --- REJECT 규칙 (계약 6절) --------------------------------------------------


def test_reject_case_with_reference_is_caught():
    assert "reject" in _rules(_validate([_reject_case(reference="무언가")]))


def test_reject_case_with_populated_field_is_caught():
    assert "reject" in _rules(_validate([_reject_case(expected_evidence_ids=["EVD_X"])]))


def test_strategy_and_behavior_mismatch_is_caught():
    assert "reject" in _rules(_validate([_reject_case(expected_behavior="ANSWER")]))


# --- expected_behavior 규칙 (계약 4절) --------------------------------------


@pytest.mark.parametrize(
    "reference,label",
    [
        ("수확 7일 전까지 사용하고 1500배로 희석한다.", "농약 사용 조건"),
        ("밑거름 표준 성분량은 질소 11.0kg/10a이다.", "비료 표준사용량"),
        ("검증상태가 PRELIMINARY_PASS이다.", "검증상태 미확정"),
        ("CR 표기만으로 저항성을 확정할 수 없다.", "품종 표기 저항성 추정"),
    ],
)
def test_warning_trigger_requires_warning(reference, label):
    row = _answer_case(reference=reference, expected_facts=[reference])
    assert label in warning_triggers(row)
    assert "behavior" in _rules(_validate([row]))


def test_warning_trigger_satisfied_by_warning_and_safety():
    reference = "수확 7일 전까지 사용하고 1500배로 희석한다."
    row = _answer_case(
        reference=reference,
        expected_facts=[reference],
        expected_behavior="ANSWER_WITH_WARNING",
        safety_assertions=["등록된 사용 기준을 준수해야 한다"],
    )
    assert _validate([row]) == []


def test_warning_without_safety_is_caught():
    assert "behavior" in _rules(
        _validate([_answer_case(expected_behavior="ANSWER_WITH_WARNING")])
    )


def test_safety_without_warning_is_caught():
    assert "behavior" in _rules(_validate([_answer_case(safety_assertions=["주의"])]))


def test_warning_without_trigger_is_allowed():
    # 계약 4절: 트리거가 없어도 약제 사용을 권하는 답변에는 경고를 붙일 수 있습니다.
    row = _answer_case(
        expected_behavior="ANSWER_WITH_WARNING",
        safety_assertions=["해당 작물에 등록된 전용약제를 선택해야 한다"],
    )
    assert _validate([row]) == []


# --- question 규칙 (계약 3절) ------------------------------------------------


def test_demonstrative_without_antecedent_is_caught():
    row = _answer_case(question="양파에 등록된 이 살균제의 상표는 무엇인가요?")
    assert "question" in _rules(_validate([row]))


def test_demonstrative_with_antecedent_is_allowed():
    row = _answer_case(
        question="배추에서 파밤나방이 발생했을 때, 이 해충 방제에 등록된 농약 품목은 무엇인가요?"
    )
    assert _validate([row]) == []


def test_empty_question_is_caught():
    assert "question" in _rules(_validate([_answer_case(question="   ")]))


# --- 중복·구조 (계약 2·5절) --------------------------------------------------


def test_identical_gold_ids_are_caught_as_duplicate():
    rows = [_answer_case(), _answer_case(case_id="T002")]
    assert "duplicate" in _rules(_validate(rows))


def test_vector_with_relation_is_caught():
    row = _answer_case(
        expected_relation_ids=["REL_X"], expected_entity_ids=["ENT_X"]
    )
    assert "strategy" in _rules(_validate([row]))


def test_graph_must_be_single_hop():
    row = _answer_case(
        query_strategy="graph",
        expected_entity_ids=["ENT_A", "ENT_B"],
        expected_relation_ids=["REL_A", "REL_B"],
    )
    assert "strategy" in _rules(_validate([row]))


def test_hybrid_must_be_two_hop():
    row = _answer_case(
        query_strategy="hybrid",
        expected_entity_ids=["ENT_A", "ENT_B"],
        expected_relation_ids=["REL_A"],
    )
    assert "strategy" in _rules(_validate([row]))


def test_graph_without_entities_is_caught():
    row = _answer_case(query_strategy="graph", expected_relation_ids=["REL_A"])
    assert "strategy" in _rules(_validate([row]))


def test_duplicate_ids_inside_one_field_are_caught():
    row = _answer_case(expected_evidence_ids=["EVD_T1", "EVD_T1"])
    assert "ids" in _rules(_validate([row]))


# --- 코퍼스 참조 -------------------------------------------------------------


def test_dangling_corpus_id_is_caught(tmp_path):
    (tmp_path / "entities.jsonl").write_text(
        '{"entity_id": "ENT_REAL"}\n', encoding="utf-8"
    )
    row = _answer_case(
        query_strategy="graph",
        expected_entity_ids=["ENT_MISSING"],
        expected_relation_ids=["REL_A"],
    )
    violations = validate([row], processed_dir=tmp_path, check_corpus=True)
    assert any(v.rule == "ids" and "미존재" in v.detail for v in violations)


# --- CLI ---------------------------------------------------------------------


def test_main_returns_zero_for_clean_file(tmp_path, capsys):
    import json

    path = tmp_path / "golden.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in [_answer_case(), _reject_case()]),
        encoding="utf-8",
    )
    assert main(["--golden-set", str(path), "--skip-corpus"]) == 0
    assert "계약 위반 없음" in capsys.readouterr().out


def test_main_returns_one_for_violating_file(tmp_path, capsys):
    import json

    path = tmp_path / "golden.jsonl"
    path.write_text(json.dumps(_answer_case(status="DRAFT")), encoding="utf-8")
    assert main(["--golden-set", str(path), "--skip-corpus"]) == 1
    assert "계약 위반" in capsys.readouterr().out


def test_violation_str_is_readable():
    assert str(Violation("GS001", "status", "detail")) == "[status] GS001: detail"
