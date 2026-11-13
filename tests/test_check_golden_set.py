import json

import pytest

from scripts.evaluation.check_golden_set import (
    check_processed_references,
    check_rules,
    check_source_drift,
    main,
    make_reference,
    regenerate_from_reviewed,
)


def _row(case_id="GS001", **over):
    row = {
        "case_id": case_id,
        "question": f"질문 {case_id}",
        "reference": "사실 하나.",
        "question_type": "relation",
        "difficulty": "easy",
        "query_strategy": "graph",
        "expected_behavior": "ANSWER",
        "expected_facts": ["사실 하나"],
        "expected_entity_ids": ["ENT_A", "ENT_B"],
        "expected_relation_ids": ["REL_A"],
        "expected_evidence_ids": ["EVD_A"],
        "expected_source_chunk_ids": ["CHK_A"],
        "safety_assertions": [],
        "status": "READY",
    }
    row.update(over)
    return row


def _reject_row(case_id="GS002"):
    return _row(
        case_id,
        question=f"범위 밖 {case_id}",
        reference="",
        query_strategy="reject",
        expected_behavior="REJECT",
        question_type="out_of_scope",
        expected_facts=[],
        expected_entity_ids=[],
        expected_relation_ids=[],
        expected_evidence_ids=[],
        expected_source_chunk_ids=[],
    )


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _processed(tmp_path):
    directory = tmp_path / "processed"
    directory.mkdir()
    _write(directory / "entities.jsonl", [{"entity_id": "ENT_A"}, {"entity_id": "ENT_B"}])
    _write(
        directory / "relations.jsonl",
        [
            {
                "relation_id": "REL_A",
                "from_id": "ENT_A",
                "to_id": "ENT_B",
                "relation_type": "HAS_VARIETY",
                "retrieval_eligible": True,
            }
        ],
    )
    _write(directory / "evidence.jsonl", [{"evidence_id": "EVD_A"}])
    _write(directory / "chunks.jsonl", [{"chunk_id": "CHK_A"}])
    return directory


def test_make_reference_appends_safety_assertion_for_warning_behavior():
    row = {
        "expected_behavior": "ANSWER_WITH_WARNING",
        "expected_facts": ["사실"],
        "safety_assertions": ["주의"],
    }

    assert make_reference(row) == "사실. 주의."


def test_make_reference_omits_safety_assertion_for_plain_answer():
    row = {
        "expected_behavior": "ANSWER",
        "expected_facts": ["사실"],
        "safety_assertions": ["주의"],
    }

    assert make_reference(row) == "사실."


def test_clean_rows_pass_rule_check():
    assert check_rules([_row("GS001"), _reject_row("GS002")]) == []


def test_rule_check_catches_reference_missing_safety_sentence():
    # 실제로 있었던 결함입니다: ANSWER_WITH_WARNING 인데 reference 에 안전 문장이 없음.
    row = _row(
        expected_behavior="ANSWER_WITH_WARNING",
        safety_assertions=["표준값을 그대로 쓰지 말 것"],
        reference="사실 하나.",
    )

    problems = check_rules([row])

    assert any("reference 가 expected_facts 와 어긋납니다" in p for p in problems)


def test_rule_check_catches_case_id_gap():
    # 실제로 있었던 결함입니다: 67줄인데 번호가 GS014, GS046 을 건너뜀.
    problems = check_rules([_row("GS001"), _row("GS003")])

    assert any("빈 번호 없이 이어지지 않습니다" in p for p in problems)
    assert any("GS002" in p for p in problems)


def test_rule_check_catches_duplicate_question_and_case_id():
    problems = check_rules([_row("GS001"), _row("GS002", question="질문 GS001")])

    assert any("중복 질문" in p for p in problems)

    problems = check_rules([_row("GS001"), _row("GS001", question="다른 질문")])
    assert any("중복 case_id" in p for p in problems)


@pytest.mark.parametrize(
    "over, needle",
    [
        ({"status": "REJECTED"}, "status 가 READY"),
        ({"query_strategy": "nope"}, "잘못된 query_strategy"),
        ({"expected_behavior": "nope"}, "잘못된 expected_behavior"),
        ({"expected_behavior": "REJECT"}, "일치하지 않습니다"),
        ({"expected_facts": [], "reference": ""}, "기대 사실이 비어 있습니다"),
        ({"expected_evidence_ids": []}, "근거 ID 가 비어 있습니다"),
        ({"question": "  "}, "질문이 비어 있습니다"),
    ],
)
def test_rule_check_catches_notebook_rule_violations(over, needle):
    problems = check_rules([_row(**over)])

    assert any(needle in p for p in problems), problems


def test_rule_check_catches_reject_row_with_payload():
    problems = check_rules([_reject_row("GS001") | {"expected_facts": ["x"]}])

    assert any("REJECT 문항의 필드가 비어 있지 않습니다" in p for p in problems)


def test_rule_check_reports_missing_required_keys():
    row = _row()
    del row["expected_evidence_ids"]

    problems = check_rules([row])

    assert any("필수 키 누락" in p for p in problems)


def test_processed_check_passes_for_connected_chain(tmp_path):
    assert check_processed_references([_row()], _processed(tmp_path)) == []


def test_processed_check_catches_unknown_ids(tmp_path):
    directory = _processed(tmp_path)
    row = _row(expected_entity_ids=["ENT_A", "ENT_MISSING"], expected_evidence_ids=["EVD_X"])

    problems = check_processed_references([row], directory)

    assert any("없는 엔티티 ID ENT_MISSING" in p for p in problems)
    assert any("없는 근거 ID EVD_X" in p for p in problems)


def test_processed_check_catches_relation_excluded_from_retrieval(tmp_path):
    directory = _processed(tmp_path)
    _write(
        directory / "relations.jsonl",
        [
            {
                "relation_id": "REL_A",
                "from_id": "ENT_A",
                "to_id": "ENT_B",
                "relation_type": "HAS_VARIETY",
                "retrieval_eligible": False,
            }
        ],
    )

    problems = check_processed_references([_row()], directory)

    assert any("검색 대상이 아닌 관계 ID" in p for p in problems)


def test_processed_check_catches_broken_chain(tmp_path):
    directory = _processed(tmp_path)
    row = _row(expected_entity_ids=["ENT_B", "ENT_A"])

    problems = check_processed_references([row], directory)

    assert any("체인이 끊깁니다" in p for p in problems)


def test_processed_check_catches_node_count_mismatch(tmp_path):
    row = _row(expected_entity_ids=["ENT_A"])

    problems = check_processed_references([row], _processed(tmp_path))

    assert any("hop 수 + 1" in p for p in problems)


def test_regenerate_numbers_cases_from_one_and_skips_rejected(tmp_path):
    team = tmp_path / "team"
    _write(
        team / "reviewed_a.jsonl",
        [
            _row("CAND_1", question="질문 1"),
            _row("CAND_2", question="질문 2", status="REJECTED"),
            _row("CAND_3", question="질문 3"),
        ],
    )

    regenerated = regenerate_from_reviewed(team)

    assert [row["case_id"] for row in regenerated] == ["GS001", "GS002"]
    assert [row["question"] for row in regenerated] == ["질문 1", "질문 3"]


def test_regenerate_rejects_unreviewed_rows(tmp_path):
    team = tmp_path / "team"
    _write(team / "reviewed_a.jsonl", [_row("CAND_1", status="PENDING")])

    with pytest.raises(ValueError, match="검수가 끝나지 않은"):
        regenerate_from_reviewed(team)


def test_regenerate_ignores_candidates_files(tmp_path):
    # candidates_*.jsonl 은 검수 전이라 통합 대상이 아닙니다.
    team = tmp_path / "team"
    _write(team / "reviewed_a.jsonl", [_row("CAND_1", question="질문 1")])
    _write(team / "candidates_b.jsonl", [_row("CAND_9", question="후보 질문")])

    regenerated = regenerate_from_reviewed(team)

    assert [row["question"] for row in regenerated] == ["질문 1"]


def test_drift_check_passes_when_golden_set_matches_sources(tmp_path):
    team = tmp_path / "team"
    _write(team / "reviewed_a.jsonl", [_row("CAND_1", question="질문 1")])
    rows = regenerate_from_reviewed(team)

    assert check_source_drift(rows, team) == []


def test_drift_check_reports_question_only_in_golden_set(tmp_path):
    team = tmp_path / "team"
    _write(team / "reviewed_a.jsonl", [_row("CAND_1", question="원본 질문")])
    rows = [_row("GS001", question="손으로 고친 질문")]

    problems = check_source_drift(rows, team)

    assert any("골든셋에만 있는 질문" in p for p in problems)
    assert any("검수 원본에만 있는 질문" in p for p in problems)


def test_drift_check_reports_field_difference(tmp_path):
    # 실제로 있었던 결함입니다: 골든셋에만 안전 경고가 추가되어 재생성하면 사라짐.
    team = tmp_path / "team"
    _write(team / "reviewed_a.jsonl", [_row("CAND_1", question="질문")])
    rows = regenerate_from_reviewed(team)
    rows[0] = rows[0] | {
        "expected_behavior": "ANSWER_WITH_WARNING",
        "safety_assertions": ["주의"],
    }

    problems = check_source_drift(rows, team)

    assert any("expected_behavior 가 검수 원본과 다릅니다" in p for p in problems)
    assert any("safety_assertions 가 검수 원본과 다릅니다" in p for p in problems)


def test_main_returns_zero_on_clean_set(tmp_path, capsys):
    team = tmp_path / "team"
    _write(team / "reviewed_a.jsonl", [_row("CAND_1", question="질문 1")])
    golden = _write(tmp_path / "golden.jsonl", regenerate_from_reviewed(team))

    exit_code = main(
        [
            "--golden-set",
            str(golden),
            "--team-dir",
            str(team),
            "--processed-dir",
            str(_processed(tmp_path)),
        ]
    )

    assert exit_code == 0
    assert "모든 검사를 통과했습니다" in capsys.readouterr().out


def test_main_returns_one_and_counts_problems(tmp_path, capsys):
    team = tmp_path / "team"
    _write(team / "reviewed_a.jsonl", [_row("CAND_1", question="질문 1")])
    golden = _write(tmp_path / "golden.jsonl", [_row("GS002", question="질문 1")])

    exit_code = main(
        [
            "--golden-set",
            str(golden),
            "--team-dir",
            str(team),
            "--skip-processed",
        ]
    )

    assert exit_code == 1
    assert "문제" in capsys.readouterr().out
