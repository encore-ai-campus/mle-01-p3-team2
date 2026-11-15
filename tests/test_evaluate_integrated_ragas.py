from __future__ import annotations

from types import SimpleNamespace

from src.retrievers.search_result import SearchResult


def _module():
    from scripts.evaluation import evaluate_integrated_ragas

    return evaluate_integrated_ragas


class _RecordingApp:
    def __init__(self, state=None, error: Exception | None = None):
        self.state = state or {}
        self.error = error
        self.inputs: list[dict] = []

    def invoke(self, payload):
        self.inputs.append(payload)
        if self.error is not None:
            raise self.error
        return self.state


def test_run_supervisor_cases_sends_only_the_question_to_the_app():
    module = _module()
    app = _RecordingApp(
        state={
            "initial_route": "VECTOR",
            "route": "VECTOR",
            "route_path": ["VECTOR"],
            "final_status": "ANSWERED",
            "final_answer": "최종 답변",
            "retrieved_results": [],
            "evidences": [],
        }
    )
    case = {
        "case_id": "GS100",
        "question": "상추 재배법은?",
        "expected_behavior": "ANSWER",
        "expected_facts": ["정답이 Supervisor에 전달되면 안 된다"],
        "expected_source_chunk_ids": ["CHK_SECRET"],
    }

    records = module.run_supervisor_cases(app, [case], top_k=5)

    assert app.inputs == [{"question": "상추 재배법은?"}]
    assert records[0]["case_id"] == "GS100"
    assert records[0]["response"] == "최종 답변"


def test_state_to_record_deduplicates_ids_and_uses_original_evidence_text():
    module = _module()
    results = [
        SearchResult(
            source="vector",
            text="검색 결과 1",
            chunk_ids=["CHK_1"],
            evidence_ids=["EVD_1"],
        ),
        SearchResult(
            source="graph",
            text="합성 그래프 경로",
            chunk_ids=["CHK_1", "CHK_2"],
            evidence_ids=["EVD_1", "EVD_2"],
        ),
    ]
    state = {
        "initial_route": "GRAPH",
        "route": "HYBRID",
        "route_path": ["GRAPH", "HYBRID"],
        "final_status": "ANSWERED",
        "final_answer": "근거 기반 답변",
        "retrieved_results": results,
        "evidences": [
            SimpleNamespace(evidence_id="EVD_1", quote="공식 원문 1"),
            SimpleNamespace(evidence_id="EVD_1", quote="공식 원문 1"),
            SimpleNamespace(evidence_id="EVD_2", quote="공식 원문 2"),
        ],
    }
    case = {
        "case_id": "GS101",
        "question": "질문",
        "expected_behavior": "ANSWER",
        "expected_source_chunk_ids": ["CHK_2"],
        "expected_evidence_ids": ["EVD_2"],
    }

    record = module.state_to_record(case, state, top_k=5)

    assert record["retrieved_chunk_ids"] == ["CHK_1", "CHK_2"]
    assert record["retrieved_evidence_ids"] == ["EVD_1", "EVD_2"]
    assert record["retrieved_contexts"] == ["공식 원문 1", "공식 원문 2"]
    assert record["chunk_hit_at_k"] == 1.0
    assert record["evidence_hit_at_k"] == 1.0
    assert record["fallback"] is True


def test_reject_case_is_structural_only_and_not_ragas_eligible():
    module = _module()
    record = module.state_to_record(
        {
            "case_id": "GS001",
            "question": "퀵정렬을 구현해줘",
            "expected_behavior": "REJECT",
        },
        {
            "initial_route": "REJECT",
            "route": "REJECT",
            "route_path": ["REJECT"],
            "final_status": "REJECTED",
            "final_answer": "범위 밖 질문입니다.",
            "retrieved_results": [],
            "evidences": [],
        },
        top_k=5,
    )

    assert record["ragas_eligible"] is False
    assert record["structural_pass"] is True


def test_run_supervisor_cases_records_errors_without_aborting():
    module = _module()
    app = _RecordingApp(error=RuntimeError("database unavailable"))

    records = module.run_supervisor_cases(
        app,
        [{"case_id": "GS102", "question": "질문", "expected_behavior": "ANSWER"}],
        top_k=5,
    )

    assert records[0]["run_status"] == "ERROR"
    assert records[0]["error_type"] == "RuntimeError"
    assert records[0]["ragas_eligible"] is False


def test_build_ragas_rows_joins_references_without_changing_run_records():
    module = _module()
    records = [
        {
            "case_id": "GS103",
            "question": "질문",
            "response": "답변",
            "retrieved_contexts": ["검색 원문"],
            "ragas_eligible": True,
        },
        {
            "case_id": "GS104",
            "question": "범위 밖",
            "response": "거절",
            "retrieved_contexts": [],
            "ragas_eligible": False,
        },
    ]
    cases = [
        {
            "case_id": "GS103",
            "expected_facts": ["사실 A", "사실 B"],
            "expected_source_chunk_ids": ["CHK_A"],
        },
        {"case_id": "GS104", "expected_facts": [], "expected_source_chunk_ids": []},
    ]
    chunks = [{"chunk_id": "CHK_A", "clean_text": "정답 원문"}]

    rows = module.build_ragas_rows(records, cases, chunks)

    assert rows == [
        {
            "case_id": "GS103",
            "user_input": "질문",
            "response": "답변",
            "retrieved_contexts": ["검색 원문"],
            "reference": "사실 A 사실 B",
            "reference_contexts": ["정답 원문"],
        }
    ]
    assert "reference" not in records[0]


def test_ragas_runtime_defaults_are_safe_for_context_precision():
    module = _module()

    args = module.parse_args([])

    assert args.ragas_timeout == 900
    assert args.ragas_max_workers == 4
    assert args.ragas_max_retries == 10


def test_require_complete_scores_rejects_partial_metric_output():
    module = _module()
    complete = {name: 0.5 for name in module.RAGAS_METRICS}
    module._require_complete_scores([{"case_id": "GS100", **complete}])

    partial = {**complete, "context_precision": None}
    try:
        module._require_complete_scores([{"case_id": "GS101", **partial}])
    except RuntimeError as exc:
        assert "GS101" in str(exc)
        assert "context_precision" in str(exc)
    else:
        raise AssertionError("partial RAGAS output must be rejected")


def test_record_keeps_guardrail_and_execution_decisions():
    """보류 원인을 레일별로 구분할 수 있는 필드를 남긴다."""
    module = _module()

    record = module.state_to_record(
        {
            "case_id": "GS019",
            "question": "등록된 농약 품목은 무엇인가요?",
            "expected_behavior": "ANSWER",
        },
        {
            "initial_route": "GRAPH",
            "route": "GRAPH",
            "route_path": ["GRAPH"],
            "final_status": "WITHHELD",
            "final_answer": "작물·병해충 정보가 부족합니다.",
            "retrieved_results": [],
            "evidences": [],
            "guardrail_decision": "FALLBACK",
            "guardrail_reason": "농약 정보 조회 요청",
            "execution_decision": "FALLBACK",
            "execution_reason": "작물·병해충 정보가 부족합니다.",
            "execution_records": [],
        },
        top_k=5,
    )

    assert record["guardrail_decision"] == "FALLBACK"
    assert record["execution_decision"] == "FALLBACK"
    assert record["execution_reason"] == "작물·병해충 정보가 부족합니다."
    assert record["execution_record_count"] == 0


def test_record_counts_listed_execution_records():
    """목록 조회로 통과한 사례는 기록 수를 남긴다."""
    module = _module()

    record = module.state_to_record(
        {
            "case_id": "GS019",
            "question": "등록된 농약 품목은 무엇인가요?",
            "expected_behavior": "ANSWER",
        },
        {
            "route_path": ["GRAPH"],
            "final_status": "ANSWERED",
            "final_answer": "등록 약제 목록입니다.",
            "retrieved_results": [],
            "evidences": [],
            "execution_decision": "WARN",
            "execution_records": [{"product_name": "A"}, {"product_name": "B"}],
        },
        top_k=5,
    )

    assert record["execution_decision"] == "WARN"
    assert record["execution_record_count"] == 2


def test_error_record_shares_the_guardrail_schema():
    """실행 실패 레코드도 같은 필드를 갖는다."""
    module = _module()

    app = _RecordingApp(error=RuntimeError("실패"))
    records = module.run_supervisor_cases(
        app,
        [{"case_id": "GS999", "question": "질문"}],
        top_k=5,
    )

    assert records[0]["execution_record_count"] == 0
    assert records[0]["guardrail_decision"] is None
    assert set(records[0]) == set(
        module.state_to_record(
            {"case_id": "GS999", "question": "질문"},
            {
                "route_path": [],
                "final_status": "ANSWERED",
                "final_answer": "답변",
                "retrieved_results": [],
                "evidences": [],
            },
            top_k=5,
        )
    )


def test_summary_breaks_down_withheld_reasons():
    """요약에 레일별 보류 집계가 포함된다."""
    module = _module()

    summary = module.summarize([
        {
            "case_id": "GS019",
            "question_type": "lookup",
            "run_status": "OK",
            "final_status": "WITHHELD",
            "execution_decision": "FALLBACK",
            "execution_reason": "작물·병해충 정보가 부족합니다.",
            "guardrail_reason": "농약 정보 조회 요청",
            "structural_pass": False,
        },
        {
            "case_id": "GS036",
            "question_type": "lookup",
            "run_status": "OK",
            "final_status": "WITHHELD",
            "execution_decision": "WARN",
            "guardrail_reason": "LLM 검증을 위한 API 키가 없습니다.",
            "structural_pass": False,
        },
        {
            "case_id": "GS020",
            "question_type": "lookup",
            "run_status": "OK",
            "final_status": "ANSWERED",
            "execution_decision": "WARN",
            "structural_pass": True,
        },
    ])

    assert summary["withheld_cases"] == 2
    assert summary["withheld_by_execution_decision"] == {
        "FALLBACK": 1,
        "WARN": 1,
    }
    assert summary["withheld_by_execution_reason"] == {
        "작물·병해충 정보가 부족합니다.": 1,
    }
    assert (
        summary["withheld_by_guardrail_reason"][
            "LLM 검증을 위한 API 키가 없습니다."
        ]
        == 1
    )


def test_case_ids_filter_selects_requested_cases(tmp_path, monkeypatch):
    """--case-ids는 지정한 케이스만 실행 대상으로 남긴다."""
    import json as _json

    from scripts.evaluation import evaluate_integrated_ragas as module

    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        "\n".join(
            _json.dumps(
                {"case_id": case_id, "question": "q", "status": "READY"},
                ensure_ascii=False,
            )
            for case_id in ("GS001", "GS002", "GS003")
        ),
        encoding="utf-8",
    )

    raw = tmp_path / "out" / "supervisor_runs.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text("", encoding="utf-8")

    seen: dict[str, list[str]] = {}

    def fake_read_jsonl(path):
        rows = [
            _json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return rows

    monkeypatch.setattr(module, "read_jsonl", fake_read_jsonl)

    def fake_run(app, cases, *, top_k, on_record=None):
        seen["case_ids"] = [case["case_id"] for case in cases]
        return []

    monkeypatch.setattr(module, "run_supervisor_cases", fake_run)
    monkeypatch.setattr(
        "scripts.run_supervisor.build_app",
        lambda: (object(), type("D", (), {"close": lambda self: None})()),
    )

    exit_code = module.main([
        "--golden-set", str(golden),
        "--output-dir", str(raw.parent),
        "--stage", "collect",
        "--case-ids", "GS003, GS001",
    ])

    assert exit_code == 0
    assert seen["case_ids"] == ["GS001", "GS003"]


def test_case_ids_filter_rejects_unknown_case(tmp_path, monkeypatch):
    """Golden Set에 없는 ID를 주면 조용히 건너뛰지 않고 알린다."""
    import json as _json

    import pytest as _pytest

    from scripts.evaluation import evaluate_integrated_ragas as module

    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        _json.dumps(
            {"case_id": "GS001", "question": "q", "status": "READY"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with _pytest.raises(ValueError, match="GS999"):
        module.main([
            "--golden-set", str(golden),
            "--output-dir", str(tmp_path / "out"),
            "--stage", "collect",
            "--case-ids", "GS999",
        ])
