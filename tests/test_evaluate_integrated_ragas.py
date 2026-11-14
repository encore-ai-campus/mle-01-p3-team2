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
