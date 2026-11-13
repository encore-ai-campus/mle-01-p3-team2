from src.retrievers.answer import (
    Citation,
    CitedAnswer,
    Claim,
)
from src.retrievers.router import QuestionRouter
from src.retrievers.search_result import SearchResult
from src.supervisor.citation_verifier import AnswerReview
from src.supervisor.graph import (
    INSUFFICIENT_EVIDENCE_WARNING,
    build_supervisor,
)

def _evidence_lookup(
    evidence_ids: list[str],
) -> dict[str, dict]:
    """Supervisor 테스트용 Evidence 조회 함수."""

    return {
        evidence_id: {
            "evidence_id": evidence_id,
            "document_id": "DOC_TEST",
            "title": "테스트 공식 문서",
            "url": "https://example.com/test",
            "text": f"공식 검색 근거 {evidence_id}",
            "chunk_id": f"CHK_{evidence_id}",
        }
        for evidence_id in evidence_ids
    }


def _cited_answer_generator(
    question,
    evidences,
) -> CitedAnswer:
    """LLM 호출 없이 정상적인 구조화 답변을 만듭니다."""

    del question

    evidence = evidences[0]

    return CitedAnswer(
        answer=evidence.quote,
        claims=[
            Claim(
                text=evidence.quote,
                citations=[
                    Citation(
                        evidence_id=evidence.evidence_id,
                        quote=evidence.quote,
                    )
                ],
            )
        ],
    )


def _support_checker(
    claim,
    contexts,
) -> AnswerReview:
    """테스트에서는 유효한 주장으로 판정합니다."""

    assert claim
    assert contexts

    return AnswerReview(
        supported=True,
        reason="테스트 근거가 주장을 직접 뒷받침합니다.",
    )


def _build_app(**kwargs):
    """Supervisor 공통 테스트 의존성을 주입합니다."""

    return build_supervisor(
        evidence_lookup=_evidence_lookup,
        cited_answer_generator=_cited_answer_generator,
        citation_support_checker=_support_checker,
        **kwargs,
    )


def _graph_result():
    return SearchResult(
        source="graph",
        text="마늘 → 대서마늘 → 남부 지역",
        evidence_ids=["EVD_1"],
        path=["마늘", "대서마늘", "남부 지역"],
        relation_types=["HAS_VARIETY", "SUITABLE_FOR_REGION"],
        path_edges=[
            {"hop": 1, "relation_type": "HAS_VARIETY", "from_entity_id": "CROP_1", "to_entity_id": "VAR_1", "relation_id": "REL_1", "original_chunks": [{"chunk_id": "CHK_1", "evidence_id": "EVD_1", "source_id": "SRC_1", "text": "마늘 품종은 대서마늘이다."}]},
            {"hop": 2, "relation_type": "SUITABLE_FOR_REGION", "from_entity_id": "VAR_1", "to_entity_id": "REG_1", "relation_id": "REL_2", "original_chunks": [{"chunk_id": "CHK_2", "evidence_id": "EVD_2", "source_id": "SRC_2", "text": "대서마늘은 남부 지역에 적합하다."}]},
        ],
        metadata={"has_complete_evidence": True},
    )


def test_supervisor_routes_and_calls_graph_worker():
    calls = []

    def graph_search(question, **kwargs):
        calls.append(("graph", question, kwargs))
        return [_graph_result()]

    def vector_search(question, **kwargs):
        calls.append(("vector", question, kwargs))
        return []

    app = _build_app(graph_search_fn=graph_search, vector_search_fn=vector_search)
    state = app.invoke({"question": "마늘 품종별 적합 지역은 어디인가?"})

    assert state["route"] == "GRAPH"
    assert state["final_status"] == "ANSWERED"
    assert state["evidence_status"] == "PASS"
    assert calls[0][0] == "graph"


def test_supervisor_retries_with_hybrid_when_evidence_is_missing():
    routes = []

    def graph_search(question, **kwargs):
        routes.append("graph")
        return [SearchResult(source="graph", text="근거 없음")]

    def vector_search(question, **kwargs):
        routes.append("vector")
        return [
            SearchResult(
                source="vector",
                text="공식 문서",
                evidence_ids=["EVD_1"],
                chunk_ids=["CHK_1"],
            )
        ]

    def hybrid_search_fn(question, **kwargs):
        routes.append("hybrid")
        return [
            SearchResult(
                source="vector",
                text="공식 문서",
                evidence_ids=["EVD_1"],
                chunk_ids=["CHK_1"],
            )
        ]

    app = _build_app(
        graph_search_fn=graph_search,
        vector_search_fn=vector_search,
        hybrid_search_fn=hybrid_search_fn,
        max_retries=1,
    )
    state = app.invoke({"question": "마늘 품종별 적합 지역은 어디인가?"})

    assert routes == ["graph", "hybrid"]
    assert state["retry_count"] == 1
    assert state["final_status"] == "ANSWERED"


def test_supervisor_withholds_after_retry_limit():
    def empty_search(question, **kwargs):
        return [SearchResult(source="graph", text="근거 없음")]

    app = _build_app(
        graph_search_fn=empty_search,
        vector_search_fn=empty_search,
        hybrid_search_fn=empty_search,
        max_retries=1,
    )
    state = app.invoke({"question": "마늘 품종별 적합 지역은 어디인가?"})

    assert state["final_status"] == "WITHHELD"
    assert state["evidence_status"] == "INSUFFICIENT"
    assert state["retry_count"] == 1

def test_supervisor_rejects_out_of_scope_quesion_with_reject_status():
    def unexpected_search(question, **kwargs): 
        raise AssertionError("REJECT 질문은 검색을 실행하지 않아야 함")

    app = _build_app(
        graph_search_fn=unexpected_search,
        vector_search_fn=unexpected_search,
        hybrid_search_fn=unexpected_search,
    )
    state = app.invoke({'question': '서울에서 데이터하기 좋은 맛집을 추천해주세요.'})

    assert state['final_status'] == 'REJECTED'
    assert state['evidence_status'] == 'NOT_APPLICABLE'
    assert not state.get('results')
    # 검색을 했다는 오해를 주는 문구가 섞이지 않아야 함
    assert '근거' not in state['final_answer']

def test_supervisor_withhold_is_not_confused_with_reject():
    """범위 안인데 근거가 없는 질문은 REJECTED가 아니라 WITHHELD입니다."""

    def empty_search(question, **kwargs):
        return [SearchResult(source="graph", text="근거 없음")]

    app = _build_app(
        graph_search_fn=empty_search,
        vector_search_fn=empty_search,
        hybrid_search_fn=empty_search,
        max_retries=1,
    )
    state = app.invoke({"question": "마늘 품종별 적합 지역은 어디인가?"})

    assert state["final_status"] == "WITHHELD"
    assert state["evidence_status"] == "INSUFFICIENT"
    assert state["route"] != "REJECT"

def test_supervisor_keeps_retrieved_results_when_evidence_is_insufficient():
    """검증에서 탈락해도 검색 원본은 남아야 Hit@K를 채점할 수 있습니다."""
    retrieved = SearchResult(source="graph", text="근거 없음")

    app = _build_app(
        graph_search_fn=lambda question, **kwargs: [retrieved],
        vector_search_fn=lambda question, **kwargs: [retrieved],
        hybrid_search_fn=lambda question, **kwargs: [retrieved],
        max_retries=0,
    )
    state = app.invoke({"question": "마늘 품종별 적합 지역은 어디인가?"})

    assert state["final_status"] == "WITHHELD"
    assert state["results"] == []            # 답변에는 쓰지 않습니다
    assert state["retrieved_results"]        # 채점용 원본은 남습니다
    assert state["rejected_results"]   # 왜 걸러졌는지 추적할 수 있어야 합니다


def test_supervisor_keeps_filtered_out_results_for_scoring_on_pass():
    """검증을 통과해도 탈락한 결과가 채점 대상에서 빠지면 안 됩니다."""
    usable = SearchResult(
        source="vector",
        text="공식 문서",
        evidence_ids=["EVD_1"],
        chunk_ids=["CHK_1"],
    )
    unusable = SearchResult(source="graph", text="근거 없음")

    app = _build_app(
        graph_search_fn=lambda question, **kwargs: [usable, unusable],
        vector_search_fn=lambda question, **kwargs: [],
    )
    state = app.invoke({"question": "마늘 품종별 적합 지역은 어디인가?"})

    assert state["evidence_status"] == "PASS"
    assert len(state["results"]) == 1
    assert len(state["retrieved_results"]) == 2


def test_supervisor_accumulates_retrieved_results_across_retries():
    first = SearchResult(source="graph", text="근거 없음")
    second = SearchResult(
        source="vector",
        text="공식 문서",
        evidence_ids=["EVD_2"],
        chunk_ids=["CHK_2"],
    )
    calls = {"n": 0}

    def graph_search(question, **kwargs):
        calls["n"] += 1
        return [first]

    app = _build_app(
        graph_search_fn=graph_search,
        vector_search_fn=lambda question, **kwargs: [],
        hybrid_search_fn=lambda question, **kwargs: [second],
        max_retries=1,
    )
    state = app.invoke({"question": "마늘 품종별 적합 지역은 어디인가?"})

    assert state["retry_count"] == 1
    assert len(state["retrieved_results"]) == 2   # 두 시도 모두 보존

def test_supervisor_does_not_duplicate_route_in_path_on_same_route_retry():
    """처음부터 HYBRID인 문항이 재시도해도 route_path에 중복이 쌓이면 안 됩니다."""
    insufficient = SearchResult(source="vector", text="근거 없음")

    app = _build_app(
        graph_search_fn=lambda question, **kwargs: [],
        vector_search_fn=lambda question, **kwargs: [],
        hybrid_search_fn=lambda question, **kwargs: [insufficient],
        max_retries=1,
    )
    state = app.invoke(
        {"question": "저온다습한 환경에서 어떤 병해가 생기고 어떻게 예방하나요?"}
    )

    assert state["initial_route"] == "HYBRID"
    # fallback이 아니라 같은 경로 재시도이므로 길이가 1이어야 합니다.
    assert state["route_path"] == ["HYBRID"]
    # 재시도 사실은 retry_count로 확인합니다.
    assert state["retry_count"] == 1


def test_supervisor_route_path_records_only_route_changes_across_retries():
    """재시도를 여러 번 해도 경로 전환은 한 번만 기록됩니다."""
    insufficient = SearchResult(source="graph", text="근거 없음")

    app = _build_app(
        graph_search_fn=lambda question, **kwargs: [insufficient],
        vector_search_fn=lambda question, **kwargs: [],
        hybrid_search_fn=lambda question, **kwargs: [insufficient],
        max_retries=2,
    )
    state = app.invoke({"question": "마늘 품종별 적합 지역은 어디인가?"})

    assert state["route_path"] == ["GRAPH", "HYBRID"]
    assert state["retry_count"] == 2

def test_supervisor_does_not_duplicate_insufficient_evidence_warning():
    """재시도를 여러 번 해도 같은 경고가 반복되지 않습니다."""
    insufficient = SearchResult(source="graph", text="근거 없음")

    app = _build_app(
        graph_search_fn=lambda question, **kwargs: [insufficient],
        vector_search_fn=lambda question, **kwargs: [insufficient],
        hybrid_search_fn=lambda question, **kwargs: [insufficient],
        max_retries=2,
    )
    state = app.invoke({"question": "마늘 품종별 적합 지역은 어디인가?"})

    assert state["retry_count"] == 2
    assert state["warnings"].count(INSUFFICIENT_EVIDENCE_WARNING) == 1


def test_supervisor_withholds_graph_path_with_ids_but_no_original_text():
    missing_original = SearchResult(
        source="graph",
        text="마늘 → 품종 → 지역",
        evidence_ids=["EVD_1"],
        path_edges=[
            {
                "hop": 1,
                "original_chunks": [],
            },
            {
                "hop": 2,
                "original_chunks": [],
            },
        ],
        metadata={
            "has_complete_evidence": False,
        },
    )

    app = _build_app(
        graph_search_fn=lambda question, **kwargs: [
            missing_original
        ],
        vector_search_fn=lambda question, **kwargs: [],
        hybrid_search_fn=lambda question, **kwargs: [
            missing_original
        ],
        max_retries=1,
    )

    state = app.invoke(
        {
            "question": (
                "마늘 품종별 적합 지역은 어디인가?"
            )
        }
    )

    assert state["retry_count"] == 1
    assert state["evidence_status"] == "INSUFFICIENT"
    assert state["final_status"] == "WITHHELD"
    assert state["results"] == []
