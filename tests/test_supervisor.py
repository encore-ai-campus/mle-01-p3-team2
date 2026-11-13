from src.retrievers.router import QuestionRouter
from src.retrievers.search_result import SearchResult
from src.supervisor.graph import build_supervisor


def _graph_result():
    return SearchResult(
        source="graph",
        text="마늘 → 대서마늘 → 남부 지역",
        evidence_ids=["EVD_1"],
        path=["마늘", "대서마늘", "남부 지역"],
        relation_types=["HAS_VARIETY", "SUITABLE_FOR_REGION"],
    )


def test_supervisor_routes_and_calls_graph_worker():
    calls = []

    def graph_search(question, **kwargs):
        calls.append(("graph", question, kwargs))
        return [_graph_result()]

    def vector_search(question, **kwargs):
        calls.append(("vector", question, kwargs))
        return []

    app = build_supervisor(graph_search_fn=graph_search, vector_search_fn=vector_search)
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
        return [SearchResult(source="vector", text="공식 문서", chunk_ids=["CHK_1"])]

    def hybrid_search_fn(question, **kwargs):
        routes.append("hybrid")
        return [SearchResult(source="vector", text="공식 문서", chunk_ids=["CHK_1"])]

    app = build_supervisor(
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

    app = build_supervisor(
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

    app = build_supervisor(
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

    app = build_supervisor(
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

    app = build_supervisor(
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
    usable = SearchResult(source="vector", text="공식 문서", chunk_ids=["CHK_1"])
    unusable = SearchResult(source="graph", text="근거 없음")

    app = build_supervisor(
        graph_search_fn=lambda question, **kwargs: [usable, unusable],
        vector_search_fn=lambda question, **kwargs: [],
    )
    state = app.invoke({"question": "마늘 품종별 적합 지역은 어디인가?"})

    assert state["evidence_status"] == "PASS"
    assert len(state["results"]) == 1
    assert len(state["retrieved_results"]) == 2


def test_supervisor_accumulates_retrieved_results_across_retries():
    first = SearchResult(source="graph", text="근거 없음")
    second = SearchResult(source="vector", text="공식 문서", chunk_ids=["CHK_1"])
    calls = {"n": 0}

    def graph_search(question, **kwargs):
        calls["n"] += 1
        return [first]

    app = build_supervisor(
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

    app = build_supervisor(
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

    app = build_supervisor(
        graph_search_fn=lambda question, **kwargs: [insufficient],
        vector_search_fn=lambda question, **kwargs: [],
        hybrid_search_fn=lambda question, **kwargs: [insufficient],
        max_retries=2,
    )
    state = app.invoke({"question": "마늘 품종별 적합 지역은 어디인가?"})

    assert state["route_path"] == ["GRAPH", "HYBRID"]
    assert state["retry_count"] == 2