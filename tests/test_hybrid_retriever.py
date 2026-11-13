from src.retrievers.hybrid_retriever import build_context, hybrid_search, merge_results
from src.retrievers.search_result import SearchResult


def _graph_result(text="마늘 → 대서마늘 → 남부 지역"):
    return SearchResult(
        source="graph",
        text=text,
        evidence_ids=["EVD_1"],
        path=text.split(" → "),
        relation_types=["HAS_VARIETY", "SUITABLE_FOR_REGION"],
        metadata={"hop_count": 2},
    )


def _vector_result(chunk_id="CHK_1"):
    return SearchResult(
        source="vector",
        text="공식 재배 문서 내용",
        score=0.91,
        chunk_ids=[chunk_id],
    )


def test_merge_results_keeps_graph_and_vector_contracts():
    results = merge_results([_graph_result()], [_vector_result()])

    assert {result.source for result in results} == {"graph", "vector"}
    graph_result = next(result for result in results if result.source == "graph")
    vector_result = next(result for result in results if result.source == "vector")
    assert graph_result.metadata["hop_count"] == 2
    assert vector_result.score == 0.91


def test_merge_results_reorders_by_hybrid_score_and_preserves_evidence():
    low_confidence_graph = _graph_result("낮은 신뢰도 경로")
    low_confidence_graph.metadata["confidence"] = 0.2
    high_score_vector = _vector_result("CHK_HIGH")

    results = merge_results([low_confidence_graph], [high_score_vector])

    assert results[0].source == "vector"
    assert results[0].chunk_ids == ["CHK_HIGH"]
    assert "hybrid_score" in results[0].metadata


def test_build_context_contains_path_and_all_evidence_identifiers():
    result = _graph_result()
    result.source_ids = ["SRC_1"]
    result.chunk_ids = ["CHK_1"]

    context = build_context([result])

    assert "마늘 → 대서마늘 → 남부 지역" in context
    assert "HAS_VARIETY → SUITABLE_FOR_REGION" in context
    assert "EVD_" in context
    assert "SRC_1" in context
    assert "CHK_1" in context


def test_hybrid_search_calls_both_searchers_and_passes_context():
    calls = []

    def graph_search(question, **kwargs):
        calls.append(("graph", question, kwargs))
        return [_graph_result()]

    def vector_search(question, **kwargs):
        calls.append(("vector", question, kwargs))
        return [_vector_result()]

    results = hybrid_search(
        "마늘 적합 지역과 재배 방법은?",
        graph_search_fn=graph_search,
        vector_search_fn=vector_search,
        crop="마늘",
        max_hops=3,
        limit=5,
    )

    assert [call[0] for call in calls] == ["graph", "vector"]
    assert calls[0][2] == {"crop": "마늘", "max_hops": 3, "limit": 10}
    assert calls[1][2] == {"k": 10}
    assert len(results) == 2
