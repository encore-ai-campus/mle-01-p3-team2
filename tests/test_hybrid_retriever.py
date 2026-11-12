from src.retrievers.hybrid_retriever import hybrid_search, merge_results
from src.retrievers.search_result import SearchResult


def _graph_result(text="마늘 → 대서마늘 → 남부 지역"):
    return SearchResult(
        source="graph",
        text=text,
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

    assert [result.source for result in results] == ["graph", "vector"]
    assert results[0].metadata["hop_count"] == 2
    assert results[1].score == 0.91


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
