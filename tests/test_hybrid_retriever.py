import pytest

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


def test_merge_results_keeps_vector_visible_when_graph_fills_result_limit():
    graph_results = [
        SearchResult(
            source="graph",
            text=f"graph path {index}",
            path=["crop", f"target {index}"],
            metadata={"confidence": 1.0, "hop_count": 2},
        )
        for index in range(10)
    ]
    vector_results = [
        SearchResult(
            source="vector",
            text=f"vector passage {index}",
            score=0.8 - index * 0.01,
            chunk_id=f"CHK_{index}",
        )
        for index in range(10)
    ]

    results = merge_results(graph_results, vector_results, limit=10)

    assert len(results) == 10
    assert any(result.source == "graph" for result in results)
    assert any(result.source == "vector" for result in results)


def test_build_context_contains_path_and_all_evidence_identifiers():
    result = _graph_result()
    result.source_ids = ["SRC_1"]
    result.chunk_ids = ["CHK_1"]
    result.path_edges = [
        {"hop": hop, "relation_type": relation_type,
         "from_entity_id": "CRP_1", "to_entity_id": "VAR_1",
         "relation_id": f"REL_{hop}",
         "original_chunks": [{"chunk_id": "CHK_1", "evidence_id": "EVD_1",
                              "source_id": "SRC_1", "text": "original passage"}]}
        for hop, relation_type in enumerate(result.relation_types, 1)
    ]

    context = build_context([result])

    assert "마늘 → 대서마늘 → 남부 지역" in context
    assert "HAS_VARIETY → SUITABLE_FOR_REGION" in context
    assert "EVD_" in context
    assert "SRC_1" in context
    assert "CHK_1" in context


def test_build_context_includes_each_hops_original_text():
    result = SearchResult(
        source="graph",
        text="마늘 → 품종 → 남부",
        path=["마늘", "품종", "남부"],
        relation_types=["HAS_VARIETY", "SUITABLE_FOR_REGION"],
        path_edges=[
            {
                "hop": 1, "relation_id": "REL_1", "relation_type": "HAS_VARIETY",
                "from_entity_id": "CRP_1", "to_entity_id": "VAR_1",
                "evidence_ids": ["EVD_1"], "chunk_ids": ["CHK_1"],
                "source_ids": ["SRC_1"],
                "original_chunks": [{"chunk_id": "CHK_1", "evidence_id": "EVD_1",
                                    "source_id": "SRC_1", "text": "품종 원문"}],
            },
            {
                "hop": 2, "relation_id": "REL_2",
                "relation_type": "SUITABLE_FOR_REGION",
                "from_entity_id": "VAR_1", "to_entity_id": "REG_1",
                "evidence_ids": ["EVD_2"], "chunk_ids": ["CHK_2"],
                "source_ids": ["SRC_2"],
                "original_chunks": [{"chunk_id": "CHK_2", "evidence_id": "EVD_2",
                                    "source_id": "SRC_2", "text": "지역 원문"}],
            },
        ],
    )

    context = build_context([result])

    assert "hop 1: HAS_VARIETY" in context
    assert "hop 2: SUITABLE_FOR_REGION" in context
    assert "품종 원문" in context
    assert "지역 원문" in context


def test_merge_results_deduplicates_shared_ids_and_keeps_original_passage():
    graph = SearchResult(
        source="graph",
        text="마늘 → 남부",
        path=["마늘", "남부"],
        relation_types=["SUITABLE_FOR_REGION"],
        evidence_ids=["EVD_1", "EVD_2"],
        chunk_ids=["CHK_1", "CHK_2"],
    )
    vector = SearchResult(
        source="vector",
        text="원문 근거",
        score=0.91,
        evidence_id="EVD_1",
        chunk_id="CHK_1",
    )

    second_chunk = SearchResult(
        source="vector", text="두 번째 원문", evidence_id="EVD_1", chunk_id="CHK_2"
    )

    results = merge_results([graph], [vector, vector, second_chunk], limit=1)

    assert len(results) == 1
    assert results[0].source == "graph"
    assert results[0].path == ["마늘", "남부"]
    assert results[0].metadata["matched_vector_hits"][0]["text"] == "원문 근거"
    assert results[0].metadata["matched_vector_hits"][0]["score"] == 0.91
    assert [hit["text"] for hit in results[0].metadata["matched_vector_hits"]] == [
        "원문 근거", "두 번째 원문"
    ]
    assert "matched_vector_hits" not in graph.metadata


def test_merge_results_deduplicates_vector_hits_by_evidence_or_chunk_id():
    first = SearchResult(
        source="vector", text="first", evidence_id="EVD_1", chunk_id="CHK_1"
    )
    same_evidence = SearchResult(
        source="vector", text="second", evidence_id="EVD_1", chunk_id="CHK_2"
    )
    same_chunk = SearchResult(
        source="vector", text="third", evidence_id="EVD_3", chunk_id="CHK_1"
    )
    distinct = SearchResult(
        source="vector", text="fourth", evidence_id="EVD_4", chunk_id="CHK_4"
    )

    results = merge_results([], [first, same_evidence, same_chunk, distinct])

    assert [result.text for result in results] == ["first", "fourth"]
    assert all("hybrid_score" in result.metadata for result in results)


def test_merge_results_matches_graph_by_either_id():
    graph = SearchResult(
        source="graph", text="A → B", path=["A", "B"],
        evidence_id="EVD_1", chunk_id="CHK_1",
    )
    evidence_match = SearchResult(
        source="vector", text="evidence match",
        evidence_id="EVD_1", chunk_id="CHK_2",
    )
    chunk_match = SearchResult(
        source="vector", text="chunk match",
        evidence_id="EVD_3", chunk_id="CHK_1",
    )

    results = merge_results([graph], [evidence_match, chunk_match])

    assert len(results) == 1
    assert [hit["text"] for hit in results[0].metadata["matched_vector_hits"]] == [
        "evidence match", "chunk match"
    ]


def test_merge_results_preserves_distinct_graph_paths_with_shared_evidence():
    first = SearchResult(
        source="graph", text="A → B", path=["A", "B"], evidence_id="EVD_1"
    )
    second = SearchResult(
        source="graph", text="A → C", path=["A", "C"], evidence_id="EVD_1"
    )

    results = merge_results([first, second], [])

    assert [result.path for result in results] == [["A", "B"], ["A", "C"]]


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


def test_hybrid_search_connects_retrievers_and_merges_shared_evidence():
    calls = []

    class StubGraphRetriever:
        def search(self, question, *, crop, max_hops, limit):
            calls.append(("graph", question, crop, max_hops, limit))
            return [SearchResult(
                source="graph", text="마늘 → 남부", path=["마늘", "남부"],
                evidence_id="EVD_1", chunk_id="CHK_1",
            )]

    class StubVectorIndex:
        def search(self, question, *, k):
            calls.append(("vector", question, k))
            return [SearchResult(
                source="vector", text="남부 지역 재배 원문", score=0.92,
                evidence_id="EVD_1", chunk_id="CHK_1",
            )]

    results = hybrid_search(
        "마늘 재배 지역은?",
        graph_retriever=StubGraphRetriever(),
        vector_index=StubVectorIndex(),
        crop="마늘",
        max_hops=2,
        graph_limit=4,
        vector_limit=3,
    )

    assert calls == [
        ("graph", "마늘 재배 지역은?", "마늘", 2, 4),
        ("vector", "마늘 재배 지역은?", 3),
    ]
    assert len(results) == 1
    assert results[0].source == "graph"
    assert results[0].metadata["matched_vector_hits"][0]["text"] == "남부 지역 재배 원문"


def test_hybrid_search_requires_both_search_backends():
    with pytest.raises(ValueError, match="graph_retriever"):
        hybrid_search("question")

    with pytest.raises(ValueError, match="vector_index"):
        hybrid_search("question", graph_search_fn=lambda *_args, **_kwargs: [])
