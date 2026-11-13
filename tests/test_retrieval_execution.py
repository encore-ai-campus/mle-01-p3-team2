import pytest
from neo4j.exceptions import ServiceUnavailable

from src.retrievers import RetrievalResponse, SearchResult, execute_search


class StubVectorIndex:
    def __init__(self):
        self.calls = []

    def search(self, question, *, k):
        self.calls.append((question, k))
        return [SearchResult(
            source="vector", text="원문 근거", score=0.91,
            evidence_id="EVD_1", chunk_id="CHK_1", warnings=["원문 확인"],
        )]


class StubGraphRetriever:
    def __init__(self):
        self.calls = []

    def search(self, question, *, crop, max_hops, limit):
        self.calls.append((question, crop, max_hops, limit))
        return [SearchResult(
            source="graph", text="마늘 → 남부", path=["마늘", "남부"],
            evidence_id="EVD_1", chunk_id="CHK_1",
        )]


def test_execute_vector_route_returns_common_response():
    vector = StubVectorIndex()

    response = execute_search(
        "재배 방법은?", route="VECTOR", vector_index=vector,
        limit=3, reason="원문 질문", warnings=["라우터 확인"],
    )

    assert isinstance(response, RetrievalResponse)
    assert response.route == "VECTOR"
    assert [hit.source for hit in response.results] == ["vector"]
    assert response.reason == "원문 질문"
    assert response.warnings == ["라우터 확인", "원문 확인"]
    assert vector.calls == [("재배 방법은?", 3)]


def test_execute_graph_route_passes_crop_and_hop_count():
    graph = StubGraphRetriever()

    response = execute_search(
        "마늘 적합 지역은?", route="GRAPH", graph_retriever=graph,
        crop="마늘", max_hops=3, limit=4,
    )

    assert response.route == "GRAPH"
    assert [hit.source for hit in response.results] == ["graph"]
    assert graph.calls == [("마늘 적합 지역은?", "마늘", 3, 4)]


def test_execute_hybrid_route_merges_both_results():
    graph = StubGraphRetriever()
    vector = StubVectorIndex()

    response = execute_search(
        "마늘 지역과 재배 근거는?", route="HYBRID",
        graph_retriever=graph, vector_index=vector,
        crop="마늘", graph_limit=4, vector_limit=3, limit=2,
    )

    assert response.route == "HYBRID"
    assert len(response.results) == 1
    assert response.results[0].source == "graph"
    assert response.results[0].metadata["matched_vector_hits"][0]["text"] == "원문 근거"
    assert response.warnings == ["원문 확인"]
    assert graph.calls == [("마늘 지역과 재배 근거는?", "마늘", 2, 4)]
    assert vector.calls == [("마늘 지역과 재배 근거는?", 3)]


def test_execute_search_warns_when_no_results_are_found():
    class EmptyVectorIndex:
        def search(self, question, *, k):
            return []

    response = execute_search(
        "재배 방법은?", route="VECTOR", vector_index=EmptyVectorIndex()
    )

    assert response.results == []
    assert response.warnings == ["VECTOR returned no results"]


def test_execute_search_records_connection_error_without_exposing_exception_text():
    class DisconnectedVectorIndex:
        def search(self, question, *, k):
            raise ServiceUnavailable("private endpoint details")

    response = execute_search(
        "재배 방법은?", route="VECTOR", vector_index=DisconnectedVectorIndex()
    )

    assert response.results == []
    assert "VECTOR connection error: ServiceUnavailable" in response.warnings
    assert "private endpoint details" not in str(response.warnings)


def test_execute_hybrid_route_returns_vector_hits_when_graph_connection_fails():
    class DisconnectedGraphRetriever:
        def search(self, question, *, crop, max_hops, limit):
            raise ServiceUnavailable("graph offline")

    response = execute_search(
        "마늘 지역과 재배 근거는?", route="HYBRID",
        graph_retriever=DisconnectedGraphRetriever(),
        vector_index=StubVectorIndex(),
    )

    assert [hit.source for hit in response.results] == ["vector"]
    assert "GRAPH connection error: ServiceUnavailable" in response.warnings
    assert "원문 확인" in response.warnings


def test_execute_hybrid_route_returns_graph_hits_when_vector_connection_fails():
    class DisconnectedVectorIndex:
        def search(self, question, *, k):
            raise ServiceUnavailable("vector offline")

    response = execute_search(
        "마늘 지역과 재배 근거는?", route="HYBRID",
        graph_retriever=StubGraphRetriever(),
        vector_index=DisconnectedVectorIndex(),
    )

    assert [hit.source for hit in response.results] == ["graph"]
    assert "VECTOR connection error: ServiceUnavailable" in response.warnings


def test_execute_graph_route_surfaces_partial_evidence_warning():
    class PartialGraphRetriever:
        def search(self, question, *, crop, max_hops, limit):
            return [SearchResult(
                source="graph", text="마늘 → 남부",
                warnings=["Graph path has partial original evidence"],
            )]

    response = execute_search(
        "마늘 지역은?", route="GRAPH", graph_retriever=PartialGraphRetriever()
    )

    assert response.warnings == ["Graph path has partial original evidence"]


def test_execute_search_does_not_mask_query_errors():
    class BrokenVectorIndex:
        def search(self, question, *, k):
            raise ValueError("invalid query")

    with pytest.raises(ValueError, match="invalid query"):
        execute_search("question", route="VECTOR", vector_index=BrokenVectorIndex())


@pytest.mark.parametrize(
    ("route", "kwargs", "message"),
    [
        ("VECTOR", {}, "vector_index"),
        ("GRAPH", {}, "graph_retriever"),
        ("HYBRID", {"vector_index": StubVectorIndex()}, "graph_retriever"),
        ("OTHER", {}, "Unsupported"),
    ],
)
def test_execute_search_rejects_missing_backends_or_unknown_route(route, kwargs, message):
    with pytest.raises(ValueError, match=message):
        execute_search("질문", route=route, **kwargs)
