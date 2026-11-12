from dataclasses import dataclass

from retrievers.graph_retriever import GraphRetriever, graph_search, path_to_result


class FakeNode(dict):
    def __init__(self, *, labels, **properties):
        super().__init__(properties)
        self.labels = set(labels)


class FakeRelationship(dict):
    def __init__(self, relation_type, **properties):
        super().__init__(properties)
        self.type = relation_type


@dataclass
class FakePath:
    nodes: list
    relationships: list


def test_path_to_result_preserves_path_and_evidence_metadata():
    path = FakePath(
        nodes=[
            FakeNode(labels={"Crop"}, entity_id="CRP_1", entity_type="Crop", name="마늘"),
            FakeNode(labels={"Variety"}, entity_id="VAR_1", entity_type="Variety", name="대서마늘"),
            FakeNode(labels={"Region"}, entity_id="REG_1", entity_type="Region", name="남부"),
        ],
        relationships=[
            FakeRelationship(
                "HAS_VARIETY",
                relation_id="REL_1",
                evidence_ids=["EVD_1"],
                source_ids=["SRC_1"],
                source_chunk_ids=["CHK_1"],
                confidence=0.9,
                retrieval_eligible=True,
            ),
            FakeRelationship(
                "SUITABLE_FOR_REGION",
                relation_id="REL_2",
                evidence_ids=["EVD_2"],
                source_ids=["SRC_2"],
                source_chunk_ids=["CHK_2"],
                confidence=1.0,
                retrieval_eligible=True,
            ),
        ],
    )

    result = path_to_result(path)

    assert result.source == "graph"
    assert result.text == "마늘 → 대서마늘 → 남부"
    assert result.score is None
    assert result.metadata["hop_count"] == 2
    assert result.metadata["node_types"] == ["Crop", "Variety", "Region"]
    assert result.relation_types == ["HAS_VARIETY", "SUITABLE_FOR_REGION"]
    assert result.metadata["relation_ids"] == ["REL_1", "REL_2"]
    assert result.evidence_ids == ["EVD_1", "EVD_2"]
    assert result.source_ids == ["SRC_1", "SRC_2"]
    assert result.chunk_ids == ["CHK_1", "CHK_2"]
    assert result.metadata["confidence"] == 0.9


class FakeResult:
    def __iter__(self):
        return iter([])


class FakeSession:
    def __init__(self):
        self.query = None
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def run(self, query, params):
        self.query = query
        self.params = params
        return FakeResult()


class FakeDriver:
    def __init__(self):
        self.session_instance = FakeSession()

    def session(self):
        return self.session_instance


def test_search_paths_uses_typed_directional_query():
    driver = FakeDriver()
    retriever = GraphRetriever(driver)

    results = retriever.search_paths(
        start_label="Crop",
        start_name="마늘",
        min_hops=2,
        max_hops=2,
        relation_types=["HAS_VARIETY", "SUITABLE_FOR_REGION"],
        limit=10,
    )

    assert results == []
    assert "(start:`Crop`)" in driver.session_instance.query
    assert ":`HAS_VARIETY`|`SUITABLE_FOR_REGION`" in driver.session_instance.query
    assert "->(target)" in driver.session_instance.query
    assert driver.session_instance.params == {"limit": 10, "start_name": "마늘"}


def test_graph_search_delegates_to_shared_retriever_interface():
    class StubRetriever:
        def search(self, question, *, crop, max_hops, limit):
            return [(question, crop, max_hops, limit)]

    assert graph_search(
        "마늘 품종과 지역",
        retriever=StubRetriever(),
        crop="마늘",
        max_hops=2,
        limit=5,
    ) == [("마늘 품종과 지역", "마늘", 2, 5)]
