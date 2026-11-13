from dataclasses import dataclass

from src.retrievers.graph_retriever import GraphRetriever, graph_search, path_to_result
from src.retrievers.search_result import SearchResult


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
    assert result.path_nodes == [
        {"entity_id": "CRP_1", "name": "마늘", "entity_type": "Crop"},
        {"entity_id": "VAR_1", "name": "대서마늘", "entity_type": "Variety"},
        {"entity_id": "REG_1", "name": "남부", "entity_type": "Region"},
    ]
    assert result.hop_count == 2
    assert result.metadata["hop_count"] == 2
    assert result.metadata["node_types"] == ["Crop", "Variety", "Region"]
    assert result.relation_types == ["HAS_VARIETY", "SUITABLE_FOR_REGION"]
    assert result.path_edges == [
        {
            "hop": 1,
            "relation_id": "REL_1",
            "relation_type": "HAS_VARIETY",
            "from_entity_id": "CRP_1",
            "to_entity_id": "VAR_1",
            "evidence_ids": ["EVD_1"],
            "chunk_ids": ["CHK_1"],
            "source_ids": ["SRC_1"],
            "original_chunks": [],
        },
        {
            "hop": 2,
            "relation_id": "REL_2",
            "relation_type": "SUITABLE_FOR_REGION",
            "from_entity_id": "VAR_1",
            "to_entity_id": "REG_1",
            "evidence_ids": ["EVD_2"],
            "chunk_ids": ["CHK_2"],
            "source_ids": ["SRC_2"],
            "original_chunks": [],
        },
    ]
    assert result.metadata["relation_ids"] == ["REL_1", "REL_2"]
    assert result.evidence_ids == ["EVD_1", "EVD_2"]
    assert result.source_ids == ["SRC_1", "SRC_2"]
    assert result.chunk_ids == ["CHK_1", "CHK_2"]
    assert result.metadata["confidence"] == 0.9
    assert result.evidence_id is None
    assert result.chunk_id is None
    assert result.warnings == []


def test_path_to_result_exposes_single_hop_ids():
    path = FakePath(
        nodes=[
            FakeNode(labels={"Crop"}, entity_id="CRP_1", name="Crop"),
            FakeNode(labels={"Region"}, entity_id="REG_1", name="Region"),
        ],
        relationships=[FakeRelationship(
            "SUITABLE_FOR_REGION",
            relation_id="REL_1",
            evidence_ids=["EVD_1"],
            source_ids=["SRC_1"],
            source_chunk_ids=["CHK_1"],
        )],
    )

    result = path_to_result(path)

    assert result.source == "graph"
    assert result.evidence_id == "EVD_1"
    assert result.chunk_id == "CHK_1"
    assert result.path == ["Crop", "Region"]
    assert [node["entity_id"] for node in result.path_nodes] == ["CRP_1", "REG_1"]
    assert result.relation_types == ["SUITABLE_FOR_REGION"]
    assert result.path_edges[0]["evidence_ids"] == ["EVD_1"]
    assert result.path_edges[0]["chunk_ids"] == ["CHK_1"]
    assert result.hop_count == 1
    assert result.metadata["hop_count"] == 1


class FakeResult:
    def __init__(self, rows=()):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


class FakeSession:
    def __init__(self, rows=(), chunk_rows=()):
        self.query = None
        self.params = None
        self.rows = rows
        self.chunk_rows = chunk_rows
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def run(self, query, params=None):
        params = params or {}
        self.calls.append((query, params))
        if query.startswith("MATCH (n:Crop)"):
            crop = params.get("crop")
            return FakeResult([{"entity_id": "CRP_FAKE", "name": crop, "attributes": "{}"}]
                              if crop else [])
        if query.startswith("MATCH (chunk:Chunk)"):
            return FakeResult(self.chunk_rows)
        self.query = query
        self.params = params
        return FakeResult(self.rows)


class FakeDriver:
    def __init__(self, rows=(), chunk_rows=()):
        self.session_instance = FakeSession(rows, chunk_rows)

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
    assert driver.session_instance.params["candidate_limit"] == 10
    assert driver.session_instance.params["start_name"] == "마늘"


def test_search_paths_supports_three_hops_and_candidate_pool():
    driver = FakeDriver()
    GraphRetriever(driver).search_paths(
        start_name="마늘", min_hops=2, max_hops=3,
        question="마늘 품종과 지역", candidate_limit=30, limit=10,
    )
    assert "*2..3" in driver.session_instance.query
    assert driver.session_instance.params["candidate_limit"] == 90


def test_graph_search_returns_shared_search_result_from_path():
    path = FakePath(
        nodes=[
            FakeNode(labels={"Crop"}, entity_id="CRP_1", name="마늘"),
            FakeNode(labels={"Region"}, entity_id="REG_1", name="남부"),
        ],
        relationships=[FakeRelationship(
            "SUITABLE_FOR_REGION",
            relation_id="REL_1",
            evidence_ids=["EVD_1"],
            source_ids=["SRC_1"],
            source_chunk_ids=["CHK_1"],
        )],
    )
    driver = FakeDriver(
        [{"path": path}],
        [{"chunk_id": "CHK_1", "evidence_id": "EVD_1",
          "source_id": "SRC_1", "text": "남부 재배 원문"}],
    )

    results = graph_search(
        "마늘 품종과 지역",
        retriever=GraphRetriever(driver),
        crop="마늘",
        max_hops=2,
        limit=5,
    )

    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].source == "graph"
    assert results[0].text == "마늘 → 남부"
    assert results[0].evidence_id == "EVD_1"
    assert results[0].chunk_id == "CHK_1"
    assert results[0].path_nodes[0]["entity_id"] == "CRP_1"
    assert results[0].relation_types == ["SUITABLE_FOR_REGION"]
    assert results[0].path_edges[0]["relation_id"] == "REL_1"
    assert results[0].path_edges[0]["original_chunks"][0]["text"] == "남부 재배 원문"
    assert results[0].hop_count == 1
    assert results[0].metadata["hop_count"] == 1
    assert driver.session_instance.params["candidate_limit"] == 90
    assert driver.session_instance.params["start_name"] == "마늘"


def test_graph_search_rejects_blank_question():
    import pytest

    with pytest.raises(ValueError, match="question"):
        graph_search(" ", retriever=GraphRetriever(FakeDriver()))


def test_two_hop_search_links_original_text_to_each_relation_in_one_chunk_query():
    path = FakePath(
        nodes=[
            FakeNode(labels={"Crop"}, entity_id="CRP_1", name="마늘"),
            FakeNode(labels={"Variety"}, entity_id="VAR_1", name="대서마늘"),
            FakeNode(labels={"Region"}, entity_id="REG_1", name="남부"),
        ],
        relationships=[
            FakeRelationship(
                "HAS_VARIETY", relation_id="REL_1", evidence_ids=["EVD_1"],
                source_ids=["SRC_1"], source_chunk_ids=["CHK_1"],
            ),
            FakeRelationship(
                "SUITABLE_FOR_REGION", relation_id="REL_2",
                evidence_ids=["EVD_2"], source_ids=["SRC_2"],
                source_chunk_ids=["CHK_2"],
            ),
        ],
    )
    driver = FakeDriver(
        [{"path": path}],
        [
            {"chunk_id": "CHK_1", "evidence_id": "EVD_1",
             "source_id": "SRC_1", "text": "마늘 품종 원문"},
            {"chunk_id": "CHK_2", "evidence_id": "EVD_2",
             "source_id": "SRC_2", "text": "대서마늘 지역 원문"},
        ],
    )

    results = GraphRetriever(driver).search_paths(min_hops=2, max_hops=2)

    assert len(results) == 1
    assert results[0].hop_count == 2
    assert [edge["original_chunks"][0]["text"] for edge in results[0].path_edges] == [
        "마늘 품종 원문", "대서마늘 지역 원문"
    ]
    assert [edge["original_chunks"][0]["evidence_id"] for edge in results[0].path_edges] == [
        "EVD_1", "EVD_2"
    ]
    assert results[0].warnings == []
    assert len(driver.session_instance.calls) == 2
    assert driver.session_instance.calls[1][1] == {"chunk_ids": ["CHK_1", "CHK_2"]}


def test_graph_search_warns_when_relation_chunk_is_missing():
    path = FakePath(
        nodes=[
            FakeNode(labels={"Crop"}, entity_id="CRP_1", name="마늘"),
            FakeNode(labels={"Region"}, entity_id="REG_1", name="남부"),
        ],
        relationships=[FakeRelationship(
            "SUITABLE_FOR_REGION", relation_id="REL_1",
            evidence_ids=["EVD_1"], source_ids=["SRC_1"],
            source_chunk_ids=["CHK_MISSING"],
        )],
    )

    result = GraphRetriever(FakeDriver([{"path": path}])).search_paths()[0]

    assert result.path_edges[0]["original_chunks"] == []
    assert "REL_1" in result.warnings[0]
    assert "CHK_MISSING" in result.warnings[0]


def test_graph_search_warns_when_relation_has_no_source_chunk_ids():
    path = FakePath(
        nodes=[
            FakeNode(labels={"Crop"}, entity_id="CRP_1", name="마늘"),
            FakeNode(labels={"Region"}, entity_id="REG_1", name="남부"),
        ],
        relationships=[FakeRelationship(
            "SUITABLE_FOR_REGION", relation_id="REL_1",
            evidence_ids=["EVD_1"], source_chunk_ids=[],
        )],
    )
    driver = FakeDriver([{"path": path}])

    result = GraphRetriever(driver).search_paths()[0]

    assert result.path_edges[0]["original_chunks"] == []
    assert result.warnings == [
        "Relation REL_1 has no linked source chunks",
        "Graph path has no original evidence",
    ]
    assert len(driver.session_instance.calls) == 1


def test_graph_search_does_not_present_invalid_chunks_as_original_evidence():
    path = FakePath(
        nodes=[
            FakeNode(labels={"Crop"}, entity_id="CRP_1", name="마늘"),
            FakeNode(labels={"Region"}, entity_id="REG_1", name="남부"),
        ],
        relationships=[FakeRelationship(
            "SUITABLE_FOR_REGION", relation_id="REL_1",
            evidence_ids=["EVD_1"], source_ids=["SRC_1"],
            source_chunk_ids=["CHK_EMPTY", "CHK_WRONG"],
        )],
    )
    driver = FakeDriver(
        [{"path": path}],
        [
            {"chunk_id": "CHK_EMPTY", "evidence_id": "EVD_1",
             "source_id": "SRC_1", "text": " "},
            {"chunk_id": "CHK_WRONG", "evidence_id": "EVD_2",
             "source_id": "SRC_1", "text": "다른 근거의 원문"},
        ],
    )

    result = GraphRetriever(driver).search_paths()[0]

    assert result.path_edges[0]["original_chunks"] == []
    assert any("empty original text" in warning for warning in result.warnings)
    assert any("evidence does not match" in warning for warning in result.warnings)


def test_two_hop_path_reports_partial_original_evidence():
    path = FakePath(
        nodes=[
            FakeNode(labels={"Crop"}, entity_id="CRP_1", name="마늘"),
            FakeNode(labels={"Variety"}, entity_id="VAR_1", name="대서마늘"),
            FakeNode(labels={"Region"}, entity_id="REG_1", name="남부"),
        ],
        relationships=[
            FakeRelationship(
                "HAS_VARIETY", relation_id="REL_1", evidence_ids=["EVD_1"],
                source_chunk_ids=["CHK_1"],
            ),
            FakeRelationship(
                "SUITABLE_FOR_REGION", relation_id="REL_2", evidence_ids=["EVD_2"],
                source_chunk_ids=["CHK_MISSING"],
            ),
        ],
    )
    driver = FakeDriver(
        [{"path": path}],
        [{"chunk_id": "CHK_1", "evidence_id": "EVD_1",
          "source_id": "SRC_1", "text": "품종 원문"}],
    )

    result = GraphRetriever(driver).search_paths(min_hops=2, max_hops=2)[0]

    assert result.path_edges[0]["original_chunks"][0]["text"] == "품종 원문"
    assert result.path_edges[1]["original_chunks"] == []
    assert "Graph path has partial original evidence" in result.warnings


def test_explicit_variety_region_question_prioritizes_named_two_hop_path():
    driver = FakeDriver()
    retriever = GraphRetriever(driver)

    retriever.search(
        "마늘 의성종 품종은 어떤 지역에 적합한가요? 2홉 경로를 알려주세요",
        crop="마늘", max_hops=2, limit=10,
    )

    query, params = next((q, p) for q, p in driver.session_instance.calls if "MATCH path =" in q)
    assert "*2..2" in query
    assert ":`HAS_VARIETY`|`SUITABLE_FOR_REGION`" in query
    assert "node_score" in query
    assert "$question CONTAINS node.name" not in query
    assert params["terms"]
    assert len(driver.session_instance.calls) >= 2  # crop anchor lookup + path query


def test_graph_search_keeps_a_two_hop_result_when_named_path_exists():
    path = FakePath(
        nodes=[
            FakeNode(labels={"Crop"}, entity_id="C", name="마늘"),
            FakeNode(labels={"Variety"}, entity_id="V", name="의성종"),
            FakeNode(labels={"Region"}, entity_id="R", name="경북 의성 및 영남 내륙"),
        ],
        relationships=[
            FakeRelationship("HAS_VARIETY", relation_id="R1", evidence_ids=["E1"], source_chunk_ids=["K1"]),
            FakeRelationship("SUITABLE_FOR_REGION", relation_id="R2", evidence_ids=["E2"], source_chunk_ids=["K2"]),
        ],
    )
    driver = FakeDriver(
        rows=[{"path": path}],
        chunk_rows=[
            {"chunk_id": "K1", "evidence_id": "E1", "source_id": "S1", "text": "의성종은 마늘 품종이다."},
            {"chunk_id": "K2", "evidence_id": "E2", "source_id": "S2", "text": "의성종은 의성 지역에 적합하다."},
        ],
    )

    results = graph_search(
        "마늘 의성종 품종은 어떤 지역에 적합한가요? 2홉 경로를 알려주세요",
        retriever=GraphRetriever(driver), crop="마늘", max_hops=2, limit=10,
    )

    assert results[0].hop_count == 2
    assert [edge["original_chunks"][0]["text"] for edge in results[0].path_edges] == [
        "의성종은 마늘 품종이다.", "의성종은 의성 지역에 적합하다.",
    ]


def test_variety_group_region_question_uses_two_hop_search():
    driver = FakeDriver()

    GraphRetriever(driver).search(
        "중만생종 양파는 어느 지역에서 재배하기에 적합한가요?",
        crop="양파",
        max_hops=2,
        limit=10,
    )

    query, params = next((q, p) for q, p in driver.session_instance.calls if "MATCH path =" in q)
    assert "*2..2" in query
    assert ":`HAS_VARIETY`|`SUITABLE_FOR_REGION`" in query
    assert "양파" in params["terms"]
