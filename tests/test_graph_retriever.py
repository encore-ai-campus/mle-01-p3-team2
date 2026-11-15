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
    def __init__(self, rows=(), chunk_rows=(), pest_rows=()):
        self.query = None
        self.params = None
        self.rows = rows
        self.chunk_rows = chunk_rows
        self.pest_rows = pest_rows
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
        if query.startswith("MATCH (n:Insect|Disease)"):
            return FakeResult(self.pest_rows)
        self.query = query
        self.params = params
        return FakeResult(self.rows)


class FakeDriver:
    def __init__(self, rows=(), chunk_rows=(), pest_rows=()):
        self.session_instance = FakeSession(rows, chunk_rows, pest_rows)

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


def test_named_pest_question_prioritizes_pesticide_relations():
    """분류어("해충") 없이 해충 이름만 적은 질문도 약제 2-hop 경로를 먼저 찾습니다."""

    driver = FakeDriver(pest_rows=[{"name": "파밤나방"}, {"name": "노균병"}])
    retriever = GraphRetriever(driver)

    retriever.search(
        "배추에서 파밤나방을 방제하려면 어떤 등록 약제를 사용하나요?",
        crop="배추", max_hops=2, limit=10,
    )

    query, _ = next(
        (q, p) for q, p in driver.session_instance.calls if "MATCH path =" in q
    )
    assert "*2..2" in query
    assert ":`HAS_REGISTERED_PESTICIDE_USE`|`CONTROLS_TARGET`" in query


def test_pesticide_question_without_known_pest_name_stays_generic():
    """그래프에 없는 대상이면 약제 관계를 강제하지 않고 일반 검색으로 갑니다."""

    driver = FakeDriver(pest_rows=[{"name": "파밤나방"}])
    retriever = GraphRetriever(driver)

    retriever.search(
        "배추에 쓸 수 있는 약제를 알려주세요",
        crop="배추", max_hops=2, limit=10,
    )

    query, _ = next(
        (q, p) for q, p in driver.session_instance.calls if "MATCH path =" in q
    )
    assert "*1..2" in query
    assert "HAS_REGISTERED_PESTICIDE_USE" not in query


def test_pest_names_are_read_from_the_graph_only_once():
    """이름 목록은 검색마다 다시 읽지 않습니다."""

    driver = FakeDriver(pest_rows=[{"name": "파밤나방"}])
    retriever = GraphRetriever(driver)

    for _ in range(3):
        retriever.search(
            "배추 파밤나방 방제 약제", crop="배추", max_hops=2, limit=10,
        )

    pest_lookups = [
        query
        for query, _ in driver.session_instance.calls
        if query.startswith("MATCH (n:Insect|Disease)")
    ]
    assert len(pest_lookups) == 1


# --- 라우터 관계 타입·hop 수가 하드 필터로 작동해 2-hop 후보가 사라지던 회귀 ---
# (multi-hop Top-5 미스 16건 진단: docs/multihop-top5-miss-analysis.md)


def _path_queries(driver):
    return [query for query, _ in driver.session_instance.calls if "MATCH path =" in query]


def test_two_hop_search_relaxes_router_relation_types_when_nothing_matches():
    """라우터가 이어지지 않는 관계 조합을 주면 타입 필터를 풀고 다시 찾는다."""

    driver = FakeDriver(pest_rows=[{"name": "벼룩잎벌레"}])
    retriever = GraphRetriever(driver)

    retriever.search(
        "배추에 발생한 벼룩잎벌레를 방제하려면 등록된 약제와 사용 방법이 어떻게 되나요?",
        crop="배추",
        max_hops=2,
        # Crop에서 나가는 간선끼리의 조합이라 2-hop으로 이어지지 않는다.
        relation_types=["HAS_REGISTERED_PESTICIDE_USE", "AFFECTED_BY_INSECT"],
        limit=10,
    )

    queries = _path_queries(driver)
    two_hop = [query for query in queries if "*2..2" in query]
    assert len(two_hop) >= 3
    assert ":`HAS_REGISTERED_PESTICIDE_USE`|`AFFECTED_BY_INSECT`" in two_hop[0]
    assert ":`HAS_REGISTERED_PESTICIDE_USE`|`CONTROLS_TARGET`" in two_hop[1]
    # 마지막 2-hop 시도는 관계 타입을 제한하지 않는다.
    assert "-[*2..2]->" in two_hop[2]


def test_pesticide_question_searches_two_hops_even_when_router_says_one_hop():
    """약제 질문에 max_hops=1이 와도 2-hop 탐색을 보장한다."""

    driver = FakeDriver(pest_rows=[{"name": "벼룩잎벌레"}])
    retriever = GraphRetriever(driver)

    retriever.search(
        "배추 벼룩잎벌레 방제 약제의 사용 기준을 알려주세요",
        crop="배추",
        max_hops=1,
        relation_types=["HAS_REGISTERED_PESTICIDE_USE"],
        limit=10,
    )

    assert any("*2..2" in query for query in _path_queries(driver))
    assert retriever.last_search_diagnostics["max_hops_used"] == 2


def test_one_hop_search_drops_relation_filter_when_it_returns_nothing():
    """시작 라벨에서 나갈 수 없는 관계 타입만 지정되면 필터를 버리고 재탐색한다."""

    driver = FakeDriver()
    retriever = GraphRetriever(driver)

    retriever.search(
        "배추에 등록된 플루페녹수론 약제는 어떤 피해를 방제하는 데 쓰이나요?",
        crop="배추",
        max_hops=1,
        # Crop에는 CONTROLS_TARGET 간선이 없어 1-hop 결과가 공집합이 된다.
        relation_types=["CONTROLS_TARGET"],
        limit=10,
    )

    one_hop = [query for query in _path_queries(driver) if "*1..1" in query]
    assert len(one_hop) == 2
    assert ":`CONTROLS_TARGET`" in one_hop[0]
    assert "-[*1..1]->" in one_hop[1]


# --- 이름 정확 일치·사용 조건 기반 재랭킹 (docs/multihop-top5-miss-analysis.md 8절) ---

from src.retrievers.graph_retriever import (  # noqa: E402
    _condition_value_score,
    _name_match_bonus,
    _normalize,
    _path_relevance_score,
    _requested_condition_values,
)


def _pesticide_path(product, *, brand="", dilution="1000", use_count="3",
                    use_timing="수확 7일 전까지", application_method="발생 초기 7일간격"):
    nodes = [
        {"entity_id": "CRP_1", "name": "배추", "entity_type": "Crop"},
        {"entity_id": "PST_1", "name": f"{product} - 벼룩잎벌레", "entity_type": "PesticideUse"},
        {"entity_id": "INS_1", "name": "벼룩잎벌레", "entity_type": "Insect"},
    ]
    edges = [
        {"hop": 1, "relation_id": "REL_1", "relation_type": "HAS_REGISTERED_PESTICIDE_USE",
         "from_entity_id": "CRP_1", "to_entity_id": "PST_1", "original_chunks": []},
        {"hop": 2, "relation_id": "REL_2", "relation_type": "CONTROLS_TARGET",
         "from_entity_id": "PST_1", "to_entity_id": "INS_1", "original_chunks": []},
    ]
    attributes = [
        {"crop": "배추"},
        {"product": product, "brand": brand, "crop": "배추", "target_name": "벼룩잎벌레",
         "dilution": dilution, "use_count": use_count, "use_timing": use_timing,
         "application_method": application_method, "registration_no": "12-배추-3"},
        {"crop": "배추"},
    ]
    return nodes, edges, attributes


def _pesticide_score(question, product, **kwargs):
    nodes, edges, attributes = _pesticide_path(product, **kwargs)
    return _path_relevance_score(nodes, edges, question, attributes)


def test_name_match_bonus_ranks_exact_above_alias_above_partial():
    """완전 일치 > 별칭 일치 > 부분 일치 순으로 보너스를 준다."""
    question = _normalize("배추에 등록된 모스피란 수화제의 사용 기준은?")
    terms = ["모스피란", "수화제", "배추"]

    exact = _name_match_bonus(question, terms, "모스피란 수화제 - 벼룩잎벌레",
                              {"product": "모스피란 수화제"})
    alias = _name_match_bonus(question, terms, "다른 약제",
                              {"product": "다른 약제", "aliases": ["모스피란"]})
    partial = _name_match_bonus(question, terms, "모스피란유 혼합 제품군 전용",
                                {"product": "혼합 제품군 전용 모스피란유제품"})
    none = _name_match_bonus(question, terms, "전혀 다른 이름", {"product": "전혀 다른 이름"})

    assert exact > alias > partial > none == 0.0


def test_name_match_bonus_counts_crop_match_separately():
    """작물명만 맞는 경우에는 식별 보너스보다 작은 값만 준다."""
    question = _normalize("배추에 쓰는 약제를 알려주세요")
    crop_only = _name_match_bonus(question, ["배추"], "어떤 약제", {"crop": "배추"})
    assert 0.0 < crop_only < 6.0


def test_requested_condition_values_reads_only_question_text():
    """질문에 적힌 값만 추출한다(Gold 값 사용 금지)."""
    requested = _requested_condition_values(
        "1000배로 희석해 10일 간격으로 최대 3회, 수확 7일 전까지 쓰려면?"
    )
    assert requested["dilution"] == 1000
    assert requested["interval_days"] == 10
    assert requested["use_count"] == 3
    assert requested["harvest_days"] == 7
    assert _requested_condition_values("배추 약제를 알려주세요") == {}


def test_condition_value_score_rewards_matching_values_only():
    """값이 맞으면 가산, 틀리면 감산한다."""
    requested = {"dilution": 1000, "use_count": 3}
    match = _condition_value_score(requested, {"dilution": "1000", "use_count": "3"})
    mismatch = _condition_value_score(requested, {"dilution": "3000", "use_count": "2"})
    assert match > 0 > mismatch


def test_named_product_outranks_sibling_path_with_same_shape():
    """질문이 품목명을 말하면 같은 모양의 형제 경로보다 높은 점수를 받는다."""
    question = "배추에 등록된 모스피란 수화제로 벼룩잎벌레를 방제하려면 어떻게 하나요?"
    named = _pesticide_score(question, "모스피란 수화제")
    sibling = _pesticide_score(question, "아바멕틴 유제")
    assert named > sibling


def test_complete_use_record_outranks_one_missing_requested_field():
    """질문이 요구한 사용 기준 필드를 모두 갖춘 기록을 우대한다."""
    question = "배추 벼룩잎벌레 약제의 희석배수와 최대 사용횟수, 수확 전 사용시기를 알려주세요"
    complete = _pesticide_score(question, "가 약제")
    missing = _pesticide_score(question, "가 약제", dilution="", use_count="")
    assert complete > missing


def test_question_stated_dilution_breaks_tie_between_sibling_paths():
    """질문에 적힌 희석배수와 같은 기록이 형제 경로보다 앞선다."""
    question = "배추 벼룩잎벌레 약제를 1000배로 희석해 쓰려면 어떤 약제를 쓰나요?"
    matching = _pesticide_score(question, "가 약제", dilution="1000")
    other = _pesticide_score(question, "가 약제", dilution="3000")
    assert matching > other
