import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from aura_graph import build_graph_elements, fetch_graph_paths, search_companies, serialize_path
from app import render_aura_graph_html


def test_build_graph_elements_merges_same_node_from_two_paths():
    paths = [
        {
            "nodes": [
                {"id": "company:a", "name": "A사", "labels": ["ParentCompany"]},
                {"id": "company:b", "name": "B사", "labels": ["ParentCompany"]},
            ],
            "relationships": [
                {"id": "rel:1", "type": "AFFILIATED_WITH", "start": "company:a", "end": "company:b"}
            ],
        },
        {
            "nodes": [
                {"id": "company:a", "name": "A사", "labels": ["ParentCompany"]},
                {"id": "company:c", "name": "C사", "labels": ["SubsidiaryCompany"]},
            ],
            "relationships": [
                {"id": "rel:2", "type": "HAS_SUBSIDIARY", "start": "company:a", "end": "company:c"}
            ],
        },
    ]

    nodes, edges = build_graph_elements(paths, center_id="company:a")

    assert {node["id"] for node in nodes} == {"company:a", "company:b", "company:c"}
    assert len(edges) == 2
    assert next(node for node in nodes if node["id"] == "company:a")["level"] == 0


def test_build_graph_elements_marks_second_hop_as_level_two():
    paths = [
        {
            "nodes": [
                {"id": "company:a", "name": "A사", "labels": ["ParentCompany"]},
                {"id": "company:b", "name": "B사", "labels": ["ParentCompany"]},
                {"id": "company:c", "name": "C사", "labels": ["SubsidiaryCompany"]},
            ],
            "relationships": [
                {"id": "rel:1", "type": "AFFILIATED_WITH", "start": "company:a", "end": "company:b"},
                {"id": "rel:2", "type": "HAS_SUBSIDIARY", "start": "company:b", "end": "company:c"},
            ],
        }
    ]

    nodes, _ = build_graph_elements(paths, center_id="company:a")
    levels = {node["id"]: node["level"] for node in nodes}

    assert levels == {"company:a": 0, "company:b": 1, "company:c": 2}


class FakeSearchDriver:
    def execute_query(self, query, **kwargs):
        self.query = query
        self.kwargs = kwargs
        return (
            [
                {
                    "id": "parent_company:1",
                    "name": "롯데쇼핑(주)",
                    "labels": ["ParentCompany"],
                }
            ],
            None,
            None,
        )


def test_search_companies_uses_parameterized_aura_query():
    driver = FakeSearchDriver()

    results = search_companies(driver, phrase="롯데", database="e8fbc6d2")

    assert results == [
        {
            "id": "parent_company:1",
            "name": "롯데쇼핑(주)",
            "labels": ["ParentCompany"],
        }
    ]
    assert driver.kwargs["phrase"] == "롯데"
    assert driver.kwargs["limit"] == 20
    assert driver.kwargs["database_"] == "e8fbc6d2"
    assert "CONTAINS" in driver.query


class FakeNode(dict):
    def __init__(self, node_id, name, label):
        super().__init__(id=node_id, name=name)
        self.labels = frozenset([label])


class FakeRelationship(dict):
    def __init__(self, relationship_id, relationship_type, start_node, end_node):
        super().__init__()
        self.element_id = relationship_id
        self.type = relationship_type
        self.start_node = start_node
        self.end_node = end_node


class FakePath:
    def __init__(self, nodes, relationships):
        self.nodes = nodes
        self.relationships = relationships


def test_serialize_path_keeps_business_ids_and_relationship_direction():
    center = FakeNode("parent_company:1", "A사", "ParentCompany")
    child = FakeNode("subsidiary_company:2", "B사", "SubsidiaryCompany")
    path = FakePath(
        [center, child],
        [FakeRelationship("rel:1", "HAS_SUBSIDIARY", center, child)],
    )

    assert serialize_path(path) == {
        "nodes": [
            {"id": "parent_company:1", "name": "A사", "labels": ["ParentCompany"]},
            {"id": "subsidiary_company:2", "name": "B사", "labels": ["SubsidiaryCompany"]},
        ],
        "relationships": [
            {
                "id": "rel:1",
                "type": "HAS_SUBSIDIARY",
                "start": "parent_company:1",
                "end": "subsidiary_company:2",
            }
        ],
    }


class FakePathDriver:
    def __init__(self, path):
        self.path = path

    def execute_query(self, query, **kwargs):
        self.query = query
        self.kwargs = kwargs
        return ([{"path": self.path}], None, None)


def test_fetch_graph_paths_limits_two_hop_query_and_serializes_results():
    center = FakeNode("parent_company:1", "A사", "ParentCompany")
    child = FakeNode("subsidiary_company:2", "B사", "SubsidiaryCompany")
    path = FakePath(
        [center, child],
        [FakeRelationship("rel:1", "HAS_SUBSIDIARY", center, child)],
    )
    driver = FakePathDriver(path)

    paths = fetch_graph_paths(
        driver,
        center_id="parent_company:1",
        database="e8fbc6d2",
        path_limit=100,
        relationship_type="HAS_SUBSIDIARY",
    )

    assert paths[0]["nodes"][1]["id"] == "subsidiary_company:2"
    assert driver.kwargs == {
        "center_id": "parent_company:1",
        "path_limit": 100,
        "relationship_type": "HAS_SUBSIDIARY",
        "database_": "e8fbc6d2",
    }
    assert "[*1..2]" in driver.query
    assert "AFFILIATED_WITH" in driver.query
    assert "HAS_SUBSIDIARY" in driver.query


def test_render_aura_graph_html_enables_dragging_and_uses_company_name():
    markup = render_aura_graph_html(
        [
            {
                "id": "parent_company:1",
                "label": "롯데쇼핑(주)",
                "group": "ParentCompany",
                "level": 0,
            }
        ],
        [],
    )

    assert "롯데쇼핑(주)".encode("unicode_escape").decode() in markup
    assert "dragNodes" in markup
    assert '"fixed": true' not in markup.lower()
    assert '"color": "#b4aaa1"' in markup.lower()


def test_render_aura_graph_html_uses_fixed_rose_periwinkle_tones_for_node_importance():
    markup = render_aura_graph_html(
        [
            {
                "id": "parent_company:1",
                "label": "A사",
                "group": "ParentCompany",
                "level": 0,
            },
            {
                "id": "subsidiary_company:2",
                "label": "B사",
                "group": "ParentCompany",
                "level": 1,
            },
            {
                "id": "subsidiary_company:3",
                "label": "C사",
                "group": "SubsidiaryCompany",
                "level": 1,
            },
        ],
        [
            {
                "source": "parent_company:1",
                "target": "subsidiary_company:2",
                "relation": "AFFILIATED_WITH",
            },
            {
                "source": "parent_company:1",
                "target": "subsidiary_company:3",
                "relation": "HAS_SUBSIDIARY",
            }
        ],
    )

    assert '"color": "#b4aaa1"' in markup.lower()
    assert "#f4b6c2" in markup.lower()
    assert "#b8c0ff" in markup.lower()


def test_render_aura_graph_html_uses_fixed_rose_periwinkle_colors():
    markup = render_aura_graph_html(
        [
            {
                "id": "parent_company:1",
                "label": "A사",
                "group": "ParentCompany",
                "level": 0,
            },
            {
                "id": "subsidiary_company:2",
                "label": "B사",
                "group": "SubsidiaryCompany",
                "level": 1,
            },
            {
                "id": "parent_company:3",
                "label": "C사",
                "group": "ParentCompany",
                "level": 1,
            },
        ],
        [
            {
                "source": "parent_company:1",
                "target": "subsidiary_company:2",
                "relation": "HAS_SUBSIDIARY",
            },
            {
                "source": "parent_company:1",
                "target": "parent_company:3",
                "relation": "AFFILIATED_WITH",
            }
        ],
    )

    assert "#f4b6c2" in markup.lower()
    assert "#b8c0ff" in markup.lower()
    assert "#cfccc4" in markup.lower()
