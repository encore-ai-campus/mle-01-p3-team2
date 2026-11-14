from __future__ import annotations

from typing import Any, Protocol


class AuraDriver(Protocol):
    def execute_query(self, query: str, **kwargs: Any) -> tuple[list[Any], Any, Any]: ...


COMPANY_SEARCH_QUERY = """
MATCH (node)
WHERE (node:ParentCompany OR node:SubsidiaryCompany)
  AND toLower(coalesce(node.name, "")) CONTAINS toLower($phrase)
RETURN node.id AS id, node.name AS name, labels(node) AS labels
ORDER BY node.name
LIMIT $limit
"""

GRAPH_PATH_QUERY = """
MATCH (center)
WHERE center.id = $center_id
MATCH path = (center)-[*1..2]-(connected)
WHERE ALL(relationship IN relationships(path)
    WHERE type(relationship) IN ["AFFILIATED_WITH", "HAS_SUBSIDIARY"])
  AND ($relationship_type IS NULL
       OR type(relationships(path)[0]) = $relationship_type)
RETURN path
LIMIT $path_limit
"""


def search_companies(
    driver: AuraDriver,
    phrase: str,
    database: str,
    limit: int = 20,
) -> list[dict[str, object]]:
    """회사명 일부와 일치하는 Aura 그래프 기업 후보를 반환한다."""
    phrase = phrase.strip()
    if not phrase:
        return []

    records, _, _ = driver.execute_query(
        COMPANY_SEARCH_QUERY,
        phrase=phrase,
        limit=limit,
        database_=database,
    )
    return [dict(record) for record in records]


def serialize_path(path: Any) -> dict[str, list[dict[str, object]]]:
    """Neo4j Path 객체를 화면과 테스트에서 공통으로 쓰는 딕셔너리로 변환한다."""
    nodes = []
    for node in path.nodes:
        node_id = node.get("id")
        if not node_id:
            raise ValueError("Graph node is missing its required id property")
        nodes.append(
            {
                "id": str(node_id),
                "name": str(node.get("name") or node_id),
                "labels": sorted(str(label) for label in node.labels),
            }
        )

    relationships = []
    for relationship in path.relationships:
        start_id = relationship.start_node.get("id")
        end_id = relationship.end_node.get("id")
        if not start_id or not end_id:
            raise ValueError("Graph relationship endpoint is missing its required node id")
        relationships.append(
            {
                "id": str(relationship.element_id),
                "type": str(relationship.type),
                "start": str(start_id),
                "end": str(end_id),
            }
        )
    return {"nodes": nodes, "relationships": relationships}


def fetch_graph_paths(
    driver: AuraDriver,
    center_id: str,
    database: str,
    path_limit: int = 100,
    relationship_type: str | None = None,
) -> list[dict[str, list[dict[str, object]]]]:
    """기준 기업에서 최대 두 단계 떨어진 관계 경로를 제한된 수만 읽는다."""
    records, _, _ = driver.execute_query(
        GRAPH_PATH_QUERY,
        center_id=center_id,
        path_limit=path_limit,
        relationship_type=relationship_type,
        database_=database,
    )
    return [serialize_path(record["path"]) for record in records]


def build_graph_elements(
    paths: list[dict[str, list[dict[str, object]]]],
    center_id: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Aura에서 받은 경로들을 화면용 고유 노드·엣지 목록으로 변환한다."""
    nodes_by_id: dict[str, dict[str, object]] = {}
    edges_by_id: dict[str, dict[str, object]] = {}

    for path in paths:
        for level, raw_node in enumerate(path["nodes"]):
            node_id = str(raw_node["id"])
            labels = raw_node.get("labels", [])
            label = str(raw_node.get("name") or node_id)
            candidate = {
                "id": node_id,
                "label": label,
                "group": labels[0] if labels else "Unknown",
                "level": level,
            }

            existing = nodes_by_id.get(node_id)
            if existing is None or level < existing["level"]:
                nodes_by_id[node_id] = candidate

        for raw_edge in path["relationships"]:
            edge_id = str(raw_edge["id"])
            edges_by_id.setdefault(
                edge_id,
                {
                    "id": edge_id,
                    "source": str(raw_edge["start"]),
                    "target": str(raw_edge["end"]),
                    "relation": str(raw_edge["type"]),
                },
            )

    return list(nodes_by_id.values()), list(edges_by_id.values())
