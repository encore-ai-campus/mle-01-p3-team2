"""농업 GraphRAG 파이프라인을 위한 Neo4j 그래프 검색기입니다."""

from __future__ import annotations

import os
import re
from typing import Any, Iterable, Sequence

from neo4j import Driver, GraphDatabase
from .search_result import SearchResult


_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


GraphSearchResult = SearchResult


def _check_identifier(value: str, kind: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid Neo4j {kind}: {value!r}")
    return value


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def path_to_result(path: Any) -> SearchResult:
    nodes = list(path.nodes)
    relationships = list(path.relationships)
    node_names = [str(node.get("name") or node.get("entity_id")) for node in nodes]
    node_ids = [str(node.get("entity_id")) for node in nodes]
    node_types = [str(node.get("entity_type") or next(iter(node.labels), "")) for node in nodes]
    relation_types = [str(rel.type) for rel in relationships]
    relation_ids = [str(rel.get("relation_id")) for rel in relationships]
    evidence_ids = sorted({item for rel in relationships for item in _as_list(rel.get("evidence_ids"))})
    source_ids = sorted({item for rel in relationships for item in _as_list(rel.get("source_ids"))})
    chunk_ids = sorted({item for rel in relationships for item in _as_list(rel.get("source_chunk_ids"))})
    confidences = [rel.get("confidence") for rel in relationships if rel.get("confidence") is not None]
    eligible = all(rel.get("retrieval_eligible", True) for rel in relationships)
    confidence = min(confidences) if confidences else None
    return SearchResult(
        source="graph",
        text=" → ".join(node_names),
        score=None,
        path=node_names,
        relation_types=relation_types,
        evidence_ids=evidence_ids,
        source_ids=source_ids,
        chunk_ids=chunk_ids,
        metadata={
            "node_ids": node_ids,
            "node_types": node_types,
            "relation_ids": relation_ids,
            "hop_count": len(relationships),
            "confidence": confidence,
            "retrieval_eligible": eligible,
        },
    )


class GraphRetriever:
    """Neo4j 그래프에서 타입이 지정된 멀티홉 경로를 검색합니다."""

    def __init__(self, driver: Driver):
        self.driver = driver

    @classmethod
    def from_env(cls) -> "GraphRetriever":
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        if not password:
            raise ValueError("NEO4J_PASSWORD is required")
        return cls(GraphDatabase.driver(uri, auth=(username, password)))

    def close(self) -> None:
        self.driver.close()

    def search_paths(
        self,
        *,
        start_label: str = "Crop",
        start_name: str | None = None,
        min_hops: int = 1,
        max_hops: int = 2,
        relation_types: Sequence[str] | None = None,
        limit: int = 20,
    ) -> list[GraphSearchResult]:
        _check_identifier(start_label, "label")
        if min_hops < 1 or max_hops < min_hops:
            raise ValueError("Require 1 <= min_hops <= max_hops")
        if limit < 1:
            raise ValueError("limit must be positive")
        rel_pattern = "|".join(
            f"`{_check_identifier(item, 'relationship type')}`"
            for item in (relation_types or [])
        )
        if rel_pattern:
            rel_pattern = f":{rel_pattern}"
        query = f"""
        MATCH path = (start:`{start_label}`)-[{rel_pattern}*{min_hops}..{max_hops}]->(target)
        {"WHERE start.name = $start_name" if start_name else ""}
        RETURN path
        LIMIT $limit
        """
        params = {"limit": limit}
        if start_name:
            params["start_name"] = start_name
        with self.driver.session() as session:
            records = session.run(query, params)
            return [path_to_result(record["path"]) for record in records]

    def search(self, question: str, *, crop: str | None = None, max_hops: int = 2, limit: int = 20) -> list[GraphSearchResult]:
        """질문과 관련된 그래프 근거를 반환합니다.

        질문 분류는 Router가 담당합니다. Router를 교체할 수 있도록
        이 메서드는 이미 추출된 작물명을 선택적으로 받습니다.
        """
        return self.search_paths(
            start_label="Crop",
            start_name=crop,
            min_hops=1,
            max_hops=max_hops,
            limit=limit,
        )


def graph_search(
    question: str,
    *,
    retriever: GraphRetriever,
    crop: str | None = None,
    max_hops: int = 2,
    limit: int = 20,
) -> list[SearchResult]:
    """Neo4j를 검색하고 공통 SearchResult 형식으로 반환합니다.

    호출 전에 Router 또는 검색 워커가 ``crop``을 추출합니다.
    검색기 인스턴스를 주입받기 때문에 LangGraph에서 재사용하고
    테스트하기 쉽습니다.
    """
    return retriever.search(
        question,
        crop=crop,
        max_hops=max_hops,
        limit=limit,
    )
