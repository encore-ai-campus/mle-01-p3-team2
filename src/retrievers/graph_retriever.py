
"""농업 GraphRAG 파이프라인을 위한 Neo4j 그래프 검색기입니다."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

from neo4j import Driver, GraphDatabase

from .search_result import (
    GraphPathEdge,
    GraphPathNode,
    OriginalChunk,
    SearchResult,
)


_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

GraphSearchResult = SearchResult


def _check_identifier(value: str, kind: str) -> str:
    """Neo4j 라벨 및 관계 타입을 검증합니다."""
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid Neo4j {kind}: {value!r}")
    return value


def _as_list(value: Any) -> list[str]:
    """Neo4j 속성을 문자열 리스트로 정규화합니다."""
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]

    return [str(value)]


def _unique_list(values: Sequence[str]) -> list[str]:
    """입력 순서를 유지하면서 중복을 제거합니다."""
    return list(dict.fromkeys(values))


def path_to_result(
    path: Any,
    *,
    chunks_by_id: Mapping[str, OriginalChunk] | None = None,
) -> SearchResult:
    """Neo4j Path를 공통 SearchResult로 변환합니다."""

    nodes = list(path.nodes)
    relationships = list(path.relationships)

    # --------------------------------------------------
    # 1. 경로 노드 정보
    # --------------------------------------------------
    path_nodes: list[GraphPathNode] = [
        {
            "entity_id": str(node.get("entity_id") or ""),
            "name": str(
                node.get("name")
                or node.get("entity_id")
                or ""
            ),
            "entity_type": str(
                node.get("entity_type")
                or next(iter(sorted(node.labels)), "")
            ),
        }
        for node in nodes
    ]

    node_names = [node["name"] for node in path_nodes]
    node_ids = [node["entity_id"] for node in path_nodes]
    node_types = [node["entity_type"] for node in path_nodes]

    # --------------------------------------------------
    # 2. 관계 및 hop 정보
    # --------------------------------------------------
    path_edges: list[GraphPathEdge] = []

    for index, rel in enumerate(relationships):
        relation_id = rel.get("relation_id")

        path_edges.append(
            {
                "hop": index + 1,
                "relation_id": (
                    str(relation_id)
                    if relation_id is not None
                    else None
                ),
                "relation_type": str(rel.type),
                "from_entity_id": path_nodes[index]["entity_id"],
                "to_entity_id": path_nodes[index + 1]["entity_id"],
                "evidence_ids": _unique_list(
                    _as_list(rel.get("evidence_ids"))
                ),
                "chunk_ids": _unique_list(
                    _as_list(rel.get("source_chunk_ids"))
                ),
                "source_ids": _unique_list(
                    _as_list(rel.get("source_ids"))
                ),
                "original_chunks": [],
            }
        )

    # --------------------------------------------------
    # 3. 관계별 원문 Chunk 연결 및 검증
    # --------------------------------------------------
    warnings: list[str] = []

    if chunks_by_id is not None:
        for edge in path_edges:
            relation = (
                edge["relation_id"]
                or edge["relation_type"]
            )

            if not edge["chunk_ids"]:
                warnings.append(
                    f"Relation {relation} has no linked source chunks"
                )

            if not edge["evidence_ids"]:
                warnings.append(
                    f"Relation {relation} has no linked evidence IDs"
                )

            for chunk_id in edge["chunk_ids"]:
                chunk = chunks_by_id.get(chunk_id)

                if chunk is None:
                    warnings.append(
                        f"Relation {relation} references "
                        f"missing chunk {chunk_id}"
                    )
                    continue

                if not chunk["text"].strip():
                    warnings.append(
                        f"Relation {relation} has empty "
                        f"original text in {chunk_id}"
                    )
                    continue

                if (
                    edge["evidence_ids"]
                    and chunk["evidence_id"]
                    not in edge["evidence_ids"]
                ):
                    warnings.append(
                        f"Relation {relation} evidence "
                        f"does not match chunk {chunk_id}"
                    )
                    continue

                if (
                    edge["source_ids"]
                    and chunk["source_id"]
                    not in edge["source_ids"]
                ):
                    warnings.append(
                        f"Relation {relation} source "
                        f"does not match chunk {chunk_id}"
                    )
                    continue

                edge["original_chunks"].append(chunk)

        # 원문 근거 연결 상태
        if warnings and any(
            edge["original_chunks"]
            for edge in path_edges
        ):
            warnings.append(
                "Graph path has partial original evidence"
            )

        elif path_edges and not any(
            edge["original_chunks"]
            for edge in path_edges
        ):
            warnings.append(
                "Graph path has no original evidence"
            )

    # --------------------------------------------------
    # 4. 경로 전체 메타데이터
    # --------------------------------------------------
    relation_types = [
        edge["relation_type"]
        for edge in path_edges
    ]

    relation_ids = [
        edge["relation_id"]
        for edge in path_edges
    ]

    evidence_ids = sorted({
        item
        for edge in path_edges
        for item in edge["evidence_ids"]
    })

    source_ids = sorted({
        item
        for edge in path_edges
        for item in edge["source_ids"]
    })

    chunk_ids = sorted({
        item
        for edge in path_edges
        for item in edge["chunk_ids"]
    })

    confidences = [
        float(rel.get("confidence"))
        for rel in relationships
        if rel.get("confidence") is not None
    ]

    confidence = min(confidences) if confidences else None

    eligible = all(
        rel.get("retrieval_eligible", True) is True
        for rel in relationships
    )

    # 모든 관계에 검증된 원문이 있는지 확인
    has_complete_evidence = bool(path_edges) and all(
        bool(edge["original_chunks"])
        for edge in path_edges
    )

    return SearchResult(
        source="graph",
        text=" → ".join(node_names),
        score=None,
        path=node_names,
        path_nodes=path_nodes,
        path_edges=path_edges,
        relation_types=relation_types,
        hop_count=len(relationships),
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
            "has_complete_evidence": has_complete_evidence,
        },
        warnings=_unique_list(warnings),
    )


class GraphRetriever:
    """Neo4j 그래프에서 멀티홉 경로와 원문 근거를 검색합니다."""

    def __init__(
        self,
        driver: Driver,
        *,
        database: str = "neo4j",
    ) -> None:
        self.driver = driver
        self.database = database

    @classmethod
    def from_env(cls) -> GraphRetriever:
        """환경변수를 사용하여 Neo4j에 연결합니다."""

        uri = os.getenv(
            "NEO4J_URI",
            "bolt://localhost:7687",
        )
        username = os.getenv(
            "NEO4J_USERNAME",
            "neo4j",
        )
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv(
            "NEO4J_DATABASE",
            "neo4j",
        )

        if not password:
            raise ValueError(
                "NEO4J_PASSWORD is required"
            )

        driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
        )

        try:
            driver.verify_connectivity()

            return cls(
                driver,
                database=database,
            )

        except Exception:
            driver.close()
            raise

    def close(self) -> None:
        """Neo4j 드라이버 연결을 종료합니다."""
        self.driver.close()

    @staticmethod
    def _fetch_paths_and_chunks(
        tx: Any,
        query: str,
        params: dict[str, Any],
    ) -> tuple[list[Any], dict[str, OriginalChunk]]:
        """하나의 읽기 트랜잭션에서 경로와 원문을 조회합니다."""

        paths = [
            record["path"]
            for record in tx.run(query, params)
        ]

        if not paths:
            return [], {}

        # 경로에 포함된 원문 Chunk ID 수집
        chunk_ids = sorted({
            chunk_id
            for path in paths
            for rel in path.relationships
            for chunk_id in _as_list(
                rel.get("source_chunk_ids")
            )
        })

        if not chunk_ids:
            return paths, {}

        # 원문 청크 일괄 조회
        chunk_query = (
            "MATCH (chunk:Chunk) WHERE chunk.chunk_id IN $chunk_ids "
            "RETURN chunk.chunk_id AS chunk_id, "
            "chunk.evidence_id AS evidence_id, "
            "chunk.source_id AS source_id, chunk.raw_text AS text"
        )

        chunks_by_id: dict[str, OriginalChunk] = {}

        for row in tx.run(
            chunk_query,
            {"chunk_ids": chunk_ids},
        ):
            chunk_id = row.get("chunk_id")

            if chunk_id is None:
                continue

            chunk_id = str(chunk_id)

            chunks_by_id[chunk_id] = {
                "chunk_id": chunk_id,
                "evidence_id": str(
                    row.get("evidence_id") or ""
                ),
                "source_id": str(
                    row.get("source_id") or ""
                ),
                "text": str(
                    row.get("text") or ""
                ),
            }

        return paths, chunks_by_id

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
        """지정된 엔티티를 시작점으로 멀티홉 경로를 검색합니다."""

        # --------------------------------------------------
        # 1. 입력값 검증
        # --------------------------------------------------
        _check_identifier(
            start_label,
            "label",
        )

        if min_hops < 1 or max_hops < min_hops:
            raise ValueError(
                "Require 1 <= min_hops <= max_hops"
            )

        if limit < 1:
            raise ValueError(
                "limit must be positive"
            )

        # 관계 타입 검증 및 중복 제거
        types = _unique_list(
            list(relation_types or [])
        )

        for relation_type in types:
            _check_identifier(
                relation_type,
                "relationship type",
            )

        rel_pattern = (
            ":" + "|".join(
                f"`{relation_type}`"
                for relation_type in types
            )
            if types
            else ""
        )

        # --------------------------------------------------
        # 2. Cypher 검색 쿼리
        # --------------------------------------------------
        where_clause = (
            "WHERE start.name = $start_name"
            if start_name is not None
            else ""
        )

        query = f"""
        MATCH path = (start:`{start_label}`)-[{rel_pattern}*{min_hops}..{max_hops}]->(target)
        {where_clause}
        RETURN path
        ORDER BY length(path) ASC, elementId(target) ASC
        LIMIT $limit
        """

        params = {
            "start_name": start_name,
            "limit": limit,
        }

        # --------------------------------------------------
        # 3. Neo4j 읽기 트랜잭션
        # --------------------------------------------------
        session_factory = self.driver.session
        session_options = {"database": self.database} if self.database != "neo4j" else {}
        with session_factory(**session_options) as session:
            if hasattr(session, "execute_read"):
                paths, chunks_by_id = session.execute_read(
                    self._fetch_paths_and_chunks,
                    query,
                    params,
                )
            else:
                paths, chunks_by_id = self._fetch_paths_and_chunks(
                    session, query, params,
                )

        # --------------------------------------------------
        # 4. SearchResult 변환
        # --------------------------------------------------
        return [
            path_to_result(
                path,
                chunks_by_id=chunks_by_id,
            )
            for path in paths
        ]

    def search(
        self,
        question: str,
        *,
        crop: str | None = None,
        max_hops: int = 2,
        limit: int = 20,
        relation_types: Sequence[str] | None = None,
    ) -> list[GraphSearchResult]:
        """질문과 Router의 검색 조건을 사용합니다."""

        if (
            not isinstance(question, str)
            or not question.strip()
        ):
            raise ValueError(
                "question must be a nonempty string"
            )

        return self.search_paths(
            start_label="Crop",
            start_name=crop,
            min_hops=1,
            max_hops=max_hops,
            relation_types=relation_types,
            limit=limit,
        )


def graph_search(
    question: str,
    *,
    retriever: GraphRetriever,
    crop: str | None = None,
    max_hops: int = 2,
    limit: int = 20,
    relation_types: Sequence[str] | None = None,
) -> list[SearchResult]:
    """Graph 검색 결과를 공통 SearchResult로 반환합니다."""

    kwargs: dict[str, Any] = {"crop": crop, "max_hops": max_hops, "limit": limit}
    if relation_types is not None:
        kwargs["relation_types"] = relation_types
    return retriever.search(question, **kwargs)
