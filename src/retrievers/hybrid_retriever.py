
"""Graph와 Vector 검색 결과를 결합하는 하이브리드 검색기입니다."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import partial
import math
import re
from typing import Any

from .graph_retriever import GraphRetriever, graph_search
from .search_result import SearchResult
from .vector_retriever import VectorIndex, vector_search


# Reciprocal Rank Fusion 설정
RRF_K = 60
GRAPH_WEIGHT = 1.1
VECTOR_WEIGHT = 1.0
QUERY_RELEVANCE_WEIGHT = 0.002

_QUERY_STOP_WORDS = {
    "무엇", "무엇인가", "어떤", "어떻게", "알려줘", "알려주세요", "대해", "대한",
    "에서", "으로", "에게", "하는", "하나요", "인가요", "있나요", "그리고",
    "경우", "관련", "방법", "정보", "것은", "것이", "있는", "하였을",
}

_RELATION_QUERY_TERMS = {
    "HAS_REGISTERED_PESTICIDE_USE": ("농약", "등록", "사용", "품목", "상표"),
    "CONTROLS_TARGET": ("방제", "방제할", "대상", "효과", "잡는"),
    "HAS_DISEASE": ("병", "병해", "병해충", "발생"),
    "HAS_PEST": ("해충", "병해충", "발생", "방제"),
    "RECOMMENDED_FOR": ("추천", "적합", "사용", "조건"),
}


def _query_terms(text: str) -> set[str]:
    return {
        token.casefold() for token in re.findall(r"[가-힣A-Za-z0-9]+", text)
        if len(token) > 1 and token.casefold() not in _QUERY_STOP_WORDS
    }


def _query_relevance(query: str | None, result: SearchResult) -> float:
    """Return lexical query coverage from result text and graph path labels."""
    if not query:
        return 0.0
    terms = _query_terms(query)
    if not terms:
        return 0.0

    labels: list[str] = [result.text]
    labels.extend(str(node.get("name", "")) for node in (result.path_nodes or []))
    relations = list(result.relation_types or [])
    relations.extend(str(edge.get("relation_type", "")) for edge in (result.path_edges or []))
    for relation in relations:
        labels.extend(_RELATION_QUERY_TERMS.get(relation, ()))
    # A graph hit may carry the matching original chunk text as vector support.
    for hit in result.metadata.get("matched_vector_hits", []):
        labels.append(str(hit.get("text", "")))
    covered = set().union(*(_query_terms(label) for label in labels))
    return len(terms & covered) / len(terms)


def _ids(
    result: SearchResult,
) -> tuple[set[str], set[str]]:
    """검색 결과의 Evidence 및 Chunk ID를 추출합니다."""

    evidence_ids = set(result.evidence_ids or [])
    chunk_ids = set(result.chunk_ids or [])

    if result.evidence_id:
        evidence_ids.add(result.evidence_id)

    if result.chunk_id:
        chunk_ids.add(result.chunk_id)

    return evidence_ids, chunk_ids


def _same_evidence(
    left: SearchResult,
    right: SearchResult,
) -> bool:
    """Evidence ID 또는 Chunk ID가 겹치는지 확인합니다."""

    left_evidence, left_chunks = _ids(left)
    right_evidence, right_chunks = _ids(right)

    return bool(
        left_evidence & right_evidence
        or left_chunks & right_chunks
    )


def _graph_key(
    result: SearchResult,
) -> tuple[Any, ...]:
    """서로 다른 Graph 경로를 구분하는 키를 생성합니다."""

    node_ids = result.metadata.get("node_ids") or []
    relation_ids = result.metadata.get("relation_ids") or []

    nodes = tuple(node_ids or result.path or [])

    relations = tuple(
        relation_id or relation_type
        for relation_id, relation_type in zip(
            relation_ids,
            result.relation_types or [],
        )
    )

    if not relations:
        relations = tuple(result.relation_types or [])

    return (
        nodes,
        relations,
        result.text if not result.path else "",
    )


def _is_eligible(
    result: SearchResult,
) -> bool:
    """검색 결과가 하이브리드 근거로 사용 가능한지 확인합니다."""

    if result.source != "graph":
        return True

    return result.metadata.get("retrieval_eligible") is not False


def _rrf_score(
    rank: int,
    weight: float = 1.0,
) -> float:
    """순위 기반 RRF 점수를 계산합니다."""

    return weight / (RRF_K + rank)


def _vector_hit(
    result: SearchResult,
) -> dict[str, Any]:
    """Vector 결과를 Graph 경로의 연결 근거로 변환합니다."""

    evidence_ids, chunk_ids = _ids(result)

    return {
        "text": result.text,
        "score": result.score,
        "evidence_ids": sorted(evidence_ids),
        "chunk_ids": sorted(chunk_ids),
        "source_ids": list(result.source_ids or []),
        "metadata": dict(result.metadata),
        "warnings": list(result.warnings or []),
    }


def _index_result(
    result: SearchResult,
    index: int,
    evidence_index: dict[str, set[int]],
    chunk_index: dict[str, set[int]],
) -> None:
    """Evidence 및 Chunk ID를 역인덱스에 등록합니다."""

    evidence_ids, chunk_ids = _ids(result)

    for evidence_id in evidence_ids:
        evidence_index[evidence_id].add(index)

    for chunk_id in chunk_ids:
        chunk_index[chunk_id].add(index)


def _find_matches(
    result: SearchResult,
    evidence_index: dict[str, set[int]],
    chunk_index: dict[str, set[int]],
) -> set[int]:
    """ID 역인덱스로 중복 근거를 검색합니다."""

    evidence_ids, chunk_ids = _ids(result)

    matches: set[int] = set()

    for evidence_id in evidence_ids:
        matches.update(evidence_index.get(evidence_id, set()))

    for chunk_id in chunk_ids:
        matches.update(chunk_index.get(chunk_id, set()))

    return matches


def merge_results(
    graph_results: list[SearchResult],
    vector_results: list[SearchResult],
    *,
    limit: int = 10,
    graph_weight: float = GRAPH_WEIGHT,
    vector_weight: float = VECTOR_WEIGHT,
    query: str | None = None,
) -> list[SearchResult]:
    """Graph 경로와 Vector 근거를 병합하고 중복을 제거합니다."""

    if limit < 1:
        raise ValueError("limit은 양수여야 합니다.")

    if (not math.isfinite(graph_weight) or not math.isfinite(vector_weight)
            or graph_weight <= 0 or vector_weight <= 0):
        raise ValueError("RRF weights must be finite positive numbers")

    merged: list[SearchResult] = []

    # 결과별 검색 순위 기록
    graph_ranks: dict[int, int] = {}
    vector_scores: dict[int, float] = {}

    # --------------------------------------------------
    # 1. Graph 경로 중복 제거
    # --------------------------------------------------
    seen_paths: set[tuple[Any, ...]] = set()

    for rank, result in enumerate(graph_results, start=1):
        if not _is_eligible(result):
            continue

        key = _graph_key(result)

        if key in seen_paths:
            continue

        seen_paths.add(key)

        index = len(merged)
        merged.append(result)
        graph_ranks[index] = rank

    # --------------------------------------------------
    # 2. Graph 근거 ID 역인덱스 구성
    # --------------------------------------------------
    graph_evidence_index: dict[str, set[int]] = defaultdict(set)
    graph_chunk_index: dict[str, set[int]] = defaultdict(set)

    for index, result in enumerate(merged):
        _index_result(
            result,
            index,
            graph_evidence_index,
            graph_chunk_index,
        )

    # --------------------------------------------------
    # 3. Vector 중복 제거 및 Graph 근거 연결
    # --------------------------------------------------
    vector_evidence_index: dict[str, set[int]] = defaultdict(set)
    vector_chunk_index: dict[str, set[int]] = defaultdict(set)

    # Vector 중복 그룹의 대표 결과
    vector_groups: list[SearchResult] = []
    vector_group_ranks: list[int] = []
    vector_group_members: list[list[SearchResult]] = []
    group_parent: list[int] = []

    def group_root(group_id: int) -> int:
        while group_parent[group_id] != group_id:
            group_parent[group_id] = group_parent[group_parent[group_id]]
            group_id = group_parent[group_id]
        return group_id

    for rank, result in enumerate(vector_results, start=1):
        matching_groups = {
            group_root(group_id)
            for group_id in _find_matches(
                result, vector_evidence_index, vector_chunk_index,
            )
        }

        if matching_groups:
            # 순위가 가장 높은 기존 결과를 대표로 사용
            group_id = min(
                matching_groups,
                key=lambda idx: vector_group_ranks[idx],
            )

            # 기존 그룹과 간접적으로 연결되는 그룹도 통합
            for other_id in matching_groups:
                if other_id == group_id:
                    continue
                group_parent[other_id] = group_id
                vector_group_members[group_id].extend(vector_group_members[other_id])
                for member in vector_group_members[other_id]:
                    _index_result(
                        member, group_id, vector_evidence_index, vector_chunk_index,
                    )

            # 새 결과의 ID도 대표 그룹에 연결
            _index_result(
                result,
                group_id,
                vector_evidence_index,
                vector_chunk_index,
            )
            vector_group_members[group_id].append(result)

            continue

        group_id = len(vector_groups)

        vector_groups.append(result)
        vector_group_ranks.append(rank)
        vector_group_members.append([result])
        group_parent.append(group_id)

        _index_result(
            result,
            group_id,
            vector_evidence_index,
            vector_chunk_index,
        )

    # --------------------------------------------------
    # 4. Vector 결과를 Graph와 연결
    # --------------------------------------------------
    for group_id, result in enumerate(vector_groups):
        if group_root(group_id) != group_id:
            continue
        rank = vector_group_ranks[group_id]
        members = vector_group_members[group_id]
        matches = set().union(*(
            _find_matches(member, graph_evidence_index, graph_chunk_index)
            for member in members
        ))

        vector_score = _rrf_score(
            rank,
            vector_weight,
        )

        if matches:
            for index in sorted(matches):
                graph = merged[index]

                supporting_hits = list(
                    graph.metadata.get("matched_vector_hits", [])
                )

                for member in members:
                    hit = _vector_hit(member)
                    if not any(
                        existing.get("chunk_ids") == hit["chunk_ids"]
                        and existing.get("evidence_ids") == hit["evidence_ids"]
                        for existing in supporting_hits
                    ):
                        supporting_hits.append(hit)

                merged[index] = replace(
                    graph,
                    metadata={
                        **graph.metadata,
                        "matched_vector_hits": supporting_hits,
                    },
                )

                # 여러 Vector 근거가 연결되면 점수 누적
                vector_scores[index] = (
                    vector_scores.get(index, 0.0)
                    + vector_score
                )

            continue

        # Graph와 겹치지 않는 독립 Vector 근거
        index = len(merged)
        merged.append(result)
        vector_scores[index] = vector_score

    # --------------------------------------------------
    # 5. RRF 기반 최종 점수 계산
    # --------------------------------------------------
    ranked: list[SearchResult] = []

    for index, result in enumerate(merged):
        graph_score = 0.0

        if index in graph_ranks:
            graph_score = _rrf_score(
                graph_ranks[index],
                graph_weight,
            )

        vector_score = vector_scores.get(index, 0.0)

        relevance = _query_relevance(query, result)
        hybrid_score = graph_score + vector_score + QUERY_RELEVANCE_WEIGHT * relevance

        ranked.append(
            replace(
                result,
                metadata={
                    **result.metadata,
                    "graph_rrf_score": graph_score,
                    "vector_rrf_score": vector_score,
                    "query_relevance": relevance,
                    "hybrid_score": hybrid_score,
                },
            )
        )

    # --------------------------------------------------
    # 6. 점수 순 정렬 및 결과 제한
    # --------------------------------------------------
    ranked.sort(
        key=lambda result: (
            -float(result.metadata["hybrid_score"]),
            -float(
                (result.score or 0.0) if result.source == "vector"
                else (result.metadata.get("confidence") or 0.0)
            ),
            0 if result.source == "graph" else 1,
        )
    )

    return ranked[:limit]


def build_context(
    results: list[SearchResult],
) -> str:
    """검색 근거를 답변 생성용 Context로 변환합니다."""

    sections: list[str] = []

    for index, result in enumerate(results, start=1):
        # 검색에서 제외해야 하는 Graph 관계는 사용하지 않음
        if not _is_eligible(result):
            continue

        # Graph는 모든 hop에 검증된 원문이 있어야 사용
        if result.source == "graph":
            edges = result.path_edges or []

            if not edges or not all(
                edge.get("original_chunks")
                for edge in edges
            ):
                continue

        lines = [
            f"[근거 {index}]",
            f"검색 출처: {result.source}",
            f"내용: {result.text}",
            f"hybrid_score: "
            f"{float(result.metadata.get('hybrid_score') or 0.0):.6f}",
            f"evidence_ids: {', '.join(result.evidence_ids or []) or '없음'}",
            f"source_ids: {', '.join(result.source_ids or []) or '없음'}",
            f"chunk_ids: {', '.join(result.chunk_ids or []) or '없음'}",
        ]

        if result.source == "graph":
            lines.extend([
                f"그래프 경로: "
                f"{' → '.join(result.path or []) or '없음'}",
                f"관계 타입: "
                f"{' → '.join(result.relation_types or []) or '없음'}",
                f"hop 수: {result.hop_count or 0}",
            ])

            for edge in result.path_edges or []:
                lines.append(
                    f"hop {edge['hop']}: "
                    f"{edge['relation_type']} "
                    f"({edge['from_entity_id']} → "
                    f"{edge['to_entity_id']}) "
                    f"relation_id={edge['relation_id'] or '없음'}"
                )

                for chunk in edge["original_chunks"]:
                    lines.append(
                        f"원문 근거: "
                        f"chunk_id={chunk['chunk_id']} "
                        f"evidence_id={chunk['evidence_id']} "
                        f"source_id={chunk['source_id']} "
                        f"내용={chunk['text']}"
                    )

            # Graph와 연결된 Vector 근거
            for hit in result.metadata.get(
                "matched_vector_hits", []
            ):
                lines.append(
                    "연결된 벡터 원문: "
                    f"chunk_ids={', '.join(hit['chunk_ids']) or '없음'} "
                    f"evidence_ids={', '.join(hit['evidence_ids']) or '없음'} "
                    f"내용={hit['text']}"
                )

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def hybrid_search(
    question: str,
    *,
    graph_retriever: GraphRetriever | None = None,
    vector_index: VectorIndex | None = None,
    graph_search_fn: Callable[..., list[SearchResult]] | None = None,
    vector_search_fn: Callable[..., list[SearchResult]] | None = None,
    crop: str | None = None,
    max_hops: int = 2,
    relation_types: Sequence[str] | None = None,
    entity_names: Sequence[str] | None = None,
    graph_limit: int = 10,
    vector_limit: int = 10,
    limit: int = 10,
    graph_weight: float = GRAPH_WEIGHT,
    vector_weight: float = VECTOR_WEIGHT,
) -> list[SearchResult]:
    """Graph와 Vector를 검색한 뒤 결과를 병합합니다."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")

    if min(graph_limit, vector_limit, limit) < 1:
        raise ValueError("검색 limit은 모두 양수여야 합니다.")

    # --------------------------------------------------
    # 1. 검색 함수 구성
    # --------------------------------------------------
    if graph_search_fn is None:
        if graph_retriever is None:
            raise ValueError(
                "graph_retriever or graph_search_fn is required"
            )

        graph_search_fn = partial(
            graph_search,
            retriever=graph_retriever,
        )

    if vector_search_fn is None:
        if vector_index is None:
            raise ValueError(
                "vector_index or vector_search_fn is required"
            )

        vector_search_fn = partial(
            vector_search,
            index=vector_index,
        )

    # --------------------------------------------------
    # 2. Graph 검색
    # --------------------------------------------------
    graph_kwargs: dict[str, Any] = {
        "crop": crop,
        "max_hops": max_hops,
        "limit": graph_limit,
    }

    if relation_types is not None:
        graph_kwargs["relation_types"] = relation_types
    if entity_names is not None:
        graph_kwargs["entity_names"] = entity_names

    graph_results = graph_search_fn(
        question,
        **graph_kwargs,
    )

    # --------------------------------------------------
    # 3. Vector 검색
    # --------------------------------------------------
    vector_results = vector_search_fn(
        question,
        k=vector_limit,
    )

    # --------------------------------------------------
    # 4. 검색 결과 병합
    # --------------------------------------------------
    return merge_results(
        graph_results,
        vector_results,
        limit=limit,
        graph_weight=graph_weight,
        vector_weight=vector_weight,
        query=question,
    )
