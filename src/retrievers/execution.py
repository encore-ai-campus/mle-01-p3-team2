
"""One call interface for vector, graph, and hybrid retrieval routes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Literal, TypeAlias

from neo4j.exceptions import ServiceUnavailable, SessionExpired

from .graph_retriever import GraphRetriever, graph_search
from .hybrid_retriever import hybrid_search
from .search_result import RetrievalResponse, SearchResult
from .vector_retriever import VectorIndex, vector_search


# ============================================================
# 1. 타입 및 기본 설정
# ============================================================

SearchRoute: TypeAlias = Literal[
    "VECTOR",
    "GRAPH",
    "HYBRID",
]

SearchFunction: TypeAlias = Callable[
    ...,
    list[SearchResult],
]

VALID_ROUTES = frozenset({
    "VECTOR",
    "GRAPH",
    "HYBRID",
})

CONNECTION_ERRORS = (
    ServiceUnavailable,
    SessionExpired,
    ConnectionError,
    TimeoutError,
)


# ============================================================
# 2. 입력값 검증
# ============================================================

def _validate_positive_int(
    value: int,
    name: str,
) -> None:
    """양의 정수인지 검증합니다."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
    ):
        raise ValueError(
            f"{name} must be a positive integer"
        )


def _validate_search_options(
    question: str,
    route: str,
    *,
    max_hops: int,
    limit: int,
    graph_limit: int,
    vector_limit: int,
) -> str:
    """검색 옵션을 검증하고 질문을 정규화합니다."""

    if not isinstance(question, str):
        raise TypeError("question must be a string")

    question = question.strip()

    if not question:
        raise ValueError(
            "question must be a nonempty string"
        )

    if route not in VALID_ROUTES:
        raise ValueError(
            f"Unsupported retrieval route: {route}"
        )

    _validate_positive_int(max_hops, "max_hops")
    _validate_positive_int(limit, "limit")
    _validate_positive_int(graph_limit, "graph_limit")
    _validate_positive_int(vector_limit, "vector_limit")

    return question


# ============================================================
# 3. 검색 함수 구성
# ============================================================

def _resolve_graph_search(
    retriever: GraphRetriever | None,
    search_fn: SearchFunction | None,
) -> SearchFunction | None:
    """Graph 검색 함수를 결정합니다."""

    if search_fn is not None:
        return search_fn

    if retriever is None:
        return None

    return partial(
        graph_search,
        retriever=retriever,
    )


def _resolve_vector_search(
    index: VectorIndex | None,
    search_fn: SearchFunction | None,
) -> SearchFunction | None:
    """Vector 검색 함수를 결정합니다."""

    if search_fn is not None:
        return search_fn

    if index is None:
        return None

    return partial(
        vector_search,
        index=index,
    )


# ============================================================
# 4. 경고 메시지 처리
# ============================================================

def _unique_warnings(
    warnings: Sequence[str],
) -> list[str]:
    """경고 메시지의 순서를 유지하며 중복을 제거합니다."""

    return list(dict.fromkeys(
        warning
        for warning in warnings
        if isinstance(warning, str) and warning.strip()
    ))


def _collect_result_warnings(
    results: list[SearchResult],
) -> list[str]:
    """검색 결과와 연결된 Vector 근거의 경고를 수집합니다."""

    collected: list[str] = []

    for result in results:
        collected.extend(result.warnings or [])

        if result.source != "graph":
            continue

        for hit in result.metadata.get(
            "matched_vector_hits", []
        ):
            if isinstance(hit, dict):
                collected.extend(
                    hit.get("warnings") or []
                )

    return _unique_warnings(collected)


# ============================================================
# 5. 공통 검색 실행
# ============================================================

def _run_backend(
    name: str,
    search: Callable[[], list[SearchResult]],
    warnings: list[str],
) -> list[SearchResult]:
    """검색 백엔드를 실행하고 연결 오류를 처리합니다."""

    try:
        results = search()

    except CONNECTION_ERRORS as exc:
        warnings.append(
            f"{name} connection error: "
            f"{type(exc).__name__}"
        )
        return []

    if not isinstance(results, list):
        raise TypeError(
            f"{name} must return list[SearchResult]"
        )

    if not all(
        isinstance(item, SearchResult)
        for item in results
    ):
        raise TypeError(
            f"{name} returned invalid SearchResult items"
        )

    if not results:
        warnings.append(
            f"{name} returned no results"
        )

    return results


# ============================================================
# 6. 통합 검색 라우터
# ============================================================

def execute_search(
    question: str,
    *,
    route: SearchRoute,
    vector_index: VectorIndex | None = None,
    graph_retriever: GraphRetriever | None = None,
    graph_search_fn: SearchFunction | None = None,
    vector_search_fn: SearchFunction | None = None,
    hybrid_search_fn: SearchFunction | None = None,
    crop: str | None = None,
    max_hops: int = 2,
    relation_types: Sequence[str] | None = None,
    limit: int = 10,
    graph_limit: int = 10,
    vector_limit: int = 10,
    reason: str = "",
    warnings: list[str] | None = None,
) -> RetrievalResponse:
    """선택된 검색 경로를 실행하고 공통 응답을 반환합니다.

    VECTOR:
        Vector 검색 결과 반환

    GRAPH:
        Graph 경로와 원문 근거 반환

    HYBRID:
        Graph + Vector 검색 후 결과 병합

    백엔드 연결 오류는 warnings에 기록합니다.
    그 외 코드 오류는 호출자에게 전달합니다.
    """

    # --------------------------------------------------------
    # 1. 입력 검증
    # --------------------------------------------------------
    question = _validate_search_options(
        question,
        route,
        max_hops=max_hops,
        limit=limit,
        graph_limit=graph_limit,
        vector_limit=vector_limit,
    )

    route_warnings: list[str] = []

    # --------------------------------------------------------
    # 2. 검색 함수 구성
    # --------------------------------------------------------
    graph_fn = _resolve_graph_search(
        graph_retriever,
        graph_search_fn,
    )

    vector_fn = _resolve_vector_search(
        vector_index,
        vector_search_fn,
    )

    # --------------------------------------------------------
    # 3. 필요한 검색기 확인
    # --------------------------------------------------------
    if route in ("GRAPH", "HYBRID") and graph_fn is None:
        raise ValueError(
            f"{route} route requires "
            "graph_retriever or graph_search_fn"
        )

    if route in ("VECTOR", "HYBRID") and vector_fn is None:
        raise ValueError(
            f"{route} route requires "
            "vector_index or vector_search_fn"
        )

    # --------------------------------------------------------
    # 4. Graph 검색 함수
    # --------------------------------------------------------
    def run_graph(
        query: str,
        *,
        crop: str | None = None,
        max_hops: int = 2,
        limit: int = 10,
        relation_types: Sequence[str] | None = None,
    ) -> list[SearchResult]:

        if graph_fn is None:
            raise ValueError(
                "Graph search function is not configured"
            )

        kwargs: dict[str, Any] = {
            "crop": crop,
            "max_hops": max_hops,
            "limit": limit,
        }

        # 기존 검색 함수와 호환되도록 선택적으로 전달
        if relation_types is not None:
            kwargs["relation_types"] = relation_types

        return _run_backend(
            "GRAPH",
            lambda: graph_fn(query, **kwargs),
            route_warnings,
        )

    # --------------------------------------------------------
    # 5. Vector 검색 함수
    # --------------------------------------------------------
    def run_vector(
        query: str,
        *,
        k: int = 10,
    ) -> list[SearchResult]:

        if vector_fn is None:
            raise ValueError(
                "Vector search function is not configured"
            )

        return _run_backend(
            "VECTOR",
            lambda: vector_fn(query, k=k),
            route_warnings,
        )

    # --------------------------------------------------------
    # 6. 검색 경로별 실행
    # --------------------------------------------------------
    if route == "VECTOR":
        results = run_vector(
            question,
            k=limit,
        )

    elif route == "GRAPH":
        results = run_graph(
            question,
            crop=crop,
            max_hops=max_hops,
            limit=limit,
            relation_types=relation_types,
        )

    else:
        # HYBRID
        selected_hybrid = hybrid_search_fn or hybrid_search

        hybrid_kwargs: dict[str, Any] = {
            "graph_search_fn": run_graph,
            "vector_search_fn": run_vector,
            "crop": crop,
            "max_hops": max_hops,
            "graph_limit": graph_limit,
            "vector_limit": vector_limit,
            "limit": limit,
        }

        if relation_types is not None:
            hybrid_kwargs["relation_types"] = relation_types

        results = selected_hybrid(
            question,
            **hybrid_kwargs,
        )

        if not isinstance(results, list) or not all(
            isinstance(item, SearchResult)
            for item in results
        ):
            raise TypeError(
                "HYBRID must return list[SearchResult]"
            )

    # --------------------------------------------------------
    # 7. 반환 결과 개수 제한
    # --------------------------------------------------------
    results = results[:limit]

    if not results:
        route_warnings.append(
            f"{route} returned no results"
        )

    # --------------------------------------------------------
    # 8. 경고 통합
    # --------------------------------------------------------
    result_warnings = _collect_result_warnings(
        results
    )

    all_warnings = _unique_warnings([
        *(warnings or []),
        *route_warnings,
        *result_warnings,
    ])

    # --------------------------------------------------------
    # 9. 공통 RetrievalResponse 반환
    # --------------------------------------------------------
    return RetrievalResponse(
        route=route,
        results=results,
        reason=reason,
        warnings=all_warnings,
    )
