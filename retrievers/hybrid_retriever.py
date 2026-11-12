"""그래프 검색 결과와 벡터 검색 결과를 결합하는 하이브리드 검색기입니다."""

from __future__ import annotations

from collections.abc import Callable

from search_result import SearchResult


def _result_key(result: SearchResult) -> tuple[str, ...]:
    """동일 근거 결과를 하나로 합치기 위한 키를 만듭니다."""
    if result.source == "vector" and result.chunk_ids:
        return ("chunk", *sorted(result.chunk_ids))
    if result.source == "graph" and result.path:
        return ("path", *result.path, *(result.relation_types or []))
    return (result.source, result.text)


def merge_results(
    graph_results: list[SearchResult],
    vector_results: list[SearchResult],
    *,
    limit: int = 10,
) -> list[SearchResult]:
    """그래프 경로와 벡터 청크를 중복 제거해 하나의 결과 목록으로 합칩니다."""
    if limit < 1:
        raise ValueError("limit은 양수여야 합니다.")

    merged: list[SearchResult] = []
    seen: set[tuple[str, ...]] = set()
    # 그래프 경로를 먼저 배치해 관계·멀티홉 근거가 결과에서 보존되도록 합니다.
    for result in [*graph_results, *vector_results]:
        key = _result_key(result)
        if key in seen:
            continue
        seen.add(key)
        merged.append(result)
        if len(merged) >= limit:
            break
    return merged


def hybrid_search(
    question: str,
    *,
    graph_search_fn: Callable[..., list[SearchResult]],
    vector_search_fn: Callable[..., list[SearchResult]],
    crop: str | None = None,
    max_hops: int = 2,
    graph_limit: int = 10,
    vector_limit: int = 10,
    limit: int = 10,
) -> list[SearchResult]:
    """그래프·벡터 검색을 실행하고 공통 결과 형식으로 결합합니다.

    라우터는 이 함수를 직접 호출하지 않습니다. 라우터가 ``HYBRID``를
    반환한 뒤 Supervisor 또는 검색 워커가 이 함수를 실행합니다.
    """
    graph_results = graph_search_fn(
        question,
        crop=crop,
        max_hops=max_hops,
        limit=graph_limit,
    )
    vector_results = vector_search_fn(question, k=vector_limit)
    return merge_results(graph_results, vector_results, limit=limit)
