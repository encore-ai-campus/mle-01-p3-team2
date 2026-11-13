"""그래프 검색 결과와 벡터 검색 결과를 결합하는 하이브리드 검색기입니다."""

from __future__ import annotations

from collections.abc import Callable

from .search_result import SearchResult


def _result_key(result: SearchResult) -> tuple[str, ...]:
    """동일 근거 결과를 하나로 합치기 위한 키를 만듭니다."""
    if result.source == "vector" and result.chunk_ids:
        return ("chunk", *sorted(result.chunk_ids))
    if result.source == "graph" and result.path:
        return ("path", *result.path, *(result.relation_types or []))
    return (result.source, result.text)


def _ranking_score(result: SearchResult) -> float:
    """그래프·벡터 결과를 공통 하이브리드 정렬 점수로 변환합니다."""
    if result.source == "vector":
        return float(result.score or 0.0)
    if result.metadata.get("retrieval_eligible") is False:
        return -1.0
    confidence = result.metadata.get("confidence")
    confidence_score = float(confidence) if confidence is not None else 0.5
    hop_count = int(result.metadata.get("hop_count", 0) or 0)
    # 멀티홉 경로를 보존하되, 근거 신뢰도가 지나치게 낮은 경로가 앞서지 않게 합니다.
    return min(1.0, confidence_score + min(hop_count, 4) * 0.02)


def merge_results(
    graph_results: list[SearchResult],
    vector_results: list[SearchResult],
    *,
    limit: int = 10,
) -> list[SearchResult]:
    """그래프 경로와 벡터 청크를 중복 제거해 하나의 결과 목록으로 합칩니다."""
    if limit < 1:
        raise ValueError("limit은 양수여야 합니다.")

    candidates: list[SearchResult] = []
    seen: set[tuple[str, ...]] = set()
    # 먼저 중복을 제거하고, 이후 점수 기준으로 정렬합니다.
    for result in [*graph_results, *vector_results]:
        key = _result_key(result)
        if key in seen:
            continue
        seen.add(key)
        result.metadata.setdefault("hybrid_score", _ranking_score(result))
        candidates.append(result)
    candidates.sort(
        key=lambda result: (
            float(result.metadata.get("hybrid_score", 0.0)),
            1 if result.source == "graph" else 0,
        ),
        reverse=True,
    )
    return candidates[:limit]


def build_context(results: list[SearchResult]) -> str:
    """검색 결과를 출처와 경로가 보존된 답변용 컨텍스트로 구성합니다."""
    sections: list[str] = []
    for index, result in enumerate(results, 1):
        evidence = ", ".join(result.evidence_ids) or "없음"
        sources = ", ".join(result.source_ids) or "없음"
        chunks = ", ".join(result.chunk_ids) or "없음"
        path = " → ".join(result.path or []) or "없음"
        relations = " → ".join(result.relation_types or []) or "없음"
        sections.append(
            f"[근거 {index}] source={result.source}\n"
            f"내용: {result.text}\n"
            f"그래프 경로: {path}\n"
            f"관계 타입: {relations}\n"
            f"evidence_ids: {evidence}\n"
            f"source_ids: {sources}\n"
            f"chunk_ids: {chunks}\n"
            f"hybrid_score: {result.metadata.get('hybrid_score', 0.0):.4f}"
        )
    return "\n\n".join(sections)


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
