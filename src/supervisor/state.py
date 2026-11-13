"""Supervisor가 공유하는 LangGraph 상태 정의입니다."""

from __future__ import annotations

from typing import TypedDict

from src.retrievers.search_result import SearchResult


class GraphState(TypedDict, total=False):
    """검색·검증·답변 노드 사이에서 공유하는 상태입니다."""

    question: str
    # 현재 검색에 사용하는 경로입니다. fallback이 일어나면 바뀝니다.
    route: str
    # Router가 처음 판단한 경로입니다. fallback이 일어나도 바뀌지 않습니다.
    initial_route: str
    # 경로 이동 이력입니다. 예: ["GRAPH", "HYBRID"]
    route_path: list[str]
    route_reason: str
    crop: str | None
    max_hops: int
    # 답변 근거로 사용할 결과. verify에서 걸러진 뒤의 것
    results: list[SearchResult]
    # 검증 전 검색 원본. verify가 덮어쓰지 않음
    # Hit@K, multi-hop 채점은 이 값을 봐야 함. 재시도 시 누적됨
    retrieved_results: list[SearchResult]
    warnings: list[str]
    # PASS | INSUFFICIENT | NOT_APPLICABLE (범위 밖이라 검증을 건너뜀)
    evidence_status: str
    retry_count: int
    max_retries: int
    answer: str
    # ANSWERED | WITHHELD | REJECTED
    final_status: str
    final_answer: str
