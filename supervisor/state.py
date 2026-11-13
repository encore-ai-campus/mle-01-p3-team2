"""Supervisor가 공유하는 LangGraph 상태 정의입니다."""

from __future__ import annotations

from typing import TypedDict

from search_result import SearchResult


class GraphState(TypedDict, total=False):
    """검색·검증·답변 노드 사이에서 공유하는 상태입니다."""

    question: str
    route: str
    route_reason: str
    crop: str | None
    max_hops: int
    results: list[SearchResult]
    warnings: list[str]
    evidence_status: str
    retry_count: int
    max_retries: int
    answer: str
    final_status: str
    final_answer: str
