
"""Supervisor가 공유하는 LangGraph 상태 정의입니다."""

from __future__ import annotations

from typing import Any, TypedDict

from src.retrievers.answer import CitedAnswer, Evidence
from src.retrievers.search_result import SearchResult


class GraphState(TypedDict, total=False):
    """검색·검증·답변 노드 사이에서 공유하는 상태입니다."""

    # 사용자 질문
    question: str

    # Router 및 검색 경로
    route: str
    initial_route: str
    route_path: list[str]
    route_reason: str
    initial_route_reason: str

    crop: str | None
    max_hops: int
    entities: list[str]
    relation_types: list[str]
    needs_official_evidence: bool
    route_confidence: float | None

    # 검색 결과
    results: list[SearchResult]
    retrieved_results: list[SearchResult]
    rejected_results: list[SearchResult]
    warnings: list[str]

    # Evidence 검증
    evidence_status: str
    retry_count: int
    max_retries: int

    # 답변
    answer: str
    final_status: str
    final_answer: str

    # 검색 시도 이력 (기존 코드의 필드명 유지)
    attemps: list[dict[str, Any]]

    # Evidence 생성
    evidences: list[Evidence]
    dropped_evidences: list[dict[str, Any]]

    # Citation 검증
    cited_answer: CitedAnswer | None
    citation_status: str
    citation_failures: list[dict[str, Any]]
    citation_conflicts: list[dict[str, Any]]
    checked_claims: int
    supported_claims: int

    # 재검색 및 답변 보류
    search_query: str
    retry_reason: str
    withhold_reason: str

    # Guardrails - Input / Dialog
    guardrail_decision: str
    guardrail_reason: str
    pesticide_use_requested: bool
    input_intent: str
    dialog_action: str

    # Guardrails - Retrieval
    retrieval_rejections: list[dict[str, object]]

    # Guardrails - 농약 검증
    product_name: str
    target_pest: str

    # Guardrails - Execution
    execution_decision: str
    execution_reason: str
    execution_record: dict[str, Any] | None
