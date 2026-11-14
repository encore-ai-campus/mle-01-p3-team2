
"""Execution Rail과 LangGraph Supervisor 연결."""

from __future__ import annotations

from typing import Any

from src.guardrails.execution_rail import (
    UsageLookup,
    CurrentVerifier,
    get_registered_usage,
)


def _stop(decision: str, reason: str) -> dict[str, Any]:
    """검증 실패 시 안전한 응답을 반환한다."""
    return {
        "execution_decision": decision,
        "execution_reason": reason,
        "guardrail_decision": decision,
        "guardrail_reason": reason,
        "final_status": (
            "BLOCKED" if decision == "BLOCK" else "WITHHELD"
        ),
        "final_answer": (
            f"{reason} 공식 등록정보와 제품 라벨의 "
            "최신 사용기준을 확인해 주세요."
        ),
    }


def execution_node(
    state: dict[str, Any],
    *,
    lookup_fn: UsageLookup | None = None,
    current_verifier: CurrentVerifier | None = None,
) -> dict[str, Any]:
    """농약 등록정보를 검증하고 실행 경로를 결정한다."""

    # 근거가 없는 질문은 DB 조회 없이 기존 흐름으로
    if state.get("evidence_status") != "PASS":
        return {
            "execution_decision": "SKIP",
            "execution_reason": "검색 근거 부족",
        }

    intent = state.get("input_intent", "GENERAL")

    # 일반 농업 질문은 Execution Rail 검사 생략
    if intent == "GENERAL":
        return {
            "execution_decision": "SKIP",
            "execution_reason": "일반 농업 질문",
        }

    if intent not in (
        "PESTICIDE_LOOKUP",
        "PESTICIDE_RECOMMENDATION",
    ):
        return _stop(
            "FALLBACK",
            "허용되지 않은 농약 조회 요청입니다.",
        )

    recommend = intent == "PESTICIDE_RECOMMENDATION"

    product_name = state.get("product_name", "")
    crop = state.get("crop", "")
    target_pest = state.get("target_pest", "")

    if not all(
        isinstance(value, str) and value.strip()
        for value in (product_name, crop, target_pest)
    ):
        return _stop(
            "FALLBACK",
            "제품명·작물·병해충 정보가 부족합니다.",
        )

    if lookup_fn is None:
        return _stop(
            "BLOCK" if recommend else "FALLBACK",
            "공식 농약 등록정보 DB가 연결되지 않았습니다.",
        )

    result = get_registered_usage(
        product_name=product_name,
        crop=crop,
        target_pest=target_pest,
        lookup_fn=lookup_fn,
        recommend=recommend,
        current_verifier=current_verifier,
    )

    if not result.allowed:
        return _stop(result.decision, result.reason)

    # 추천은 DB 조회뿐 아니라 최종 답변 수치 검증도 필요
    if recommend:
        return _stop(
            "FALLBACK",
            "답변의 농약 사용조건 일치 검증이 "
            "연결되지 않아 추천을 보류합니다.",
        )

    return {
        "execution_decision": result.decision,
        "execution_reason": result.reason,
        "execution_record": result.record,
        "warnings": list(dict.fromkeys([
            *state.get("warnings", []),
            *result.warnings,
        ])),
    }


def after_execution(state: dict[str, Any]) -> str:
    """검증 통과 시 답변 생성, 실패 시 종료."""

    if state.get("execution_decision") in (
        "SKIP",
        "ALLOW",
        "WARN",
    ):
        return "answer"

    return "end"
