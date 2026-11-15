
"""Execution Rail과 LangGraph Supervisor 연결.

정보 조회와 실제 사용 추천의 필수 조건을 분리한다.

- 목록 조회(제품명 없음): 작물 + 병해충만 필수
- 특정 제품 조회: 제품명 + 작물 + 병해충 일치 검사
- 사용 추천: 제품명 + 작물 + 병해충 + 최신 공식 검증 필수
"""

from __future__ import annotations

from typing import Any

from src.guardrails.execution_rail import (
    UsageListLookup,
    UsageLookup,
    CurrentVerifier,
    get_registered_usage,
    list_registered_usages,
)
from src.guardrails.pesticide_identity import (
    identity_state_update,
)


LOOKUP_INTENTS = (
    "PESTICIDE_LOOKUP",
    "PESTICIDE_RECOMMENDATION",
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


def _text(value: Any) -> str:
    """상태 값에서 공백을 제거한 문자열을 얻는다."""
    return value.strip() if isinstance(value, str) else ""


def _pass(
    result: Any,
    state: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    """검증을 통과한 결과를 상태 업데이트로 변환한다."""

    return {
        **identity,
        "execution_decision": result.decision,
        "execution_reason": result.reason,
        "execution_record": result.record,
        "execution_records": list(result.records),
        "warnings": list(dict.fromkeys([
            *state.get("warnings", []),
            *result.warnings,
        ])),
    }


def execution_node(
    state: dict[str, Any],
    *,
    lookup_fn: UsageLookup | None = None,
    list_lookup_fn: UsageListLookup | None = None,
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

    if intent not in LOOKUP_INTENTS:
        return _stop(
            "FALLBACK",
            "허용되지 않은 농약 조회 요청입니다.",
        )

    recommend = intent == "PESTICIDE_RECOMMENDATION"

    # 검색 결과에서 작물·병해충·제품명을 먼저 채운다.
    identity = identity_state_update(state)
    merged = {**state, **identity}

    product_name = _text(merged.get("product_name"))
    crop = _text(merged.get("crop"))
    target_pest = _text(merged.get("target_pest"))

    # 조회·추천 모두 작물 + 병해충은 필수
    if not (crop and target_pest):
        return {
            **identity,
            **_stop(
                "FALLBACK",
                "작물·병해충 정보가 부족합니다.",
            ),
        }

    # 1. 제품명이 없는 정보 조회는 등록 약제 목록 조회로 처리
    if not recommend and not product_name:
        if list_lookup_fn is None:
            return {
                **identity,
                **_stop(
                    "FALLBACK",
                    "공식 농약 등록정보 DB가 연결되지 않았습니다.",
                ),
            }

        result = list_registered_usages(
            crop=crop,
            target_pest=target_pest,
            list_lookup_fn=list_lookup_fn,
        )

        if not result.allowed:
            return {
                **identity,
                **_stop(result.decision, result.reason),
            }

        return _pass(result, state, identity)

    # 2. 추천은 제품명까지 특정돼야 한다
    if recommend and not product_name:
        return {
            **identity,
            **_stop(
                "FALLBACK",
                "추천 대상 제품명을 확인하지 못했습니다.",
            ),
        }

    if lookup_fn is None:
        return {
            **identity,
            **_stop(
                "BLOCK" if recommend else "FALLBACK",
                "공식 농약 등록정보 DB가 연결되지 않았습니다.",
            ),
        }

    # 3. 특정 제품 조회·추천은 3요소 정확 일치 검사
    result = get_registered_usage(
        product_name=product_name,
        crop=crop,
        target_pest=target_pest,
        lookup_fn=lookup_fn,
        recommend=recommend,
        current_verifier=current_verifier,
    )

    if not result.allowed:
        return {
            **identity,
            **_stop(result.decision, result.reason),
        }

    # 추천은 DB 조회뿐 아니라 최종 답변 수치 검증도 필요
    if recommend:
        return {
            **identity,
            **_stop(
                "FALLBACK",
                "답변의 농약 사용조건 일치 검증이 "
                "연결되지 않아 추천을 보류합니다.",
            ),
        }

    return _pass(result, state, identity)


def after_execution(state: dict[str, Any]) -> str:
    """검증 통과 시 답변 생성, 실패 시 종료."""

    if state.get("execution_decision") in (
        "SKIP",
        "ALLOW",
        "WARN",
    ):
        return "answer"

    return "end"
