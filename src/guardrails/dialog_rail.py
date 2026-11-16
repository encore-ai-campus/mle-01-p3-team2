
"""농업 GraphRAG Dialog Rail - 대화 흐름 제어."""

from __future__ import annotations

from typing import Any

from src.guardrails.nemo_dialog_adapter import (
    NEMO_CHECK_ERROR_REASON,
    check_nemo_dialog,
)


SEARCH_INTENTS = {
    "GENERAL",
    "PESTICIDE_LOOKUP",
    "PESTICIDE_RECOMMENDATION",
}

STOP_INTENTS = {
    "DANGEROUS_USE",
    "HARMFUL_INTENT",
    "INJECTION",
    "EXPOSURE",
    "UNCERTAIN",
    "INVALID",
}


def dialog_node(state: dict[str, Any]) -> dict[str, Any]:
    """Input Rail과 NeMo의 판단으로 대화 진행 여부를 결정한다."""

    intent = state.get("input_intent")

    # 1. 기존 Input Rail에서 차단한 요청은 그대로 종료
    if intent in STOP_INTENTS:
        return {
            "dialog_action": "STOP",
            "final_status": state.get(
                "final_status", "WITHHELD"
            ),
            "final_answer": state.get(
                "final_answer",
                "요청을 안전하게 처리할 수 없습니다.",
            ),
        }

    # 2. 알 수 없는 분류 결과는 검색하지 않는다.
    if intent not in SEARCH_INTENTS:
        return {
            "dialog_action": "STOP",
            "guardrail_decision": "FALLBACK",
            "guardrail_reason": "알 수 없는 입력 분류 결과",
            "final_status": "WITHHELD",
            "final_answer": "요청을 분류하지 못해 답변을 보류합니다.",
        }

    # 3. 정상 요청도 NeMo에서 한 번 더 검사한다.
    question = state.get("question", "")
    allowed, reason = check_nemo_dialog(question)

    if not allowed:
        # 검사 자체를 수행하지 못한 경우(인증·네트워크 등 인프라 오류)와
        # 질문이 정책을 위반해 차단된 경우를 구분해 메시지를 달리한다.
        if reason == NEMO_CHECK_ERROR_REASON:
            return {
                "dialog_action": "STOP",
                "guardrail_decision": "FALLBACK",
                "guardrail_reason": reason,
                "final_status": "ERROR",
                "final_answer": (
                    "일시적인 오류로 안전성 검사를 완료하지 못했습니다. "
                    "잠시 후 다시 시도해 주세요."
                ),
            }

        return {
            "dialog_action": "STOP",
            "guardrail_decision": "FALLBACK",
            "guardrail_reason": reason,
            "final_status": "WITHHELD",
            "final_answer": (
                "안전성 검사를 통과하지 못해 답변을 보류합니다."
            ),
        }

    # 4. 모든 입력 검사를 통과한 경우에만 검색 진행
    return {
        "dialog_action": "SEARCH",
        "pesticide_use_requested": (
            intent == "PESTICIDE_RECOMMENDATION"
        ),
    }


def after_dialog(state: dict[str, Any]) -> str:
    """안전 검사를 통과한 요청만 검색기로 전달한다."""

    if state.get("dialog_action") == "SEARCH":
        return "route"

    return "end"
