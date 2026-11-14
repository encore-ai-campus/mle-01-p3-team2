
"""Supervisor Output Rail 통합 테스트."""

from types import SimpleNamespace
from unittest.mock import patch

from src.guardrails.supervisor_adapter import guardrails_node


def make_state(answer: str, intent: str = "PESTICIDE_LOOKUP"):
    """검증용 가상 Supervisor State."""
    return {
        "question": "테스트 농약의 등록정보를 알려주세요.",
        "answer": answer,
        "results": [{"text": "가상의 농약 등록정보"}],
        "input_intent": intent,
        "pesticide_use_requested": (
            intent == "PESTICIDE_RECOMMENDATION"
        ),
        "execution_decision": "WARN",
        "execution_record": {
            "product_name": "테스트제품",
            "crop": "마늘",
            "target_pest": "녹병",
            "dilution": "1,000배",
            "use_count": "3회",
            "use_timing": "수확 7일 전",
        },
        "warnings": [],
    }


def approved_grounding():
    """기존 근거 검증이 통과한 상황을 재현."""
    return SimpleNamespace(
        allowed=True,
        decision="ALLOW",
        reason="근거 일치",
        warnings=[],
        fallback_text=None,
    )


def test_correct_usage_answer_passes():
    """사용조건 숫자가 일치하면 최종 답변을 허용한다."""
    state = make_state(
        "수집된 등록정보에 희석배수는 1,000배로 표시됩니다."
    )

    with patch(
        "src.guardrails.supervisor_adapter.validate_request",
        return_value=approved_grounding(),
    ):
        result = guardrails_node(state)

    assert result["final_status"] == "ANSWERED"
    assert result["guardrail_decision"] == "ALLOW"


def test_wrong_dilution_blocks_answer():
    """잘못된 희석배수는 최종 출력에서 차단한다."""
    state = make_state("희석배수는 500배입니다.")

    with patch(
        "src.guardrails.supervisor_adapter.validate_request",
        return_value=approved_grounding(),
    ):
        result = guardrails_node(state)

    assert result["final_status"] == "BLOCKED"
    assert "500배" not in result["final_answer"]


def test_missing_execution_result_withholds():
    """Execution Rail을 거치지 않은 농약 답변은 보류한다."""
    state = make_state("희석배수는 1,000배입니다.")
    state["execution_decision"] = "SKIP"

    result = guardrails_node(state)

    assert result["final_status"] == "WITHHELD"


def test_recommendation_remains_withheld():
    """숫자가 맞아도 미검증 농약 추천은 보류한다."""
    state = make_state(
        "이 농약의 사용을 추천합니다.",
        intent="PESTICIDE_RECOMMENDATION",
    )

    with patch(
        "src.guardrails.supervisor_adapter.validate_request",
        return_value=approved_grounding(),
    ):
        result = guardrails_node(state)

    assert result["final_status"] == "WITHHELD"


def test_existing_grounding_block_preserved():
    """기존 Guardrails 차단 결과를 그대로 유지한다."""
    state = make_state("희석배수는 1,000배입니다.")

    denied = SimpleNamespace(
        allowed=False,
        decision="FALLBACK",
        reason="검색 근거 불일치",
        warnings=[],
        fallback_text="근거를 확인할 수 없습니다.",
    )

    with patch(
        "src.guardrails.supervisor_adapter.validate_request",
        return_value=denied,
    ):
        result = guardrails_node(state)

    assert result["final_status"] == "WITHHELD"
    assert result["final_answer"] == "근거를 확인할 수 없습니다."
