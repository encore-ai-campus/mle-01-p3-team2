
"""LangGraph Supervisor와 Guardrails 연결 어댑터."""

from typing import Any

from src.guardrails.contracts import GuardrailRequest
from src.guardrails.pipeline import validate_request
from src.guardrails.output_usage_validator import (
    validate_pesticide_output,
)
from src.guardrails.verification_notice import annotate_answer


def _withhold(reason: str) -> dict[str, Any]:
    """검증에 실패한 답변을 안전하게 보류한다."""
    return {
        "guardrail_decision": "FALLBACK",
        "guardrail_reason": reason,
        "final_status": "WITHHELD",
        "final_answer": (
            "답변의 안전성과 정확성을 확인하지 못했습니다. "
            "공식 자료를 다시 확인해 주세요."
        ),
    }


def guardrails_node(state: dict[str, Any]) -> dict[str, Any]:
    """Supervisor 답변에 근거 검증과 농약 사용조건 검증을 적용한다."""

    question = state.get("question", "")
    answer = state.get("final_answer") or state.get("answer", "")
    results = state.get("results", [])

    # 1. 답변과 검색 근거 확인
    if not answer or not results:
        return _withhold("답변 또는 검색 근거 없음")

    # 2. 기존 Input Rail의 농약 관련 판단 확인
    intent = state.get("input_intent")
    pesticide_use_requested = state.get("pesticide_use_requested")

    if not isinstance(pesticide_use_requested, bool):
        question_text = question.strip()

        general_keywords = (
            "품종", "적합 지역", "재배 지역",
            "생육", "파종", "수확 시기",
            "재배 환경", "토양", "기온",
        )

        pesticide_keywords = (
            "농약", "살충제", "살균제",
            "제초제", "희석", "살포",
            "약제", "방제약",
        )

        is_general = any(
            keyword in question_text
            for keyword in general_keywords
        )
        is_pesticide = any(
            keyword in question_text
            for keyword in pesticide_keywords
        )

        if is_general and not is_pesticide:
            pesticide_use_requested = False
        else:
            return _withhold(
                "농약 사용 추천 의도 분류 결과 없음"
            )

    # 3. 농약 조회와 추천을 구분
    is_lookup = intent == "PESTICIDE_LOOKUP"
    is_recommendation = (
        intent == "PESTICIDE_RECOMMENDATION"
    )

    # 농약 관련 답변은 별도의 사용조건 검증 대상
    is_pesticide_request = (
        is_lookup
        or is_recommendation
        or pesticide_use_requested
    )

    # 4. Execution Rail에서 검증한 등록정보 확인
    execution_record = state.get("execution_record")
    execution_records = state.get("execution_records") or []
    execution_decision = state.get("execution_decision")

    if is_pesticide_request:
        # Execution Rail을 우회한 농약 답변은 허용하지 않음
        if execution_decision not in ("ALLOW", "WARN"):
            return _withhold(
                "Execution Rail의 농약 등록정보 검증이 "
                "완료되지 않았습니다."
            )

    # 5. 농약 사용조건 검증
    #
    # 공식 등록정보로 수치를 대조하는 검사를 근거 일치 판정보다 먼저
    # 돌린다. 판정 모델에게 "이 항목은 이미 대조했다"는 사실을 함께
    # 넘겨, 근거 줄을 혼동해 같은 수치를 불일치로 보는 것을 줄인다.
    #
    # 등록정보의 검증상태도 모델이 빠뜨려도 근거에 있는 사실이므로
    # 규칙으로 먼저 덧붙인다.
    usage_warnings: list[str] = []
    verified_conditions: list[str] = []

    if is_pesticide_request:
        answer = annotate_answer(answer, execution_records)

        try:
            usage_result = validate_pesticide_output(
                answer=answer,
                record=execution_record,
                records=execution_records,
                pesticide_use_requested=True,
                recommendation_requested=is_recommendation,
            )
        except Exception:
            return _withhold("농약 사용조건 검증 실행 실패")

        if not usage_result["allowed"]:
            decision = usage_result["decision"]

            return {
                "guardrail_decision": (
                    "BLOCK" if decision == "BLOCK" else "FALLBACK"
                ),
                "guardrail_reason": usage_result["reason"],
                "warnings": list(state.get("warnings", [])),
                "final_status": (
                    "BLOCKED" if decision == "BLOCK" else "WITHHELD"
                ),
                "final_answer": (
                    "농약 사용조건을 안전하게 확인하지 못했습니다. "
                    "최신 공식 등록정보와 제품 라벨을 확인해 주세요."
                ),
            }

        usage_warnings = list(usage_result.get("warnings") or [])

        # 대조를 마친 항목만 판정 모델에 사실로 알린다.
        unverified = set(usage_result.get("unverified_fields") or [])
        verified_conditions = [
            name
            for name in ("희석배수", "사용횟수", "수확 전 사용시기")
            if name not in unverified
        ]

    # 6. 검색 근거 일치 검증
    request = GuardrailRequest(
        question=question,
        answer=answer,
        evidence=results,
        pesticide_use_requested=pesticide_use_requested,
        metadata={"verified_conditions": verified_conditions},
    )

    try:
        result = validate_request(request)
    except Exception:
        return _withhold(
            "기존 Guardrails 검증 실행 실패"
        )

    warnings = list(
        dict.fromkeys([
            *state.get("warnings", []),
            *result.warnings,
        ])
    )

    base_update = {
        "guardrail_decision": result.decision,
        "guardrail_reason": result.reason,
        "warnings": warnings,

        # 어느 판정이 보류를 만들었는지 평가에서 되짚기 위해 남긴다.
        "guardrail_details": dict(
            getattr(result, "details", None) or {}
        ),
    }

    # 기존 검증이 거절한 답변은 신규 검사로 허용할 수 없음
    if not (
        result.allowed
        and result.decision in ("ALLOW", "WARN")
    ):
        return {
            **base_update,
            "final_status": (
                "BLOCKED"
                if result.decision == "BLOCK"
                else "WITHHELD"
            ),
            "final_answer": (
                result.fallback_text
                or "안전한 답변을 확인할 수 없어 응답을 보류합니다."
            ),
        }

    # 7. 검증하지 못한 사용조건과 등록 유효성 주의사항을 전달한다.
    if is_pesticide_request:
        warnings = list(dict.fromkeys([
            *warnings,
            *usage_warnings,
        ]))

        if execution_decision == "WARN":
            warnings = list(dict.fromkeys([
                *warnings,
                "농약 등록정보의 최신 유효성을 확인해야 합니다.",
            ]))

    # 8. 모든 검증을 통과한 답변만 반환
    final_answer = answer

    if warnings:
        final_answer += (
            "\n\n[주의사항]\n"
            + "\n".join(warnings)
        )

    return {
        **base_update,
        "warnings": warnings,
        "final_status": "ANSWERED",
        "final_answer": final_answer,
    }
