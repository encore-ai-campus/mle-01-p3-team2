
"""Unified Guardrails validation pipeline."""

from dataclasses import replace

from .contracts import GuardrailRequest, ValidationResult
from .evidence_validator import validate_evidence
from .pesticide_validator import validate_pesticide
from .output_validator import validate_output


def validate_request(
    request: GuardrailRequest,
) -> ValidationResult:
    """
    GraphRAG 검증 파이프라인.

    1. 검색 근거 검증
    2. 농약 안전성 검증
    3. 답변 근거 일치 검증
    4. 최종 결정 및 경고 병합
    """

    # 1. Evidence 검증
    evidence_result = validate_evidence(request)

    if not evidence_result.allowed:
        return evidence_result

    # 2. 농약 검증
    pesticide_result = validate_pesticide(request)

    if not pesticide_result.allowed:
        return pesticide_result

    # 3. Output 검증: LLM-as-Judge 포함
    output_result = validate_output(request)

    warnings = tuple(
        dict.fromkeys(
            pesticide_result.warnings + output_result.warnings
        )
    )

    details = dict(output_result.details)

    details["evidence"] = evidence_result.details
    details["pesticide"] = pesticide_result.details

    # 4. 답변 검증 실패
    if not output_result.allowed:
        return replace(
            output_result,
            warnings=warnings,
            details=details,
        )

    # 5. 모든 검증 통과, 경고 여부 결정
    decision = (
        "WARN"
        if warnings or pesticide_result.decision == "WARN"
        else "ALLOW"
    )

    return replace(
        output_result,
        decision=decision,
        warnings=warnings,
        details=details,
    )
