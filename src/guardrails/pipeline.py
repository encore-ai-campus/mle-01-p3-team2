"""Ordered orchestration of the independent guardrail validators."""

from dataclasses import replace

from .contracts import GuardrailRequest, ValidationResult
from .evidence_validator import validate_evidence
from .output_validator import validate_output
from .pesticide_validator import validate_pesticide


def validate_request(request: GuardrailRequest) -> ValidationResult:
    """Run evidence, pesticide, and output checks in fail-closed order."""

    evidence_result = validate_evidence(request)
    if not evidence_result.allowed:
        return evidence_result

    pesticide_result = validate_pesticide(request)
    if not pesticide_result.allowed:
        return pesticide_result

    output_result = validate_output(request)
    warnings = pesticide_result.warnings + output_result.warnings
    details = dict(output_result.details)
    details["evidence"] = evidence_result.details
    details["pesticide"] = pesticide_result.details

    if not output_result.allowed:
        return replace(output_result, warnings=warnings, details=details)

    decision = "WARN" if warnings or pesticide_result.decision == "WARN" else output_result.decision
    return replace(output_result, decision=decision, warnings=warnings, details=details)
