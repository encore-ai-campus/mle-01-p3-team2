"""Fail-closed validation for pesticide registration status."""

from collections.abc import Mapping
import re
from typing import Any

from .contracts import GuardrailRequest, ValidationResult


_STATUS_KEYS = (
    "pesticide_verification_status",
    "verification_status",
    "registration_status",
    "검증상태",
)
_STATUS_FROM_TEXT = re.compile(r"검증상태\s*[:=]\s*([A-Za-z][A-Za-z0-9_-]*)", re.IGNORECASE)
_CURRENT = "VERIFIED_CURRENT"
_UNKNOWN = "UNKNOWN"
_UNCONFIRMED = frozenset(
    {_UNKNOWN, "PRELIMINARY_PASS", "REVIEW_REQUIRED", "EXPIRED", "REJECTED"}
)


def _as_mapping(item: Any) -> dict[str, Any] | None:
    if isinstance(item, Mapping):
        return dict(item)
    field_names = ("text", "metadata")
    if all(hasattr(item, name) for name in field_names):
        return {name: getattr(item, name) for name in field_names}
    return None


def _normalize_status(value: Any) -> str:
    normalized = re.sub(r"[\s-]+", "_", str(value).strip().upper())
    return normalized or _UNKNOWN


def _statuses_from_metadata(metadata: Mapping[str, Any]) -> list[str]:
    statuses: list[str] = []
    for key in _STATUS_KEYS:
        if key in metadata and metadata[key] not in (None, ""):
            values = metadata[key]
            if not isinstance(values, (list, tuple)):
                values = (values,)
            statuses.extend(_normalize_status(value) for value in values)
    return statuses


def extract_pesticide_statuses(request: GuardrailRequest) -> tuple[str, ...]:
    """Extract all explicit pesticide statuses, preserving evidence order."""

    statuses: list[str] = []
    request_metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
    statuses.extend(_statuses_from_metadata(request_metadata))

    for item in request.evidence:
        normalized = _as_mapping(item)
        if normalized is None:
            continue
        metadata = normalized.get("metadata", {})
        text = normalized.get("text", "")
        evidence_type = metadata.get("evidence_type", "") if isinstance(metadata, Mapping) else ""
        is_pesticide = "pesticide" in str(evidence_type).lower() or "검증상태" in str(text)
        if not is_pesticide:
            continue

        evidence_statuses = _statuses_from_metadata(metadata) if isinstance(metadata, Mapping) else []
        if isinstance(text, str):
            evidence_statuses.extend(
                _normalize_status(match.group(1))
                for match in _STATUS_FROM_TEXT.finditer(text)
            )
        statuses.extend(evidence_statuses or [_UNKNOWN])

    return tuple(statuses)


def _has_pesticide_evidence(request: GuardrailRequest) -> bool:
    request_metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
    if _statuses_from_metadata(request_metadata):
        return True
    for item in request.evidence:
        normalized = _as_mapping(item)
        if normalized is None:
            continue
        metadata = normalized.get("metadata", {})
        evidence_type = metadata.get("evidence_type", "") if isinstance(metadata, Mapping) else ""
        text = normalized.get("text", "")
        if "pesticide" in str(evidence_type).lower() or "검증상태" in str(text):
            return True
    return bool(request.pesticide_use_requested)


def validate_pesticide(request: GuardrailRequest) -> ValidationResult:
    """Allow recommendations only when every relevant status is current."""

    statuses = list(extract_pesticide_statuses(request))
    has_pesticide_evidence = _has_pesticide_evidence(request)
    if not has_pesticide_evidence:
        return ValidationResult(
            allowed=True,
            decision="ALLOW",
            reason="농약 근거가 아니므로 등록상태 검증을 적용하지 않음",
            details={"statuses": statuses},
        )

    if not statuses:
        statuses = [_UNKNOWN]

    unconfirmed = [status for status in statuses if status != _CURRENT]
    if request.pesticide_use_requested and unconfirmed:
        return ValidationResult(
            allowed=False,
            decision="BLOCK",
            reason="농약 등록상태가 현재 유효하다고 확인되지 않음",
            fallback_text="현재 등록 유효성을 확인할 수 없어 사용 추천을 제한합니다. 공식 등록사항과 제품 라벨을 다시 확인해 주세요.",
            details={"statuses": statuses, "unconfirmed_statuses": unconfirmed},
        )

    if unconfirmed:
        return ValidationResult(
            allowed=True,
            decision="WARN",
            reason="농약 등록상태 재확인 필요",
            warnings=("현재 공식 등록 유효성과 제품 라벨을 재확인해야 합니다.",),
            details={"statuses": statuses, "unconfirmed_statuses": unconfirmed},
        )

    return ValidationResult(
        allowed=True,
        decision="ALLOW",
        reason="농약 등록상태가 현재 유효하다고 확인됨",
        details={"statuses": statuses},
    )
