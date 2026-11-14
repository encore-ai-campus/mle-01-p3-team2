
"""Pesticide registration and structured usage validation."""

from collections.abc import Mapping
from typing import Any
import re
import unicodedata

from .contracts import GuardrailRequest, ValidationResult


_STATUS_KEYS = (
    "pesticide_verification_status",
    "verification_status",
    "registration_status",
    "검증상태",
)

_STATUS_FROM_TEXT = re.compile(
    r"검증상태\s*[:=]\s*([A-Za-z][A-Za-z0-9_-]*)",
    re.IGNORECASE,
)

_CURRENT = "VERIFIED_CURRENT"
_UNKNOWN = "UNKNOWN"

# 원본 데이터와 정확히 비교할 사용조건
USAGE_FIELDS = (
    "product_name",
    "crop",
    "target_pest",
    "dilution",
    "application_amount",
    "application_timing",
    "max_applications",
    "preharvest_interval",
)

# 현재 공식 등록·적용조건의 검증을 나타내는 필드
# 이 값들은 신뢰할 수 있는 공식 DB 조회 결과에서만 채워야 한다.
OFFICIAL_FIELDS = (
    "official_verified",
    "official_source_id",
    "registration_checked_at",
    "verified_product_crop_target",
    "verified_usage_conditions",
)


def _as_mapping(item: Any) -> dict[str, Any] | None:
    if isinstance(item, Mapping):
        return dict(item)

    if hasattr(item, "text") and hasattr(item, "metadata"):
        return {
            "text": getattr(item, "text"),
            "metadata": getattr(item, "metadata"),
        }

    return None


def _normalize_status(value: Any) -> str:
    value = re.sub(
        r"[\s-]+",
        "_",
        str(value).strip().upper(),
    )
    return value or _UNKNOWN


def _normalize_value(value: Any) -> str | None:
    if not isinstance(value, (str, int, float)):
        return None

    text = unicodedata.normalize("NFKC", str(value))
    text = " ".join(text.strip().split()).casefold()

    return text or None


def _statuses_from_metadata(
    metadata: Mapping[str, Any],
) -> list[str]:
    statuses = []

    for key in _STATUS_KEYS:
        if key not in metadata:
            continue

        value = metadata[key]

        if value is None or value == "":
            continue

        values = value if isinstance(value, (list, tuple)) else (value,)

        statuses.extend(
            _normalize_status(item)
            for item in values
        )

    return statuses


def _pesticide_records(
    request: GuardrailRequest,
) -> list[dict[str, Any]]:
    """농약과 관련된 검색 근거만 추출한다."""

    records = []

    if not isinstance(request.evidence, (list, tuple)):
        return records

    for item in request.evidence:
        normalized = _as_mapping(item)

        if normalized is None:
            continue

        metadata = normalized.get("metadata", {})
        text = normalized.get("text", "")

        if not isinstance(metadata, Mapping):
            metadata = {}

        evidence_type = str(
            metadata.get("evidence_type", "")
        ).lower()

        is_pesticide = (
            "pesticide" in evidence_type
            or "검증상태" in str(text)
        )

        if not is_pesticide:
            continue

        records.append({
            "text": text if isinstance(text, str) else "",
            "metadata": dict(metadata),
        })

    return records


def extract_pesticide_statuses(
    request: GuardrailRequest,
) -> tuple[str, ...]:
    """요청 및 검색 근거의 모든 농약 검증상태를 추출한다."""

    statuses = []

    request_metadata = (
        request.metadata
        if isinstance(request.metadata, Mapping)
        else {}
    )

    statuses.extend(
        _statuses_from_metadata(request_metadata)
    )

    for record in _pesticide_records(request):
        metadata = record["metadata"]
        text = record["text"]

        record_statuses = _statuses_from_metadata(metadata)

        record_statuses.extend(
            _normalize_status(match.group(1))
            for match in _STATUS_FROM_TEXT.finditer(text)
        )

        statuses.extend(
            record_statuses or [_UNKNOWN]
        )

    return tuple(statuses)


def _check_usage_values(
    source_usage: Any,
    answer_usage: Any,
) -> ValidationResult:
    """구조화된 원본과 출력용 농약 사용조건을 비교한다."""

    if (
        not isinstance(source_usage, Mapping)
        or not isinstance(answer_usage, Mapping)
    ):
        return ValidationResult(
            allowed=False,
            decision="FALLBACK",
            reason="농약 사용조건의 비교 데이터 형식이 올바르지 않음",
            fallback_text=(
                "농약 사용조건의 정확성을 확인하지 못했습니다."
            ),
            details={"usage_validation": "INVALID_FORMAT"},
        )

    if not answer_usage:
        return ValidationResult(
            allowed=False,
            decision="FALLBACK",
            reason="비교할 농약 사용조건이 없음",
            fallback_text="농약 사용조건을 확인할 수 없습니다.",
            details={"usage_validation": "EMPTY"},
        )

    unsupported_fields = [
        field
        for field in answer_usage
        if field not in USAGE_FIELDS
    ]

    mismatches = []

    for field, answer_value in answer_usage.items():
        if field not in USAGE_FIELDS:
            continue

        source_value = source_usage.get(field)

        original = _normalize_value(source_value)
        generated = _normalize_value(answer_value)

        if (
            original is None
            or generated is None
            or original != generated
        ):
            mismatches.append({
                "field": field,
                "source": source_value,
                "answer": answer_value,
            })

    if mismatches or unsupported_fields:
        return ValidationResult(
            allowed=False,
            decision="FALLBACK",
            reason="농약 사용조건이 원본 데이터와 일치하지 않음",
            fallback_text=(
                "조회된 원본 자료와 사용조건이 일치하지 않아 "
                "해당 답변을 제공할 수 없습니다."
            ),
            details={
                "usage_validation": "MISMATCH",
                "mismatches": mismatches,
                "unsupported_fields": unsupported_fields,
            },
        )

    return ValidationResult(
        allowed=True,
        decision="ALLOW",
        reason="농약 사용조건 원본 값 일치",
        details={
            "usage_validation": "MATCH",
            "checked_fields": list(answer_usage),
        },
    )


def _official_conditions_verified(
    metadata: Mapping[str, Any],
) -> bool:
    """
    공식 검증 플래그의 완전성을 확인한다.

    주의: 플래그 자체의 진위나 날짜의 최신성을 조회하지 않는다.
    공식 DB를 호출하는 신뢰된 코드에서만 값을 설정해야 한다.
    """

    if metadata.get("official_verified") is not True:
        return False

    if metadata.get("verified_product_crop_target") is not True:
        return False

    if metadata.get("verified_usage_conditions") is not True:
        return False

    source_id = metadata.get("official_source_id")
    checked_at = metadata.get("registration_checked_at")

    return (
        isinstance(source_id, str)
        and bool(source_id.strip())
        and isinstance(checked_at, str)
        and bool(checked_at.strip())
    )


def validate_pesticide(
    request: GuardrailRequest,
) -> ValidationResult:
    """
    농약 정보 조회와 현재 사용 추천을 구분한다.

    결정:
    ALLOW    - 일반 정보 또는 신뢰된 검증 조건 통과
    WARN     - 수집 기록 제공 가능, 현재 유효성 미확인
    FALLBACK - 구조화된 사용조건 불일치
    BLOCK    - 미검증 농약의 현재 사용 추천 제한
    """

    metadata = (
        request.metadata
        if isinstance(request.metadata, Mapping)
        else {}
    )

    records = _pesticide_records(request)

    statuses = list(
        extract_pesticide_statuses(request)
    )

    has_pesticide_evidence = bool(
        records or _statuses_from_metadata(metadata)
    )

    recommendation_requested = bool(
        request.pesticide_use_requested
    )

    # 1. 일반 농업 질문
    if not has_pesticide_evidence and not recommendation_requested:
        return ValidationResult(
            allowed=True,
            decision="ALLOW",
            reason="농약 관련 질문이 아니므로 등록상태 검증 제외",
            details={"statuses": []},
        )

    if not statuses:
        statuses = [_UNKNOWN]

    unconfirmed = [
        status
        for status in statuses
        if status != _CURRENT
    ]

    status_conflict = len(set(statuses)) > 1

    # 2. 현재 사용 추천 요청
    if recommendation_requested:
        # 상태 표기뿐 아니라 신뢰된 공식 검증 정보가 필요
        official_ok = (
            not unconfirmed
            and not status_conflict
            and len(records) == 1
            and _official_conditions_verified(
                records[0]["metadata"]
            )
        )

        if not official_ok:
            return ValidationResult(
                allowed=False,
                decision="BLOCK",
                reason=(
                    "현재 공식 등록상태 및 적용조건 검증이 "
                    "충분하지 않아 농약 사용 추천을 제한함"
                ),
                fallback_text=(
                    "수집된 농약 정보는 참고할 수 있지만 "
                    "현재 사용 가능 여부는 확인되지 않았습니다. "
                    "실제 사용 전 최신 공식 등록사항과 "
                    "제품 라벨을 확인해 주세요."
                ),
                details={
                    "statuses": statuses,
                    "unconfirmed_statuses": unconfirmed,
                    "status_conflict": status_conflict,
                    "official_verified": False,
                },
            )

    # 3. 구조화된 사용조건 비교
    # 두 필드가 모두 있을 때에만 실제 비교 수행
    source_usage = metadata.get("pesticide_source_usage")
    answer_usage = metadata.get("pesticide_answer_usage")

    if source_usage is not None or answer_usage is not None:
        usage_result = _check_usage_values(
            source_usage,
            answer_usage,
        )

        if not usage_result.allowed:
            return usage_result

    # 4. 현재 사용 추천
    if recommendation_requested:
        # 공식 최신 검증 조건은 만족했지만,
        # 구조화된 사용조건 비교를 생략한 경우는 허용하지 않음
        if source_usage is None or answer_usage is None:
            return ValidationResult(
                allowed=False,
                decision="BLOCK",
                reason="농약 사용조건의 원본 대조 정보가 없음",
                fallback_text=(
                    "현재 사용조건을 충분히 검증하지 못해 "
                    "농약 사용을 추천할 수 없습니다."
                ),
                details={
                    "statuses": statuses,
                    "usage_validation": "NOT_PROVIDED",
                },
            )

        return ValidationResult(
            allowed=True,
            decision="ALLOW",
            reason="등록상태 및 구조화된 사용조건 검증 통과",
            details={
                "statuses": statuses,
                "official_verified": True,
                "usage_validation": "MATCH",
            },
        )

    # 5. 수집된 농약 정보 조회
    if unconfirmed or status_conflict:
        return ValidationResult(
            allowed=True,
            decision="WARN",
            reason="농약 등록상태 재확인 필요",
            warnings=(
                "제공된 내용은 수집된 농약 정보이며, "
                "현재 공식 등록 유효성은 확인되지 않았습니다. "
                "실제 사용 전 최신 공식 등록사항과 "
                "제품 라벨의 적용 작물·병해충·사용기준을 "
                "확인해 주세요.",
            ),
            details={
                "statuses": statuses,
                "unconfirmed_statuses": unconfirmed,
                "status_conflict": status_conflict,
            },
        )

    # VERIFIED_CURRENT 표기가 있어도 정보 조회에서는
    # 검증 시점과 공식 근거가 부족하면 경고 유지
    if not (
        len(records) == 1
        and _official_conditions_verified(
            records[0]["metadata"]
        )
    ):
        return ValidationResult(
            allowed=True,
            decision="WARN",
            reason="공식 등록 확인 근거 불충분",
            warnings=(
                "현재 등록 유효성을 확인할 공식 검증 정보가 "
                "충분하지 않습니다. 실제 사용 전 최신 공식 "
                "등록사항과 제품 라벨을 확인해 주세요.",
            ),
            details={"statuses": statuses},
        )

    return ValidationResult(
        allowed=True,
        decision="ALLOW",
        reason="농약 정보 검증 통과",
        details={"statuses": statuses},
    )
