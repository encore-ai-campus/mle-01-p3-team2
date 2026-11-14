
"""농약 공식 등록정보 조회용 Execution Rail."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
import unicodedata


# 실제 DB 연결 시 주입할 읽기 전용 조회 함수
UsageLookup = Callable[
    [str, str, str],
    list[Mapping[str, Any]],
]

# 최신 공식 등록사항을 별도로 확인하는 함수
CurrentVerifier = Callable[
    [Mapping[str, Any]],
    bool,
]

IDENTITY_FIELDS = (
    "product_name",
    "crop",
    "target_pest",
)

USAGE_FIELDS = (
    "dilution",
    "application_amount",
    "application_timing",
    "max_applications",
    "preharvest_interval",
)


@dataclass
class ExecutionResult:
    allowed: bool
    decision: str
    reason: str
    record: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)


def _normalize(value: Any) -> str:
    """공백과 유니코드 표현만 정규화한다."""
    if not isinstance(value, str):
        return ""

    value = unicodedata.normalize("NFKC", value)
    return " ".join(value.strip().split()).casefold()



def _matches(
    record: Mapping[str, Any],
    product_name: str,
    crop: str,
    target_pest: str,
) -> bool:
    """품목명 또는 상표명 + 작물 + 병해충 일치 검사."""

    product_match = any(
        _normalize(record.get(field)) == _normalize(product_name)
        for field in ("product_name", "brand")
        if _normalize(record.get(field))
    )

    crop_match = (
        _normalize(record.get("crop")) == _normalize(crop)
    )

    pest_match = (
        _normalize(record.get("target_pest"))
        == _normalize(target_pest)
    )

    return (
        product_match
        and crop_match
        and pest_match
        and bool(_normalize(crop))
        and bool(_normalize(target_pest))
    )



def _conflicting_records(
    records: list[Mapping[str, Any]],
) -> bool:
    """동일 조합의 조회 결과 간 상태·사용조건 충돌 검사."""

    if len(records) < 2:
        return False

    fields = (
        *IDENTITY_FIELDS,
        *USAGE_FIELDS,
        "verification_status",
        "registration_status",
    )

    def signature(record: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(
            _normalize(record.get(field))
            for field in fields
        )

    return len({signature(record) for record in records}) > 1


def get_registered_usage(
    *,
    product_name: str,
    crop: str,
    target_pest: str,
    lookup_fn: UsageLookup,
    recommend: bool = False,
    current_verifier: CurrentVerifier | None = None,
) -> ExecutionResult:
    """
    읽기 전용 농약 등록정보 조회 및 실행 검증.

    lookup_fn:
        신뢰된 내부 데이터 접근 계층에서 제공하는
        읽기 전용 조회 함수.

    current_verifier:
        최신 공식 등록정보와 실제 적용조건을
        독립적으로 확인하는 검증 함수.

    recommend:
        True  = 현재 농약 사용 추천 요청
        False = 수집된 농약 정보 조회

    중요:
        문서 메타데이터의 VERIFIED_CURRENT 표기만으로
        실제 사용 추천을 허용하지 않는다.
    """

    # 1. 필수 조회 조건 검사
    identifiers = (product_name, crop, target_pest)

    if not all(
        isinstance(value, str) and value.strip()
        for value in identifiers
    ):
        return ExecutionResult(
            allowed=False,
            decision="FALLBACK",
            reason="제품명·작물·병해충 정보가 부족합니다.",
        )

    # 2. 읽기 전용 DB 조회
    try:
        raw_records = lookup_fn(
            product_name,
            crop,
            target_pest,
        )
    except Exception:
        return ExecutionResult(
            allowed=False,
            decision="FALLBACK",
            reason="농약 등록정보 조회에 실패했습니다.",
        )

    if not isinstance(raw_records, list):
        return ExecutionResult(
            allowed=False,
            decision="FALLBACK",
            reason="DB 조회 결과 형식이 올바르지 않습니다.",
        )

    # 3. DB 반환값 형식 및 조합 확인
    if any(
        not isinstance(item, Mapping)
        for item in raw_records
    ):
        return ExecutionResult(
            allowed=False,
            decision="FALLBACK",
            reason="조회 결과에 잘못된 형식의 자료가 있습니다.",
        )

    matched = [
        item
        for item in raw_records
        if _matches(
            item,
            product_name,
            crop,
            target_pest,
        )
    ]

    if not matched:
        return ExecutionResult(
            allowed=False,
            decision="BLOCK" if recommend else "FALLBACK",
            reason=(
                "요청한 제품·작물·병해충 조합에 "
                "일치하는 등록 기록을 확인하지 못했습니다."
            ),
        )

    # 4. 동일 조합의 정보 충돌 검사
    if _conflicting_records(matched):
        return ExecutionResult(
            allowed=False,
            decision="FALLBACK",
            reason=(
                "조회된 농약 기록 사이에 등록상태 "
                "또는 사용조건 불일치가 있습니다."
            ),
        )

    record = dict(matched[0])
    # DB에서 명시적으로 추천 불가 처리한 레코드는 차단
    if recommend and record.get("recommendation_eligible") is not True:
        return ExecutionResult(
            allowed=False,
            decision="BLOCK",
            reason=(
                "이 농약 기록은 현재 사용 추천 대상으로 "
                "승인되지 않았습니다."
            ),
        )

    # 5. 정보 조회는 수집 기록으로 제공 가능
    if not recommend:
        return ExecutionResult(
            allowed=True,
            decision="WARN",
            reason="수집된 농약 등록정보 조회",
            record=record,
            warnings=[
                "현재 공식 등록 유효성은 별도로 "
                "확인되지 않았습니다. 실제 사용 전 "
                "최신 공식 등록정보와 제품 라벨을 "
                "확인해 주세요."
            ],
        )

    # 6. 현재 사용 추천은 공식 최신 검증 필수
    if current_verifier is None:
        return ExecutionResult(
            allowed=False,
            decision="BLOCK",
            reason=(
                "현재 공식 등록상태와 적용조건을 "
                "검증할 조회 기능이 연결되지 않았습니다."
            ),
        )

    try:
        verified = current_verifier(record) is True
    except Exception:
        verified = False

    if not verified:
        return ExecutionResult(
            allowed=False,
            decision="BLOCK",
            reason=(
                "현재 공식 등록상태 및 적용조건을 "
                "확인하지 못했습니다."
            ),
        )

    return ExecutionResult(
        allowed=True,
        decision="ALLOW",
        reason="최신 공식 등록·적용조건 검증 통과",
        record=record,
    )
