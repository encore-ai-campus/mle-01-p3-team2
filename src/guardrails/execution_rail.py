
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

# 작물 + 병해충만으로 등록 기록 목록을 받는 조회 함수
UsageListLookup = Callable[
    [str, str],
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

NOT_CURRENT_WARNING = (
    "현재 공식 등록 유효성은 별도로 "
    "확인되지 않았습니다. 실제 사용 전 "
    "최신 공식 등록정보와 제품 라벨을 "
    "확인해 주세요."
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
    records: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _normalize(value: Any) -> str:
    """공백과 유니코드 표현만 정규화한다."""
    if not isinstance(value, str):
        return ""

    value = unicodedata.normalize("NFKC", value)
    return " ".join(value.strip().split()).casefold()



def _scope_matches(
    record: Mapping[str, Any],
    crop: str,
    target_pest: str,
) -> bool:
    """작물 + 병해충 일치 검사."""

    crop_match = (
        _normalize(record.get("crop")) == _normalize(crop)
    )

    pest_match = (
        _normalize(record.get("target_pest"))
        == _normalize(target_pest)
    )

    return (
        crop_match
        and pest_match
        and bool(_normalize(crop))
        and bool(_normalize(target_pest))
    )


def _product_matches(
    record: Mapping[str, Any],
    product_name: str,
) -> bool:
    """품목명 또는 상표명 일치 검사."""

    if not _normalize(product_name):
        return False

    return any(
        _normalize(record.get(field)) == _normalize(product_name)
        for field in ("product_name", "brand")
        if _normalize(record.get(field))
    )


def _matches(
    record: Mapping[str, Any],
    product_name: str,
    crop: str,
    target_pest: str,
) -> bool:
    """품목명 또는 상표명 + 작물 + 병해충 일치 검사."""

    return (
        _product_matches(record, product_name)
        and _scope_matches(record, crop, target_pest)
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
            records=[record],
            warnings=[NOT_CURRENT_WARNING],
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
        records=[record],
    )


def _product_key(
    record: Mapping[str, Any],
) -> tuple[str, str]:
    """제품 단위 식별키(품목명, 상표명)."""

    return (
        _normalize(record.get("product_name")),
        _normalize(record.get("brand")),
    )


def _dedupe(
    records: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """동일한 내용의 기록을 한 건으로 합친다."""

    fields = (
        *IDENTITY_FIELDS,
        *USAGE_FIELDS,
        "brand",
        "registration_no",
        "registration_status",
        "verification_status",
    )

    unique: dict[tuple[str, ...], dict[str, Any]] = {}

    for record in records:
        signature = tuple(
            _normalize(record.get(field))
            for field in fields
        )

        if signature not in unique:
            unique[signature] = dict(record)

    return list(unique.values())


def list_registered_usages(
    *,
    crop: str,
    target_pest: str,
    list_lookup_fn: UsageListLookup,
    max_records: int = 500,
) -> ExecutionResult:
    """
    작물 + 병해충으로 등록된 약제 기록 목록을 조회한다.

    제품명을 특정하지 않는 정보 조회·목록 질문 전용 경로다.
    제품명이 없다는 이유로 보류하지 않고, 작물·병해충이
    일치하는 기록이 없을 때만 보류한다.

    list_lookup_fn:
        신뢰된 내부 데이터 접근 계층의 읽기 전용 조회 함수.

    중요:
        목록 조회 결과는 실제 사용 추천이 아니다.
        최신 공식 등록 유효성은 항상 경고로 함께 전달한다.
    """

    # 1. 목록 조회의 필수 조건은 작물 + 병해충
    if not all(
        isinstance(value, str) and value.strip()
        for value in (crop, target_pest)
    ):
        return ExecutionResult(
            allowed=False,
            decision="FALLBACK",
            reason="작물·병해충 정보가 부족합니다.",
        )

    # 2. 읽기 전용 DB 조회
    try:
        raw_records = list_lookup_fn(crop, target_pest)
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

    if any(
        not isinstance(item, Mapping)
        for item in raw_records
    ):
        return ExecutionResult(
            allowed=False,
            decision="FALLBACK",
            reason="조회 결과에 잘못된 형식의 자료가 있습니다.",
        )

    # 3. 작물·병해충이 정확히 일치하는 기록만 사용
    matched = _dedupe([
        item
        for item in raw_records
        if _scope_matches(item, crop, target_pest)
    ])

    if not matched:
        return ExecutionResult(
            allowed=False,
            decision="FALLBACK",
            reason=(
                "요청한 작물·병해충 조합에 "
                "일치하는 등록 기록을 확인하지 못했습니다."
            ),
        )

    # 4. 제품별로 사용조건 충돌을 검사해 충돌 제품만 제외
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for record in matched:
        groups.setdefault(
            _product_key(record),
            [],
        ).append(record)

    records: list[dict[str, Any]] = []
    warnings = [NOT_CURRENT_WARNING]
    conflicted = 0

    for key in sorted(groups):
        group = groups[key]

        if _conflicting_records(group):
            conflicted += 1
            continue

        records.append(group[0])

    if not records:
        return ExecutionResult(
            allowed=False,
            decision="FALLBACK",
            reason=(
                "조회된 농약 기록 사이에 등록상태 "
                "또는 사용조건 불일치가 있습니다."
            ),
        )

    if conflicted:
        warnings.append(
            f"등록상태 또는 사용조건이 서로 다른 제품 "
            f"{conflicted}건은 목록에서 제외했습니다."
        )

    # 5. 목록이 너무 길면 상위 기록만 전달하고 알린다.
    #
    # 이 목록은 최종 답변의 사용조건을 대조하는 기준이 된다. 품목명
    # 순으로 앞에서 자르면 답변이 인용한 제품이 기준에서 빠져, 다른
    # 제품의 등록값과 비교되는 오탐이 난다. 그래서 상한을 넉넉히 둔다.
    if len(records) > max_records:
        warnings.append(
            f"일치하는 등록 기록 {len(records)}건 중 "
            f"{max_records}건만 제공합니다."
        )
        records = records[:max_records]

    return ExecutionResult(
        allowed=True,
        decision="WARN",
        reason="수집된 농약 등록정보 목록 조회",
        # 여러 제품의 수치를 한 기록으로 섞어 검증하지 않도록
        # 단건일 때만 record를 채운다.
        record=records[0] if len(records) == 1 else None,
        records=records,
        warnings=warnings,
    )
