
"""농약 사용조건에 대한 Output Rail 검증."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping, Sequence


# 1. 숫자로 표현된 농약 사용조건 검증
#
# 답변 쪽은 단위를 요구해 오탐을 줄이고, 등록정보 쪽은 단위 표기가
# 없는 값("1000")도 비교할 수 있도록 별도 패턴을 쓴다.
PATTERNS = {
    # 배수 변화를 뜻하는 "2배 증가" 같은 표현은 희석배수가 아니다.
    "dilution": re.compile(
        r"(\d[\d,]*)\s*배"
        r"(?!\s*(?:이상\s*)?(?:증가|감소|늘|더|많|높|빠르|크))"
    ),
    "use_count": re.compile(r"(\d+)\s*회"),
    "harvest_days": re.compile(r"수확\s*(\d+)\s*일\s*전"),
}

# "N회"가 사용횟수를 뜻하는지 판단하는 주변 문맥.
# "2~3회 발생" 같은 발생 횟수를 사용횟수로 오인하지 않게 한다.
USE_COUNT_CONTEXT = re.compile(
    r"사용|살포|처리|방제|적용|횟수|뿌리|관주|경엽"
)

USE_COUNT_SUFFIX = re.compile(r"^\s*(?:이내|이하|까지)")

# 숫자만 저장된 등록정보 필드에서 값을 읽는 패턴
BARE_NUMBER = re.compile(r"\d[\d,]*")

# 사용조건별 비교 방식
#   exact     : 등록값과 정확히 일치해야 한다(희석배수).
#   max_limit : 등록 상한을 넘으면 안 된다(사용횟수).
#   min_limit : 등록 일수보다 짧으면 안 된다(수확 전 사용시기).
COMPARISONS = {
    "dilution": "exact",
    "use_count": "max_limit",
    "harvest_days": "min_limit",
}

FIELD_NAMES = {
    "dilution": "희석배수",
    "use_count": "사용횟수",
    "harvest_days": "수확 전 사용시기",
}

# Execution Rail과 Neo4j 어댑터의 필드명이 다르므로
# 비교 대상 필드를 별칭까지 함께 확인한다.
FIELD_SOURCES = {
    "dilution": ("dilution",),
    "use_count": ("use_count", "max_applications"),
    "harvest_days": ("use_timing", "preharvest_interval"),
}

# 제품을 식별할 수 있는 필드. 좁은 식별자부터 본다.
# 등록번호 > 상표명 > 품목명 순이며, 같은 품목이 상표별로 사용조건이
# 다른 자료가 있어 순서가 중요하다.
IDENTIFIER_FIELDS = ("registration_no", "brand", "product_name")

# 답변 한 줄이 등록번호로 특정 제품을 가리키는지 확인한다.
ROW_REGISTRATION = re.compile(r"\d+\s*-\s*[가-힣]+\s*-\s*\d+")


# 2. 위험한 자연어 사용 지시 탐지
UNSAFE_PATTERNS = {
    "농약 농도 임의 강화": re.compile(
        r"(?:희석액|희석\s*배수|농약|약액)"
        r".{0,25}?"
        r"(?:더\s*진하게|더\s*강하게|농도를\s*높여|두\s*배\s*강하게)"
        r".{0,20}?"
        r"(?:살포|뿌리|사용|만들|조절)"
    ),

    "사용횟수 제한 초과": re.compile(
        r"(?:사용\s*횟수|살포\s*횟수)"
        r".{0,35}?"
        r"(?:초과|넘겨|제한보다\s*많이|더\s*많이)"
        r".{0,20}?"
        r"(?:살포|사용|뿌리)"
    ),

    "미검증 농약 안전성 단정": re.compile(
        r"(?:등록\s*상태.{0,12}?(?:불분명|미확인)"
        r"|검증되지\s*않|확인되지\s*않|미검증)"
        r".{0,35}?"
        r"(?:사용해도\s*안전|사용해도\s*괜찮|사용\s*가능|바로\s*사용)"
    ),

    "공식 검증 없는 사용 권장": re.compile(
        r"(?:공식\s*확인\s*없이|등록\s*확인\s*없이|검증\s*없이)"
        r".{0,35}?"
        r"(?:사용을\s*권장|사용을\s*추천|살포를\s*권장|사용해도\s*안전)"
    ),

    "사용조건 무시 및 안전성 단정": re.compile(
        r"(?:사용\s*조건|사용\s*기준|안전\s*기준)"
        r".{0,15}?"
        r"(?:무시|어겨|지키지\s*않)"
        r".{0,20}?"
        r"(?:안전|괜찮|문제없)"
    ),
}


# 위험 지시를 부정하는 표현
NEGATION_PATTERN = re.compile(
    r"(?:하지\s*마|지\s*마|말아|"
    r"해서는\s*안|하면\s*안|"
    r"지\s*않|금지|잘못된|"
    r"안전하지\s*않|아닙니다)"
)


def _numbers(
    text: str,
    pattern: re.Pattern,
) -> set[int]:
    """사용조건에 해당하는 숫자를 추출한다."""
    return {
        int(value.replace(",", ""))
        for value in pattern.findall(text)
    }


def _answer_values(
    row: str,
    field: str,
) -> set[int]:
    """답변 한 줄에서 해당 사용조건의 숫자를 추출한다."""

    if field != "use_count":
        return _numbers(row, PATTERNS[field])

    # 사용횟수는 "N회"만으로 판단하지 않고 주변 문맥을 확인한다.
    values: set[int] = set()

    for match in PATTERNS["use_count"].finditer(row):
        before = row[max(0, match.start() - 25):match.start()]
        after = row[match.end():match.end() + 15]

        if (
            USE_COUNT_CONTEXT.search(before)
            or USE_COUNT_CONTEXT.search(after)
            or USE_COUNT_SUFFIX.match(after)
        ):
            values.add(int(match.group(1)))

    return values


def _normalize(value: Any) -> str:
    """제품명 비교용 정규화."""
    if not isinstance(value, str):
        return ""

    value = unicodedata.normalize("NFKC", value)
    return " ".join(value.split()).casefold()


def _reference_values(
    record: Mapping[str, Any],
    field: str,
    pattern: re.Pattern,
) -> set[int]:
    """등록정보에서 해당 사용조건의 숫자를 모은다.

    등록정보는 "1,000배"처럼 단위를 붙인 값과 "1000"처럼 숫자만
    저장한 값이 섞여 있다. 단위를 붙인 표기를 먼저 읽고, 읽을 값이
    없으면 숫자만 저장된 값으로 다시 읽는다.
    """

    values: set[int] = set()

    for source in FIELD_SOURCES[field]:
        raw = str(record.get(source) or "")

        if not raw.strip():
            continue

        matched = _numbers(raw, pattern)

        # 단위 표기가 없는 필드는 숫자만으로 비교한다.
        values |= matched or _numbers(raw, BARE_NUMBER)

    return values


def _conflicts(
    field: str,
    answer_values: set[int],
    reference_values: set[int],
) -> bool:
    """사용조건의 의미에 맞게 답변 숫자와 등록값을 비교한다."""

    comparison = COMPARISONS[field]

    # 사용횟수는 상한이므로 등록 상한을 넘을 때만 불일치다.
    if comparison == "max_limit":
        return max(answer_values) > max(reference_values)

    # 수확 전 사용시기는 하한이므로 등록 일수보다
    # 짧게 안내할 때만 불일치다.
    if comparison == "min_limit":
        return min(answer_values) < min(reference_values)

    return not answer_values.issubset(reference_values)


def _records_for_row(
    row: str,
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """답변 한 줄이 가리키는 제품의 기록만 고른다.

    같은 품목이 상표별로 사용조건이 다르고, 같은 상표명을 여러
    품목이 함께 쓰기도 한다. 그래서 좁은 식별자부터 맞춰본다.

    답변이 등록번호로 제품을 특정했는데 그 등록번호의 기록이 비교
    대상에 없으면 빈 목록을 돌려준다. 다른 제품의 등록값과 비교해
    불일치로 단정하는 것보다, 대조하지 못했다고 남기는 쪽이 맞다.
    """

    text = _normalize(row)

    def matching(
        field: str,
        pool: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        return [
            record
            for record in pool
            if (name := _normalize(record.get(field))) and name in text
        ]

    by_registration = matching("registration_no", records)

    # 등록번호로 특정한 제품의 기록이 없으면 비교하지 않는다.
    if not by_registration and ROW_REGISTRATION.search(row):
        return []

    candidates = by_registration or list(records)

    # 남은 식별자로 한 단계씩 좁힌다. 좁혀지지 않으면 그대로 둔다.
    for field in ("brand", "product_name"):
        narrowed = matching(field, candidates)

        if narrowed:
            candidates = narrowed

    return candidates


def _registration_conflict(
    row: str,
    records: Sequence[Mapping[str, Any]],
) -> str | None:
    """등록번호와 함께 적힌 상표가 다른 제품의 것인지 검사한다.

    같은 품목이 상표별로 따로 등록돼 있어, 답변이 한 제품의
    등록번호에 다른 제품의 상표를 붙이는 일이 생긴다. 등록번호는
    제품을 하나로 특정하므로 규칙으로 대조할 수 있다.
    """

    text = _normalize(row)

    for cited in ROW_REGISTRATION.findall(row):
        number = _normalize(cited)

        owners = [
            record
            for record in records
            if _normalize(record.get("registration_no")) == number
        ]

        # 등록정보에 없는 등록번호는 여기서 판단하지 않는다.
        if not owners:
            continue

        own_brands = {
            name
            for record in owners
            if (name := _normalize(record.get("brand")))
        }

        if not own_brands:
            continue

        # 이 줄에 등장한 다른 제품의 상표
        others = {
            name
            for record in records
            if (name := _normalize(record.get("brand")))
            and name not in own_brands
            and name in text
        }

        # 올바른 상표도 함께 적혀 있으면 여러 제품을 나열한 줄로 본다.
        if others and not (own_brands & {
            name for name in own_brands if name in text
        }):
            return cited

    return None


def _rows(answer: str) -> list[str]:
    """제품별 비교가 가능하도록 답변을 줄·문장 단위로 나눈다."""

    rows = [
        row.strip()
        for chunk in answer.splitlines()
        for row in re.split(r"(?<=[.!?。！？])\s+", chunk)
    ]

    return [row for row in rows if row]


def _detect_unsafe_instruction(answer: str) -> str | None:
    """위험한 농약 사용 지시와 안전 경고를 구분한다."""

    # 문장 단위로 검사하여 다른 문장의 표현이
    # 차단 판단에 영향을 주지 않도록 한다.
    sentences = re.split(r"[.!?。！？\n]+", answer)

    for sentence in sentences:
        text = sentence.strip()

        if not text:
            continue

        for name, pattern in UNSAFE_PATTERNS.items():
            for match in pattern.finditer(text):

                # 위험 표현 뒤에 이어지는 금지·부정 표현 확인
                following = text[
                    match.end():match.end() + 50
                ]

                if NEGATION_PATTERN.search(following):
                    continue

                return name

    return None


def _reject(
    decision: str,
    reason: str,
) -> dict[str, Any]:
    """검증 실패 결과를 반환한다."""
    return {
        "allowed": False,
        "decision": decision,
        "reason": reason,
        "warnings": [],
        "unverified_fields": [],
    }


def validate_pesticide_output(
    answer: str,
    record: Mapping[str, Any] | None = None,
    *,
    records: Sequence[Mapping[str, Any]] | None = None,
    pesticide_use_requested: bool = False,
    recommendation_requested: bool = False,
) -> dict[str, Any]:
    """농약 사용조건의 숫자 및 위험한 자연어 지시를 검사한다.

    record:
        단일 제품 조회 결과.

    records:
        작물·병해충 기준 목록 조회 결과. 제품별 답변 줄을
        해당 제품 기록과 각각 비교한다.
    """

    # 공식 최신 검증이 완료되지 않은 추천은 허용하지 않음
    if recommendation_requested:
        return _reject(
            "WITHHELD",
            "농약 추천에 필요한 최신 공식 등록정보와 "
            "전체 사용조건 검증이 완료되지 않았습니다.",
        )

    # 일반 농업 질문은 별도 검증 대상
    if not pesticide_use_requested:
        return {
            "allowed": True,
            "decision": "PASS",
            "reason": "농약 사용조건 검증 대상이 아닙니다.",
            "warnings": [],
            "unverified_fields": [],
        }

    if not isinstance(answer, str) or not answer.strip():
        return _reject(
            "WITHHELD",
            "검증할 답변이 없습니다.",
        )

    # 단건·목록 결과를 하나의 비교 대상 목록으로 모은다.
    reference_records: list[Mapping[str, Any]] = []

    if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        reference_records = [
            item
            for item in records
            if isinstance(item, Mapping) and item
        ]

    if isinstance(record, Mapping) and record:
        if not any(item is record for item in reference_records):
            reference_records.insert(0, record)

    if not reference_records:
        return _reject(
            "WITHHELD",
            "농약 등록정보가 없어 사용조건을 검증할 수 없습니다.",
        )

    # 1차: 위험한 자연어 사용 지시 검사
    unsafe_reason = _detect_unsafe_instruction(answer)

    if unsafe_reason:
        return _reject(
            "BLOCK",
            f"위험한 농약 사용 표현이 발견됐습니다: {unsafe_reason}",
        )

    # 2차: 희석배수·사용횟수·수확 전 일수를 제품별로 검증
    unverified_fields: list[str] = []
    deviating_fields: list[str] = []

    for row in _rows(answer):
        row_records = _records_for_row(row, reference_records)

        # 등록번호에 다른 제품의 상표를 붙였는지 먼저 본다.
        mismatched = _registration_conflict(row, reference_records)

        if mismatched:
            return _reject(
                "BLOCK",
                f"등록번호 {mismatched}의 상표가 "
                "등록정보와 일치하지 않습니다.",
            )

        for field, pattern in PATTERNS.items():
            answer_values = _answer_values(row, field)

            if not answer_values:
                continue

            reference_values: set[int] = set()

            for item in row_records:
                reference_values |= _reference_values(
                    item,
                    field,
                    pattern,
                )

            # 등록정보에 해당 항목 값이 없으면 참·거짓을 판단할 수
            # 없다. 불일치가 확인된 것이 아니므로 답변을 보류하지
            # 않고 검증하지 못한 항목으로만 남긴다.
            if not reference_values:
                if FIELD_NAMES[field] not in unverified_fields:
                    unverified_fields.append(FIELD_NAMES[field])
                continue

            if _conflicts(field, answer_values, reference_values):
                return _reject(
                    "BLOCK",
                    f"{FIELD_NAMES[field]}이(가) "
                    "등록정보와 일치하지 않습니다.",
                )

            # 안전한 방향이지만 등록 표기와 다른 값은 경고로 남긴다.
            if (
                COMPARISONS[field] != "exact"
                and not answer_values.issubset(reference_values)
                and FIELD_NAMES[field] not in deviating_fields
            ):
                deviating_fields.append(FIELD_NAMES[field])

    warnings: list[str] = []

    if unverified_fields:
        warnings.append(
            f"{', '.join(unverified_fields)}은(는) 등록정보에 "
            "값이 없어 대조하지 못했습니다. 제품 라벨을 "
            "확인해 주세요."
        )

    if deviating_fields:
        warnings.append(
            f"{', '.join(deviating_fields)}은(는) 등록 표기와 "
            "다르게 안내됐습니다. 등록정보 기준을 "
            "확인해 주세요."
        )

    return {
        "allowed": True,
        "decision": "WARN" if warnings else "PASS",
        "reason": (
            "검사 가능한 사용조건에서 불일치가 발견되지 않았습니다. "
            "전체 사용조건이나 최신 등록상태를 보증하지 않습니다."
        ),
        "warnings": warnings,
        "unverified_fields": unverified_fields,
    }
