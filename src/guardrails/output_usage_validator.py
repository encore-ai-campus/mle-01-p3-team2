
"""농약 사용조건에 대한 Output Rail 검증."""

from __future__ import annotations

import re
from typing import Any, Mapping


# 1. 숫자로 표현된 농약 사용조건 검증
PATTERNS = {
    "dilution": re.compile(r"(\d[\d,]*)\s*배"),
    "use_count": re.compile(r"(\d+)\s*회"),
    "harvest_days": re.compile(r"수확\s*(\d+)\s*일\s*전"),
}

FIELD_NAMES = {
    "dilution": "희석배수",
    "use_count": "사용횟수",
    "harvest_days": "수확 전 사용시기",
}


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
    }


def validate_pesticide_output(
    answer: str,
    record: Mapping[str, Any] | None,
    *,
    pesticide_use_requested: bool = False,
    recommendation_requested: bool = False,
) -> dict[str, Any]:
    """농약 사용조건의 숫자 및 위험한 자연어 지시를 검사한다."""

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
        }

    if not isinstance(answer, str) or not answer.strip():
        return _reject(
            "WITHHELD",
            "검증할 답변이 없습니다.",
        )

    if not isinstance(record, Mapping) or not record:
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

    # 2차: 희석배수·사용횟수·수확 전 일수 검증
    for field, pattern in PATTERNS.items():
        answer_values = _numbers(answer, pattern)

        if not answer_values:
            continue

        reference = (
            record.get("use_timing", "")
            if field == "harvest_days"
            else record.get(field, "")
        )

        reference_values = _numbers(
            str(reference or ""),
            pattern,
        )

        if not reference_values:
            return _reject(
                "WITHHELD",
                f"{FIELD_NAMES[field]}에 대한 "
                "비교 가능한 등록정보가 없습니다.",
            )

        if not answer_values.issubset(reference_values):
            return _reject(
                "BLOCK",
                f"{FIELD_NAMES[field]}이(가) "
                "등록정보와 일치하지 않습니다.",
            )

    return {
        "allowed": True,
        "decision": "PASS",
        "reason": (
            "검사 가능한 사용조건에서 불일치가 발견되지 않았습니다. "
            "전체 사용조건이나 최신 등록상태를 보증하지 않습니다."
        ),
    }
