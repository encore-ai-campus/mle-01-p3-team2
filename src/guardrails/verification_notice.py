"""답변이 언급한 농약의 검증상태를 규칙으로 덧붙인다.

Execution Rail이 조회한 등록정보에는 검증상태가 들어 있다. 답변
생성 모델이 이 상태를 빠뜨리면 LLM-as-Judge가 "최신 근거 없이 현재
사용 가능하다고 단정했다"고 보고 답변 전체를 보류한다.

상태 표기는 등록정보에 이미 있는 사실이므로 모델에 맡기지 않고
규칙으로 덧붙인다. 답변이 실제로 언급한 제품만 대상으로 한다.
"""

from __future__ import annotations

import unicodedata
from typing import Any, Mapping, Sequence


# 최신 등록 여부가 확인되지 않은 검증상태
REVIEW_REQUIRED = "REVIEW_REQUIRED"
PRELIMINARY_PASS = "PRELIMINARY_PASS"

NOTICE_TEMPLATES = {
    REVIEW_REQUIRED: (
        "{products}{josa} 검증상태가 REVIEW_REQUIRED로, "
        "현재 등록·사용 가능 여부가 확인되지 않았습니다. "
        "사용 전 최신 공식 등록정보를 확인해 주세요."
    ),
    PRELIMINARY_PASS: (
        "{products}의 사용기준은 수집된 등록정보 기준이며, "
        "최신 등록 여부는 별도로 확인되지 않았습니다."
    ),
}

# 안내 문구가 너무 길어지지 않도록 제품 수를 제한한다.
MAX_PRODUCTS = 5

PRODUCT_FIELDS = ("product_name", "brand")


def _josa(word: str, with_final: str, without_final: str) -> str:
    """끝 글자의 종성에 맞는 조사를 고른다."""

    last = word.strip()[-1:] if word.strip() else ""

    if not ("가" <= last <= "힣"):
        return with_final

    has_final = (ord(last) - ord("가")) % 28 != 0

    return with_final if has_final else without_final


def _normalize(value: Any) -> str:
    if not isinstance(value, str):
        return ""

    return " ".join(
        unicodedata.normalize("NFKC", value).split()
    ).casefold()


def _status(record: Mapping[str, Any]) -> str:
    value = record.get("verification_status")

    return str(value).strip().upper() if value else ""


def _mentioned_label(
    record: Mapping[str, Any],
    answer_text: str,
) -> str | None:
    """답변이 이 제품을 언급했으면 표시할 이름을 돌려준다."""

    for field in PRODUCT_FIELDS:
        name = record.get(field)

        if not isinstance(name, str) or not name.strip():
            continue

        if _normalize(name) in answer_text:
            return name.strip()

    return None


def build_verification_notices(
    answer: str,
    records: Sequence[Mapping[str, Any]] | None,
) -> list[str]:
    """답변이 언급한 제품의 검증상태 안내 문구를 만든다."""

    if not isinstance(answer, str) or not answer.strip():
        return []

    if not records:
        return []

    answer_text = _normalize(answer)

    # 상태별로 언급된 제품 이름을 모은다.
    grouped: dict[str, list[str]] = {
        REVIEW_REQUIRED: [],
        PRELIMINARY_PASS: [],
    }

    for record in records:
        if not isinstance(record, Mapping):
            continue

        status = _status(record)

        if status not in grouped:
            continue

        label = _mentioned_label(record, answer_text)

        if label and label not in grouped[status]:
            grouped[status].append(label)

    notices = []

    for status, products in grouped.items():
        if not products:
            continue

        shown = products[:MAX_PRODUCTS]
        text = ", ".join(shown)

        if len(products) > len(shown):
            text += f" 등 {len(products)}개 품목"

        notices.append(
            NOTICE_TEMPLATES[status].format(
                products=text,
                josa=_josa(text, "은", "는"),
            )
        )

    return notices


def annotate_answer(
    answer: str,
    records: Sequence[Mapping[str, Any]] | None,
) -> str:
    """검증상태 안내를 답변 본문에 덧붙인다.

    근거 일치 검증 전에 붙여, 검증상태를 밝히지 않았다는 이유로
    답변 전체가 보류되지 않게 한다.
    """

    notices = build_verification_notices(answer, records)

    if not notices:
        return answer

    return answer + "\n\n[등록정보 검증상태]\n" + "\n".join(notices)
