
"""GraphRAG 검색 결과 보안 검증 레일."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.retrievers.search_result import SearchResult


# 검색 문서가 AI의 행동이나 안전 규칙을 변경하라고 지시하는 패턴
INJECTION_PATTERNS = (
    re.compile(
        r"(이전|기존|시스템|개발자).{0,30}"
        r"(지시|규칙|프롬프트).{0,30}"
        r"(무시|해제|삭제|변경|덮어)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(안전\s*규칙|가드레일|guardrails?).{0,25}"
        r"(무시|해제|우회|비활성화)",
        re.IGNORECASE,
    ),
    re.compile(
        r"ignore\s+(all\s+)?(previous|prior|system)"
        r"\s+instructions",
        re.IGNORECASE,
    ),
    re.compile(
        r"(assistant|챗봇|AI).{0,30}"
        r"(반드시|무조건|항상).{0,30}"
        r"(권장|추천|출력|답변하라|따라라)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(system|developer)\s*:"
        r".{0,100}(ignore|override|disable)",
        re.IGNORECASE | re.DOTALL,
    ),
)

# 출처가 불확실할 수 있는 유형
UNTRUSTED_SOURCE_TYPES = {
    "blog",
    "forum",
    "advertisement",
    "seller_ad",
    "community",
    "personal_blog",
}

# 농약 정보 레코드 여부를 판단하는 메타데이터
PESTICIDE_TYPES = {
    "pesticide",
    "pesticide_registration",
    "pesticide_usage",
    "pesticide_use_validated_record",
}


@dataclass
class RetrievalCheckResult:
    """검색 결과 검증 후 사용할 데이터."""

    accepted: list[SearchResult] = field(default_factory=list)
    rejected: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _is_pesticide_record(result: SearchResult) -> bool:
    """메타데이터 또는 텍스트로 농약 관련 근거를 식별한다."""

    metadata = result.metadata or {}

    evidence_type = str(
        metadata.get("evidence_type", "")
    ).strip().lower()

    return (
        evidence_type in PESTICIDE_TYPES
        or "pesticide" in evidence_type
        or "검증상태" in result.text
    )


def _contains_injection(text: str) -> bool:
    """검색 문서 내 명령 주입 의심 패턴을 검사한다."""

    return any(
        pattern.search(text)
        for pattern in INJECTION_PATTERNS
    )


def _has_evidence_identifier(result: SearchResult) -> bool:
    """근거 식별자가 최소 1개 존재하는지 검사한다."""

    ids = (
        result.evidence_ids
        + result.source_ids
        + result.chunk_ids
    )

    return any(
        isinstance(identifier, str)
        and bool(identifier.strip())
        for identifier in ids
    )


def _source_type(result: SearchResult) -> str:
    metadata = result.metadata or {}

    return str(
        metadata.get("source_type", "")
    ).strip().lower()


def filter_retrieval_results(
    results: list[SearchResult],
    *,
    pesticide_use_requested: bool = False,
) -> RetrievalCheckResult:
    """
    검색 결과를 검사하여 사용 가능한 근거만 반환한다.

    - 악성 지시문 의심 결과 제외
    - 근거 식별자 누락 결과 제외
    - 신뢰할 수 없는 출처는 농약 사용 추천 근거에서 제외
    - 일반 농업 정보의 정상 검색을 최대한 유지
    """

    checked = RetrievalCheckResult()

    for index, result in enumerate(results):

        # 1. 자료형 검사
        if not isinstance(result, SearchResult):
            checked.rejected.append({
                "index": index,
                "reason": "INVALID_RESULT_TYPE",
            })
            continue

        # 2. 검색 근거 식별자 검사
        if not _has_evidence_identifier(result):
            checked.rejected.append({
                "index": index,
                "reason": "MISSING_EVIDENCE_ID",
            })
            continue

        # 3. 비어 있거나 과도하게 큰 문서 검사
        if (
            not isinstance(result.text, str)
            or not result.text.strip()
            or len(result.text) > 20000
        ):
            checked.rejected.append({
                "index": index,
                "reason": "INVALID_DOCUMENT_TEXT",
            })
            continue

        # 4. 기존 검색 적격성 검사
        metadata = result.metadata or {}

        if metadata.get("retrieval_eligible") is False:
            checked.rejected.append({
                "index": index,
                "reason": "NOT_RETRIEVAL_ELIGIBLE",
            })
            continue

        # 5. 간접 프롬프트 인젝션 검사
        if _contains_injection(result.text):
            checked.rejected.append({
                "index": index,
                "reason": "SUSPECTED_PROMPT_INJECTION",
            })
            checked.warnings.append(
                "검색 문서에서 안전 규칙 변경 지시가 "
                "의심되어 해당 근거를 제외했습니다."
            )
            continue

        # 6. 농약 관련 출처 확인
        if _is_pesticide_record(result):

            source_type = _source_type(result)

            # 현재 농약 사용 추천에 한해서
            # 명시적으로 신뢰할 수 없는 출처는 제외
            if (
                pesticide_use_requested
                and source_type in UNTRUSTED_SOURCE_TYPES
            ):
                checked.rejected.append({
                    "index": index,
                    "reason": "UNTRUSTED_PESTICIDE_SOURCE",
                })
                continue

            # 수집 기록의 출처가 명확하지 않다면 경고
            if not result.source_ids:
                checked.warnings.append(
                    "일부 농약 기록의 출처 식별자가 "
                    "없어 공식 등록정보 확인이 필요합니다."
                )

            if source_type in UNTRUSTED_SOURCE_TYPES:
                checked.warnings.append(
                    "일부 농약 정보에는 비공식 자료가 "
                    "포함되어 있습니다. 현재 사용기준의 "
                    "근거로 삼지 마세요."
                )

        checked.accepted.append(result)

    checked.warnings = list(
        dict.fromkeys(checked.warnings)
    )

    return checked
