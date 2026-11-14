"""구조화 답변의 주장과 인용을 검증합니다"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.retrievers.answer import CitedAnswer, Evidence

CitationStatus = Literal["PASS", "PARTIAL", "FAIL"]

FailureCode = Literal[
    "UNKNOWN_EVIDENCE_ID",
    "QUOTE_NOT_FOUND",
    "NOT_SUPPORTED",
    "MISSING_CITATION",
    "NO_CLAIMS",
]


class AnswerReview(BaseModel):
    """주장이 주어진 인용 근거로 뒷받침되는지에 대한 판정입니다."""

    supported: bool
    reason: str

    conflict: bool = False
    conflict_reason: str | None = None
    conflicting_evidence_ids: list[str] = Field(default_factory=list)

class CitationConflict(BaseModel):
    """동일 주장에 연결된 근거 사이의 충돌입니다."""

    claim_index: int
    claim: str
    evidence_ids: list[str]
    reason: str
    safety_critical: bool = False


class CitationFailure(BaseModel):
    """인용 검증 실패 한 건입니다."""

    code: FailureCode
    claim_index: int
    citation_index: int | None = None
    evidence_id: str | None = None
    reason: str


class CitationVerificationResult(BaseModel):
    """답변 전체의 인용 검증 결과입니다."""

    status: CitationStatus
    conflicts: list[CitationConflict] = Field(default_factory=list)
    failures: list[CitationFailure] = Field(default_factory=list)
    checked_claims: int = 0
    supported_claims: int = 0


SupportChecker = Callable[
    [str, list[dict[str, str]]],
    AnswerReview,
]

_WHITESPACE = re.compile(r"\s+")


def normalize_quote(text: str) -> str:
    """유니코드와 공백 차이만 정규화합니다."""

    normalized = unicodedata.normalize("NFKC", text)
    return _WHITESPACE.sub(" ", normalized).strip().casefold()


def quote_exists(citation_quote: str, source_text: str) -> bool:
    """인용 구절이 실제 원문에 포함되는지 확인합니다."""

    if not citation_quote.strip() or not source_text.strip():
        return False

    # 원문 그대로 존재하는 경우를 우선 인정
    if citation_quote in source_text:
        return True

    normalized_quote = normalize_quote(citation_quote)
    normalized_source = normalize_quote(source_text)

    return bool(
        normalized_quote
        and normalized_quote in normalized_source
    )

_SAFETY_KEYWORDS = (
    "농약",
    "비료",
    "희석",
    "사용량",
    "사용횟수",
    "수확 전",
    "안전사용",
    "등록상태",
)


def _is_safety_critical_claim(claim: str) -> bool:
    return any(
        keyword in claim
        for keyword in _SAFETY_KEYWORDS
    )

def verify_citations(
    answer: CitedAnswer,
    evidences: list[Evidence],
    support_checker: SupportChecker,
) -> CitationVerificationResult:
    """ID, 인용 구절, 주장 지원 여부를 순서대로 검증합니다."""

    evidence_map = {
        evidence.evidence_id: evidence
        for evidence in evidences
    }

    failures: list[CitationFailure] = []
    conflicts: list[CitationConflict] = []
    supported_claims = 0

    if not answer.claims:
        return CitationVerificationResult(
            status="FAIL",
            failures=[
                CitationFailure(
                    code="NO_CLAIMS",
                    claim_index=-1,
                    reason="검증할 주요 주장이 없습니다.",
                )
            ],
            conflicts=[],
            checked_claims=0,
            supported_claims=0,
        )

    for claim_index, claim in enumerate(answer.claims):
        if not claim.citations:
            failures.append(
                CitationFailure(
                    code="MISSING_CITATION",
                    claim_index=claim_index,
                    reason="주장에 연결된 인용이 없습니다.",
                )
            )
            continue

        valid_contexts: list[dict[str, str]] = []

        for citation_index, citation in enumerate(
            claim.citations
        ):
            evidence = evidence_map.get(
                citation.evidence_id
            )

            # 1단계: evidence_id 존재 확인
            if evidence is None:
                failures.append(
                    CitationFailure(
                        code="UNKNOWN_EVIDENCE_ID",
                        claim_index=claim_index,
                        citation_index=citation_index,
                        evidence_id=citation.evidence_id,
                        reason=(
                            "인용한 evidence_id가 "
                            "검색 근거에 존재하지 않습니다."
                        ),
                    )
                )
                continue

            # 2단계: 인용 구절이 원문에 존재하는지 확인
            if not quote_exists(
                citation.quote,
                evidence.quote,
            ):
                failures.append(
                    CitationFailure(
                        code="QUOTE_NOT_FOUND",
                        claim_index=claim_index,
                        citation_index=citation_index,
                        evidence_id=citation.evidence_id,
                        reason=(
                            "인용 구절이 해당 근거 원문에 "
                            "존재하지 않습니다."
                        ),
                    )
                )
                continue

            valid_contexts.append(
                {
                    "evidence_id": evidence.evidence_id,
                    "citation_quote": citation.quote,
                    "source_text": evidence.quote,
                }
            )

        # ID 또는 인용 구절 검사에서 전부 실패했다면
        # LLM support_checker를 호출하지 않음
        if not valid_contexts:
            continue

        # 3단계: 유효한 인용들이 주장을 뒷받침하는지 확인
        review = support_checker(
            claim.text,
            valid_contexts,
        )

        if review.conflict:
            conflicts.append(
                CitationConflict(
                    claim_index=claim_index,
                    claim=claim.text,
                    evidence_ids=(
                        review.conflicting_evidence_ids
                    ),
                    reason=(
                        review.conflict_reason
                        or review.reason
                        or "근거 사이의 충돌이 확인되었습니다."
                    ),
                    safety_critical=(
                        _is_safety_critical_claim(
                            claim.text
                        )
                    ),
                )
            )
            continue

        if review.supported:
            supported_claims += 1
        else:
            failures.append(
                CitationFailure(
                    code="NOT_SUPPORTED",
                    claim_index=claim_index,
                    reason=review.reason,
                )
            )

    # 중요: 이 부분은 for claim 반복문 바깥에 있어야 함
    if (
        not failures
        and not conflicts
        and supported_claims == len(answer.claims)
    ):
        status: CitationStatus = "PASS"

    elif supported_claims > 0:
        status = "PARTIAL"

    else:
        status = "FAIL"

    # 중요: 최종 return도 for claim 반복문 바깥
    return CitationVerificationResult(
        status=status,
        failures=failures,
        conflicts=conflicts,
        checked_claims=len(answer.claims),
        supported_claims=supported_claims,
    )

class LLMSupportChecker:
    """주장과 유효한 인용 근거의 의미 일치를 판정합니다."""

    def __init__(self, model: Any):
        self.model = model
        self.structured_model = model.with_structured_output(
            AnswerReview
        )

    def __call__(
        self,
        claim: str,
        contexts: list[dict[str, str]],
    ) -> AnswerReview:
        evidence_block = "\n\n".join(
            (
                f"evidence_id: {context['evidence_id']}\n"
                f"인용 구절: {context['citation_quote']}\n"
                f"근거 원문: {context['source_text']}"
            )
            for context in contexts
        )

        prompt = (
            "다음 주장이 제시된 검색 근거만으로 직접 뒷받침되는지 "
            "판정하세요.\n"
            "외부 지식을 사용하지 마세요.\n"
            "여러 인용이 있을 경우 함께 결합해서 판단하세요.\n\n"

            "충돌 판정 규칙:\n"
            "1. 같은 대상과 조건에 대해 수치, 날짜, 사용량, 희석배수, "
            "사용횟수 또는 등록상태가 다르면 conflict=true입니다.\n"
            "2. 서로 다른 작물·제품·시기·조건의 값은 충돌이 아닙니다.\n"
            "3. 충돌한 경우 conflicting_evidence_ids에 양쪽 ID를 넣으세요.\n"
            "4. 충돌 여부가 불분명하면 conflict=false로 두고 "
            "supported 판단을 수행하세요.\n\n"

            f"주장:\n{claim}\n\n"
            f"검색 근거:\n{evidence_block}"
        )

        result = self.structured_model.invoke(prompt)

        if isinstance(result, AnswerReview):
            return result

        return AnswerReview.model_validate(result)