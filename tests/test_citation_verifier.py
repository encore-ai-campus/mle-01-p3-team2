from src.retrievers.answer import (
    Citation,
    CitedAnswer,
    Claim,
    Evidence,
)
from src.supervisor.citation_verifier import (
    AnswerReview,
    verify_citations,
)


def _evidence() -> Evidence:
    return Evidence(
        evidence_id="EVD_1",
        document_id="DOC_1",
        title="배추 등록 정보",
        url="https://example.com",
        quote=(
            "작물=배추; 병해충=무름병; "
            "사용적기=정식전; 최대 사용횟수=1회"
        ),
        source="vector",
    )

def test_valid_citation_passes():
    answer = CitedAnswer(
        answer="해당 약제는 정식 전에 사용합니다.",
        claims=[
            Claim(
                text="사용 적기는 정식 전이다.",
                citations=[
                    Citation(
                        evidence_id="EVD_1",
                        quote="사용적기=정식전",
                    )
                ],
            )
        ],
    )

    def support_checker(claim, contexts):
        return AnswerReview(
            supported=True,
            reason="원문이 주장을 직접 뒷받침합니다.",
        )

    result = verify_citations(
        answer,
        [_evidence()],
        support_checker,
    )

    assert result.status == "PASS"
    assert result.failures == []

def test_unknown_evidence_id_fails_without_llm_call():
    answer = CitedAnswer(
        answer="답변",
        claims=[
            Claim(
                text="사용 적기는 정식 전이다.",
                citations=[
                    Citation(
                        evidence_id="EVD_UNKNOWN",
                        quote="사용적기=정식전",
                    )
                ],
            )
        ],
    )

    def support_checker(claim, contexts):
        raise AssertionError("ID 검증 실패 시 LLM을 호출하면 안 됩니다.")

    result = verify_citations(
        answer,
        [_evidence()],
        support_checker,
    )

    assert result.status == "FAIL"
    assert result.failures[0].code == "UNKNOWN_EVIDENCE_ID"

def test_quote_not_found_fails_without_llm_call():
    answer = CitedAnswer(
        answer="답변",
        claims=[
            Claim(
                text="사용 적기는 수확 7일 전이다.",
                citations=[
                    Citation(
                        evidence_id="EVD_1",
                        quote="수확 7일 전까지",
                    )
                ],
            )
        ],
    )

    def support_checker(claim, contexts):
        raise AssertionError("구절 검증 실패 시 LLM을 호출하면 안 됩니다.")

    result = verify_citations(
        answer,
        [_evidence()],
        support_checker,
    )

    assert result.status == "FAIL"
    assert result.failures[0].code == "QUOTE_NOT_FOUND"

def test_existing_quote_that_does_not_support_claim_fails():
    answer = CitedAnswer(
        answer="해당 약제는 수확 7일 전까지 사용합니다.",
        claims=[
            Claim(
                text="사용 시기는 수확 7일 전까지이다.",
                citations=[
                    Citation(
                        evidence_id="EVD_1",
                        quote="작물=배추",
                    )
                ],
            )
        ],
    )

    def support_checker(claim, contexts):
        return AnswerReview(
            supported=False,
            reason=(
                "인용은 대상 작물만 설명하며 "
                "수확 전 사용기한을 뒷받침하지 않습니다."
            ),
        )

    result = verify_citations(
        answer,
        [_evidence()],
        support_checker,
    )

    assert result.status == "FAIL"
    assert result.failures[0].code == "NOT_SUPPORTED"

def test_partially_supported_claims_return_partial():
    answer = CitedAnswer(
        answer="첫 번째 주장은 맞고 두 번째 주장은 근거가 부족합니다.",
        claims=[
            Claim(
                text="사용 적기는 정식 전이다.",
                citations=[
                    Citation(
                        evidence_id="EVD_1",
                        quote="사용적기=정식전",
                    )
                ],
            ),
            Claim(
                text="사용 시기는 수확 7일 전까지이다.",
                citations=[
                    Citation(
                        evidence_id="EVD_1",
                        quote="작물=배추",
                    )
                ],
            ),
        ],
    )

    def support_checker(claim, contexts):
        if claim == "사용 적기는 정식 전이다.":
            return AnswerReview(
                supported=True,
                reason="원문이 사용 적기를 직접 뒷받침합니다.",
            )

        return AnswerReview(
            supported=False,
            reason="인용이 수확 전 사용기한을 뒷받침하지 않습니다.",
        )

    result = verify_citations(
        answer,
        [_evidence()],
        support_checker,
    )

    assert result.status == "PARTIAL"
    assert result.supported_claims == 1
    assert len(result.failures) == 1
    assert result.failures[0].code == "NOT_SUPPORTED"