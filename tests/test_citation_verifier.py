from src.retrievers.answer import (
    Citation,
    CitedAnswer,
    Claim,
    Evidence,
)
from src.supervisor.citation_verifier import (
    AnswerReview,
    quote_exists,
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

# --- 조사·어미 차이로 정상 인용이 QUOTE_NOT_FOUND로 떨어지던 회귀 (GS025) ---

_GS025_SOURCE = (
    "증상: 주로 어린 유묘에 발생하여 피해를 주는데 병이 어느 정도 진전되면 "
    "생육이 정지되고 지상부가 시드는 증상으로 나타난다. 병든주를 뽑아보면 "
    "인경 겉부분이 연한 흑색을 띠며 내부를 잘라보면 검은색 가루의 후막포자가 "
    "무수히 들어 있는 것을 볼 수 있다."
)


def test_quote_exists_allows_minor_particle_difference():
    """원문 '진전되면'을 '진전이 되면'으로 옮긴 인용도 같은 구절로 인정한다."""
    quote = (
        "주로 어린 유묘에 발생하여 피해를 주는데 병이 어느 정도 진전이 되면 "
        "생육이 정지되고 지상부가 시드는 증상으로 나타난다."
    )
    assert quote_exists(quote, _GS025_SOURCE) is True


def test_quote_exists_still_rejects_fabricated_quote():
    """원문에 없는 내용을 지어낸 인용은 계속 걸러낸다."""
    quote = (
        "양파 깜부기병은 티람 수화제를 1000배 희석해 종자소독하면 "
        "완전히 방제된다."
    )
    assert quote_exists(quote, _GS025_SOURCE) is False


def test_quote_exists_rejects_partially_overlapping_fabrication():
    """앞부분만 베끼고 뒤를 지어낸 인용도 임계값에서 걸러진다."""
    quote = (
        "어린 유묘에 발생하여 피해를 주는데 병이 진전되면 작물이 전멸하고 "
        "토양이 영구적으로 오염된다."
    )
    assert quote_exists(quote, _GS025_SOURCE) is False


def test_citable_text_includes_graph_path():
    """그래프 관계에만 있는 사실은 경로를 인용해야 하므로 대조 대상에 포함한다."""
    from src.supervisor.citation_verifier import citable_text

    evidence = Evidence(
        evidence_id="EVD_X",
        document_id="DOC_X",
        title="병해 상세",
        url="",
        quote=_GS025_SOURCE,
        source="graph",
        graph_paths=[["양파", "깜부기병"]],
    )
    text = citable_text(evidence)
    assert "그래프 경로: 양파 → 깜부기병" in text
    assert _GS025_SOURCE in text
    assert quote_exists("그래프 경로: 양파 → 깜부기병", text) is True


def test_citable_text_without_graph_path_is_quote_only():
    """벡터 근거에는 그래프 경로가 없으므로 원문만 대조한다."""
    from src.supervisor.citation_verifier import citable_text

    evidence = Evidence(
        evidence_id="EVD_Y",
        document_id="DOC_Y",
        title="재배 설명",
        url="",
        quote=_GS025_SOURCE,
        source="vector",
    )
    assert citable_text(evidence) == _GS025_SOURCE
