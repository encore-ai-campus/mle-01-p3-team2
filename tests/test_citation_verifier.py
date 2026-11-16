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


def test_support_checker_receives_graph_path_for_relation_claim():
    """그래프 관계에만 있는 병명 주장도 지원 판정기가 경로를 받아 통과한다.

    verify_citations 가 지원 판정 단계에서 evidence.quote 만 넘기면, 병명이
    청크 본문이 아니라 그래프 관계에만 있는 경우 판정기가 "근거에 없다"고
    보고 NOT_SUPPORTED 로 떨어진다(E2E 2번 회귀). source_text 로 citable_text
    를 넘겨 경로가 함께 전달되는지 확인한다.
    """
    evidence = Evidence(
        evidence_id="EVD_REL",
        document_id="DOC_REL",
        title="양파 병해 상세",
        url="",
        quote=(
            "어린 유묘에서 생육이 정지되고 지상부가 시들며 "
            "인경 내부에 검은색 가루가 보인다."
        ),
        source="graph",
        graph_paths=[["양파", "깜부기병"]],
    )

    answer = CitedAnswer(
        answer="해당 증상은 깜부기병입니다.",
        claims=[
            Claim(
                text="해당 증상을 보이는 병은 깜부기병이다.",
                citations=[
                    Citation(
                        evidence_id="EVD_REL",
                        quote="그래프 경로: 양파 → 깜부기병",
                    )
                ],
            )
        ],
    )

    seen_contexts: list[dict[str, str]] = []

    def support_checker(claim, contexts):
        seen_contexts.extend(contexts)
        # 병명이 그래프 경로로 전달되면 지원 판정이 가능하다.
        supported = any(
            "깜부기병" in context["source_text"]
            for context in contexts
        )
        return AnswerReview(
            supported=supported,
            reason="그래프 관계가 병명을 뒷받침합니다."
            if supported
            else "근거에 병명이 없습니다.",
        )

    result = verify_citations(answer, [evidence], support_checker)

    # 지원 판정기가 받은 source_text 에 그래프 경로(병명)가 있어야 한다.
    assert seen_contexts
    assert "깜부기병" in seen_contexts[0]["source_text"]
    assert result.status == "PASS"
    assert result.failures == []


def test_gs031_repairs_unknown_id_from_unique_matching_quote():
    """존재하지 않는 ID도 인용문이 한 근거에만 있으면 안전하게 교정한다."""
    from src.supervisor.citation_verifier import repair_unknown_citation_ids

    answer = CitedAnswer(
        answer="진주시에서 파종한 상추 품종은 공심채입니다.",
        claims=[
            Claim(
                text="진주시에서 파종한 상추 품종은 공심채이다.",
                citations=[
                    Citation(
                        evidence_id="EVD_UNKNOWN",
                        quote="상추 품종 '공심채'을 파종한 실측 기록이다.",
                    )
                ],
            )
        ],
    )
    evidence = Evidence(
        evidence_id="EVD_GS031",
        document_id="DOC_GS031",
        title="진주시 파종 기록",
        url="",
        quote=(
            "경상남도 진주시의 익명 농가·육묘장에서 20231219에 "
            "상추 품종 '공심채'을 파종한 실측 기록이다."
        ),
        source="graph",
    )

    repaired = repair_unknown_citation_ids(answer, [evidence])

    assert repaired.claims[0].citations[0].evidence_id == "EVD_GS031"
    assert answer.claims[0].citations[0].evidence_id == "EVD_UNKNOWN"


def test_unknown_id_is_not_repaired_when_quote_matches_multiple_evidences():
    """인용문이 여러 근거에 있으면 임의의 ID로 바꾸지 않는다."""
    from src.supervisor.citation_verifier import repair_unknown_citation_ids

    answer = CitedAnswer(
        answer="공심채입니다.",
        claims=[
            Claim(
                text="품종은 공심채이다.",
                citations=[
                    Citation(
                        evidence_id="EVD_UNKNOWN",
                        quote="품종은 공심채이다.",
                    )
                ],
            )
        ],
    )
    evidences = [
        Evidence(
            evidence_id=f"EVD_{index}",
            document_id=f"DOC_{index}",
            title="기록",
            url="",
            quote="품종은 공심채이다.",
        )
        for index in (1, 2)
    ]

    repaired = repair_unknown_citation_ids(answer, evidences)

    assert repaired.claims[0].citations[0].evidence_id == "EVD_UNKNOWN"
