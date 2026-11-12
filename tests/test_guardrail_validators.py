from src.retrievers.search_result import SearchResult

from src.guardrails.contracts import GuardrailRequest
from src.guardrails.evidence_validator import validate_evidence
from src.guardrails.output_validator import validate_output
from src.guardrails.pesticide_validator import validate_pesticide
from src.guardrails.pipeline import validate_request


def _valid_result() -> SearchResult:
    return SearchResult(
        source="vector",
        text="마늘 잎마름병은 4~5월 다습한 환경에서 심해진다.",
        evidence_ids=["EVD_18E55EAE2F72ED3A"],
        source_ids=["SRC_2DE42643BECD363B"],
        chunk_ids=["CHK_C6C71D52D933D8ED"],
        metadata={"evidence_type": "disease_record"},
    )


def _pesticide_result(status: str | None, tag: str = "A") -> SearchResult:
    metadata = {"evidence_type": "pesticide_use_validated_record"}
    if status is not None:
        metadata["verification_status"] = status
    return SearchResult(
        source="graph",
        text=f"작물=마늘; 품목=테스트 약제; 검증상태={status or '미확인'}",
        evidence_ids=[f"EVD_{tag}AAAAAAAAAAAAAAA"],
        source_ids=[f"SRC_{tag}BBBBBBBBBBBBBBB"],
        chunk_ids=[f"CHK_{tag}CCCCCCCCCCCCCCC"],
        metadata=metadata,
    )


def test_valid_mock_search_result_is_allowed():
    result = validate_evidence(
        GuardrailRequest(
            question="마늘 잎마름병은 언제 심해지나요?",
            answer="4~5월 다습한 환경에서 심해집니다.",
            evidence=[_valid_result()],
        )
    )

    assert result.allowed is True
    assert result.decision == "ALLOW"


def test_empty_evidence_is_blocked():
    result = validate_evidence(
        GuardrailRequest(question="질문", answer="답변", evidence=[])
    )

    assert result.allowed is False
    assert result.decision == "BLOCK"
    assert "근거" in result.reason


def test_invalid_source_text_and_ids_are_blocked_without_raising():
    invalid = SearchResult(
        source="database",  # type: ignore[arg-type]
        text=" ",
        evidence_ids=["bad-id", "bad-id"],
        source_ids=[],
        chunk_ids=[],
    )

    result = validate_evidence(
        GuardrailRequest(question="질문", answer="답변", evidence=[invalid])
    )

    assert result.allowed is False
    assert result.decision == "BLOCK"
    assert result.details["invalid_items"]


def test_request_and_result_round_trip_to_dict():
    request = GuardrailRequest(
        question="질문",
        answer="답변",
        evidence=[_valid_result()],
        expected_facts=["4~5월"],
        pesticide_use_requested=True,
        metadata={"request_id": "REQ_ABC123"},
    )

    result = validate_evidence(request)
    payload = result.to_dict()

    assert payload["allowed"] is True
    assert payload["decision"] == "ALLOW"
    assert isinstance(payload["warnings"], list)
    assert payload["details"]["checked_items"] == 1


def test_verified_current_allows_pesticide_recommendation():
    result = validate_pesticide(
        GuardrailRequest(
            question="마늘에 사용할 농약을 추천해 주세요.",
            answer="테스트 약제를 검토할 수 있습니다.",
            evidence=[_pesticide_result("VERIFIED_CURRENT")],
            pesticide_use_requested=True,
        )
    )

    assert result.allowed is True
    assert result.decision == "ALLOW"


def test_unknown_status_blocks_pesticide_recommendation():
    result = validate_pesticide(
        GuardrailRequest(
            question="마늘에 사용할 농약을 추천해 주세요.",
            answer="테스트 약제를 사용하세요.",
            evidence=[_pesticide_result(None)],
            pesticide_use_requested=True,
        )
    )

    assert result.allowed is False
    assert result.decision == "BLOCK"
    assert "공식" in (result.fallback_text or "")
    assert result.details["statuses"] == ["UNKNOWN"]


def test_unconfirmed_statuses_are_fail_closed_for_recommendations():
    for status in ("PRELIMINARY_PASS", "REVIEW_REQUIRED", "EXPIRED"):
        result = validate_pesticide(
            GuardrailRequest(
                question="농약 사용을 추천해 주세요.",
                answer="사용할 수 있습니다.",
                evidence=[_pesticide_result(status)],
                pesticide_use_requested=True,
            )
        )

        assert result.allowed is False, status
        assert result.decision == "BLOCK", status


def test_non_recommendation_fact_answer_warns_but_can_continue():
    result = validate_pesticide(
        GuardrailRequest(
            question="등록 정보의 검증상태를 알려 주세요.",
            answer="검증상태는 PRELIMINARY_PASS입니다.",
            evidence=[_pesticide_result("PRELIMINARY_PASS")],
            pesticide_use_requested=False,
        )
    )

    assert result.allowed is True
    assert result.decision == "WARN"
    assert result.warnings


def test_missing_status_in_one_of_multiple_pesticide_records_blocks():
    result = validate_request(
        GuardrailRequest(
            question="농약 사용을 추천해 주세요.",
            answer="사용할 수 있습니다.",
            evidence=[
                _pesticide_result("VERIFIED_CURRENT", tag="A"),
                _pesticide_result(None, tag="B"),
            ],
            pesticide_use_requested=True,
        )
    )

    assert result.allowed is False
    assert result.decision == "BLOCK"


def test_conflicting_metadata_statuses_block_recommendation():
    evidence = _pesticide_result("VERIFIED_CURRENT")
    evidence.metadata["registration_status"] = "EXPIRED"

    result = validate_pesticide(
        GuardrailRequest(
            question="농약 사용을 추천해 주세요.",
            answer="사용할 수 있습니다.",
            evidence=[evidence],
            pesticide_use_requested=True,
        )
    )

    assert result.allowed is False
    assert result.decision == "BLOCK"


def test_multiple_text_status_markers_block_on_conflict():
    evidence = _pesticide_result("VERIFIED_CURRENT")
    evidence.text = "품목=테스트 약제; 검증상태=VERIFIED_CURRENT; 검증상태=EXPIRED"

    result = validate_pesticide(
        GuardrailRequest(
            question="농약 사용을 추천해 주세요.",
            answer="사용할 수 있습니다.",
            evidence=[evidence],
            pesticide_use_requested=True,
        )
    )

    assert result.allowed is False
    assert result.decision == "BLOCK"


def test_matching_expected_facts_are_allowed():
    result = validate_output(
        GuardrailRequest(
            question="마늘 잎마름병은 언제 심해지나요?",
            answer="4~5월 다습한 환경에서 심해집니다.",
            evidence=[_valid_result()],
            expected_facts=["4~5월", "다습한 환경"],
        )
    )

    assert result.allowed is True
    assert result.decision == "ALLOW"


def test_fact_missing_from_answer_returns_fallback():
    result = validate_output(
        GuardrailRequest(
            question="질문",
            answer="생육 후기에 심해집니다.",
            evidence=[_valid_result()],
            expected_facts=["4~5월"],
        )
    )

    assert result.allowed is False
    assert result.decision == "FALLBACK"
    assert result.fallback_text


def test_answer_fact_not_supported_by_evidence_returns_fallback():
    result = validate_output(
        GuardrailRequest(
            question="질문",
            answer="9월 건조한 환경에서 심해집니다.",
            evidence=[_valid_result()],
            expected_facts=["9월"],
        )
    )

    assert result.allowed is False
    assert result.decision == "FALLBACK"
    assert "근거" in result.reason


def test_empty_answer_returns_fallback():
    result = validate_output(
        GuardrailRequest(
            question="질문",
            answer=" ",
            evidence=[_valid_result()],
            expected_facts=["4~5월"],
        )
    )

    assert result.allowed is False
    assert result.decision == "FALLBACK"


def test_pipeline_stops_on_empty_evidence_and_merges_warning():
    result = validate_request(
        GuardrailRequest(
            question="질문",
            answer="답변",
            evidence=[],
            pesticide_use_requested=False,
        )
    )

    assert result.allowed is False
    assert result.decision == "BLOCK"
    assert "근거" in result.reason


def test_unsupported_extra_instruction_returns_fallback():
    result = validate_output(
        GuardrailRequest(
            question="질문",
            answer="4~5월 다습한 환경에서 심해집니다. 임의로 10회 살포하세요.",
            evidence=[_valid_result()],
            expected_facts=["4~5월", "다습한 환경"],
        )
    )

    assert result.allowed is False
    assert result.decision == "FALLBACK"


def test_empty_expected_fact_returns_fallback():
    result = validate_output(
        GuardrailRequest(
            question="질문",
            answer="관련 없는 답변입니다.",
            evidence=[_valid_result()],
            expected_facts=[""],
        )
    )

    assert result.allowed is False
    assert result.decision == "FALLBACK"



def test_missing_expected_facts_uses_llm_judge(monkeypatch):
    """expected_facts가 없어도 Judge 검증을 통해 답변을 허용한다."""

    from src.guardrails.contracts import ValidationResult

    def mock_judge(question, answer, evidence):
        assert question == "질문"
        assert "4~5월" in answer
        assert "4~5월" in evidence

        return ValidationResult(
            allowed=True,
            decision="ALLOW",
            reason="모의 LLM-as-Judge 검증 통과",
            details={"validation_method": "llm_judge"},
        )

    monkeypatch.setattr(
        "src.guardrails.output_validator._judge_with_llm",
        mock_judge,
    )

    result = validate_output(
        GuardrailRequest(
            question="질문",
            answer="4~5월 다습한 환경에서 심해집니다.",
            evidence=[_valid_result()],
        )
    )

    assert result.allowed is True
    assert result.decision == "ALLOW"
    assert result.details["validation_method"] == "llm_judge"


def test_numeric_decimal_difference_returns_fallback():
    evidence = SearchResult(
        source="vector",
        text="희석배수는 1.0 mL이다.",
        evidence_ids=["EVD_1111111111111111"],
        source_ids=["SRC_2222222222222222"],
        chunk_ids=["CHK_3333333333333333"],
    )

    result = validate_output(
        GuardrailRequest(
            question="희석배수는 얼마인가요?",
            answer="희석배수는 10 mL입니다.",
            evidence=[evidence],
            expected_facts=["1.0 mL"],
        )
    )

    assert result.allowed is False
    assert result.decision == "FALLBACK"


def test_malformed_evidence_types_are_blocked_without_raising():
    malformed_items = [
        {"source": [], "text": "내용", "evidence_ids": ["EVD_AAAAAAAAAAAAAAAA"]},
        {"source": "vector", "text": "내용", "evidence_ids": [{}]},
    ]

    for item in malformed_items:
        result = validate_evidence(
            GuardrailRequest(question="질문", answer="답변", evidence=[item])
        )
        assert result.allowed is False
        assert result.decision == "BLOCK"


def test_non_iterable_evidence_is_blocked_without_raising():
    result = validate_evidence(
        GuardrailRequest(question="질문", answer="답변", evidence=42)  # type: ignore[arg-type]
    )

    assert result.allowed is False
    assert result.decision == "BLOCK"
