import pytest

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


def test_verified_current_without_official_evidence_blocks_recommendation():
    """등록상태 표시만으로는 현재 농약 사용을 추천하지 않는다."""

    result = validate_pesticide(
        GuardrailRequest(
            question="마늘에 사용할 농약을 추천해 주세요.",
            answer="테스트 약제를 검토할 수 있습니다.",
            evidence=[_pesticide_result("VERIFIED_CURRENT")],
            pesticide_use_requested=True,
        )
    )

    assert result.allowed is False
    assert result.decision == "BLOCK"
    assert result.details["official_verified"] is False


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

    def mock_judge(question, answer, evidence, verified_conditions=()):
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


class _StubCompletions:
    """OpenAI 판정 응답을 흉내내는 스텁."""

    def __init__(self, payload):
        self._payload = payload

    def create(self, **_kwargs):
        message = type("M", (), {"content": self._payload})()
        choice = type("C", (), {"message": message})()
        return type("R", (), {"choices": [choice]})()


class _StubClient:
    def __init__(self, payload):
        self.chat = type(
            "Chat", (), {"completions": _StubCompletions(payload)}
        )()


def _judge_request():
    from src.guardrails.contracts import GuardrailRequest

    return GuardrailRequest(
        question="배추 배추흰나비 방제 약제의 사용 기준은?",
        answer="베스탄은 발병 초부터 10일 간격으로 살포합니다.",
        evidence=[
            {
                "source": "graph",
                "text": (
                    "작물=배추; 병해충=배추흰나비; 상표=베스탄; "
                    "사용적기=발병 초 10일 간격"
                ),
                "evidence_ids": ["ev_1"],
                "source_ids": [],
                "chunk_ids": [],
                "metadata": {},
            }
        ],
    )


def _run_judge(monkeypatch, payload, request=None):
    import src.guardrails.output_validator as module

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        module,
        "OpenAI",
        lambda **_kwargs: _StubClient(payload),
    )

    return module.validate_output(request or _judge_request())


def test_judge_wording_severity_allows_with_warning(monkeypatch):
    """표현 차이만 있으면 보류하지 않고 경고로 전달한다."""
    result = _run_judge(
        monkeypatch,
        '{"allowed": false, "severity": "WORDING", '
        '"reason": "같은 간격을 다르게 서술함"}',
    )

    assert result.allowed is True
    assert result.decision == "WARN"
    assert result.warnings
    assert result.details["judge_severity"] == "WORDING"


def test_judge_conflict_severity_falls_back(monkeypatch):
    """근거와 어긋나는 수치는 기존대로 보류한다."""
    result = _run_judge(
        monkeypatch,
        '{"allowed": false, "severity": "CONFLICT", '
        '"reason": "희석배수가 다름"}',
    )

    assert result.allowed is False
    assert result.decision == "FALLBACK"
    assert result.details["judge_severity"] == "CONFLICT"


def test_judge_allowed_keeps_allow_decision(monkeypatch):
    """근거와 일치하면 그대로 통과한다."""
    result = _run_judge(
        monkeypatch,
        '{"allowed": true, "severity": "", "reason": "일치"}',
    )

    assert result.allowed is True
    assert result.decision == "ALLOW"


def test_judge_registered_value_objection_is_deferred(monkeypatch):
    """등록 수치 이견은 결정론 검증 담당이므로 보류하지 않는다."""
    result = _run_judge(
        monkeypatch,
        '{"allowed": false, "category": "REGISTERED_VALUE", '
        '"severity": "CONFLICT", "reason": "희석배수가 다름"}',
    )

    assert result.allowed is True
    assert result.decision == "WARN"
    assert result.details["judge_category"] == "REGISTERED_VALUE"
    assert result.warnings


def test_judge_verification_status_objection_is_deferred(monkeypatch):
    """검증상태 이견은 규칙 삽입 담당이므로 보류하지 않는다."""
    result = _run_judge(
        monkeypatch,
        '{"allowed": false, "category": "VERIFICATION_STATUS", '
        '"severity": "UNSUPPORTED", '
        '"reason": "REVIEW_REQUIRED인데 사용 가능하다고 단정"}',
    )

    assert result.allowed is True
    assert result.decision == "WARN"


def test_judge_completeness_objection_is_deferred(monkeypatch):
    """답변 완전성은 판정 대상이 아니다."""
    result = _run_judge(
        monkeypatch,
        '{"allowed": false, "category": "COMPLETENESS", '
        '"severity": "UNSUPPORTED", "reason": "일부 항목 누락"}',
    )

    assert result.allowed is True


@pytest.mark.parametrize(
    "category",
    ["FABRICATED", "WRONG_TARGET", "OVERGENERALIZED"],
)
def test_judge_in_scope_objection_still_withholds(monkeypatch, category):
    """판정 범위 안의 사유는 그대로 보류한다."""
    result = _run_judge(
        monkeypatch,
        '{"allowed": false, "category": "' + category + '", '
        '"severity": "UNSUPPORTED", "reason": "근거 없음"}',
    )

    assert result.allowed is False
    assert result.decision == "FALLBACK"
    assert result.details["judge_category"] == category


def test_explicit_usability_claim_without_notice_still_withholds(monkeypatch):
    """현재 사용 가능하다는 명시적 단정은 기존처럼 보류한다."""
    request = GuardrailRequest(
        question="배추흰나비 약제를 지금 사용해도 되나요?",
        answer="베스탄은 현재 사용 가능합니다.",
        evidence=_judge_request().evidence,
    )

    result = _run_judge(
        monkeypatch,
        '{"allowed": false, "category": "USABILITY_CLAIM", '
        '"severity": "UNSUPPORTED", "reason": "현재 사용 가능 단정"}',
        request,
    )

    assert result.allowed is False
    assert result.decision == "FALLBACK"


def test_judge_unknown_category_still_withholds(monkeypatch):
    """분류를 알 수 없으면 안전한 쪽으로 보류한다."""
    result = _run_judge(
        monkeypatch,
        '{"allowed": false, "reason": "분류 없음"}',
    )

    assert result.allowed is False
    assert result.decision == "FALLBACK"


def test_verified_conditions_reach_the_judge(monkeypatch):
    """이미 대조한 항목을 판정 모델에 사실로 넘긴다."""
    import src.guardrails.output_validator as module

    from src.guardrails.contracts import GuardrailRequest

    captured = {}

    class _Capturing(_StubCompletions):
        def create(self, **kwargs):
            captured["user"] = kwargs["messages"][1]["content"]
            return super().create(**kwargs)

    class _Client:
        def __init__(self, **_kwargs):
            self.chat = type(
                "Chat", (), {"completions": _Capturing(
                    '{"allowed": true, "category": "NONE", "reason": "ok"}'
                )}
            )()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(module, "OpenAI", lambda **_kw: _Client())

    base = _judge_request()
    request = GuardrailRequest(
        question=base.question,
        answer=base.answer,
        evidence=base.evidence,
        metadata={"verified_conditions": ["희석배수", "사용횟수"]},
    )

    module.validate_output(request)

    assert "이미 확인된 사항" in captured["user"]
    assert "희석배수, 사용횟수" in captured["user"]


def test_gs035_usability_claim_without_explicit_claim_is_deferred(monkeypatch):
    """등록 사용기준 목록을 현재 사용 가능 단정으로 오인해도 통과한다."""
    request = GuardrailRequest(
        question="배추 복숭아혹진딧물 등록 약제와 사용기준은?",
        answer=(
            "배추의 복숭아혹진딧물 등록 약제와 사용기준은 다음과 "
            "같습니다.\n스미사이딘: 발생초기, 2000배 희석."
        ),
        evidence=[
            {
                "text": (
                    "작물=배추; 병해충=복숭아혹진딧물; "
                    "상표=스미사이딘; 등록번호=3-살충-154; "
                    "사용적기=발생초기; 희석배수=2000"
                )
            }
        ],
    )

    result = _run_judge(
        monkeypatch,
        '{"allowed": false, "category": "USABILITY_CLAIM", '
        '"severity": "UNSUPPORTED", '
        '"reason": "지금 사용해도 된다고 단정함"}',
        request,
    )

    assert result.allowed is True
    assert result.decision == "WARN"
    assert result.details["judge_category"] == "USABILITY_CLAIM"


def test_review_required_notice_downgrades_usability_claim(monkeypatch):
    """REVIEW_REQUIRED 확인 안내가 있으면 사용 가능성 이견을 경고로 낮춘다."""
    request = GuardrailRequest(
        question="배추흰나비 등록 약제는?",
        answer=(
            "일품은 현재 사용 가능합니다. 검증상태는 REVIEW_REQUIRED입니다."
            "\n\n[등록정보 검증상태]\n"
            "일품은 현재 등록·사용 가능 여부가 확인되지 않았습니다. "
            "사용 전 최신 공식 등록정보를 확인해 주세요."
        ),
        evidence=[
            {
                "text": (
                    "작물=배추; 병해충=배추흰나비; 상표=일품; "
                    "등록번호=45-살충-25; 검증상태=REVIEW_REQUIRED"
                )
            }
        ],
    )

    result = _run_judge(
        monkeypatch,
        '{"allowed": false, "category": "USABILITY_CLAIM", '
        '"severity": "CONFLICT", "reason": "사용 가능성을 단정함"}',
        request,
    )

    assert result.allowed is True
    assert result.decision == "WARN"


def test_gs036_exact_brand_registration_pairs_override_fabricated(monkeypatch):
    """동일 품목의 복수 상표를 혼동한 FABRICATED 오판을 무효화한다."""
    request = GuardrailRequest(
        question="마늘 잎마름병 등록 약제는?",
        answer=(
            "플루디옥소닐 액상수화제(상표: 에이팜썬더, "
            "등록번호: 2-살균-279): 희석배수 2000.\n"
            "플루디옥소닐 액상수화제(상표: 신기록-D, "
            "등록번호: 64-살균-15): 희석배수 2000."
        ),
        evidence=[
            {
                "text": (
                    "작물=마늘; 병해충=잎마름병; "
                    "품목=플루디옥소닐 액상수화제; 상표=에이팜썬더; "
                    "등록번호=2-살균-279; 희석배수=2000"
                )
            },
            {
                "text": (
                    "작물=마늘; 병해충=잎마름병; "
                    "품목=플루디옥소닐 액상수화제; 상표=신기록-D; "
                    "등록번호=64-살균-15; 희석배수=2000"
                )
            },
        ],
    )

    result = _run_judge(
        monkeypatch,
        '{"allowed": false, "category": "FABRICATED", '
        '"severity": "CONFLICT", '
        '"reason": "에이팜썬더의 상표와 등록번호가 근거와 다름"}',
        request,
    )

    assert result.allowed is True
    assert result.decision == "WARN"
    assert result.details["judge_category"] == "FABRICATED"


def test_gs066_usage_instruction_is_not_current_usability_claim(monkeypatch):
    """등록 사용법의 '사용하며'를 현재 사용 가능 단정으로 보지 않는다."""
    request = GuardrailRequest(
        question="고추 탄저병 등록 약제와 사용기준은?",
        answer=(
            "베노밀 수화제(밧사피, 등록번호 1-살균-53)는 "
            "발병초부터 10일 간격으로 사용하며, 최대 4회 이내, "
            "수확 2일 전까지 사용합니다."
        ),
        evidence=[
            {
                "text": (
                    "작물=고추; 병해충=탄저병; 품목=베노밀 수화제; "
                    "상표=밧사피; 등록번호=1-살균-53; "
                    "사용적기=발병초부터 10일간격; "
                    "수확 전 사용시기=수확 2일 전까지; "
                    "최대 사용횟수=4회 이내"
                )
            }
        ],
    )

    result = _run_judge(
        monkeypatch,
        '{"allowed": false, "category": "USABILITY_CLAIM", '
        '"severity": "UNSUPPORTED", '
        '"reason": "사용 가능성을 단정적으로 표현함"}',
        request,
    )

    assert result.allowed is True
    assert result.decision == "WARN"
    assert result.details["deterministic_override"] == (
        "NO_EXPLICIT_USABILITY_CLAIM"
    )


def test_gs012_grounded_usage_row_overrides_overgeneralized(monkeypatch):
    """제품별 사용적기가 근거와 같으면 OVERGENERALIZED 오판을 낮춘다."""
    request = GuardrailRequest(
        question="배추 파밤나방 등록 약제와 사용기준은?",
        answer=(
            "비펜트린 입상수화제·팡자비·46-살충-449 | "
            "다발생기 | 수확 14일 전까지 | 4000배 | 3회 이내"
        ),
        evidence=[
            {
                "text": (
                    "작물=배추; 병해충=파밤나방; "
                    "품목=비펜트린 입상수화제; 상표=팡자비; "
                    "등록번호=46-살충-449; 사용적기=다발생기; "
                    "수확 전 사용시기=수확 14일 전까지; "
                    "최대 사용횟수=3회 이내; 희석배수=4000"
                )
            }
        ],
        metadata={
            "verified_conditions": [
                "희석배수", "사용횟수", "수확 전 사용시기"
            ]
        },
    )

    result = _run_judge(
        monkeypatch,
        '{"allowed": false, "category": "OVERGENERALIZED", '
        '"severity": "CONFLICT", '
        '"reason": "팡자비를 애벌레 발생초기로 일반화함"}',
        request,
    )

    assert result.allowed is True
    assert result.decision == "WARN"
    assert result.details["deterministic_override"] == (
        "REGISTERED_USAGE_ROWS_VERIFIED"
    )


def test_gs036_grounded_registration_rows_override_overgeneralized(monkeypatch):
    """복수 제품의 등록번호·수치가 근거와 같으면 judge 오판을 낮춘다."""
    request = GuardrailRequest(
        question="마늘 잎마름병 방제용 품목과 사용조건은?",
        answer=(
            "트리플록시스트로빈 액상수화제(헌티드, 등록번호 "
            "63-살균-71): 발생초기에 10일 간격으로 사용하고, 2000배, "
            "수확 7일 전, 3회.\n"
            "프로피코나졸.테부코나졸 유현탁제(이차메, 등록번호 "
            "8-살균-310): 발병초부터 10일 간격, 2000배, "
            "수확 14일 전까지, 3회 이내."
        ),
        evidence=[
            {
                "text": (
                    "작물=마늘; 병해충=잎마름병; "
                    "품목=트리플록시스트로빈 액상수화제; 상표=헌티드; "
                    "등록번호=63-살균-71; 사용적기=발생초기 10일간격; "
                    "수확 전 사용시기=수확7일전; 최대 사용횟수=3회; "
                    "희석배수=2000"
                )
            },
            {
                "text": (
                    "작물=마늘; 병해충=잎마름병; "
                    "품목=프로피코나졸.테부코나졸 유현탁제; "
                    "상표=이차메; 등록번호=8-살균-310; "
                    "사용적기=발병초 10일간격; "
                    "수확 전 사용시기=수확 14일 전까지; "
                    "최대 사용횟수=3회 이내; 희석배수=2000"
                )
            },
        ],
        metadata={
            "verified_conditions": [
                "희석배수", "사용횟수", "수확 전 사용시기"
            ]
        },
    )

    result = _run_judge(
        monkeypatch,
        '{"allowed": false, "category": "OVERGENERALIZED", '
        '"severity": "CONFLICT", '
        '"reason": "등록번호와 희석배수가 근거와 다름"}',
        request,
    )

    assert result.allowed is True
    assert result.decision == "WARN"


def test_wrong_usage_timing_still_blocks_overgeneralized(monkeypatch):
    """식별자·수치가 맞아도 사용적기를 바꾸면 계속 보류한다."""
    request = GuardrailRequest(
        question="배추 파밤나방 등록 약제와 사용기준은?",
        answer=(
            "비펜트린 입상수화제·팡자비·46-살충-449 | "
            "애벌레 발생초기 | 수확 14일 전까지 | 4000배 | 3회 이내"
        ),
        evidence=[
            {
                "text": (
                    "작물=배추; 병해충=파밤나방; "
                    "품목=비펜트린 입상수화제; 상표=팡자비; "
                    "등록번호=46-살충-449; 사용적기=다발생기; "
                    "수확 전 사용시기=수확 14일 전까지; "
                    "최대 사용횟수=3회 이내; 희석배수=4000"
                )
            }
        ],
        metadata={
            "verified_conditions": [
                "희석배수", "사용횟수", "수확 전 사용시기"
            ]
        },
    )

    result = _run_judge(
        monkeypatch,
        '{"allowed": false, "category": "OVERGENERALIZED", '
        '"severity": "CONFLICT", "reason": "사용적기가 다름"}',
        request,
    )

    assert result.allowed is False
    assert result.decision == "FALLBACK"
