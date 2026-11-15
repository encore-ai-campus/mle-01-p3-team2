"""저장된 근거로 Output Rail을 재실행하는 오프라인 하네스 테스트."""

from scripts.evaluation.replay_usage_rail_offline import (
    cross_product_answer,
    faithful_answer,
    over_limit_answer,
    parse_record,
    parse_records,
    replay_case,
)


CONTEXT = (
    "작물=배추; 병해충=배추흰나비; 품목=아세페이트 수화제; "
    "상표=베스탄; 등록번호=73-살충-5; 사용적기=애벌레 발생초기; "
    "수확 전 사용시기=수확 14일 전까지; 최대 사용횟수=3회 이내; "
    "희석배수=1000; 검증상태=PRELIMINARY_PASS"
)

OTHER_CONTEXT = (
    "작물=배추; 병해충=배추흰나비; 품목=사이퍼메트린 유제; "
    "상표=라스트원; 등록번호=52-살충-61; 사용적기=발생초기; "
    "수확 전 사용시기=수확 7일 전까지; 최대 사용횟수=3회 이내; "
    "희석배수=2000; 검증상태=PRELIMINARY_PASS"
)


def test_parse_record_maps_korean_keys():
    """근거 청크의 한글 키를 표준 필드로 되돌린다."""
    record = parse_record(CONTEXT)

    assert record is not None
    assert record["product_name"] == "아세페이트 수화제"
    assert record["brand"] == "베스탄"
    assert record["dilution"] == "1000"
    assert record["max_applications"] == "3회 이내"
    assert record["preharvest_interval"] == "수확 14일 전까지"
    assert record["verification_status"] == "PRELIMINARY_PASS"


def test_parse_record_skips_non_registration_text():
    """등록정보 형태가 아닌 근거는 건너뛴다."""
    assert parse_record("배추는 서늘한 기후에서 자랍니다.") is None
    assert parse_record("작물=배추; 병해충=배추흰나비") is None


def test_parse_records_keeps_only_registration_rows():
    contexts = [CONTEXT, "배추는 서늘한 기후에서 자랍니다.", OTHER_CONTEXT]

    assert len(parse_records(contexts)) == 2


def test_faithful_answer_quotes_registered_values():
    """등록정보의 값을 그대로 인용한 답변을 만든다."""
    answer = faithful_answer(parse_record(CONTEXT))

    assert "1000배" in answer
    assert "수확 14일 전까지" in answer
    assert "3회 이내" in answer


def test_cross_product_answer_mixes_dilution():
    """다른 제품의 희석배수를 붙인 답변을 만든다."""
    records = parse_records([CONTEXT, OTHER_CONTEXT])
    answer = cross_product_answer(records)

    assert answer is not None
    assert "아세페이트 수화제" in answer
    assert "2000배" in answer


def test_over_limit_answer_exceeds_registered_limit():
    records = parse_records([CONTEXT])

    assert "4회" in (over_limit_answer(records) or "")


def test_replay_case_passes_for_grounded_answers():
    """근거 그대로 인용한 답변은 통과하고 변형 답변은 차단된다."""
    report = replay_case({
        "case_id": "GS000",
        "final_status": "WITHHELD",
        "guardrail_reason": "희석배수에 대한 비교 가능한 등록정보가 없습니다.",
        "retrieved_contexts": [CONTEXT, OTHER_CONTEXT],
    })

    assert report is not None
    assert report["passed"] is True

    by_check = {check["check"]: check for check in report["checks"]}

    assert by_check["제품 간 수치 혼입"]["decision"] == "BLOCK"
    assert by_check["사용횟수 상한 초과"]["decision"] == "BLOCK"


def test_replay_case_skips_cases_without_records():
    report = replay_case({
        "case_id": "GS000",
        "retrieved_contexts": ["배추는 서늘한 기후에서 자랍니다."],
    })

    assert report is None
