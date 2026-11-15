"""검증상태 안내 문구 생성 테스트."""

from src.guardrails.verification_notice import (
    annotate_answer,
    build_verification_notices,
)


REVIEW_RECORD = {
    "product_name": "펜발러레이트 유제",
    "brand": "일품",
    "verification_status": "REVIEW_REQUIRED",
    "review_reason": "CANCELLATION_HISTORY_FOUND",
}

PRELIMINARY_RECORD = {
    "product_name": "아세페이트 수화제",
    "brand": "베스탄",
    "verification_status": "PRELIMINARY_PASS",
}


def test_review_required_product_gets_notice():
    """답변이 언급한 REVIEW_REQUIRED 품목을 안내한다."""
    notices = build_verification_notices(
        "일품은 수확 7일 전까지 3회 이내로 살포합니다.",
        [REVIEW_RECORD],
    )

    assert len(notices) == 1
    assert "REVIEW_REQUIRED" in notices[0]
    assert "일품" in notices[0]


def test_preliminary_pass_product_gets_collected_data_notice():
    notices = build_verification_notices(
        "베스탄은 1000배로 희석해 살포합니다.",
        [PRELIMINARY_RECORD],
    )

    assert len(notices) == 1
    assert "수집된 등록정보 기준" in notices[0]


def test_unmentioned_product_is_not_annotated():
    """답변이 언급하지 않은 제품은 안내하지 않는다."""
    notices = build_verification_notices(
        "배추는 서늘한 기후에서 자랍니다.",
        [REVIEW_RECORD, PRELIMINARY_RECORD],
    )

    assert notices == []


def test_item_name_mention_is_detected():
    """상표명 대신 품목명으로 언급해도 찾아낸다."""
    notices = build_verification_notices(
        "펜발러레이트 유제는 등록된 약제입니다.",
        [REVIEW_RECORD],
    )

    assert len(notices) == 1


def test_statuses_are_grouped_separately():
    """검증상태별로 안내 문구를 나눈다."""
    notices = build_verification_notices(
        "일품과 베스탄이 등록돼 있습니다.",
        [REVIEW_RECORD, PRELIMINARY_RECORD],
    )

    assert len(notices) == 2


def test_many_products_are_summarized():
    """제품이 많으면 개수로 줄여 안내한다."""
    records = [
        {
            "product_name": f"약제{index}",
            "brand": f"상표{index}",
            "verification_status": "REVIEW_REQUIRED",
        }
        for index in range(8)
    ]

    answer = " ".join(f"약제{index}" for index in range(8))
    notices = build_verification_notices(answer, records)

    assert len(notices) == 1
    assert "등 8개 품목" in notices[0]


def test_annotate_answer_appends_section():
    """안내가 있으면 답변 본문에 절을 덧붙인다."""
    answer = annotate_answer(
        "일품은 등록된 약제입니다.",
        [REVIEW_RECORD],
    )

    assert "[등록정보 검증상태]" in answer
    assert answer.startswith("일품은 등록된 약제입니다.")


def test_annotate_answer_without_records_is_unchanged():
    assert annotate_answer("답변", None) == "답변"
    assert annotate_answer("답변", []) == "답변"


def test_annotate_answer_ignores_unknown_status():
    """확인된 상태 표기가 없으면 덧붙이지 않는다."""
    record = {
        "product_name": "약제",
        "brand": "상표",
        "verification_status": "VERIFIED_CURRENT",
    }

    assert annotate_answer("약제는 등록돼 있습니다.", [record]) == (
        "약제는 등록돼 있습니다."
    )


def test_particle_matches_final_consonant():
    """끝 글자 종성에 맞는 조사를 쓴다."""
    with_final = build_verification_notices(
        "일품은 등록 약제입니다.",
        [REVIEW_RECORD],
    )[0]

    without_final = build_verification_notices(
        "미네토스타는 등록 약제입니다.",
        [{
            "product_name": "미네토스타",
            "verification_status": "REVIEW_REQUIRED",
        }],
    )[0]

    assert with_final.startswith("일품은 ")
    assert without_final.startswith("미네토스타는 ")
