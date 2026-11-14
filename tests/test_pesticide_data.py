from __future__ import annotations

import pytest

from src.streamlit.crop_data import build_graph
from src.streamlit.pesticide_data import (
    ANY,
    build_record,
    build_records,
    facet_values,
    filter_records,
    targets_for_crop,
    toxicity_values,
)


def pesticide_attributes(**overrides):
    base = {
        "brand": "아리큐어",
        "product": "메타플루미존 액상수화제",
        "company": "인바이오(주)",
        "purpose": "살충제",
        "formulation": "액상수화제",
        "target_name": "담배거세미나방",
        "dilution": "2000",
        "use_quantity": "-",
        "application_method": "다발생기",
        "use_timing": "수확7일전 까지",
        "use_count": "2회 이내",
        "human_livestock_toxicity": "보통독성(Ⅲ)",
        "ecological_toxicity": "Ⅰ급",
        "registration_no": "19-살충-4",
        "registration_valid_until": "20280330",
        "validation_status": "PRELIMINARY_PASS",
        "cancellation_dates": "",
        "audit_disposition": None,
        "official_query_url": None,
        "canonical_crop": "배추",
    }
    base.update(overrides)
    return base


# 이 환경에서는 pytest fixture가 임시 디렉터리 권한 문제로 실패해서
# 평범한 헬퍼 함수로 표본 데이터를 만듭니다.
def sample_records():
    entities = [
        {"entity_id": "CRP_1", "entity_type": "Crop", "name": "배추", "attributes": {}},
        {"entity_id": "CRP_2", "entity_type": "Crop", "name": "고추", "attributes": {}},
        {
            "entity_id": "PST_1",
            "entity_type": "PesticideUse",
            "name": "메타플루미존 액상수화제 - 담배거세미나방",
            "attributes": pesticide_attributes(),
        },
        {
            "entity_id": "PST_2",
            "entity_type": "PesticideUse",
            "name": "비펜트린 수화제 - 배추좀나방",
            "attributes": pesticide_attributes(
                brand="강심장",
                product="비펜트린 수화제",
                target_name="배추좀나방",
                formulation="수화제",
                human_livestock_toxicity="저독성(Ⅳ)",
            ),
        },
        {
            "entity_id": "PST_3",
            "entity_type": "PesticideUse",
            "name": "아족시스트로빈 액상수화제 - 탄저병",
            "attributes": pesticide_attributes(
                brand="오티바",
                product="아족시스트로빈 액상수화제",
                target_name="탄저병",
                purpose="살균제",
                canonical_crop="고추",
                cancellation_dates="2024-08-26",
                audit_disposition="HOLD_RECOMMENDATION",
                official_query_url="https://psis.rda.go.kr/example",
            ),
        },
    ]
    relations = [
        {
            "from_id": "CRP_1",
            "to_id": "PST_1",
            "relation_type": "HAS_REGISTERED_PESTICIDE_USE",
        },
        {
            "from_id": "CRP_1",
            "to_id": "PST_2",
            "relation_type": "HAS_REGISTERED_PESTICIDE_USE",
        },
        {
            "from_id": "CRP_2",
            "to_id": "PST_3",
            "relation_type": "HAS_REGISTERED_PESTICIDE_USE",
        },
    ]
    return build_records(build_graph(entities, relations))


def test_build_records_links_each_use_to_its_crop() -> None:
    records = sample_records()

    assert len(records) == 3
    assert {r.crop for r in records} == {"배추", "고추"}


def test_build_record_formats_valid_until_and_drops_placeholder() -> None:
    record = build_record("배추", pesticide_attributes())

    assert record.valid_until == "2028-03-30"
    # "-"는 값이 없다는 뜻이라 빈 문자열로 둡니다.
    assert record.quantity == ""
    assert record.dilution == "2000"


def test_build_record_flags_cancellation_and_hold() -> None:
    clean = build_record("배추", pesticide_attributes())
    cancelled = build_record("배추", pesticide_attributes(cancellation_dates="2024-08-26"))
    held = build_record("배추", pesticide_attributes(audit_disposition="HOLD_RECOMMENDATION"))

    assert clean.needs_review is False
    assert cancelled.needs_review is True
    assert held.needs_review is True


def test_filter_records_by_crop_and_target() -> None:
    records = sample_records()

    found = filter_records(records, crop="배추", target="배추좀나방")

    assert [r.brand for r in found] == ["강심장"]


def test_filter_records_by_keyword_matches_brand_product_and_company() -> None:
    records = sample_records()

    assert len(filter_records(records, keyword="아족시스트로빈")) == 1
    assert len(filter_records(records, keyword="인바이오")) == 3
    assert len(filter_records(records, keyword="없는약")) == 0


def test_filter_records_can_hide_records_needing_review() -> None:
    records = sample_records()

    found = filter_records(records, hide_needs_review=True)

    assert len(found) == 2
    assert all(not r.needs_review for r in found)


def test_filter_records_combines_conditions() -> None:
    records = sample_records()

    found = filter_records(records, purpose="살충제", toxicity="저독성(Ⅳ)")

    assert [r.brand for r in found] == ["강심장"]


def test_targets_for_crop_only_lists_that_crops_targets() -> None:
    records = sample_records()

    assert targets_for_crop(records, "배추") == [ANY, "담배거세미나방", "배추좀나방"]
    assert targets_for_crop(records, "고추") == [ANY, "탄저병"]


def test_facet_values_start_with_any_and_are_sorted() -> None:
    records = sample_records()

    assert facet_values(records, "purpose") == [ANY, "살균제", "살충제"]


def test_toxicity_values_follow_risk_order_not_alphabet() -> None:
    records = sample_records()

    assert toxicity_values(records) == [ANY, "저독성(Ⅳ)", "보통독성(Ⅲ)"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("20311206", "2031-12-06"), ("", ""), ("미정", "미정")],
)
def test_build_record_valid_until_variants(raw: str, expected: str) -> None:
    record = build_record("배추", pesticide_attributes(registration_valid_until=raw))

    assert record.valid_until == expected
