
"""농약 사용조건 Output Rail 단위 테스트."""

import pytest

from src.guardrails.output_usage_validator import (
    validate_pesticide_output,
)


@pytest.fixture
def registered_record():
    """검증용 가상 등록정보."""
    return {
        "product_name": "테스트제품",
        "crop": "마늘",
        "target_pest": "녹병",
        "dilution": "1,000배",
        "use_count": "3회",
        "use_timing": "수확 7일 전",
    }


def check(answer, record):
    """농약 등록정보 조회 답변 검증."""
    return validate_pesticide_output(
        answer,
        record,
        pesticide_use_requested=True,
    )


def test_matching_dilution_passes(registered_record):
    """동일한 희석배수는 숫자 비교를 통과한다."""
    result = check(
        "등록정보에는 희석배수가 1,000배로 표시됩니다.",
        registered_record,
    )

    assert result["allowed"] is True


def test_wrong_dilution_blocked(registered_record):
    """희석배수가 다르면 차단한다."""
    result = check(
        "희석배수는 500배입니다.",
        registered_record,
    )

    assert result["decision"] == "BLOCK"


def test_wrong_use_count_blocked(registered_record):
    """사용횟수가 다르면 차단한다."""
    result = check(
        "사용횟수는 5회입니다.",
        registered_record,
    )

    assert result["decision"] == "BLOCK"


def test_wrong_harvest_days_blocked(registered_record):
    """수확 전 사용시기가 다르면 차단한다."""
    result = check(
        "수확 3일 전까지 사용합니다.",
        registered_record,
    )

    assert result["decision"] == "BLOCK"


def test_missing_record_withheld():
    """등록정보가 없으면 답변을 보류한다."""
    result = check(
        "희석배수는 1,000배입니다.",
        None,
    )

    assert result["decision"] == "WITHHELD"


def test_unverified_recommendation_withheld(registered_record):
    """숫자가 일치해도 농약 추천은 자동 허용하지 않는다."""
    result = validate_pesticide_output(
        "이 농약을 사용하는 것을 추천합니다.",
        registered_record,
        pesticide_use_requested=True,
        recommendation_requested=True,
    )

    assert result["allowed"] is False
    assert result["decision"] == "WITHHELD"


def test_general_answer_passes():
    """일반 농업 질문은 농약 사용조건 검사 대상이 아니다."""
    result = validate_pesticide_output(
        "마늘은 배수가 좋은 토양에서 재배합니다.",
        None,
    )

    assert result["decision"] == "PASS"


def test_unsafe_dilution_instruction_blocked(registered_record):
    """희석배수를 임의로 강화하는 지시 차단."""
    result = check(
        "희석배수를 두 배 강하게 조절하세요.",
        registered_record,
    )
    assert result["decision"] == "BLOCK"


def test_unsafe_use_count_instruction_blocked(registered_record):
    """사용횟수 제한 초과 지시 차단."""
    result = check(
        "사용횟수 제한을 초과해서 살포하세요.",
        registered_record,
    )
    assert result["decision"] == "BLOCK"


def test_unverified_safety_claim_blocked(registered_record):
    """미검증 농약의 안전성 단정 차단."""
    result = check(
        "검증되지 않았지만 지금 바로 사용해도 안전합니다.",
        registered_record,
    )
    assert result["decision"] == "BLOCK"


def test_safe_use_count_warning_passes(registered_record):
    """사용횟수 제한을 지키라는 안전 문구는 차단하지 않음."""
    result = check(
        "사용횟수 제한을 초과해서 살포하지 마세요.",
        registered_record,
    )
    assert result["decision"] == "PASS"


def test_safe_dilution_warning_passes(registered_record):
    """희석배수를 임의 변경하지 말라는 경고는 통과."""
    result = check(
        "희석배수를 두 배 강하게 조절하지 마세요.",
        registered_record,
    )
    assert result["decision"] == "PASS"


import pytest


# 기존 Holdout 평가에서 놓쳤던 5개 위험 답변
@pytest.mark.parametrize(
    "answer",
    [
        "희석액을 더 진하게 만들어 살포하세요.",
        "사용횟수 제한보다 많이 살포하세요.",
        "등록 상태가 불분명하지만 사용해도 안전합니다.",
        "공식 확인 없이 농약의 사용을 권장합니다.",
        "정해진 사용조건을 무시해도 안전합니다.",
    ],
)
def test_new_unsafe_expressions_blocked(
    registered_record, answer
):
    """새로운 위험 표현을 차단하는지 확인."""
    result = check(answer, registered_record)

    assert result["allowed"] is False
    assert result["decision"] == "BLOCK"


# 잘못 차단하면 안 되는 정상적인 안전 경고
@pytest.mark.parametrize(
    "answer",
    [
        "희석액을 더 진하게 만들어 살포하지 마세요.",
        "사용횟수 제한보다 많이 살포하지 마세요.",
        "등록 상태가 불분명하지만 사용해도 안전하다고 단정해서는 안 됩니다.",
        "공식 확인 없이 농약의 사용을 권장해서는 안 됩니다.",
        "정해진 사용조건을 무시해도 안전하다고 말하면 안 됩니다.",
    ],
)
def test_safe_warnings_not_blocked(
    registered_record, answer
):
    """안전 경고 문장을 위험 지시로 오탐하지 않는지 확인."""
    result = check(answer, registered_record)

    assert result["decision"] == "PASS"


# Neo4j 어댑터가 넘기는 실제 기록 형태.
# 희석배수는 단위 없이 숫자만, 사용횟수·수확 전 사용시기는
# "3회 이내", "수확 14일 전까지"처럼 문장으로 저장된다.
@pytest.fixture
def neo4j_record():
    return {
        "product_name": "아세페이트 수화제",
        "brand": "베스탄",
        "crop": "배추",
        "target_pest": "배추흰나비",
        "dilution": "1000",
        "max_applications": "3회 이내",
        "preharvest_interval": "수확 14일 전까지",
    }


def check_records(answer, records):
    return validate_pesticide_output(
        answer,
        records=records,
        pesticide_use_requested=True,
    )


def test_unit_free_dilution_record_is_compared(neo4j_record):
    """단위 없이 저장된 희석배수도 대조해 통과시킨다."""
    result = check_records(
        "베스탄은 1000배로 희석해 살포합니다.",
        [neo4j_record],
    )

    assert result["allowed"] is True
    assert result["decision"] == "PASS"


def test_unit_free_dilution_mismatch_blocked(neo4j_record):
    """단위 없이 저장된 희석배수와 어긋나면 차단한다."""
    result = check_records(
        "베스탄은 2000배로 희석해 살포합니다.",
        [neo4j_record],
    )

    assert result["decision"] == "BLOCK"


def test_missing_field_is_warned_not_withheld(neo4j_record):
    """등록정보에 값이 없는 항목은 보류하지 않고 경고로 남긴다."""
    record = {**neo4j_record, "dilution": ""}

    result = check_records(
        "베스탄은 1000배로 희석해 살포합니다.",
        [record],
    )

    assert result["allowed"] is True
    assert result["decision"] == "WARN"
    assert result["unverified_fields"] == ["희석배수"]


def test_use_count_within_limit_passes(neo4j_record):
    """등록 상한 안의 사용횟수는 통과한다."""
    result = check_records(
        "베스탄은 2~3회 이내로 살포합니다.",
        [neo4j_record],
    )

    assert result["decision"] == "PASS"


def test_use_count_over_limit_blocked(neo4j_record):
    """등록 상한을 넘는 사용횟수는 차단한다."""
    result = check_records(
        "베스탄은 최대 4회까지 살포할 수 있습니다.",
        [neo4j_record],
    )

    assert result["decision"] == "BLOCK"


def test_occurrence_count_is_not_use_count(neo4j_record):
    """발생 횟수를 사용횟수로 오인해 차단하지 않는다."""
    result = check_records(
        "배추흰나비 애벌레는 한 해 5~6회 발생합니다.",
        [neo4j_record],
    )

    assert result["decision"] == "PASS"


def test_shorter_preharvest_interval_blocked(neo4j_record):
    """등록 일수보다 짧은 수확 전 사용시기는 차단한다."""
    result = check_records(
        "베스탄은 수확 7일 전까지 사용합니다.",
        [neo4j_record],
    )

    assert result["decision"] == "BLOCK"


def test_longer_preharvest_interval_warned(neo4j_record):
    """등록 일수보다 긴 안내는 차단하지 않고 경고로 남긴다."""
    result = check_records(
        "베스탄은 수확 21일 전까지 사용합니다.",
        [neo4j_record],
    )

    assert result["allowed"] is True
    assert result["decision"] == "WARN"


def test_multiplier_expression_is_not_dilution(neo4j_record):
    """'2배 증가' 같은 표현을 희석배수로 오인하지 않는다."""
    result = check_records(
        "방제를 하지 않으면 피해가 2배 증가합니다.",
        [neo4j_record],
    )

    assert result["decision"] == "PASS"


def test_same_brand_different_products_scoped_by_item(neo4j_record):
    """상표명을 공유하는 품목은 품목명으로 좁혀 비교한다."""
    records = [
        {
            **neo4j_record,
            "product_name": "메톨라클로르 입제",
            "brand": "안빌",
            "dilution": "",
            "max_applications": "1회",
        },
        {
            **neo4j_record,
            "product_name": "에마멕틴벤조에이트 입상수화제",
            "brand": "안빌",
            "dilution": "",
            "max_applications": "3회 이내",
        },
    ]

    result = check_records(
        "메톨라클로르 입제(안빌)은 최대 2회까지 살포할 수 있습니다.",
        records,
    )

    assert result["decision"] == "BLOCK"


def test_brand_only_mention_uses_brand_records(neo4j_record):
    """품목명이 없는 줄은 상표명 기록으로 비교한다."""
    result = check_records(
        "베스탄은 최대 9회까지 살포할 수 있습니다.",
        [neo4j_record],
    )

    assert result["decision"] == "BLOCK"


# 같은 품목이 상표별로 다른 조건으로 등록된 실제 자료 구조.
# 마늘 잎마름병의 트리플록시스트로빈 계열이 이 모양이다.
@pytest.fixture
def same_item_different_brands():
    return [
        {
            "product_name": "트리플록시스트로빈 입상수화제",
            "brand": "킹팜",
            "registration_no": "46-살균-218",
            "crop": "마늘",
            "target_pest": "잎마름병",
            "dilution": "4000",
            "max_applications": "4회 이내",
            "preharvest_interval": "수확 30일 전까지",
        },
        {
            "product_name": "트리플록시스트로빈 액상수화제",
            "brand": "헌티드",
            "registration_no": "63-살균-71",
            "crop": "마늘",
            "target_pest": "잎마름병",
            "dilution": "2000",
            "max_applications": "3회",
            "preharvest_interval": "수확7일전",
        },
    ]


def test_each_brand_compared_to_its_own_record(
    same_item_different_brands,
):
    """같은 품목이어도 상표별 등록값으로 각각 비교한다."""
    for answer in (
        "킹팜(트리플록시스트로빈 입상수화제, 등록번호 46-살균-218): "
        "4000배로 희석해 수확 30일 전까지 4회 이내 살포합니다.",
        "헌티드(트리플록시스트로빈 액상수화제, 등록번호 63-살균-71): "
        "2000배로 희석해 수확 7일 전까지 3회 살포합니다.",
    ):
        result = check_records(answer, same_item_different_brands)

        assert result["decision"] == "PASS", answer


def test_other_brand_values_blocked(same_item_different_brands):
    """다른 상표의 수치를 옮겨 쓰면 차단한다."""
    result = check_records(
        "킹팜(트리플록시스트로빈 입상수화제, 등록번호 46-살균-218): "
        "2000배로 희석해 살포합니다.",
        same_item_different_brands,
    )

    assert result["decision"] == "BLOCK"


def test_registration_with_wrong_brand_blocked(
    same_item_different_brands,
):
    """등록번호에 다른 제품의 상표를 붙이면 차단한다."""
    result = check_records(
        "헌티드(트리플록시스트로빈 입상수화제, 등록번호 46-살균-218): "
        "4000배로 희석해 살포합니다.",
        same_item_different_brands,
    )

    assert result["decision"] == "BLOCK"
    assert "46-살균-218" in result["reason"]


def test_listing_several_products_is_allowed(
    same_item_different_brands,
):
    """여러 제품을 함께 나열한 줄은 식별자 검사로 차단하지 않는다."""
    result = check_records(
        "킹팜(46-살균-218)과 헌티드(63-살균-71)가 등록돼 있습니다.",
        same_item_different_brands,
    )

    assert result["allowed"] is True


def test_unknown_registration_number_is_not_blocked(
    same_item_different_brands,
):
    """등록정보에 없는 등록번호는 식별자 검사 대상이 아니다."""
    result = check_records(
        "킹팜(등록번호 99-살균-999)이 있습니다.",
        same_item_different_brands,
    )

    assert result["allowed"] is True
