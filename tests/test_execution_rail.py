
"""Execution Rail 단위 테스트."""

from src.guardrails.execution_rail import (
    get_registered_usage,
)


# 실제 농약 데이터가 아닌 가상 테스트 기록
RECORD = {
    "product_name": "테스트 약제 A",
    "crop": "마늘",
    "target_pest": "잎마름병",
    "dilution": "2000배",
    "max_applications": "3회",
    "verification_status": "PRELIMINARY_PASS",
}


def lookup(*args):
    return [RECORD.copy()]


def query(**kwargs):
    return get_registered_usage(
        product_name="테스트 약제 A",
        crop="마늘",
        target_pest="잎마름병",
        lookup_fn=lookup,
        **kwargs,
    )


def test_lookup_returns_warning():
    result = query()

    assert result.allowed is True
    assert result.decision == "WARN"
    assert result.warnings


def test_recommendation_without_official_check_blocked():
    result = query(recommend=True)

    assert result.allowed is False
    assert result.decision == "BLOCK"


def test_wrong_crop_not_matched():
    result = get_registered_usage(
        product_name="테스트 약제 A",
        crop="양파",
        target_pest="잎마름병",
        lookup_fn=lookup,
    )

    assert result.allowed is False
    assert result.decision == "FALLBACK"


def test_conflicting_records_rejected():
    def conflicting_lookup(*args):
        other = RECORD.copy()
        other["dilution"] = "1000배"
        return [RECORD.copy(), other]

    result = get_registered_usage(
        product_name="테스트 약제 A",
        crop="마늘",
        target_pest="잎마름병",
        lookup_fn=conflicting_lookup,
    )

    assert result.allowed is False
    assert result.decision == "FALLBACK"


def test_database_failure_returns_fallback():
    def broken_lookup(*args):
        raise RuntimeError("DB unavailable")

    result = get_registered_usage(
        product_name="테스트 약제 A",
        crop="마늘",
        target_pest="잎마름병",
        lookup_fn=broken_lookup,
    )

    assert result.decision == "FALLBACK"


def test_verified_recommendation_can_pass():
    # 테스트 전용: 모든 검증 조건을 충족한 가상 레코드
    approved_record = RECORD.copy()
    approved_record["recommendation_eligible"] = True

    result = get_registered_usage(
        product_name="테스트 약제 A",
        crop="마늘",
        target_pest="잎마름병",
        lookup_fn=lambda *args: [approved_record],
        recommend=True,
        current_verifier=lambda record: True,
    )

    assert result.allowed is True
    assert result.decision == "ALLOW"


def test_missing_lookup_information_rejected():
    result = get_registered_usage(
        product_name="",
        crop="마늘",
        target_pest="잎마름병",
        lookup_fn=lookup,
    )

    assert result.allowed is False
    assert result.decision == "FALLBACK"
