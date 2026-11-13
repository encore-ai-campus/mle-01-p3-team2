from __future__ import annotations

import pytest

from src.streamlit.crop_data import build_graph
from src.streamlit.region_data import (
    build_profiles,
    build_weather_month,
    province_values,
    sigungu_profiles,
    summarize_weather,
    variety_regions,
)


def weather_attributes(year_month: str, high: float, low: float, rain: float, **extra):
    attributes = {
        "year_month": year_month,
        "monthly_max_temperature_c": high,
        "monthly_min_temperature_c": low,
        "monthly_precipitation_mm": rain,
        "monthly_mean_humidity_pct": 70.0,
        "completeness": {"강수량일수완전여부": True, "최고기온일수완전여부": True},
    }
    attributes.update(extra)
    return attributes


# 이 환경에서는 pytest fixture가 임시 디렉터리 권한 문제로 실패해서
# 평범한 헬퍼 함수로 표본 그래프를 만듭니다.
def sample_profiles():
    entities = [
        {
            "entity_id": "REG_1",
            "entity_type": "Region",
            "name": "전라남도 무안군",
            "attributes": {
                "province": "전라남도",
                "region_type": "시군구",
                "sigungu": "무안군",
            },
        },
        {
            "entity_id": "REG_2",
            "entity_type": "Region",
            "name": "전라남도 나주시",
            "attributes": {
                "province": "전라남도",
                "region_type": "시군구",
                "sigungu": "나주시",
            },
        },
        # 같은 이름이지만 속성이 없는 서술형 노드입니다.
        {
            "entity_id": "REG_3",
            "entity_type": "Region",
            "name": "전라남도 무안군",
            "attributes": {},
        },
        {"entity_id": "REG_4", "entity_type": "Region", "name": "남부지방", "attributes": {}},
        {
            "entity_id": "WTH_1",
            "entity_type": "WeatherObservation",
            "name": "무안군 2024-08",
            "attributes": weather_attributes("2024-08", 37.3, 22.0, 344.8),
        },
        {
            "entity_id": "WTH_2",
            "entity_type": "WeatherObservation",
            "name": "무안군 2025-01",
            "attributes": weather_attributes(
                "2025-01",
                8.0,
                -9.9,
                10.0,
                completeness={"강수량일수완전여부": False},
            ),
        },
        {
            "entity_id": "VAR_1",
            "entity_type": "Variety",
            "name": "남도마늘",
            "attributes": {"crop": "마늘", "characteristics": "남부 적응"},
        },
        {
            "entity_id": "GRW_1",
            "entity_type": "GrowthObservation",
            "name": "양파 생육 관측",
            "attributes": {},
        },
    ]
    relations = [
        {
            "from_id": "REG_1",
            "to_id": "WTH_1",
            "relation_type": "REGION_HAS_WEATHER_OBSERVATION",
        },
        {
            "from_id": "REG_1",
            "to_id": "WTH_2",
            "relation_type": "REGION_HAS_WEATHER_OBSERVATION",
        },
        {"from_id": "GRW_1", "to_id": "REG_3", "relation_type": "OBSERVED_IN_REGION"},
        {"from_id": "VAR_1", "to_id": "REG_4", "relation_type": "SUITABLE_FOR_REGION"},
    ]
    return build_profiles(build_graph(entities, relations))


def by_sigungu(profiles, sigungu):
    return next(p for p in profiles if p.sigungu == sigungu)


def test_province_values_only_lists_administrative_regions() -> None:
    # '남부지방'처럼 속성이 없는 노드는 시도 목록에 들어가면 안 됩니다.
    assert province_values(sample_profiles()) == ["전라남도"]


def test_sigungu_profiles_are_scoped_to_the_province() -> None:
    found = sigungu_profiles(sample_profiles(), "전라남도")

    assert sorted(p.sigungu for p in found) == ["나주시", "무안군"]


def test_weather_is_attached_and_sorted_by_month() -> None:
    profile = by_sigungu(sample_profiles(), "무안군")

    assert [m.year_month for m in profile.weather] == ["2024-08", "2025-01"]


def test_observation_count_merges_same_named_descriptive_node() -> None:
    # 생육 관측은 속성 없는 쪽 노드에 달려 있지만 시군구 쪽으로 합칩니다.
    assert by_sigungu(sample_profiles(), "무안군").observation_count == 1
    assert by_sigungu(sample_profiles(), "나주시").observation_count == 0


def test_build_weather_month_marks_incomplete_records() -> None:
    complete = build_weather_month(weather_attributes("2024-08", 30.0, 20.0, 100.0))
    partial = build_weather_month(
        weather_attributes("2024-09", 30.0, 20.0, 100.0, completeness={"a": False})
    )

    assert complete.complete is True
    assert partial.complete is False


def test_build_weather_month_parses_missing_values_as_none() -> None:
    month = build_weather_month({"year_month": "2024-08"})

    assert month.max_temp is None
    assert month.precipitation is None
    assert month.month == 8
    assert month.year == "2024"


def test_summarize_weather_averages_and_picks_extremes() -> None:
    profile = by_sigungu(sample_profiles(), "무안군")

    summary = summarize_weather(profile.weather)

    assert summary.months == 2
    assert summary.avg_max_temp == pytest.approx(22.6)
    assert summary.avg_min_temp == pytest.approx(6.0)
    assert summary.total_precipitation == pytest.approx(354.8)
    assert summary.hottest.year_month == "2024-08"
    assert summary.coldest.year_month == "2025-01"
    assert summary.wettest.year_month == "2024-08"


def test_summarize_weather_handles_empty_input() -> None:
    summary = summarize_weather([])

    assert summary.months == 0
    assert summary.avg_max_temp is None
    assert summary.hottest is None


def test_variety_regions_lists_only_nodes_with_varieties() -> None:
    found = variety_regions(sample_profiles())

    assert [r.name for r in found] == ["남부지방"]
    assert [(v.crop, v.name) for v in found[0].varieties] == [("마늘", "남도마늘")]
