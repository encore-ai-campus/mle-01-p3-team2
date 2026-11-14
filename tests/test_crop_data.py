from __future__ import annotations

import pytest

from src.streamlit.crop_data import (
    Entity,
    build_env_conditions,
    build_graph,
    build_profiles,
    format_env_value,
)


# 이 환경에서는 pytest fixture가 임시 디렉터리 권한 문제로 실패해서
# 평범한 헬퍼 함수로 표본 그래프를 만듭니다.
def sample_graph():
    entities = [
        {"entity_id": "CRP_1", "entity_type": "Crop", "name": "마늘", "attributes": {}},
        {
            "entity_id": "VAR_1",
            "entity_type": "Variety",
            "name": "남도마늘",
            "attributes": {"cultivation_type": "난지형", "characteristics": "남부 적응"},
        },
        {
            "entity_id": "SOW_1",
            "entity_type": "SowingRecommendation",
            "name": "남도마늘 파종 권고",
            "attributes": {"sowing_time": "9월 하순 ~ 10월 중순"},
        },
        {
            "entity_id": "REG_1",
            "entity_type": "Region",
            "name": "남부지방",
            "attributes": {},
        },
        {
            "entity_id": "ENV_1",
            "entity_type": "EnvironmentFact",
            "name": "마늘 생육 환경조건",
            "attributes": {
                "category": "temperature",
                "stage": "생육",
                "min_value": 18,
                "max_value": 20,
                "unit": "℃",
                "evidence_summary": "생육 적온 18~20℃",
            },
        },
        {
            "entity_id": "DIS_1",
            "entity_type": "Disease",
            "name": "잎마름병",
            "attributes": {"symptoms": "잎 끝이 마른다"},
        },
        {
            "entity_id": "INS_1",
            "entity_type": "Insect",
            "name": "고자리파리",
            "attributes": {"damage": "뿌리를 가해한다"},
        },
        {
            "entity_id": "PST_1",
            "entity_type": "PesticideUse",
            "name": "살균제",
            "attributes": {},
        },
    ]
    relations = [
        {"from_id": "CRP_1", "to_id": "VAR_1", "relation_type": "HAS_VARIETY"},
        {
            "from_id": "CRP_1",
            "to_id": "ENV_1",
            "relation_type": "HAS_ENVIRONMENT_REQUIREMENT",
        },
        {"from_id": "CRP_1", "to_id": "DIS_1", "relation_type": "AFFECTED_BY_DISEASE"},
        {"from_id": "CRP_1", "to_id": "INS_1", "relation_type": "AFFECTED_BY_INSECT"},
        {
            "from_id": "CRP_1",
            "to_id": "PST_1",
            "relation_type": "HAS_REGISTERED_PESTICIDE_USE",
        },
        {
            "from_id": "VAR_1",
            "to_id": "SOW_1",
            "relation_type": "HAS_SOWING_RECOMMENDATION",
        },
        {"from_id": "VAR_1", "to_id": "REG_1", "relation_type": "SUITABLE_FOR_REGION"},
    ]
    return build_graph(entities, relations)


def test_build_graph_reads_entities_and_relations() -> None:
    graph = sample_graph()

    assert len(graph.entities) == 8
    assert len(graph.relations) == 7
    assert graph.entities["CRP_1"].name == "마늘"


def test_build_profiles_counts_each_related_type() -> None:
    profile = build_profiles(sample_graph())["마늘"]

    assert profile.counts["병해"] == 1
    assert profile.counts["해충"] == 1
    assert profile.counts["품종"] == 1
    assert profile.counts["등록 농약"] == 1


def test_build_profiles_attaches_sowing_and_region_to_variety() -> None:
    variety = build_profiles(sample_graph())["마늘"].varieties[0]

    assert variety.name == "남도마늘"
    assert [s.attributes["sowing_time"] for s in variety.sowings] == [
        "9월 하순 ~ 10월 중순"
    ]
    assert [r.name for r in variety.regions] == ["남부지방"]


def test_build_profiles_uses_stage_as_environment_label() -> None:
    conditions = build_profiles(sample_graph())["마늘"].environment

    assert len(conditions) == 1
    assert conditions[0].label == "생육"
    assert conditions[0].value == "18 ~ 20 ℃"
    assert conditions[0].category == "temperature"


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        ({"min_value": 15, "max_value": 25, "unit": "℃"}, "15 ~ 25 ℃"),
        ({"min_value": 20, "max_value": 20, "unit": "℃"}, "20 ℃"),
        ({"min_value": 30, "max_value": "", "unit": "일"}, "30 일 이상"),
        ({"min_value": "", "max_value": 5, "unit": "pH"}, "5 pH 이하"),
        # 숫자가 아닌 한쪽 값에는 "이상/이하"를 붙이지 않습니다.
        ({"min_value": "well category", "max_value": "", "unit": ""}, "well category"),
        ({"min_value": "", "max_value": "", "unit": "text"}, ""),
    ],
)
def test_format_env_value(attributes: dict, expected: str) -> None:
    assert format_env_value(attributes) == expected


def test_build_env_conditions_falls_back_to_field_label() -> None:
    fact = Entity(
        entity_id="ENV_2",
        entity_type="EnvironmentFact",
        name="상추 optimal_soil_ph 환경조건",
        attributes={
            "field": "optimal_soil_ph",
            "min_value": 6,
            "max_value": 7,
            "unit": "pH",
        },
    )

    condition = build_env_conditions([fact])[0]

    assert condition.label == "적정 토양 pH"
    # category가 비어 있어도 field 이름으로 분류합니다.
    assert condition.category == "soil"
