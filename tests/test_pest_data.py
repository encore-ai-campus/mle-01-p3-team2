from __future__ import annotations

from src.streamlit.crop_data import build_graph
from src.streamlit.pest_data import (
    ANY,
    DISEASE,
    INSECT,
    build_entries,
    count_entries,
    crop_values,
    filter_entries,
)


# 이 환경에서는 pytest fixture가 임시 디렉터리 권한 문제로 실패해서
# 평범한 헬퍼 함수로 표본 데이터를 만듭니다.
def sample_entries():
    entities = [
        {"entity_id": "CRP_1", "entity_type": "Crop", "name": "배추", "attributes": {}},
        {"entity_id": "CRP_2", "entity_type": "Crop", "name": "상추", "attributes": {}},
        {
            "entity_id": "DIS_1",
            "entity_type": "Disease",
            "name": "노균병",
            "attributes": {
                "crop": "배추",
                "symptoms": "잎에 황색 반점이 생긴다",
                "development_condition": "서늘하고 습할 때",
                "prevention_method": "통풍을 좋게 한다",
            },
        },
        {
            "entity_id": "DIS_2",
            "entity_type": "Disease",
            "name": "칼라병",
            "attributes": {
                "crop": "상추",
                "aliases": ["TSWV", "토마토반점위조바이러스"],
                "symptoms": "잎이 변색된다",
                # 발생 조건과 예방 방법이 비어 있는 경우입니다.
                "development_condition": "",
                "prevention_method": None,
            },
        },
        {
            "entity_id": "INS_1",
            "entity_type": "Insect",
            "name": "배추좀나방",
            "attributes": {
                "damage": "잎을 갉아먹는다",
                "ecology": "연 4~5회 발생",
                "prevention": "방충망을 친다",
            },
        },
        {
            "entity_id": "PST_1",
            "entity_type": "PesticideUse",
            "name": "약제 1",
            "attributes": {},
        },
        {
            "entity_id": "PST_2",
            "entity_type": "PesticideUse",
            "name": "약제 2",
            "attributes": {},
        },
    ]
    relations = [
        {"from_id": "CRP_1", "to_id": "DIS_1", "relation_type": "AFFECTED_BY_DISEASE"},
        {"from_id": "CRP_2", "to_id": "DIS_2", "relation_type": "AFFECTED_BY_DISEASE"},
        {"from_id": "CRP_1", "to_id": "INS_1", "relation_type": "AFFECTED_BY_INSECT"},
        {"from_id": "PST_1", "to_id": "DIS_1", "relation_type": "CONTROLS_TARGET"},
        {"from_id": "PST_2", "to_id": "DIS_1", "relation_type": "CONTROLS_TARGET"},
    ]
    return build_entries(build_graph(entities, relations))


def by_name(entries, name):
    return next(entry for entry in entries if entry.name == name)


def test_build_entries_covers_diseases_and_insects() -> None:
    entries = sample_entries()

    assert len(entries) == 3
    assert by_name(entries, "노균병").kind == DISEASE
    assert by_name(entries, "배추좀나방").kind == INSECT


def test_build_entries_takes_crop_from_relation() -> None:
    # 해충에는 crop 속성이 없어서 Crop -> Insect 관계로만 알 수 있습니다.
    assert by_name(sample_entries(), "배추좀나방").crops == ("배추",)


def test_build_entries_counts_registered_pesticides() -> None:
    entries = sample_entries()

    assert by_name(entries, "노균병").pesticide_count == 2
    assert by_name(entries, "배추좀나방").pesticide_count == 0


def test_build_entries_skips_empty_sections() -> None:
    entry = by_name(sample_entries(), "칼라병")

    assert [label for label, _ in entry.sections] == ["증상"]


def test_build_entries_uses_insect_specific_section_labels() -> None:
    entry = by_name(sample_entries(), "배추좀나방")

    assert [label for label, _ in entry.sections] == ["피해 양상", "생태", "방제"]


def test_filter_entries_by_crop_and_kind() -> None:
    entries = sample_entries()

    found = filter_entries(entries, crop="배추", kind=INSECT)

    assert [e.name for e in found] == ["배추좀나방"]


def test_filter_entries_keyword_matches_aliases() -> None:
    entries = sample_entries()

    assert [e.name for e in filter_entries(entries, keyword="TSWV")] == ["칼라병"]
    assert [e.name for e in filter_entries(entries, keyword="노균")] == ["노균병"]


def test_filter_entries_can_require_registered_pesticides() -> None:
    found = filter_entries(sample_entries(), only_with_pesticide=True)

    assert [e.name for e in found] == ["노균병"]


def test_crop_values_lists_every_crop_once() -> None:
    assert crop_values(sample_entries()) == [ANY, "배추", "상추"]


def test_count_entries_splits_by_kind() -> None:
    counts = count_entries(sample_entries())

    assert counts.total == 3
    assert counts.diseases == 2
    assert counts.insects == 1
    assert counts.with_pesticide == 1
