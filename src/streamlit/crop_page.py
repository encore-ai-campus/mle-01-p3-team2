"""작물 도감 화면을 그립니다."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from src.streamlit.crop_data import (
    ENV_CATEGORY_LABEL,
    CropProfile,
    Entity,
    VarietyInfo,
    build_profiles,
)
from src.streamlit.data_source import get_graph


VARIETY_TAG_FIELDS = (
    ("cultivation_type", ""),
    ("harvest_period", "수확 "),
    ("storage_quality", "저장성 "),
    ("resistance", "저항성 "),
    ("bolting_type", ""),
    ("origin", ""),
)


@st.cache_resource(show_spinner="작물 도감을 펼치는 중...")
def get_profiles() -> dict[str, CropProfile]:
    """그래프에서 작물별 도감 내용을 한 번만 만듭니다."""

    return build_profiles(get_graph())


def _esc(value: Any) -> str:
    text = "" if value is None else str(value)
    return html.escape(text.strip())


def _card(title: str, body: str) -> None:
    st.markdown(
        f"<div class='vb-card'><h3>{title}</h3>{body}</div>",
        unsafe_allow_html=True,
    )


def render_hero(profile: CropProfile) -> None:
    """작물 이름과 보유 데이터 건수를 보여 줍니다."""

    stats = "".join(
        f"<div class='vb-stat'>"
        f"<div class='vb-stat-label'>{_esc(label)}</div>"
        f"<div class='vb-stat-value'>{value:,}</div>"
        f"</div>"
        for label, value in profile.counts.items()
    )
    st.markdown(
        f"""
<div class="vb-card">
    <div class="vb-hero">
        <div class="vb-hero-emoji">{profile.emoji}</div>
        <div class="vb-hero-name">{_esc(profile.name)}</div>
    </div>
    <div class="vb-stats">{stats}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_environment(profile: CropProfile) -> None:
    """생육 환경 조건을 범주별 표로 보여 줍니다."""

    if not profile.environment:
        _card("🌤️ 생육 환경", "<p>등록된 환경 조건이 없습니다.</p>")
        return

    rows = []
    for condition in profile.environment:
        category = ENV_CATEGORY_LABEL.get(condition.category, condition.category)
        rows.append(
            "<tr>"
            f"<td class='vb-cat'>{_esc(category)}</td>"
            f"<td>{_esc(condition.label)}</td>"
            f"<td class='vb-val'>{_esc(condition.value) or '-'}</td>"
            f"<td class='vb-note'>{_esc(condition.note)}</td>"
            "</tr>"
        )
    table = (
        "<table class='vb-table'>"
        "<tr><th>구분</th><th>항목</th><th>값</th><th>설명</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    _card(f"🌤️ 생육 환경 ({len(profile.environment)}건)", table)


def _clean_tag(value: str) -> str:
    """'by_cultivation_type > 노지_터널' 같은 분류 경로를 사람이 읽게 다듬습니다."""

    text = value.split(">")[-1].strip()
    return text.replace("_", " ").strip()


def _variety_tags(info: VarietyInfo) -> str:
    tags = []
    for key, prefix in VARIETY_TAG_FIELDS:
        value = info.attributes.get(key)
        if isinstance(value, str) and value.strip():
            tags.append(f"<span class='vb-tag'>{_esc(prefix + _clean_tag(value))}</span>")

    for sowing in info.sowings:
        sowing_time = sowing.attributes.get("sowing_time")
        if sowing_time:
            tags.append(
                f"<span class='vb-tag vb-tag-sow'>🌱 {_esc(sowing_time)}</span>"
            )

    for region in info.regions:
        tags.append(f"<span class='vb-tag vb-tag-region'>📍 {_esc(region.name)}</span>")

    return "".join(tags)


def render_varieties(profile: CropProfile) -> None:
    """품종 카드를 격자로 보여 줍니다."""

    if not profile.varieties:
        _card("🌾 품종", "<p>등록된 품종이 없습니다.</p>")
        return

    cards = []
    for info in profile.varieties:
        description = info.attributes.get("characteristics") or ""
        cards.append(
            "<div class='vb-variety'>"
            f"<div class='vb-variety-name'>{_esc(info.name)}</div>"
            f"{_variety_tags(info)}"
            f"<div class='vb-variety-desc'>{_esc(description)}</div>"
            "</div>"
        )
    _card(
        f"🌾 품종 ({len(profile.varieties)}종)",
        f"<div class='vb-grid'>{''.join(cards)}</div>",
    )


def render_variety_groups(profile: CropProfile) -> None:
    """품종군 설명을 보여 줍니다."""

    if not profile.variety_groups:
        return

    cards = []
    for group in profile.variety_groups:
        category = group.attributes.get("group_category") or ""
        cards.append(
            "<div class='vb-variety'>"
            f"<div class='vb-variety-name'>{_esc(group.name)}</div>"
            + (f"<span class='vb-tag'>{_esc(category)}</span>" if category else "")
            + f"<div class='vb-variety-desc'>"
            f"{_esc(group.attributes.get('characteristics'))}</div>"
            "</div>"
        )
    _card(
        f"🧺 품종군 ({len(profile.variety_groups)}종)",
        f"<div class='vb-grid'>{''.join(cards)}</div>",
    )


def _npk_cells(values: Any) -> str:
    if not isinstance(values, dict):
        return "<td>-</td><td>-</td><td>-</td>"
    return "".join(
        f"<td class='vb-val'>{_esc(values.get(key, '-'))}</td>"
        for key in ("nitrogen", "phosphate", "potash")
    )


def render_fertilizers(profile: CropProfile) -> None:
    """비료 표준사용량을 표로 보여 줍니다."""

    if not profile.fertilizers:
        return

    rows = []
    for standard in profile.fertilizers:
        attributes = standard.attributes
        condition = attributes.get("cultivation_condition") or "-"
        unit = attributes.get("unit") or "kg/10a"
        rows.append(
            f"<tr><td class='vb-cat'>{_esc(condition)}</td>"
            f"<td>밑거름</td>{_npk_cells(attributes.get('basal'))}"
            f"<td class='vb-note'>{_esc(unit)}</td></tr>"
        )
        if attributes.get("topdressing"):
            rows.append(
                f"<tr><td class='vb-cat'>{_esc(condition)}</td>"
                f"<td>웃거름</td>{_npk_cells(attributes.get('topdressing'))}"
                f"<td class='vb-note'>{_esc(unit)}</td></tr>"
            )

    table = (
        "<table class='vb-table'>"
        "<tr><th>재배 조건</th><th>시비</th><th>질소</th><th>인산</th>"
        "<th>칼리</th><th>단위</th></tr>" + "".join(rows) + "</table>"
    )
    _card("🧪 비료 표준사용량", table)


def _render_detail_list(entities: list[Entity], fields: tuple[tuple[str, str], ...]) -> None:
    for entity in entities:
        with st.expander(entity.name):
            wrote = False
            for key, label in fields:
                value = entity.attributes.get(key)
                if isinstance(value, str) and value.strip():
                    st.markdown(f"**{label}**")
                    st.markdown(value.strip().replace("\n", "  \n"))
                    wrote = True
            if not wrote:
                st.markdown("등록된 상세 설명이 없습니다.")


def render_diseases(profile: CropProfile) -> None:
    """병해 목록을 펼침 카드로 보여 줍니다."""

    if not profile.diseases:
        return
    st.markdown(
        f"<div class='vb-panel-title'>🦠 병해 {len(profile.diseases)}종</div>",
        unsafe_allow_html=True,
    )
    _render_detail_list(
        profile.diseases,
        (
            ("symptoms", "증상"),
            ("development_condition", "발생 조건"),
            ("prevention_method", "예방 방법"),
        ),
    )


def render_insects(profile: CropProfile) -> None:
    """해충 목록을 펼침 카드로 보여 줍니다."""

    if not profile.insects:
        return
    st.markdown(
        f"<div class='vb-panel-title'>🐛 해충 {len(profile.insects)}종</div>",
        unsafe_allow_html=True,
    )
    _render_detail_list(
        profile.insects,
        (
            ("damage", "피해 양상"),
            ("ecology", "생태"),
            ("prevention", "방제"),
        ),
    )


def render_crop_page() -> None:
    """작물 도감 전체를 그립니다."""

    profiles = get_profiles()
    if not profiles:
        st.markdown(
            "<div class='vb-soon-card'><h3>작물 데이터를 찾지 못했습니다</h3>"
            "<p>data/processed/entities.jsonl 이 있는지 확인해 주세요.</p></div>",
            unsafe_allow_html=True,
        )
        return

    names = list(profiles)
    if st.session_state.get("selected_crop") not in names:
        st.session_state.selected_crop = names[0]

    with st.container(key="vbcrop"):
        columns = st.columns(len(names))
        for column, name in zip(columns, names):
            with column:
                is_active = name == st.session_state.selected_crop
                if st.button(
                    f"{profiles[name].emoji} {name}",
                    key=f"crop_{name}",
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.selected_crop = name
                    st.rerun()

    profile = profiles[st.session_state.selected_crop]
    render_hero(profile)
    render_environment(profile)
    render_varieties(profile)
    render_variety_groups(profile)
    render_fertilizers(profile)
    render_diseases(profile)
    render_insects(profile)
