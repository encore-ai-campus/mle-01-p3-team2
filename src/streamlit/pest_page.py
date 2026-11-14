"""병해충 도감 화면을 그립니다."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from src.streamlit.data_source import get_graph
from src.streamlit.pest_data import (
    ANY,
    KINDS,
    PestEntry,
    build_entries,
    count_entries,
    crop_values,
    filter_entries,
)


PAGE_SIZE = 20


@st.cache_resource(show_spinner="병해충 도감을 펼치는 중...")
def get_entries() -> list[PestEntry]:
    """병해충 항목을 한 번만 만들어 캐시합니다."""

    return build_entries(get_graph())


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value).strip())


def jump_to_pesticides(entry: PestEntry) -> None:
    """이 병해충 조건을 농약 검색 화면에 미리 넣고 넘어갑니다."""

    st.session_state.pst_crop = entry.crops[0] if entry.crops else ANY
    st.session_state.pst_target = entry.name
    st.session_state.pst_keyword = ""
    st.session_state.pst_purpose = ANY
    st.session_state.pst_formulation = ANY
    st.session_state.pst_toxicity = ANY
    st.session_state.pst_hide_review = False
    st.session_state.page = "pesticide"


def render_filters(entries: list[PestEntry]) -> dict[str, Any]:
    """검색 조건 입력을 그리고 고른 값을 돌려줍니다."""

    with st.container(key="vbfilter"):
        row = st.columns([1, 1, 2])
        with row[0]:
            crop = st.selectbox("작물", crop_values(entries), key="pest_crop")
        with row[1]:
            kind = st.selectbox("종류", list(KINDS), key="pest_kind")
        with row[2]:
            keyword = st.text_input(
                "이름·별칭 검색",
                key="pest_keyword",
                placeholder="예: 노균병, 진딧물, TSWV",
            )

        only_with_pesticide = st.checkbox(
            "등록된 약제가 있는 것만 보기", key="pest_only_pesticide"
        )

    return {
        "crop": crop,
        "kind": kind,
        "keyword": keyword,
        "only_with_pesticide": only_with_pesticide,
    }


def render_summary(found: list[PestEntry]) -> None:
    """검색 결과 건수를 숫자 칩으로 보여 줍니다."""

    counts = count_entries(found)
    stats = {
        "검색 결과": counts.total,
        "병해": counts.diseases,
        "해충": counts.insects,
        "약제 있음": counts.with_pesticide,
    }
    chips = "".join(
        f"<div class='vb-stat'><div class='vb-stat-label'>{_esc(label)}</div>"
        f"<div class='vb-stat-value'>{value:,}</div></div>"
        for label, value in stats.items()
    )
    st.markdown(
        f"<div class='vb-card'><div class='vb-stats' style='margin-top:0'>{chips}</div></div>",
        unsafe_allow_html=True,
    )


def render_entry(entry: PestEntry, index: int) -> None:
    """항목 하나를 펼침 카드로 그립니다."""

    crop_part = f" · {entry.crop_label}" if entry.crops else ""
    drug_part = f"  💊 {entry.pesticide_count}건" if entry.pesticide_count else ""
    with st.expander(f"{entry.emoji} {entry.name}{crop_part}{drug_part}"):
        if entry.aliases:
            tags = "".join(
                f"<span class='vb-tag'>{_esc(alias)}</span>" for alias in entry.aliases
            )
            st.markdown(f"<div>{tags}</div>", unsafe_allow_html=True)

        for label, body in entry.sections:
            st.markdown(f"**{label}**")
            st.markdown(body.replace("\n", "  \n"))

        if entry.pesticide_count:
            st.button(
                f"🧪 등록된 약제 {entry.pesticide_count}건 보기",
                key=f"pest_jump_{index}",
                on_click=jump_to_pesticides,
                args=(entry,),
            )


def render_results(found: list[PestEntry]) -> None:
    """검색 결과를 쪽 단위로 그립니다."""

    if not found:
        st.markdown(
            "<div class='vb-soon-card'><h3>조건에 맞는 병해충이 없어요</h3>"
            "<p>필터를 조금 풀어 보거나 검색어를 지워 보세요.</p></div>",
            unsafe_allow_html=True,
        )
        return

    pages = (len(found) + PAGE_SIZE - 1) // PAGE_SIZE
    page = 1
    if pages > 1:
        page = st.number_input(
            f"쪽 (총 {pages}쪽, {len(found):,}건)",
            min_value=1,
            max_value=pages,
            value=1,
            step=1,
            key="pest_page",
        )

    start = (int(page) - 1) * PAGE_SIZE
    for offset, entry in enumerate(found[start : start + PAGE_SIZE]):
        render_entry(entry, start + offset)


def render_pest_page() -> None:
    """병해충 도감 전체를 그립니다."""

    entries = get_entries()
    if not entries:
        st.markdown(
            "<div class='vb-soon-card'><h3>병해충 데이터를 찾지 못했습니다</h3>"
            "<p>data/processed/entities.jsonl 이 있는지 확인해 주세요.</p></div>",
            unsafe_allow_html=True,
        )
        return

    selections = render_filters(entries)
    found = filter_entries(entries, **selections)
    render_summary(found)
    render_results(found)
