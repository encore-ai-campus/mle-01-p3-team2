"""홈(랜딩) 화면을 그립니다.

각 기능(작물 도감·병해충 도감·농약 검색·지역별 정보)을 스마트팜 데이터셋풍
카드로 미리 보여 주고, '자세히 알아보기' 버튼으로 해당 페이지로 넘깁니다.
"""

from __future__ import annotations

import html

import streamlit as st

from src.streamlit.crop_data import CropProfile
from src.streamlit.crop_page import get_profiles as get_crop_profiles
from src.streamlit.data_source import get_graph
from src.streamlit.pest_data import build_entries, count_entries
from src.streamlit.pesticide_data import ANY as PST_ANY
from src.streamlit.pesticide_data import build_records, facet_values
from src.streamlit.region_data import (
    administrative,
    build_profiles as build_region_profiles,
)


# 홈 작물 카드에 보여 줄 작물 (설향/방울토마토/파프리카 자리)
HOME_CROPS = ("양파", "마늘", "고추")


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value).strip())


def _go(page: str) -> None:
    """해당 기능 페이지로 이동합니다."""

    st.session_state.page = page


def render_section_header(accent: str, rest: str, subtitle: str) -> None:
    """'스마트팜 데이터셋'처럼 강조어 + 제목 + 설명 머리말을 그립니다."""

    st.markdown(
        f"""
<div class="vb-sec-head">
    <div class="vb-sec-title"><span class="vb-sec-accent">{_esc(accent)}</span> {_esc(rest)}</div>
    <div class="vb-sec-sub">{_esc(subtitle)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _metric_tiles(items: list[tuple[str, object]]) -> str:
    """라벨·값 쌍을 숫자 타일 묶음 HTML로 만듭니다."""

    def _fmt(value: object) -> str:
        return f"{value:,}" if isinstance(value, int) else _esc(value)

    return "".join(
        f"<div class='vb-home-metric'>"
        f"<div class='l'>{_esc(label)}</div>"
        f"<div class='v'>{_fmt(value)}</div>"
        f"</div>"
        for label, value in items
    )


def _crop_card_html(profile: CropProfile) -> str:
    """작물 하나를 데이터셋 카드로 만듭니다 (6개 지표)."""

    tiles = _metric_tiles(list(profile.counts.items()))
    return (
        "<div class='vb-home-card'>"
        f"<div class='vb-home-head'>{profile.emoji} {_esc(profile.name)}</div>"
        f"<div class='vb-home-metrics'>{tiles}</div>"
        "</div>"
    )


def _summary_card_html(emoji: str, title: str, items: list[tuple[str, object]]) -> str:
    """단일 요약 카드(병해충·농약·지역)를 만듭니다."""

    tiles = _metric_tiles(items)
    return (
        "<div class='vb-home-card'>"
        f"<div class='vb-home-head'>{emoji} {_esc(title)}</div>"
        f"<div class='vb-home-metrics vb-home-metrics-wide'>{tiles}</div>"
        "</div>"
    )


def _go_button(label: str, page: str, key: str) -> None:
    """가운데 정렬된 '자세히 알아보기' 버튼을 그립니다."""

    with st.container(key=f"vbgo_{key}"):
        columns = st.columns([1, 2, 1])
        with columns[1]:
            st.button(
                label,
                key=f"home_go_{key}",
                on_click=_go,
                args=(page,),
                use_container_width=True,
            )


def render_crop_section() -> None:
    """작물 도감 미리보기."""

    render_section_header(
        "작물", "도감", "작물별 병해·해충·품종·등록농약·재배 관리·생육 관측을 한눈에 봐요."
    )

    profiles = get_crop_profiles()
    crops = [name for name in HOME_CROPS if name in profiles]
    if not crops:
        crops = list(profiles)[:3]

    if crops:
        columns = st.columns(len(crops))
        for column, name in zip(columns, crops):
            with column:
                st.markdown(_crop_card_html(profiles[name]), unsafe_allow_html=True)

    _go_button("🥬 작물 도감 자세히 알아보기  ›", "crop", "crop")


def render_pest_section() -> None:
    """병해충 도감 미리보기."""

    render_section_header(
        "병해충", "도감", "작물에 생기는 병해와 해충의 증상·예방법을 찾아봐요."
    )

    counts = count_entries(build_entries(get_graph()))
    card = _summary_card_html(
        "🐛",
        "병해충 데이터",
        [
            ("전체 항목", counts.total),
            ("병해", counts.diseases),
            ("해충", counts.insects),
            ("등록 약제 보유", counts.with_pesticide),
        ],
    )
    st.markdown(card, unsafe_allow_html=True)
    _go_button("🐛 병해충 도감 자세히 알아보기  ›", "pest", "pest")


def render_pesticide_section() -> None:
    """농약 검색 미리보기."""

    render_section_header(
        "농약", "검색", "작물·병해충별로 등록된 약제와 안전사용기준을 조회해요."
    )

    records = build_records(get_graph())
    crop_count = max(len(facet_values(records, "crop")) - 1, 0)  # ANY 제외
    purpose_count = max(len(facet_values(records, "purpose")) - 1, 0)
    card = _summary_card_html(
        "🧪",
        "등록 농약 데이터",
        [
            ("전체 등록", len(records)),
            ("제품 종류", len({record.product for record in records if record.product})),
            ("대상 작물", crop_count),
            ("용도 구분", purpose_count),
        ],
    )
    st.markdown(card, unsafe_allow_html=True)
    _go_button("🧪 농약 검색 자세히 알아보기  ›", "pesticide", "pesticide")


def render_region_section() -> None:
    """지역별 정보 미리보기."""

    render_section_header(
        "지역별", "정보", "시군구별 월별 기온·강수량과 품종 적합 지역을 살펴봐요."
    )

    profiles = build_region_profiles(get_graph())
    admin = administrative(profiles)
    provinces = {profile.province for profile in admin if profile.province}
    observations = sum(profile.observation_count for profile in profiles)
    card = _summary_card_html(
        "🗺️",
        "지역 데이터",
        [
            ("시도", len(provinces)),
            ("시군구", len(admin)),
            ("생육·파종 관측", observations),
        ],
    )
    st.markdown(card, unsafe_allow_html=True)
    _go_button("🗺️ 지역별 정보 자세히 알아보기  ›", "region", "region")


def render_home_page() -> None:
    """홈 화면 전체를 그립니다."""

    render_crop_section()
    st.markdown("<div class='vb-sec-gap'></div>", unsafe_allow_html=True)
    render_pest_section()
    st.markdown("<div class='vb-sec-gap'></div>", unsafe_allow_html=True)
    render_pesticide_section()
    st.markdown("<div class='vb-sec-gap'></div>", unsafe_allow_html=True)
    render_region_section()
