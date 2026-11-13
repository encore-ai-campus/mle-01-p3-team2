"""지역별 정보 화면을 그립니다."""

from __future__ import annotations

import html
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from src.streamlit.data_source import get_graph
from src.streamlit.region_data import (
    RegionProfile,
    WeatherMonth,
    build_profiles,
    province_values,
    sigungu_profiles,
    summarize_weather,
    variety_regions,
)


TEMP_MAX_COLOR = "#e07a35"
TEMP_MIN_COLOR = "#3f8fd1"
RAIN_COLOR = "#7fc1e8"
GRID_COLOR = "#e6d7af"
AXIS_COLOR = "#6d6033"


@st.cache_resource(show_spinner="지역 정보를 불러오는 중...")
def get_profiles() -> list[RegionProfile]:
    """지역 화면 내용을 한 번만 만들어 캐시합니다."""

    return build_profiles(get_graph())


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value).strip())


def _fmt(value: float | None, unit: str = "") -> str:
    return "-" if value is None else f"{value:g}{unit}"


def weather_frame(months: list[WeatherMonth]) -> pd.DataFrame:
    """월별 관측을 차트용 표로 바꿉니다."""

    return pd.DataFrame(
        {
            "연월": [month.year_month for month in months],
            "최고기온": [month.max_temp for month in months],
            "최저기온": [month.min_temp for month in months],
            "강수량": [month.precipitation for month in months],
            "습도": [month.humidity for month in months],
        }
    )


def render_region_picker(profiles: list[RegionProfile]) -> RegionProfile | None:
    """시도와 시군구를 고르게 합니다."""

    provinces = province_values(profiles)
    if not provinces:
        return None

    with st.container(key="vbfilter"):
        columns = st.columns(2)
        with columns[0]:
            province = st.selectbox("시도", provinces, key="rgn_province")

        candidates = sigungu_profiles(profiles, province)
        names = [profile.sigungu for profile in candidates]
        # 시도를 바꾸면 이전 시군구 선택이 목록에 없을 수 있습니다.
        if st.session_state.get("rgn_sigungu") not in names:
            st.session_state.rgn_sigungu = names[0] if names else None

        with columns[1]:
            sigungu = st.selectbox("시군구", names, key="rgn_sigungu")

    return next(
        (profile for profile in candidates if profile.sigungu == sigungu), None
    )


def render_summary(profile: RegionProfile) -> None:
    """지역 이름과 기상 요약을 보여 줍니다."""

    summary = summarize_weather(profile.weather)
    stats = {
        "관측 개월": f"{summary.months}",
        "평균 최고기온": _fmt(summary.avg_max_temp, "℃"),
        "평균 최저기온": _fmt(summary.avg_min_temp, "℃"),
        "평균 습도": _fmt(summary.avg_humidity, "%"),
        "누적 강수량": _fmt(summary.total_precipitation, "mm"),
    }
    chips = "".join(
        f"<div class='vb-stat'><div class='vb-stat-label'>{_esc(label)}</div>"
        f"<div class='vb-stat-value' style='font-size:19px'>{_esc(value)}</div></div>"
        for label, value in stats.items()
    )

    extremes = []
    if summary.hottest:
        extremes.append(
            f"🔥 가장 더웠던 달 {_esc(summary.hottest.year_month)} "
            f"({_fmt(summary.hottest.max_temp, '℃')})"
        )
    if summary.coldest:
        extremes.append(
            f"❄️ 가장 추웠던 달 {_esc(summary.coldest.year_month)} "
            f"({_fmt(summary.coldest.min_temp, '℃')})"
        )
    if summary.wettest:
        extremes.append(
            f"🌧️ 비가 가장 많았던 달 {_esc(summary.wettest.year_month)} "
            f"({_fmt(summary.wettest.precipitation, 'mm')})"
        )
    extreme_line = (
        f"<div class='vb-region-extremes'>{' · '.join(extremes)}</div>"
        if extremes
        else ""
    )

    observation = (
        f"<span class='vb-tag'>생육·파종 관측 {profile.observation_count:,}건</span>"
        if profile.observation_count
        else ""
    )

    st.markdown(
        f"""
<div class="vb-card">
    <div class="vb-hero">
        <div class="vb-hero-emoji">🗺️</div>
        <div>
            <div class="vb-hero-name">{_esc(profile.sigungu)}</div>
            <div style="font-size:13.5px;color:#6d6033">{_esc(profile.province)}</div>
        </div>
        <div style="margin-left:auto">{observation}</div>
    </div>
    <div class="vb-stats">{chips}</div>
    {extreme_line}
</div>
""",
        unsafe_allow_html=True,
    )


def _base_axis(title: str) -> alt.Axis:
    return alt.Axis(
        title=title,
        labelColor=AXIS_COLOR,
        titleColor=AXIS_COLOR,
        gridColor=GRID_COLOR,
        domainColor=GRID_COLOR,
        tickColor=GRID_COLOR,
    )


def render_temperature_chart(frame: pd.DataFrame) -> None:
    """월별 최고/최저 기온 꺾은선 그래프를 그립니다."""

    melted = frame.melt(
        id_vars="연월",
        value_vars=["최고기온", "최저기온"],
        var_name="구분",
        value_name="기온",
    ).dropna(subset=["기온"])
    if melted.empty:
        return

    chart = (
        alt.Chart(melted)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("연월:O", axis=_base_axis("")),
            y=alt.Y("기온:Q", axis=_base_axis("기온 (℃)")),
            color=alt.Color(
                "구분:N",
                scale=alt.Scale(
                    domain=["최고기온", "최저기온"],
                    range=[TEMP_MAX_COLOR, TEMP_MIN_COLOR],
                ),
                legend=alt.Legend(title="", labelColor=AXIS_COLOR),
            ),
            tooltip=["연월", "구분", "기온"],
        )
        .properties(height=240, background="transparent")
    )
    st.altair_chart(chart, use_container_width=True)


def render_rain_chart(frame: pd.DataFrame) -> None:
    """월별 강수량 막대 그래프를 그립니다."""

    data = frame.dropna(subset=["강수량"])
    if data.empty:
        return

    chart = (
        alt.Chart(data)
        .mark_bar(color=RAIN_COLOR, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("연월:O", axis=_base_axis("")),
            y=alt.Y("강수량:Q", axis=_base_axis("강수량 (mm)")),
            tooltip=["연월", "강수량", "습도"],
        )
        .properties(height=200, background="transparent")
    )
    st.altair_chart(chart, use_container_width=True)


def render_weather(profile: RegionProfile) -> None:
    """월별 기상 차트와 표를 보여 줍니다."""

    if not profile.weather:
        st.markdown(
            "<div class='vb-soon-card'><h3>이 지역은 기상 관측이 없어요</h3>"
            "<p>다른 시군구를 골라 보세요.</p></div>",
            unsafe_allow_html=True,
        )
        return

    frame = weather_frame(profile.weather)

    st.markdown(
        "<div class='vb-panel-title'>🌡️ 월별 기온</div>", unsafe_allow_html=True
    )
    render_temperature_chart(frame)

    st.markdown(
        "<div class='vb-panel-title'>🌧️ 월별 강수량</div>", unsafe_allow_html=True
    )
    render_rain_chart(frame)

    incomplete = [month for month in profile.weather if not month.complete]
    rows = "".join(
        "<tr>"
        f"<td class='vb-cat'>{_esc(month.year_month)}</td>"
        f"<td class='vb-val'>{_fmt(month.max_temp, '℃')}</td>"
        f"<td class='vb-val'>{_fmt(month.min_temp, '℃')}</td>"
        f"<td class='vb-val'>{_fmt(month.humidity, '%')}</td>"
        f"<td class='vb-val'>{_fmt(month.precipitation, 'mm')}</td>"
        f"<td class='vb-note'>{'' if month.complete else '일부 결측'}</td>"
        "</tr>"
        for month in profile.weather
    )
    note = (
        f"<p style='font-size:13px;color:#6d6033;margin-top:8px'>"
        f"관측 일수가 모자란 달이 {len(incomplete)}개 있어요.</p>"
        if incomplete
        else ""
    )
    st.markdown(
        "<div class='vb-card'><h3>📋 월별 기록</h3>"
        "<table class='vb-table'>"
        "<tr><th>연월</th><th>최고기온</th><th>최저기온</th>"
        "<th>평균습도</th><th>강수량</th><th>비고</th></tr>"
        f"{rows}</table>{note}</div>",
        unsafe_allow_html=True,
    )


def render_variety_regions(profiles: list[RegionProfile]) -> None:
    """품종 적합 지역을 따로 모아 보여 줍니다."""

    regions = variety_regions(profiles)
    if not regions:
        return

    cards = []
    for region in regions:
        tags = "".join(
            f"<span class='vb-tag'>{_esc(variety.crop)} · {_esc(variety.name)}</span>"
            for variety in region.varieties
        )
        cards.append(
            "<div class='vb-variety'>"
            f"<div class='vb-variety-name'>📍 {_esc(region.name)}</div>"
            f"<div>{tags}</div>"
            "</div>"
        )

    st.markdown(
        "<div class='vb-card'><h3>🌾 품종 적합 지역</h3>"
        "<p style='font-size:13.5px;color:#6d6033;margin:-6px 0 10px 0'>"
        "품종 자료에 적힌 재배 권장 지역이에요. 위 시군구 목록과는 따로 적혀 있어요.</p>"
        f"<div class='vb-grid'>{''.join(cards)}</div></div>",
        unsafe_allow_html=True,
    )


def render_region_page() -> None:
    """지역별 정보 전체를 그립니다."""

    profiles = get_profiles()
    if not profiles:
        st.markdown(
            "<div class='vb-soon-card'><h3>지역 데이터를 찾지 못했습니다</h3>"
            "<p>data/processed/entities.jsonl 이 있는지 확인해 주세요.</p></div>",
            unsafe_allow_html=True,
        )
        return

    profile = render_region_picker(profiles)
    if profile is not None:
        render_summary(profile)
        render_weather(profile)

    render_variety_regions(profiles)
