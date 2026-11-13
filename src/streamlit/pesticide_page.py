"""농약 검색 화면을 그립니다."""

from __future__ import annotations

import html
import math
from typing import Any

import streamlit as st

from src.streamlit.data_source import get_graph
from src.streamlit.pesticide_data import (
    ANY,
    PSIS_SEARCH_URL,
    PesticideRecord,
    build_records,
    facet_values,
    filter_records,
    targets_for_crop,
    toxicity_values,
)


PAGE_SIZE = 20

TOXICITY_CLASS = {
    "저독성(Ⅳ)": "vb-tox-low",
    "보통독성(Ⅲ)": "vb-tox-mid",
    "고독성(Ⅱ)": "vb-tox-high",
    "맹독성(Ⅰ)": "vb-tox-high",
}

PURPOSE_EMOJI = {
    "살충제": "🐛",
    "살균제": "🦠",
    "제초제": "🌿",
    "살균.살충제": "🧫",
    "생장조정제": "📈",
    "기타": "📦",
}


@st.cache_resource(show_spinner="등록 농약 정보를 정리하는 중...")
def get_records() -> list[PesticideRecord]:
    """등록 농약 레코드를 한 번만 만들어 캐시합니다."""

    return build_records(get_graph())


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value).strip())


def render_notice() -> None:
    """이 화면이 추천이 아니라 조회라는 점을 먼저 알립니다."""

    st.markdown(
        f"""
<div class="vb-notice">
    <b>⚠️ 이 목록은 추천이 아니라 등록 정보 조회 결과예요.</b><br>
    수록된 자료는 모두 검증 상태가 <code>PRELIMINARY_PASS</code> 또는
    <code>REVIEW_REQUIRED</code>라서 최신 등록 여부가 확인되지 않았어요.
    실제로 약을 치기 전에는
    <a href="{PSIS_SEARCH_URL}" target="_blank" rel="noopener">농약안전정보시스템</a>에서
    꼭 다시 확인해 주세요.
</div>
""",
        unsafe_allow_html=True,
    )


def render_filters(records: list[PesticideRecord]) -> dict[str, Any]:
    """검색 조건 입력을 그리고 고른 값을 돌려줍니다."""

    with st.container(key="vbfilter"):
        top = st.columns([1, 2])
        with top[0]:
            crop = st.selectbox("작물", facet_values(records, "crop"), key="pst_crop")
        with top[1]:
            keyword = st.text_input(
                "상표명·품목명·회사 검색",
                key="pst_keyword",
                placeholder="예: 아족시스트로빈, 아리",
            )

        target_options = targets_for_crop(records, crop)
        # 작물을 바꾸면 이전 병해충 선택이 목록에 없을 수 있습니다.
        if st.session_state.get("pst_target") not in target_options:
            st.session_state.pst_target = ANY

        row = st.columns(4)
        with row[0]:
            target = st.selectbox("병해충", target_options, key="pst_target")
        with row[1]:
            purpose = st.selectbox(
                "용도", facet_values(records, "purpose"), key="pst_purpose"
            )
        with row[2]:
            formulation = st.selectbox(
                "제형", facet_values(records, "formulation"), key="pst_formulation"
            )
        with row[3]:
            toxicity = st.selectbox(
                "인축 독성", toxicity_values(records), key="pst_toxicity"
            )

        hide_needs_review = st.checkbox(
            "취소·보류 이력이 있는 건 숨기기", key="pst_hide_review"
        )

    return {
        "crop": crop,
        "target": target,
        "purpose": purpose,
        "formulation": formulation,
        "toxicity": toxicity,
        "keyword": keyword,
        "hide_needs_review": hide_needs_review,
    }


def render_summary(found: list[PesticideRecord], total: int) -> None:
    """검색 결과 건수를 숫자 칩으로 보여 줍니다."""

    review_count = sum(1 for record in found if record.needs_review)
    stats = {
        "검색 결과": len(found),
        "전체 등록": total,
        "주의 필요": review_count,
        "제품 종류": len({record.product for record in found}),
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


def _record_card(record: PesticideRecord) -> str:
    badges = []
    if record.purpose:
        emoji = PURPOSE_EMOJI.get(record.purpose, "💊")
        badges.append(f"<span class='vb-tag'>{emoji} {_esc(record.purpose)}</span>")
    if record.formulation:
        badges.append(f"<span class='vb-tag'>{_esc(record.formulation)}</span>")
    if record.human_toxicity:
        css = TOXICITY_CLASS.get(record.human_toxicity, "vb-tox-mid")
        badges.append(
            f"<span class='vb-tag {css}'>인축 {_esc(record.human_toxicity)}</span>"
        )
    if record.eco_toxicity:
        badges.append(f"<span class='vb-tag'>어독성 {_esc(record.eco_toxicity)}</span>")

    rows = [
        ("대상 병해충", record.target),
        ("희석배수", f"{record.dilution}배" if record.dilution else ""),
        ("사용량", record.quantity),
        ("사용 적기", record.application_timing),
        ("안전사용기준", " / ".join(x for x in (record.safety_timing, record.safety_count) if x)),
        ("등록번호", record.registration_no),
        ("유효기간", record.valid_until),
        ("회사", record.company),
    ]
    detail = "".join(
        f"<tr><td class='vb-cat'>{_esc(label)}</td><td>{_esc(value)}</td></tr>"
        for label, value in rows
        if value
    )

    warning = ""
    if record.needs_review:
        link = (
            f" <a href='{_esc(record.official_url)}' target='_blank' rel='noopener'>"
            "공식 조회</a>"
            if record.official_url
            else ""
        )
        cancelled = (
            f"취소·변경 이력 {_esc(record.cancellation_date)}"
            if record.cancellation_date
            else "추천 보류로 표시된 항목"
        )
        warning = f"<div class='vb-warn'>⚠️ {cancelled} — 사용 전 확인 필요{link}</div>"

    return (
        "<div class='vb-pest'>"
        f"<div class='vb-pest-head'>"
        f"<span class='vb-pest-brand'>{_esc(record.brand)}</span>"
        f"<span class='vb-pest-crop'>{_esc(record.crop)}</span>"
        "</div>"
        f"<div class='vb-pest-product'>{_esc(record.product)}</div>"
        f"<div>{''.join(badges)}</div>"
        f"{warning}"
        f"<table class='vb-table vb-pest-table'>{detail}</table>"
        "</div>"
    )


def render_results(found: list[PesticideRecord]) -> None:
    """검색 결과를 쪽 단위로 그립니다."""

    if not found:
        st.markdown(
            "<div class='vb-soon-card'><h3>조건에 맞는 농약이 없어요</h3>"
            "<p>필터를 조금 풀어 보거나 검색어를 지워 보세요.</p></div>",
            unsafe_allow_html=True,
        )
        return

    pages = math.ceil(len(found) / PAGE_SIZE)
    page = 1
    if pages > 1:
        page = st.number_input(
            f"쪽 (총 {pages}쪽, {len(found):,}건)",
            min_value=1,
            max_value=pages,
            value=1,
            step=1,
            key="pst_page",
        )

    start = (int(page) - 1) * PAGE_SIZE
    cards = "".join(_record_card(record) for record in found[start : start + PAGE_SIZE])
    st.markdown(f"<div class='vb-pest-grid'>{cards}</div>", unsafe_allow_html=True)


def render_pesticide_page() -> None:
    """농약 검색 전체를 그립니다."""

    records = get_records()
    if not records:
        st.markdown(
            "<div class='vb-soon-card'><h3>농약 데이터를 찾지 못했습니다</h3>"
            "<p>data/processed/entities.jsonl 이 있는지 확인해 주세요.</p></div>",
            unsafe_allow_html=True,
        )
        return

    render_notice()
    selections = render_filters(records)
    found = filter_records(records, **selections)
    render_summary(found, len(records))
    render_results(found)
