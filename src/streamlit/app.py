"""프로젝트 GraphRAG Supervisor를 사용하는 게임풍 농업 정보 앱입니다."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_supervisor import build_app
from src.streamlit.chatbot import generate_answer
from src.streamlit.crop_page import render_crop_page
from src.streamlit.pest_page import render_pest_page
from src.streamlit.pesticide_page import render_pesticide_page
from src.streamlit.region_page import render_region_page
from src.streamlit.theme import (
    MENU_ITEMS,
    build_css,
    mascot_data_uri,
    mascot_path,
    render_board_html,
    render_greeting_html,
    render_user_bubble_html,
)


SAMPLE_QUESTIONS = (
    "마늘 파종 시기는 언제인가요?",
    "배추 무름병은 어떻게 예방하나요?",
    "고추 탄저병에 쓸 수 있는 약제를 알려줘",
    "양파가 잘 자라는 온도는?",
)

# "15~25℃" 같은 범위 표기가 마크다운 취소선으로 해석되는 것을 막습니다.
_TILDE_PATTERN = re.compile(r"(?<!\\)~")


def escape_tildes(text: str) -> str:
    """답변 안의 물결표를 마크다운이 삼키지 않도록 escape 합니다."""

    return _TILDE_PATTERN.sub(r"\\~", text)


@st.cache_resource
def get_supervisor_resources():
    """Supervisor와 Neo4j 드라이버를 Streamlit 프로세스에서 재사용합니다."""

    return build_app()


def ask_supervisor(question: str) -> str:
    """Supervisor를 불러 답변 문자열을 돌려줍니다."""

    try:
        supervisor, _driver = get_supervisor_resources()
    except Exception as exc:
        return f"챗봇을 시작하지 못했습니다: {exc}"

    return generate_answer(supervisor, question)


def render_user_message(text: str) -> None:
    """사용자 질문을 오른쪽 초록 말풍선으로 그립니다."""

    st.markdown(render_user_bubble_html(text), unsafe_allow_html=True)


def render_assistant_message(text: str) -> None:
    """마스코트 아바타와 함께 답변 말풍선을 그립니다."""

    with st.chat_message("assistant", avatar=mascot_path()):
        st.markdown(escape_tildes(text))


def render_history() -> None:
    """지금까지의 대화를 말풍선으로 다시 그립니다."""

    for message in st.session_state.messages:
        if message["role"] == "user":
            render_user_message(message["content"])
        else:
            render_assistant_message(message["content"])


def handle_question(question: str) -> None:
    """질문을 기록하고 마스코트가 답하도록 합니다."""

    question = question.strip()
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    render_user_message(question)

    with st.chat_message("assistant", avatar=mascot_path()):
        with st.spinner("부기가 밭에서 자료를 찾아오는 중..."):
            answer = ask_supervisor(question)
        st.markdown(escape_tildes(answer))

    st.session_state.messages.append({"role": "assistant", "content": answer})


def render_nav() -> None:
    """상단 메뉴 버튼 줄을 그립니다."""

    with st.container(key="vbnav"):
        columns = st.columns(len(MENU_ITEMS))
        for column, (key, label, enabled) in zip(columns, MENU_ITEMS):
            with column:
                is_active = key == st.session_state.page
                if st.button(
                    label if enabled else f"{label} 🔒",
                    key=f"nav_{key}",
                    disabled=not enabled,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.page = key
                    st.rerun()


def render_sidebar() -> None:
    """예시 질문과 대화 초기화 버튼을 사이드바에 놓습니다."""

    with st.sidebar:
        if st.session_state.page == "crop":
            st.markdown("### 🌿 도감 보는 법")
            st.markdown(
                "위쪽 작물 버튼을 눌러 작물을 바꾸고, "
                "병해·해충 항목을 펼치면 증상과 예방법을 볼 수 있어요."
            )
            return

        if st.session_state.page == "region":
            st.markdown("### 🗺️ 지역 보는 법")
            st.markdown(
                "시도와 시군구를 고르면 2024~2025년 월별 기온·강수량이 나와요. "
                "아래쪽에는 품종 자료에 적힌 재배 권장 지역도 모아 뒀어요."
            )
            return

        if st.session_state.page == "pest":
            st.markdown("### 🐛 도감 보는 법")
            st.markdown(
                "작물과 종류로 추리거나 이름으로 찾아보세요. "
                "항목을 펼치면 증상·생태·예방법이 나오고, "
                "💊 표시가 있으면 등록된 약제로 바로 넘어갈 수 있어요."
            )
            return

        if st.session_state.page == "pesticide":
            st.markdown("### 🧪 농약 찾는 법")
            st.markdown(
                "작물과 병해충을 고르면 등록된 약제가 나와요. "
                "희석배수와 안전사용기준을 꼭 같이 확인하고, "
                "⚠️ 표시가 붙은 건은 사용 전에 공식 조회로 확인해 주세요."
            )
            return

        if st.session_state.page != "chat":
            return

        st.markdown("### 🌿 이런 걸 물어봐요")
        for index, sample in enumerate(SAMPLE_QUESTIONS):
            if st.button(sample, key=f"sample_{index}", use_container_width=True):
                st.session_state.pending_question = sample
                st.rerun()

        st.divider()
        if st.button("🧹 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()


def render_chat_page() -> None:
    """챗봇 화면을 그립니다."""

    if not st.session_state.messages:
        st.markdown(render_greeting_html(), unsafe_allow_html=True)

    render_history()

    question = st.chat_input("궁금한 걸 물어보세요!")
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question:
        handle_question(question)


def render_locked_page(label: str) -> None:
    """아직 만들지 않은 메뉴의 안내를 보여 줍니다."""

    uri = mascot_data_uri()
    image = f"<img src='{uri}' alt='부기'/>" if uri else ""
    st.markdown(
        f"<div class='vb-soon-card'>{image}"
        f"<h3>{label}은(는) 아직 준비 중이야!</h3>"
        "<p>조금만 기다려 줘. 지금은 '물어보기'와 '작물 도감'을 쓸 수 있어.</p></div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="농업 정보",
        page_icon=mascot_path(),
        layout="centered",
    )
    st.markdown(build_css(), unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "page" not in st.session_state:
        st.session_state.page = "chat"

    st.markdown(render_board_html(), unsafe_allow_html=True)
    render_nav()
    render_sidebar()

    if st.session_state.page == "chat":
        render_chat_page()
    elif st.session_state.page == "crop":
        render_crop_page()
    elif st.session_state.page == "pest":
        render_pest_page()
    elif st.session_state.page == "pesticide":
        render_pesticide_page()
    elif st.session_state.page == "region":
        render_region_page()
    else:
        labels = {key: label for key, label, _ in MENU_ITEMS}
        render_locked_page(labels.get(st.session_state.page, "이 메뉴"))


if __name__ == "__main__":
    main()
