"""프로젝트 GraphRAG Supervisor를 사용하는 게임풍 농업 정보 앱입니다."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import streamlit as st
import logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_supervisor import build_app
from src.streamlit.chatbot import generate_answer
from src.streamlit.crop_page import render_crop_page
from src.streamlit.home_page import render_home_page
from src.streamlit.pest_page import render_pest_page
from src.streamlit.pesticide_page import render_pesticide_page
from src.streamlit.region_page import render_region_page
from src.streamlit.theme import (
    MENU_ITEMS,
    build_css,
    mascot_data_uri,
    mascot_path,
    render_board_html,
    render_float_ask_html,
    render_greeting_html,
    render_user_bubble_html,
)
from src.streamlit.chatbot import generate_answer
from src.retrievers.answer.renderer import SOURCES_HEADING

from src.retrievers.answer.renderer import SOURCES_HEADING, WITHHOLD_DETAIL_HEADING


SAMPLE_QUESTIONS = (
    "마늘 파종 시기는 언제인가요?",
    "배추 무름병은 어떻게 예방하나요?",
    "고추 탄저병에 쓸 수 있는 약제를 알려줘",
    "양파가 잘 자라는 온도는?",
)

# "15~25℃" 같은 범위 표기가 마크다운 취소선으로 해석되는 것을 막습니다.
_TILDE_PATTERN = re.compile(r"(?<!\\)~")
_MD_LINK_PATTERN = re.compile(r"\[[^\]]*\]\([^)]*\)")

def escape_tildes(text: str) -> str:
    """답변 안의 물결표를 escape 하되, 마크다운 링크 URL은 건드리지 않습니다."""

    # 마크다운 링크 구간을 보호하고 나머지에서만 ~ 를 escape 한다.
    links: list[str] = []

    def _stash(match: "re.Match[str]") -> str:
        links.append(match.group(0))
        return f"\x00{len(links) - 1}\x00"

    stashed = _MD_LINK_PATTERN.sub(_stash, text)
    escaped = _TILDE_PATTERN.sub(r"\\~", stashed)
    return re.sub(r"\x00(\d+)\x00", lambda m: links[int(m.group(1))], escaped)


@st.cache_resource(show_spinner=False)
def get_supervisor_resources():
    """Supervisor와 Neo4j 드라이버를 Streamlit 프로세스에서 재사용합니다."""

    return build_app()


def ask_supervisor(question: str) -> str:
    """Supervisor를 불러 답변 문자열을 돌려줍니다."""

    try:
        supervisor, _driver = get_supervisor_resources()
    except Exception:
        logger.exception("Supervisor 리소스 초기화 실패")
        return "챗봇을 시작하지 못했습니다. 잠시 후 다시 시도해 주세요."

    return generate_answer(supervisor, question)


def render_user_message(text: str) -> None:
    """사용자 질문을 오른쪽 초록 말풍선으로 그립니다."""

    st.markdown(render_user_bubble_html(text), unsafe_allow_html=True)

def render_answer_with_sources(text: str) -> None:
    """본문은 그대로, 출처 블록은 접을 수 있는 토글로 그립니다."""

    detail = ""
    if WITHHOLD_DETAIL_HEADING in text:
        text, detail = text.split(WITHHOLD_DETAIL_HEADING, 1)

    if SOURCES_HEADING in text:
        body, sources = text.split(SOURCES_HEADING, 1)
        st.markdown(escape_tildes(body.strip()))
        with st.expander("📚 출처 보기"):
            st.markdown(escape_tildes(sources.strip()))
    else:
        st.markdown(escape_tildes(text))

    if detail.strip():
        with st.expander("🔎 보류 사유 자세히 보기"):
            st.markdown(escape_tildes(detail.strip()))


def render_assistant_message(text: str) -> None:
    """마스코트 아바타와 함께 답변 말풍선을 그립니다."""

    with st.chat_message("assistant", avatar=mascot_path()):
        render_answer_with_sources(text)


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
        render_answer_with_sources(escape_tildes(answer))

    st.session_state.messages.append({"role": "assistant", "content": answer})


VALID_PAGES = {"home", "chat", "crop", "pest", "pesticide", "region"}


def apply_nav_query() -> None:
    """로고·플로팅 링크(?nav=...)가 넘긴 이동 요청을 처리합니다."""

    nav = st.query_params.get("nav")
    if not nav:
        return
    # 다음 rerun에서 다시 덮어쓰지 않도록 파라미터를 지웁니다.
    del st.query_params["nav"]
    if nav in VALID_PAGES:
        st.session_state.page = nav


def render_header() -> None:
    """왼쪽 로고 배너를 그립니다. '농업 정보'를 누르면 홈으로 돌아갑니다."""

    st.markdown(render_board_html(), unsafe_allow_html=True)


def render_float_menu() -> None:
    """오른쪽 가장자리에 '농업정보 물어보기' 탭을 고정합니다."""

    st.markdown(render_float_ask_html(), unsafe_allow_html=True)


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


# Aura 기업 그래프 색은 데이터에 따라 흔들리지 않게 고정값으로 둡니다.
#   중심 기업은 로즈, 중심과 이어진 모회사(계열)는 페리윙클, 자회사·주변 노드는 회색입니다.
GRAPH_BORDER_COLOR = "#b4aaa1"
GRAPH_CENTER_COLOR = "#f4b6c2"
GRAPH_AFFILIATE_COLOR = "#b8c0ff"
GRAPH_SUBSIDIARY_COLOR = "#cfccc4"
GRAPH_VIS_SCRIPT = (
    "https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/dist/vis-network.min.js"
)


def _graph_node_color(node: dict[str, object]) -> str:
    """노드의 중요도(중심 여부·회사 종류)에 따라 고정 색을 돌려줍니다."""

    try:
        level = int(node.get("level") or 0)
    except (TypeError, ValueError):
        level = 0
    if level == 0:
        return GRAPH_CENTER_COLOR
    if str(node.get("group")) == "ParentCompany":
        return GRAPH_AFFILIATE_COLOR
    return GRAPH_SUBSIDIARY_COLOR


def render_aura_graph_html(
    nodes: list[dict[str, object]],
    edges: list[dict[str, object]],
    height: int = 560,
) -> str:
    """Aura 그래프를 vis-network 마크업으로 만듭니다.

    build_graph_elements() 가 만든 노드·엣지를 그대로 받습니다. 노드는 사용자가
    끌어 옮길 수 있어야 하므로 좌표를 고정하지 않습니다.
    """

    vis_nodes = [
        {
            "id": node["id"],
            "label": str(node.get("label") or node["id"]),
            "title": str(node.get("label") or node["id"]),
            "group": str(node.get("group") or "Unknown"),
            "level": int(node.get("level") or 0),
            "color": {
                "background": _graph_node_color(node),
                "border": GRAPH_BORDER_COLOR,
                "highlight": {
                    "background": _graph_node_color(node),
                    "border": GRAPH_BORDER_COLOR,
                },
            },
        }
        for node in nodes
    ]
    vis_edges = [
        {
            "from": edge["source"],
            "to": edge["target"],
            "label": str(edge.get("relation") or ""),
            "title": str(edge.get("relation") or ""),
        }
        for edge in edges
    ]
    options = {
        "nodes": {
            "shape": "dot",
            "size": 22,
            "borderWidth": 2,
            "font": {"size": 15, "color": "#4a443e"},
        },
        "edges": {
            "color": GRAPH_BORDER_COLOR,
            "width": 1.4,
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
            "font": {"size": 11, "color": "#7a736b", "strokeWidth": 3},
            "smooth": {"enabled": True, "type": "dynamic"},
        },
        "interaction": {
            "dragNodes": True,
            "dragView": True,
            "zoomView": True,
            "hover": True,
            "tooltipDelay": 120,
        },
        "physics": {
            "enabled": True,
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 220},
        },
    }

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<script src="{GRAPH_VIS_SCRIPT}"></script>
<style>
  html, body {{ margin: 0; padding: 0; background: transparent; }}
  #aura-graph {{ width: 100%; height: {height}px; }}
</style>
</head>
<body>
<div id="aura-graph"></div>
<script>
  const nodes = new vis.DataSet({json.dumps(vis_nodes)});
  const edges = new vis.DataSet({json.dumps(vis_edges)});
  const options = {json.dumps(options)};
  new vis.Network(document.getElementById("aura-graph"), {{nodes, edges}}, options);
</script>
</body>
</html>
"""


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
        layout="wide",
    )
    st.markdown(build_css(), unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "page" not in st.session_state:
        st.session_state.page = "home"

    apply_nav_query()

    render_header()
    render_float_menu()
    render_sidebar()

    if st.session_state.page == "home":
        render_home_page()
    elif st.session_state.page == "chat":
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
