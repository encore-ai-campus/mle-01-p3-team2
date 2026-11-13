"""게임풍 농업 정보 UI의 정적 자원 로딩과 CSS 생성을 담당합니다."""

from __future__ import annotations

import base64
import html
from functools import lru_cache
from pathlib import Path


STATIC_DIR = Path(__file__).resolve().parent / "static"

MASCOT_FILENAME = "mascot.png"
BODY_FONT_FILENAME = "GangwonEduAll-Bold.woff2"
TITLE_FONT_FILENAME = "GangwonEduAll-Bold.woff2"

BODY_FONT_FAMILY = "GangwonEduAllBold"
TITLE_FONT_FAMILY = "GangwonEduAllBold"


@lru_cache(maxsize=None)
def encode_asset(filename: str) -> str:
    """static 디렉터리의 파일을 base64 문자열로 돌려줍니다."""

    path = STATIC_DIR / filename
    if not path.is_file():
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def mascot_path() -> str:
    """st.chat_message(avatar=...)에 넘길 마스코트 이미지 경로입니다."""

    return str(STATIC_DIR / MASCOT_FILENAME)


def mascot_data_uri() -> str:
    """HTML <img>에 직접 넣을 수 있는 마스코트 data URI입니다."""

    encoded = encode_asset(MASCOT_FILENAME)
    if not encoded:
        return ""
    return f"data:image/png;base64,{encoded}"


def _font_face(family: str, filename: str, weight: int) -> str:
    encoded = encode_asset(filename)
    if not encoded:
        return ""
    return f"""
@font-face {{
    font-family: '{family}';
    src: url(data:font/woff2;base64,{encoded}) format('woff2');
    font-weight: {weight};
    font-display: swap;
}}
"""


def font_faces() -> str:
    """번들된 강원교육 폰트의 @font-face 선언을 만듭니다."""

    return "".join(
        [
            _font_face(BODY_FONT_FAMILY, BODY_FONT_FILENAME, 400),
            _font_face(TITLE_FONT_FAMILY, TITLE_FONT_FILENAME, 700),
        ]
    )


PALETTE = {
    "sky_top": "#58c6f0",
    "sky_bottom": "#b7ecfb",
    "grass_light": "#8fd44a",
    "grass_dark": "#4f9e22",
    "panel_dark": "#1f2a16",
    "panel_darker": "#162010",
    "wood": "#b4762f",
    "wood_dark": "#8a561c",
    "leaf": "#7fd13b",
    "leaf_dark": "#3f7d14",
    "cream": "#fdf7e3",
    "ink": "#36300f",
}


def build_css() -> str:
    """앱 전체에 주입할 게임풍 스타일시트를 만듭니다."""

    p = PALETTE
    return f"""
<style>
{font_faces()}

/* ---------- 바탕: 하늘 + 구름 + 잔디 ---------- */
.stApp {{
    /* 모든 레이어가 viewport 기준(fixed)이라 아래쪽 잔디밭이 항상 화면에 붙어 있습니다. */
    background:
        radial-gradient(ellipse 34% 150px at 22% 100%, #a6e45c 0 70%, transparent 71%),
        radial-gradient(ellipse 30% 130px at 74% 100%, #a6e45c 0 70%, transparent 71%),
        linear-gradient(180deg,
            transparent 0 calc(100% - 165px),
            {p["grass_light"]} calc(100% - 150px),
            {p["grass_dark"]} 100%),
        radial-gradient(circle at 12% 14%, rgba(255,255,255,.85) 0 54px, transparent 55px),
        radial-gradient(circle at 18% 11%, rgba(255,255,255,.85) 0 38px, transparent 39px),
        radial-gradient(circle at 79% 10%, rgba(255,255,255,.75) 0 46px, transparent 47px),
        radial-gradient(circle at 85% 13%, rgba(255,255,255,.75) 0 32px, transparent 33px),
        linear-gradient(180deg, {p["sky_top"]} 0%, {p["sky_bottom"]} 60%, #d8f4ff 100%);
    background-attachment: fixed;
}}

html, body, .stApp, [class*="st-"] {{
    font-family: '{BODY_FONT_FAMILY}', 'Pretendard', sans-serif;
}}

/* Material 아이콘은 리거처 폰트라 본문 폰트를 덮어쓰면 글자가 그대로 보입니다. */
span[data-testid="stIconMaterial"],
.material-icons,
.material-icons-outlined,
[class*="material-symbols"] {{
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}}

header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}

.block-container {{
    max-width: 1080px;
    padding-top: 1.2rem;
    padding-bottom: 9rem;
}}

/* ---------- 상단 나무 간판 ---------- */
.vb-board {{
    background: linear-gradient(180deg, {p["panel_dark"]} 0%, {p["panel_darker"]} 100%);
    border: 4px solid {p["wood_dark"]};
    border-radius: 26px;
    box-shadow: 0 10px 0 rgba(0,0,0,.22), inset 0 2px 0 rgba(255,255,255,.08);
    padding: 18px 26px;
    margin-bottom: 18px;
}}

.vb-logo {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 44px;
    line-height: 1.1;
    letter-spacing: -1px;
    color: #ffffff;
    text-shadow:
        -3px -3px 0 {p["leaf_dark"]}, 3px -3px 0 {p["leaf_dark"]},
        -3px  3px 0 {p["leaf_dark"]}, 3px  3px 0 {p["leaf_dark"]},
        0 7px 0 rgba(0,0,0,.35);
    margin: 0;
}}
.vb-logo .vb-logo-leaf {{ color: {p["leaf"]}; }}

.vb-tagline {{
    color: #cfe9a6;
    font-size: 15px;
    margin-top: 10px;
}}

/* ---------- 메뉴 칩 ---------- */
.vb-menu {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 16px;
}}
.vb-chip {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    border-radius: 999px;
    font-size: 15px;
    border: 2px solid rgba(255,255,255,.18);
    background: rgba(255,255,255,.06);
    color: #b9d79a;
}}
.vb-chip.is-active {{
    background: linear-gradient(180deg, #a6e24f 0%, {p["leaf_dark"]} 100%);
    border-color: #e9ffce;
    color: #16300a;
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    box-shadow: 0 4px 0 rgba(0,0,0,.3);
}}
.vb-chip .vb-soon {{
    font-size: 11px;
    opacity: .65;
}}

/* ---------- 대화 영역 나무 패널 ---------- */
.vb-panel-title {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 20px;
    color: #ffffff;
    text-shadow: 0 2px 0 rgba(0,0,0,.35);
    margin: 4px 0 10px 6px;
}}

/* 채팅 말풍선 */
div[data-testid="stChatMessage"] {{
    background: transparent;
    padding: 2px 0;
    gap: 12px;
}}
div[data-testid="stChatMessageAvatarCustom"],
div[data-testid="stChatMessage"] img[data-testid="stChatMessageAvatarCustom"] {{
    background: #ffffff;
    border: 3px solid {p["leaf_dark"]};
    border-radius: 50%;
    box-shadow: 0 4px 0 rgba(0,0,0,.22);
    overflow: hidden;
}}
div[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {{
    background: {p["cream"]};
    border: 3px solid {p["wood"]};
    border-radius: 18px;
    padding: 14px 18px;
    color: {p["ink"]};
    box-shadow: 0 5px 0 rgba(0,0,0,.18);
}}
div[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] p {{
    font-size: 16px;
    line-height: 1.75;
    margin-bottom: .4rem;
}}
/* 사용자 말풍선: 직접 그린 초록 말풍선 */
.vb-user-row {{
    display: flex;
    justify-content: flex-end;
    align-items: flex-start;
    gap: 10px;
    margin: 10px 0;
}}
.vb-user-bubble {{
    max-width: 78%;
    background: linear-gradient(180deg, #e8ffcd 0%, #c6ee98 100%);
    border: 3px solid {p["leaf_dark"]};
    border-radius: 18px;
    padding: 12px 18px;
    color: {p["ink"]};
    font-size: 16px;
    line-height: 1.7;
    box-shadow: 0 5px 0 rgba(0,0,0,.18);
    word-break: break-word;
}}
.vb-user-face {{
    flex: 0 0 44px;
    width: 44px;
    height: 44px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    background: linear-gradient(180deg, #ffd978 0%, #e8a32b 100%);
    border: 3px solid {p["wood_dark"]};
    box-shadow: 0 4px 0 rgba(0,0,0,.22);
}}

/* ---------- 마스코트 인사 카드 ---------- */
.vb-greet {{
    display: flex;
    align-items: center;
    gap: 18px;
    background: {p["cream"]};
    border: 4px solid {p["wood"]};
    border-radius: 24px;
    box-shadow: 0 8px 0 rgba(0,0,0,.2);
    padding: 18px 22px;
    margin-bottom: 14px;
}}
.vb-greet img {{
    width: 92px;
    height: 92px;
    flex: 0 0 92px;
    animation: vb-bob 2.4s ease-in-out infinite;
}}
@keyframes vb-bob {{
    0%, 100% {{ transform: translateY(0); }}
    50% {{ transform: translateY(-7px); }}
}}
.vb-greet-name {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 19px;
    color: {p["leaf_dark"]};
    margin-bottom: 4px;
}}
.vb-greet-text {{
    color: {p["ink"]};
    font-size: 15px;
    line-height: 1.7;
}}

/* ---------- 입력창 ---------- */
div[data-testid="stBottom"],
div[data-testid="stBottom"] > div,
div[data-testid="stBottomBlockContainer"] {{
    background: transparent !important;
}}
div[data-testid="stBottomBlockContainer"] {{
    padding-bottom: 1.4rem;
}}
div[data-testid="stChatInput"] {{
    background: {p["cream"]};
    border: 4px solid {p["wood"]};
    border-radius: 999px;
    box-shadow: 0 6px 0 rgba(0,0,0,.25);
}}
div[data-testid="stChatInput"] textarea {{
    font-family: '{BODY_FONT_FAMILY}', sans-serif;
    font-size: 16px;
    color: {p["ink"]};
}}
div[data-testid="stChatInput"] textarea::placeholder {{ color: #a4915c; }}

/* ---------- 버튼 ---------- */
.stButton > button {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    background: linear-gradient(180deg, #a6e24f 0%, #4f9e22 100%);
    color: #15300a;
    border: 3px solid #2f6a0f;
    border-radius: 999px;
    box-shadow: 0 5px 0 rgba(0,0,0,.28);
    padding: .45rem 1.1rem;
    transition: transform .08s ease;
}}
.stButton > button:hover {{
    background: linear-gradient(180deg, #b8ef64 0%, #5cb328 100%);
    color: #15300a;
    border-color: #2f6a0f;
}}
.stButton > button:active {{
    transform: translateY(4px);
    box-shadow: 0 1px 0 rgba(0,0,0,.28);
}}

/* ---------- 사이드바 ---------- */
section[data-testid="stSidebar"] > div {{
    background: linear-gradient(180deg, {p["panel_dark"]} 0%, {p["panel_darker"]} 100%);
    border-right: 4px solid {p["wood_dark"]};
}}
section[data-testid="stSidebar"] * {{ color: #dff0c4; }}

/* ---------- 상단 메뉴 (Streamlit 버튼으로 만든 칩) ---------- */
.st-key-vbnav {{
    background: linear-gradient(180deg, {p["panel_dark"]} 0%, {p["panel_darker"]} 100%);
    border: 4px solid {p["wood_dark"]};
    border-top: none;
    border-radius: 0 0 26px 26px;
    box-shadow: 0 10px 0 rgba(0,0,0,.22);
    padding: 12px 18px 14px 18px;
    margin: -22px 0 18px 0;
}}
.st-key-vbnav .stButton > button {{
    width: 100%;
    font-size: 14px;
    padding: .4rem .5rem;
    background: rgba(255,255,255,.06);
    border: 2px solid rgba(255,255,255,.18);
    color: #b9d79a;
    box-shadow: none;
}}
.st-key-vbnav .stButton > button:hover {{
    background: rgba(255,255,255,.14);
    border-color: rgba(255,255,255,.35);
    color: #eaffd2;
}}
.st-key-vbnav .stButton > button[kind="primary"] {{
    background: linear-gradient(180deg, #a6e24f 0%, {p["leaf_dark"]} 100%);
    border-color: #e9ffce;
    color: #16300a;
    box-shadow: 0 4px 0 rgba(0,0,0,.3);
}}

/* ---------- 작물 선택 줄 ---------- */
.st-key-vbcrop {{ margin-bottom: 14px; }}
.st-key-vbcrop .stButton > button {{
    width: 100%;
    font-size: 16px;
    background: {p["cream"]};
    border: 3px solid {p["wood"]};
    color: {p["ink"]};
}}
.st-key-vbcrop .stButton > button:hover {{
    background: #fffdf2;
    border-color: {p["wood_dark"]};
    color: {p["ink"]};
}}
.st-key-vbcrop .stButton > button[kind="primary"] {{
    background: linear-gradient(180deg, #a6e24f 0%, {p["leaf_dark"]} 100%);
    border-color: #2f6a0f;
    color: #15300a;
}}

/* ---------- 공통 크림 카드 ---------- */
.vb-card {{
    background: {p["cream"]};
    border: 4px solid {p["wood"]};
    border-radius: 24px;
    box-shadow: 0 8px 0 rgba(0,0,0,.2);
    padding: 18px 22px;
    margin-bottom: 16px;
    color: {p["ink"]};
}}
.vb-card h3 {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 19px;
    color: {p["leaf_dark"]};
    margin: 0 0 12px 0;
}}

/* 작물 머리말 */
.vb-hero {{
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
}}
.vb-hero-emoji {{
    font-size: 46px;
    line-height: 1;
}}
.vb-hero-name {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 30px;
    color: {p["leaf_dark"]};
}}

/* 숫자 칩 */
.vb-stats {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 14px;
}}
.vb-stat {{
    flex: 1 1 110px;
    background: linear-gradient(180deg, #ffffff 0%, #f1e8c8 100%);
    border: 3px solid {p["wood"]};
    border-radius: 16px;
    padding: 10px 8px;
    text-align: center;
}}
.vb-stat-label {{ font-size: 12px; color: #7a6a33; }}
.vb-stat-value {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 24px;
    color: {p["leaf_dark"]};
    line-height: 1.2;
}}

/* 표 */
.vb-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14.5px;
}}
.vb-table th {{
    text-align: left;
    background: #efe3bd;
    color: #5f5218;
    padding: 8px 10px;
    border: 2px solid #ddcb9c;
    font-weight: normal;
}}
.vb-table td {{
    padding: 8px 10px;
    border: 2px solid #e6d7af;
    vertical-align: top;
}}
.vb-table .vb-cat {{
    white-space: nowrap;
    color: {p["leaf_dark"]};
}}
.vb-table .vb-val {{ white-space: nowrap; font-weight: 600; }}
.vb-table .vb-note {{ color: #6d6033; font-size: 13.5px; }}

/* 품종 카드 */
.vb-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 12px;
}}
.vb-variety {{
    background: linear-gradient(180deg, #ffffff 0%, #f6efd6 100%);
    border: 3px solid #d8c492;
    border-radius: 16px;
    padding: 12px 14px;
}}
.vb-variety-name {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 16px;
    color: {p["leaf_dark"]};
    margin-bottom: 4px;
}}
.vb-tag {{
    display: inline-block;
    font-size: 11.5px;
    padding: 2px 9px;
    border-radius: 999px;
    background: #e3f3c6;
    border: 1px solid #a8cf74;
    color: #3f6418;
    margin: 0 4px 4px 0;
}}
.vb-tag.vb-tag-sow {{ background: #ffeec2; border-color: #e0b857; color: #7a5410; }}
.vb-tag.vb-tag-region {{ background: #d9edff; border-color: #78b4e4; color: #1c4c74; }}
.vb-variety-desc {{ font-size: 13.5px; line-height: 1.6; color: #534a22; margin-top: 4px; }}

/* 병해충 접이식 (1.59에서 stExpander는 details를 감싼 div입니다) */
[data-testid="stExpander"] details {{
    background: linear-gradient(180deg, #ffffff 0%, #f6efd6 100%) !important;
    border: 3px solid #d8c492 !important;
    border-radius: 16px !important;
    margin-bottom: 8px;
    box-shadow: 0 4px 0 rgba(0,0,0,.14);
}}
[data-testid="stExpander"] summary {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    color: {p["leaf_dark"]} !important;
}}
[data-testid="stExpander"] summary:hover {{
    background: #fffdf2 !important;
    border-radius: 13px;
}}
[data-testid="stExpander"] p,
[data-testid="stExpander"] li,
[data-testid="stExpander"] strong {{
    color: {p["ink"]};
    font-size: 14.5px;
    line-height: 1.7;
}}
[data-testid="stExpander"] strong {{ color: {p["leaf_dark"]}; }}

/* ---------- 농약 검색 ---------- */
.vb-notice {{
    background: linear-gradient(180deg, #fff6d8 0%, #ffe9a8 100%);
    border: 4px solid #e0a52b;
    border-radius: 20px;
    box-shadow: 0 6px 0 rgba(0,0,0,.18);
    padding: 14px 18px;
    margin-bottom: 14px;
    color: #5c3f07;
    font-size: 14.5px;
    line-height: 1.7;
}}
.vb-notice code {{
    background: rgba(0,0,0,.07);
    border-radius: 5px;
    padding: 1px 5px;
    font-size: 13px;
}}
.vb-notice a {{ color: #0b5c9c; }}

.st-key-vbfilter {{
    background: {p["cream"]};
    border: 4px solid {p["wood"]};
    border-radius: 24px;
    box-shadow: 0 8px 0 rgba(0,0,0,.2);
    padding: 14px 20px 6px 20px;
    margin-bottom: 16px;
}}
.st-key-vbfilter label, .st-key-vbfilter p {{
    color: {p["ink"]} !important;
    font-size: 13.5px !important;
}}

.vb-pest-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
}}
.vb-pest {{
    background: {p["cream"]};
    border: 3px solid {p["wood"]};
    border-radius: 18px;
    box-shadow: 0 5px 0 rgba(0,0,0,.16);
    padding: 14px 16px;
    color: {p["ink"]};
}}
.vb-pest-head {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px;
}}
.vb-pest-brand {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 18px;
    color: {p["leaf_dark"]};
}}
.vb-pest-crop {{
    font-size: 12px;
    background: #e3f3c6;
    border: 1px solid #a8cf74;
    border-radius: 999px;
    padding: 1px 9px;
    color: #3f6418;
    white-space: nowrap;
}}
.vb-pest-product {{
    font-size: 13px;
    color: #6d6033;
    margin: 2px 0 8px 0;
    line-height: 1.5;
}}
.vb-pest-table {{ margin-top: 10px; }}
.vb-pest-table td {{ padding: 5px 8px; font-size: 13.5px; }}
.vb-pest-table .vb-cat {{ width: 38%; color: #7a6a33; }}

.vb-tag.vb-tox-low {{ background: #e3f3c6; border-color: #a8cf74; color: #3f6418; }}
.vb-tag.vb-tox-mid {{ background: #ffeec2; border-color: #e0b857; color: #7a5410; }}
.vb-tag.vb-tox-high {{ background: #ffd9d4; border-color: #e07a68; color: #8a2c17; }}

.vb-warn {{
    margin-top: 8px;
    background: #ffe2dc;
    border: 2px solid #e07a68;
    border-radius: 12px;
    padding: 7px 10px;
    font-size: 12.5px;
    color: #8a2c17;
    line-height: 1.6;
}}
.vb-warn a {{ color: #0b5c9c; }}

/* ---------- 지역별 정보 ---------- */
.vb-region-extremes {{
    margin-top: 12px;
    font-size: 13.5px;
    color: #6d6033;
    line-height: 1.8;
}}
.stAltairChart, [data-testid="stVegaLiteChart"] {{
    background: {p["cream"]};
    border: 4px solid {p["wood"]};
    border-radius: 20px;
    box-shadow: 0 6px 0 rgba(0,0,0,.18);
    padding: 14px 16px;
    margin-bottom: 16px;
}}

/* ---------- 준비중 안내 카드 ---------- */
.vb-soon-card {{
    background: {p["cream"]};
    border: 4px dashed {p["wood"]};
    border-radius: 24px;
    padding: 40px 24px;
    text-align: center;
    color: {p["ink"]};
    box-shadow: 0 8px 0 rgba(0,0,0,.18);
}}
.vb-soon-card img {{ width: 110px; margin-bottom: 12px; }}
.vb-soon-card h3 {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    color: {p["leaf_dark"]};
    margin: 0 0 6px 0;
}}
</style>
"""


MENU_ITEMS: tuple[tuple[str, str, bool], ...] = (
    ("chat", "🌱 물어보기", True),
    ("crop", "🥬 작물 도감", True),
    ("pest", "🐛 병해충 도감", True),
    ("pesticide", "🧪 농약 검색", True),
    ("region", "🗺️ 지역별 정보", True),
)


def render_board_html() -> str:
    """로고가 들어간 상단 간판을 만듭니다. 메뉴는 아래 버튼 줄이 담당합니다."""

    return """
<div class="vb-board" style="border-radius:26px 26px 0 0; box-shadow:none; margin-bottom:0;">
    <div class="vb-logo"><span class="vb-logo-leaf">농업</span> 정보</div>
    <div class="vb-tagline">작물·병해충·농약 질문에 근거를 찾아 답해 드려요.</div>
</div>
"""


def render_user_bubble_html(text: str) -> str:
    """사용자 질문을 오른쪽 초록 말풍선으로 그립니다."""

    safe = html.escape(text).replace("\n", "<br>")
    return f"""
<div class="vb-user-row">
    <div class="vb-user-bubble">{safe}</div>
    <div class="vb-user-face">🧑‍🌾</div>
</div>
"""


def render_greeting_html(
    name: str = "부기",
    text: str = (
        "안녕! 나는 농업 정보 도우미 <b>부기</b>야.<br>"
        "상추·양파·마늘·배추·고추에 대해 궁금한 걸 아래에 물어봐 줘!"
    ),
) -> str:
    """마스코트 인사 카드를 만듭니다."""

    uri = mascot_data_uri()
    img = f"<img src='{uri}' alt='{name}'/>" if uri else ""
    return f"""
<div class="vb-greet">
    {img}
    <div>
        <div class="vb-greet-name">{name}</div>
        <div class="vb-greet-text">{text}</div>
    </div>
</div>
"""
