"""농업 정보 UI의 정적 자원 로딩과 CSS 생성을 담당합니다."""

from __future__ import annotations

import base64
import html
from functools import lru_cache
from pathlib import Path


STATIC_DIR = Path(__file__).resolve().parent / "source"

MASCOT_FILENAME = "mascot.png"
BACKGROUND_FILENAME = "배경.png"
# 로고 전용 손글씨 폰트
LOGO_FONT_FILENAME = "강원교육튼튼.ttf"
# 일반 본문 / 강조(볼드)
BODY_FONT_FILENAME = "KoPubWorld Dotum Medium.ttf"
TITLE_FONT_FILENAME = "KoPubWorld Dotum Bold.ttf"

LOGO_FONT_FAMILY = "GangwonTteunTteun"
BODY_FONT_FAMILY = "KoPubDotumMedium"
TITLE_FONT_FAMILY = "KoPubDotumBold"


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


def background_data_uri() -> str:
    """메인 화면 히어로 배경의 data URI를 돌려줍니다."""

    encoded = encode_asset(BACKGROUND_FILENAME)
    if not encoded:
        return ""
    return f"data:image/png;base64,{encoded}"


_FONT_FORMATS = {
    ".woff2": ("font/woff2", "woff2"),
    ".woff": ("font/woff", "woff"),
    ".ttf": ("font/ttf", "truetype"),
    ".otf": ("font/otf", "opentype"),
}


def _font_face(family: str, filename: str, weight: int) -> str:
    encoded = encode_asset(filename)
    if not encoded:
        return ""
    mime, fmt = _FONT_FORMATS.get(Path(filename).suffix.lower(), ("font/ttf", "truetype"))
    return f"""
@font-face {{
    font-family: '{family}';
    src: url(data:{mime};base64,{encoded}) format('{fmt}');
    font-weight: {weight};
    font-display: swap;
}}
"""


def font_faces() -> str:
    """번들된 폰트(로고·본문·강조)의 @font-face 선언을 만듭니다."""

    return "".join(
        [
            _font_face(LOGO_FONT_FAMILY, LOGO_FONT_FILENAME, 700),
            _font_face(BODY_FONT_FAMILY, BODY_FONT_FILENAME, 400),
            _font_face(TITLE_FONT_FAMILY, TITLE_FONT_FILENAME, 700),
        ]
    )


PALETTE = {
    # 흰색을 중심으로 한 차분한 정보 서비스 팔레트
    "sky_top": "#ffffff",
    "sky_bottom": "#ffffff",
    "grass_light": "#dff3d0",
    "grass_dark": "#bfe6a8",
    "panel_dark": "#ffffff",
    "panel_darker": "#f4f9f3",
    "wood": "#d7e8cf",
    "wood_dark": "#bcd9ad",
    "leaf": "#56c06a",
    "leaf_dark": "#2f9e55",
    "cream": "#ffffff",
    "ink": "#2b3a2f",
}


def build_css() -> str:
    """앱 전체에 주입할 단정한 정보 서비스 스타일시트를 만듭니다."""

    p = PALETTE
    return f"""
<style>
{font_faces()}

/* ---------- 바탕 ---------- */
html, body, .stApp, [data-testid="stAppViewContainer"] {{
    background: #ffffff;
}}

html, body, .stApp, [class*="st-"] {{
    font-family: '{BODY_FONT_FAMILY}', 'Pretendard', sans-serif;
    font-size: 18px;
}}

/* 강조 텍스트는 KoPub Dotum Bold로 */
strong, b, .stMarkdown strong, .stMarkdown b {{
    font-family: '{TITLE_FONT_FAMILY}', 'Pretendard', sans-serif;
    font-weight: 700;
}}

/* Material 아이콘은 리거처 폰트라 본문 폰트를 덮어쓰면 글자가 그대로 보입니다. */
span[data-testid="stIconMaterial"],
.material-icons,
.material-icons-outlined,
[class*="material-symbols"] {{
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}}

header[data-testid="stHeader"] {{ background: #ffffff; }}
#MainMenu, footer {{ visibility: hidden; }}

.block-container {{
    max-width: 1500px;
    padding-top: 4rem;
    padding-bottom: 9rem;
    padding-left: 3rem;
    padding-right: 3rem;
}}

/* ---------- 상단 헤더 ---------- */
.vb-board {{
    background: #ffffff;
    border: none;
    border-radius: 0;
    box-shadow: none;
    padding: 20px 4px;
    margin: 0 0 -8px 0;
}}

.vb-logo {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 34px;
    line-height: 1.1;
    letter-spacing: -1px;
    color: {p["ink"]};
    margin: 0;
}}
.vb-logo .vb-logo-leaf {{ color: {p["leaf_dark"]}; }}

.vb-tagline {{
    color: #6b8569;
    font-size: 18px;
    margin-top: 10px;
}}

/* ---------- 메인 히어로 ---------- */
.vb-home-hero {{
    position: relative;
    width: 100%;
    min-height: 380px;
    aspect-ratio: 16 / 5.3;
    max-height: 470px;
    display: flex;
    align-items: center;
    overflow: hidden;
    border-radius: 28px 28px 92px 28px;
    background-color: #315f2f;
    background-position: center 55%;
    background-size: cover;
    margin: -4px 0 72px;
    isolation: isolate;
}}
.vb-home-hero::before {{
    content: "";
    position: absolute;
    inset: 0;
    z-index: -1;
    background: linear-gradient(90deg, rgba(18,45,24,.88) 0%, rgba(27,63,30,.70) 40%, rgba(20,47,25,.12) 72%);
}}
.vb-home-hero-content {{
    width: min(670px, 62%);
    padding: 64px 72px;
    color: #ffffff;
}}
.vb-home-hero-kicker {{
    font-size: 18px;
    margin-bottom: 12px;
    color: #e1f2dd;
}}
.vb-home-hero-title {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: clamp(38px, 4vw, 58px);
    line-height: 1.18;
    letter-spacing: -1.8px;
    margin: 0;
}}
.vb-home-hero-copy {{
    max-width: 570px;
    margin: 20px 0 30px;
    font-size: 19px;
    line-height: 1.7;
    color: rgba(255,255,255,.9);
}}
.vb-home-hero-actions {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}}
.vb-home-hero-button {{
    display: inline-flex;
    align-items: center;
    min-height: 44px;
    padding: 0 20px;
    border-radius: 999px;
    background: #ffffff;
    border: 1px solid #ffffff;
    color: #243b29 !important;
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 15px;
    text-decoration: none !important;
}}
.vb-home-hero-button.is-primary {{
    background: #17281c;
    border-color: #17281c;
    color: #ffffff !important;
}}
.vb-home-hero-button:hover {{
    transform: translateY(-1px);
    filter: brightness(.97);
}}

/* 가로형 배너: 왼쪽 로고 */
.vb-banner {{
    display: flex;
    align-items: center;
    gap: 16px;
}}
.vb-logo-link,
.vb-logo-link:hover,
.vb-logo-link * {{
    text-decoration: none !important;
}}
.vb-logo-link {{
    display: inline-flex;
    align-items: center;
    cursor: pointer;
    color: {p["ink"]} !important;
}}
.vb-logo-link .vb-logo {{ color: {p["ink"]} !important; }}
.vb-logo-link .vb-logo-leaf {{ color: {p["leaf_dark"]} !important; }}
.vb-logo-link:hover .vb-logo {{ opacity: .85; }}
.vb-banner .vb-logo-mark {{
    width: 56px;
    height: 56px;
    flex: 0 0 56px;
    border-radius: 50%;
    background: #ffffff;
    border: 1px solid #d7e8cf;
    padding: 4px;
}}

/* 오른쪽 가장자리 고정: 농업정보 물어보기 탭 */
.vb-float-ask,
.vb-float-ask:hover,
.vb-float-ask * {{
    text-decoration: none !important;
}}
.vb-float-ask {{
    position: fixed;
    right: 0;
    top: 40%;
    z-index: 1000;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    width: 96px;
    padding: 16px 10px;
    background: #ffffff;
    border: 1px solid #e4efdd;
    border-right: none;
    border-radius: 8px 0 0 8px;
    box-shadow: 0 2px 8px rgba(15,23,42,.08);
    color: {p["ink"]};
    font-size: 16px;
    line-height: 1.4;
    text-align: center;
    text-decoration: none;
}}
.vb-float-ask:hover {{
    background: #f7faf7;
    box-shadow: 0 2px 10px rgba(15,23,42,.10);
}}
.vb-float-ic {{
    width: 42px;
    height: 42px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    background: #eef7ef;
    border: 1px solid #bfe8c6;
}}
.vb-float-ask {{ color: {p["ink"]} !important; }}
.vb-float-tx {{ color: {p["ink"]} !important; font-weight: 500; }}

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
    padding: 9px 18px;
    border-radius: 999px;
    font-size: 17px;
    border: 2px solid #e2ecdb;
    background: #f6faf3;
    color: #5f7a5d;
}}
.vb-chip.is-active {{
    background: {p["leaf_dark"]};
    border-color: #bfe8c6;
    color: #ffffff;
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    box-shadow: none;
}}
.vb-chip .vb-soon {{
    font-size: 12px;
    opacity: .65;
}}

/* ---------- 패널 제목 ---------- */
.vb-panel-title {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 23px;
    color: {p["ink"]};
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
    border: 3px solid {p["leaf"]};
    border-radius: 50%;
    box-shadow: none;
    overflow: hidden;
}}
div[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {{
    background: {p["cream"]};
    border: 2px solid #e4efdd;
    border-radius: 18px;
    padding: 16px 20px;
    color: {p["ink"]};
    box-shadow: none;
}}
div[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] p {{
    font-size: 18px;
    line-height: 1.8;
    margin-bottom: .4rem;
}}
/* 사용자 말풍선: 초록 그라데이션 */
.vb-user-row {{
    display: flex;
    justify-content: flex-end;
    align-items: flex-start;
    gap: 10px;
    margin: 10px 0;
}}
.vb-user-bubble {{
    max-width: 78%;
    background: #eef7ef;
    border: 1px solid #cfe4d2;
    border-radius: 12px;
    padding: 14px 20px;
    color: {p["ink"]};
    font-size: 18px;
    line-height: 1.75;
    box-shadow: none;
    word-break: break-word;
}}
.vb-user-face {{
    flex: 0 0 46px;
    width: 46px;
    height: 46px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 23px;
    background: #fff2cf;
    border: 1px solid #e6b35a;
}}

/* ---------- 마스코트 인사 카드 ---------- */
.vb-greet {{
    display: flex;
    align-items: center;
    gap: 18px;
    background: #ffffff;
    border: 1px solid #dfe7df;
    border-radius: 12px;
    box-shadow: none;
    padding: 20px 24px;
    margin-bottom: 14px;
}}
.vb-greet img {{
    width: 94px;
    height: 94px;
    flex: 0 0 94px;
    animation: vb-bob 2.4s ease-in-out infinite;
}}
@keyframes vb-bob {{
    0%, 100% {{ transform: translateY(0); }}
    50% {{ transform: translateY(-7px); }}
}}
.vb-greet-name {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 22px;
    color: {p["leaf_dark"]};
    margin-bottom: 4px;
}}
.vb-greet-text {{
    color: {p["ink"]};
    font-size: 18px;
    line-height: 1.75;
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
    background: #ffffff;
    border: 2px solid #d7e8cf;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(15,23,42,.06);
}}
/* 포커스 시 빨간 테두리 대신 초록으로 */
div[data-testid="stChatInput"]:focus-within {{
    border-color: {p["leaf"]};
}}
/* 내부 래퍼(stChatInput 직속 자식 div)의 회색 배경·자체 테두리를 지워
   pill 안에 깔끔히 들어가게 합니다. 이 테두리가 포커스 시 빨갛게 변하던 원인입니다. */
div[data-testid="stChatInput"] > div,
div[data-testid="stChatInput"] > div:focus-within {{
    width: 100%;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}}
/* 포커스 시 어떤 내부 요소든 빨간 테두리가 보이지 않도록 모든 테두리색을 투명 처리 */
div[data-testid="stChatInput"] div {{
    border-color: transparent !important;
}}
div[data-testid="stChatInput"] textarea {{
    font-family: '{BODY_FONT_FAMILY}', sans-serif;
    font-size: 18px;
    color: {p["ink"]};
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}}
div[data-testid="stChatInput"] textarea:focus {{
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}}
div[data-testid="stChatInput"] textarea::placeholder {{ color: #9fb59b; }}
/* 전송 버튼: 초록 */
div[data-testid="stChatInput"] button {{
    background: {p["leaf_dark"]} !important;
    border-radius: 8px !important;
}}
div[data-testid="stChatInput"] button:hover {{
    background: #278b4a !important;
}}
div[data-testid="stChatInput"] button svg,
div[data-testid="stChatInput"] button [data-testid="stIconMaterial"] {{
    color: #ffffff !important;
    fill: #ffffff !important;
}}

/* ---------- 버튼 ---------- */
.stButton > button {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 17px;
    background: {p["leaf_dark"]};
    color: #ffffff;
    border: none;
    border-radius: 8px;
    box-shadow: none;
    padding: .5rem 1.2rem;
    transition: transform .08s ease, box-shadow .12s ease;
}}
.stButton > button:hover {{
    background: #278b4a;
    color: #ffffff;
    box-shadow: none;
}}
.stButton > button:active {{
    transform: translateY(2px);
    box-shadow: none;
}}

/* ---------- 사이드바 ---------- */
section[data-testid="stSidebar"] > div {{
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}}
section[data-testid="stSidebar"] * {{ color: {p["ink"]}; }}
section[data-testid="stSidebar"] h3 {{ color: {p["leaf_dark"]}; }}

/* ---------- 상단 메뉴 (Streamlit 버튼으로 만든 칩) ---------- */
.st-key-vbnav {{
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-top: none;
    border-radius: 0 0 10px 10px;
    box-shadow: none;
    padding: 12px 18px 14px 18px;
    margin: -20px 0 18px 0;
}}
.st-key-vbnav .stButton > button {{
    width: 100%;
    font-size: 17px;
    padding: .5rem .5rem;
    background: #f6faf3;
    border: 2px solid #e2ecdb;
    color: #5f7a5d;
    box-shadow: none;
}}
.st-key-vbnav .stButton > button:hover {{
    background: #eef6e9;
    border-color: #cfe4c3;
    color: {p["leaf_dark"]};
}}
.st-key-vbnav .stButton > button[kind="primary"] {{
    background: {p["leaf_dark"]};
    border-color: #bfe8c6;
    color: #ffffff;
    box-shadow: none;
}}

/* ---------- 작물 선택 줄 ---------- */
.st-key-vbcrop {{ margin-bottom: 14px; }}
.st-key-vbcrop .stButton > button {{
    width: 100%;
    font-size: 17px;
    background: {p["cream"]};
    border: 2px solid #d7e8cf;
    color: {p["ink"]};
}}
.st-key-vbcrop .stButton > button:hover {{
    background: #f3faef;
    border-color: #bcd9ad;
    color: {p["leaf_dark"]};
}}
.st-key-vbcrop .stButton > button[kind="primary"] {{
    background: {p["leaf_dark"]};
    border-color: #bfe8c6;
    color: #ffffff;
}}

/* ---------- 공통 카드 ---------- */
.vb-card {{
    background: {p["cream"]};
    border: 1px solid #dfe7df;
    border-radius: 10px;
    box-shadow: none;
    padding: 20px 24px;
    margin-bottom: 16px;
    color: {p["ink"]};
}}
.vb-card h3 {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 22px;
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
    font-size: 50px;
    line-height: 1;
}}
.vb-hero-name {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 34px;
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
    background: #f7faf7;
    border: 1px solid #dfe7df;
    border-radius: 8px;
    padding: 12px 8px;
    text-align: center;
}}
.vb-stat-label {{ font-size: 14px; color: #7a8f76; }}
.vb-stat-value {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 27px;
    color: {p["leaf_dark"]};
    line-height: 1.2;
}}

/* 표 */
.vb-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 16px;
}}
.vb-table th {{
    text-align: left;
    background: #eef7e8;
    color: #4f6b4a;
    padding: 9px 11px;
    border: 1px solid #dcebd3;
    font-weight: normal;
}}
.vb-table td {{
    padding: 9px 11px;
    border: 1px solid #e6efe0;
    vertical-align: top;
}}
.vb-table .vb-cat {{
    white-space: nowrap;
    color: {p["leaf_dark"]};
}}
.vb-table .vb-val {{ white-space: nowrap; font-weight: 600; }}
.vb-table .vb-note {{ color: #6b7d66; font-size: 15px; }}

/* 품종 카드 */
.vb-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 12px;
}}
.vb-variety {{
    background: #ffffff;
    border: 1px solid #dfe7df;
    border-radius: 8px;
    padding: 14px 16px;
}}
.vb-variety-name {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 18px;
    color: {p["leaf_dark"]};
    margin-bottom: 4px;
}}
.vb-tag {{
    display: inline-block;
    font-size: 13px;
    padding: 3px 11px;
    border-radius: 999px;
    background: #e5f4db;
    border: 1px solid #b4dca0;
    color: #3f6b2f;
    margin: 0 4px 4px 0;
}}
.vb-tag.vb-tag-sow {{ background: #fff2d4; border-color: #ecc878; color: #8a6318; }}
.vb-tag.vb-tag-region {{ background: #e0f0ff; border-color: #96c7ec; color: #1f567f; }}
.vb-variety-desc {{ font-size: 15px; line-height: 1.65; color: #4c5c48; margin-top: 4px; }}

/* 병해충 접이식 (1.59에서 stExpander는 details를 감싼 div입니다) */
[data-testid="stExpander"] details {{
    background: #ffffff !important;
    border: 1px solid #dfe7df !important;
    border-radius: 8px !important;
    margin-bottom: 8px;
    box-shadow: none;
}}
[data-testid="stExpander"] summary {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 17px;
    color: {p["leaf_dark"]} !important;
}}
[data-testid="stExpander"] summary:hover {{
    background: #f3faef !important;
    border-radius: 13px;
}}
[data-testid="stExpander"] p,
[data-testid="stExpander"] li,
[data-testid="stExpander"] strong {{
    color: {p["ink"]};
    font-size: 16px;
    line-height: 1.75;
}}
[data-testid="stExpander"] strong {{ color: {p["leaf_dark"]}; }}

/* ---------- 농약 검색 ---------- */
.vb-notice {{
    background: #fffaf0;
    border: 1px solid #e5c36a;
    border-radius: 8px;
    box-shadow: none;
    padding: 16px 20px;
    margin-bottom: 14px;
    color: #6b4e12;
    font-size: 16px;
    line-height: 1.75;
}}
.vb-notice code {{
    background: rgba(0,0,0,.06);
    border-radius: 5px;
    padding: 1px 5px;
    font-size: 14px;
}}
.vb-notice a {{ color: #0b6cbc; }}

.st-key-vbfilter {{
    background: {p["cream"]};
    border: 1px solid #dfe7df;
    border-radius: 10px;
    box-shadow: none;
    padding: 16px 22px 8px 22px;
    margin-bottom: 16px;
}}
.st-key-vbfilter label, .st-key-vbfilter p {{
    color: {p["ink"]} !important;
    font-size: 15px !important;
}}

.vb-pest-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
}}
.vb-pest {{
    background: {p["cream"]};
    border: 1px solid #dfe7df;
    border-radius: 8px;
    box-shadow: none;
    padding: 16px 18px;
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
    font-size: 20px;
    color: {p["leaf_dark"]};
}}
.vb-pest-crop {{
    font-size: 13px;
    background: #e5f4db;
    border: 1px solid #b4dca0;
    border-radius: 999px;
    padding: 2px 10px;
    color: #3f6b2f;
    white-space: nowrap;
}}
.vb-pest-product {{
    font-size: 14px;
    color: #6b7d66;
    margin: 2px 0 8px 0;
    line-height: 1.55;
}}
.vb-pest-table {{ margin-top: 10px; }}
.vb-pest-table td {{ padding: 6px 9px; font-size: 15px; }}
.vb-pest-table .vb-cat {{ width: 38%; color: #7a8f76; }}

.vb-tag.vb-tox-low {{ background: #e5f4db; border-color: #b4dca0; color: #3f6b2f; }}
.vb-tag.vb-tox-mid {{ background: #fff2d4; border-color: #ecc878; color: #8a6318; }}
.vb-tag.vb-tox-high {{ background: #ffe0da; border-color: #ec8a78; color: #96331c; }}

.vb-warn {{
    margin-top: 8px;
    background: #ffe8e2;
    border: 1px solid #ec8a78;
    border-radius: 12px;
    padding: 8px 11px;
    font-size: 14px;
    color: #96331c;
    line-height: 1.65;
}}
.vb-warn a {{ color: #0b6cbc; }}

/* ---------- 지역별 정보 ---------- */
.vb-region-extremes {{
    margin-top: 12px;
    font-size: 15px;
    color: #6b7d66;
    line-height: 1.85;
}}
.stAltairChart, [data-testid="stVegaLiteChart"] {{
    background: {p["cream"]};
    border: 2px solid #e4efdd;
    border-radius: 8px;
    box-shadow: none;
    padding: 16px 18px;
    margin-bottom: 16px;
}}

/* ---------- 홈(랜딩) 섹션 ---------- */
[class*="st-key-home_feature_"] {{
    border: 1px solid rgba(47, 83, 55, .10);
    border-radius: 24px;
    box-shadow: 0 8px 24px rgba(31, 55, 36, .07);
    padding: 28px 30px 32px;
    margin-bottom: 34px;
}}
.st-key-home_feature_crop {{
    background: linear-gradient(145deg, #fbfef9 0%, #edf7e9 100%);
}}
.st-key-home_feature_pest {{
    background: linear-gradient(145deg, #fffdf8 0%, #f8f1df 100%);
}}
.st-key-home_feature_pesticide {{
    background: linear-gradient(145deg, #fbfdff 0%, #eaf4f5 100%);
}}
.st-key-home_feature_region {{
    background: linear-gradient(145deg, #fdfcff 0%, #f0eef8 100%);
}}
.vb-sec-head {{
    text-align: center;
    margin: 6px 0 20px 0;
}}
.vb-sec-title {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 34px;
    color: {p["ink"]};
    letter-spacing: -.5px;
}}
.vb-sec-accent {{ color: {p["leaf_dark"]}; }}
.vb-sec-sub {{
    color: #6b8569;
    font-size: 18px;
    margin-top: 8px;
}}
.vb-sec-gap {{ height: 40px; }}

.vb-home-card {{
    background: {p["cream"]};
    border: 1px solid #dfe7df;
    border-radius: 10px;
    box-shadow: none;
    padding: 20px 22px;
    margin-bottom: 8px;
    height: 100%;
}}
.vb-home-head {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 24px;
    color: {p["leaf_dark"]};
    display: flex;
    align-items: center;
    gap: 8px;
    padding-bottom: 12px;
    border-bottom: 2px solid #eef4e9;
    margin-bottom: 14px;
}}
.vb-home-metrics {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
}}
.vb-home-metrics-wide {{
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
}}
.vb-home-metric {{
    background: #f7faf7;
    border: 1px solid #dfe7df;
    border-radius: 8px;
    padding: 12px 8px;
    text-align: center;
}}
.vb-home-metric .l {{
    font-size: 14px;
    color: #7a8f76;
    margin-bottom: 4px;
}}
.vb-home-metric .v {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 23px;
    color: {p["leaf_dark"]};
    line-height: 1.2;
}}

/* 홈 '자세히 알아보기' 버튼: 짙은 알약 버튼 */
[class*="st-key-vbgo_"] {{ margin-top: 14px; }}
[class*="st-key-vbgo_"] .stButton > button {{
    background: #26362c;
    color: #ffffff;
    font-size: 18px;
    padding: .6rem 1.2rem;
    box-shadow: none;
}}
[class*="st-key-vbgo_"] .stButton > button:hover {{
    background: {p["leaf_dark"]};
    color: #ffffff;
    box-shadow: none;
}}

/* ---------- 준비중 안내 카드 ---------- */
.vb-soon-card {{
    background: {p["cream"]};
    border: 1px solid #dfe7df;
    border-radius: 10px;
    padding: 42px 26px;
    text-align: center;
    color: {p["ink"]};
    box-shadow: none;
}}
.vb-soon-card img {{ width: 112px; margin-bottom: 12px; }}
.vb-soon-card h3 {{
    font-family: '{TITLE_FONT_FAMILY}', sans-serif;
    font-size: 22px;
    color: {p["leaf_dark"]};
    margin: 0 0 6px 0;
}}

@media (max-width: 760px) {{
    .block-container {{
        padding-top: 4.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }}
    .vb-home-hero {{
        min-height: 500px;
        aspect-ratio: auto;
        align-items: flex-end;
        border-radius: 22px 22px 56px 22px;
        background-position: 56% center;
        margin-bottom: 48px;
    }}
    .vb-home-hero::before {{
        background: linear-gradient(0deg, rgba(15,37,20,.92) 0%, rgba(20,47,25,.55) 58%, rgba(20,47,25,.08) 100%);
    }}
    .vb-home-hero-content {{
        width: 100%;
        padding: 38px 28px;
    }}
    .vb-home-hero-title {{ font-size: 36px; }}
    .vb-home-hero-copy {{ font-size: 16px; }}
}}
</style>
"""


MENU_ITEMS: tuple[tuple[str, str, bool], ...] = (
    ("home", "🏠 홈", True),
    ("chat", "🌱 물어보기", True),
    ("crop", "🥬 작물 도감", True),
    ("pest", "🐛 병해충 도감", True),
    ("pesticide", "🧪 농약 검색", True),
    ("region", "🗺️ 지역별 정보", True),
)


def render_board_html() -> str:
    """로고가 들어간 상단 간판을 만듭니다. 메뉴는 아래 버튼 줄이 담당합니다."""

    uri = mascot_data_uri()
    badge = f"<img class='vb-logo-mark' src='{uri}' alt='' />" if uri else ""
    return f"""
<div class="vb-board vb-banner">
    <a class="vb-logo-link" href="?nav=home" target="_self" aria-label="홈으로 이동">{badge}</a>
    <div class="vb-logo"><span class="vb-logo-leaf">농부의</span> 길잡이</div>
</div>
"""


def render_float_ask_html() -> str:
    """오른쪽 가장자리에 고정되는 '농업정보 물어보기' 탭을 만듭니다."""

    return """
<a class="vb-float-ask" href="?nav=chat" target="_self">
    <div class="vb-float-ic">🌱</div>
    <div class="vb-float-tx">농업정보<br>물어보기</div>
</a>
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
