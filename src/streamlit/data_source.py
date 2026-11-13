"""Streamlit 프로세스에서 그래프 데이터를 한 번만 읽어 재사용합니다."""

from __future__ import annotations

import streamlit as st

from src.streamlit.crop_data import Graph, load_graph


@st.cache_resource(show_spinner="농업 데이터를 불러오는 중...")
def get_graph() -> Graph:
    """data/processed JSONL에서 만든 그래프를 캐시해 돌려줍니다."""

    return load_graph()
