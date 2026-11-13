"""주장과 검색 근거가 연결된 구조화 답변 모델입니다."""
from __future__ import annotations
from pydantic import BaseModel, Field

class Citation(BaseModel):
    """하나의 주장에 연결된 인용입니다."""
    evidence_id : str = Field(
        description="주장을 뒷받침하는 검색 근거 ID"
    )
    quote: str = Field(
        description="해당 근거 원문에서 그대로 복사한 인용 구절"
    )

class Claim(BaseModel):
    """검증 가능한 하나의 주요 주장"""
    text: str = Field(
        description="검색 근거로 확인 가능한 하나의 주요 주장"
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="이 주장을 뒷받침하는 인용 목록"
    )

class CitedAnswer(BaseModel):
    """답변 본문과 주장별 인용을 포함한 LLM 출력"""
    answer : str = Field(
        description="사용자에게 보여줄 전체 한국어 답변"
    )
    claims : list[Claim] = Field(
        default_factory=list,
        description="답변에 포함된 검증 가능한 주요 주장"
    )