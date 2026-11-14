
"""농업 GraphRAG 입력 안전성 검증 레일."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InputDecision:
    intent: str
    allowed: bool
    pesticide_use_requested: bool | None
    reason: str
    response: str = ""


PESTICIDE = re.compile(
    r"농약|살충제|살균제|제초제|약제|방제약|"
    r"희석배수|살포량|사용횟수|수확\s*전\s*사용",
    re.IGNORECASE,
)

EXPOSURE = re.compile(
    r"(농약|살충제|살균제|제초제|약제).{0,35}"
    r"(먹었|마셨|삼켰|흡입했|들이마셨|"
    r"눈에\s*들어|피부에\s*묻|중독|노출)"
    r"|"
    r"(먹었|마셨|삼켰|흡입했|들이마셨|"
    r"눈에\s*들어|중독).{0,35}"
    r"(농약|살충제|살균제|제초제|약제)",
    re.IGNORECASE,
)

INJECTION = re.compile(
    r"(이전|기존|시스템|개발자).{0,15}"
    r"(규칙|지시|프롬프트).{0,15}"
    r"(무시|삭제|해제|덮어)"
    r"|"
    r"(안전장치|가드레일|guardrail).{0,15}"
    r"(우회|해제|무시)"
    r"|"
    r"ignore\s+(all\s+)?(previous|system)\s+instructions",
    re.IGNORECASE,
)

DANGEROUS = re.compile(
    r"(농약|살충제|살균제|제초제|약제).{0,50}"
    r"(사람을\s*해치|동물을\s*죽이|음식물\s*오염|"
    r"치사량|독살)"
    r"|"
    r"(치사량|독살|음식물\s*오염).{0,50}"
    r"(농약|살충제|살균제|제초제|약제)"
    r"|"
    r"(두\s*배|2\s*배|고농도|권장량\s*초과|"
    r"진하게\s*희석).{0,30}"
    r"(살포|뿌리|사용|섞|효과)"
    r"|"
    r"(살포|뿌리|사용|섞).{0,30}"
    r"(두\s*배|2\s*배|고농도|권장량\s*초과)",
    re.IGNORECASE,
)

RECOMMENDATION = re.compile(
    r"추천|골라\s*줘|선택해\s*줘|"
    r"사용해도|써도\s*돼|살포해도|"
    r"지금\s*사용|어떤\s*농약.{0,20}(써|사용|뿌리)|"
    r"무슨\s*약.{0,20}(써|사용|뿌리)",
    re.IGNORECASE,
)

LOOKUP = re.compile(
    r"등록|조회|정보|기록|기준|라벨|"
    r"희석배수|사용량|사용횟수|수확\s*전|"
    r"보관|폐기|보호구|작용기작|제형|"
    r"무엇|뭐야|알려|설명",
    re.IGNORECASE,
)

EMERGENCY_RESPONSE = (
    "농약 노출 또는 중독이 의심됩니다. "
    "즉시 추가 노출을 피하고 119 또는 의료기관에 "
    "도움을 요청하세요. 제품명과 제품 용기·라벨을 "
    "가능하면 의료진에게 알려주세요. "
    "임의로 구토를 유도하지 마세요."
)

BLOCK_RESPONSE = (
    "위험하거나 허용되지 않은 농약 사용 방법은 "
    "안내할 수 없습니다. 공식 등록정보와 "
    "제품 라벨의 안전사용기준을 확인해 주세요."
)

INJECTION_RESPONSE = (
    "안전 규칙을 변경하거나 우회하는 요청은 "
    "처리할 수 없습니다."
)

UNCERTAIN_RESPONSE = (
    "농약 정보 조회인지 실제 사용 추천인지 "
    "확인하기 어렵습니다. 조회하려는 정보나 "
    "사용 목적을 구체적으로 알려주세요."
)


def classify_input(question: str) -> InputDecision:
    """사용자 질문을 안전성 및 농약 사용 의도별로 분류한다."""

    if not isinstance(question, str) or not question.strip():
        return InputDecision(
            intent="INVALID",
            allowed=False,
            pesticide_use_requested=None,
            reason="질문이 비어 있거나 형식이 올바르지 않음",
            response="질문을 입력해 주세요.",
        )

    question = question.strip()

    # 1. 응급 상황 최우선
    if EXPOSURE.search(question):
        return InputDecision(
            intent="EXPOSURE",
            allowed=False,
            pesticide_use_requested=None,
            reason="농약 노출 또는 중독 의심",
            response=EMERGENCY_RESPONSE,
        )

    # 2. 직접 프롬프트 인젝션
    if INJECTION.search(question):
        return InputDecision(
            intent="INJECTION",
            allowed=False,
            pesticide_use_requested=None,
            reason="안전 규칙 우회 지시 탐지",
            response=INJECTION_RESPONSE,
        )

    # 3. 위험한 농약 사용 요청
    if DANGEROUS.search(question):
        return InputDecision(
            intent="DANGEROUS_USE",
            allowed=False,
            pesticide_use_requested=True,
            reason="위험한 농약 사용 요청 탐지",
            response=BLOCK_RESPONSE,
        )

    # 4. 일반 농업 질문
    if not PESTICIDE.search(question):
        return InputDecision(
            intent="GENERAL",
            allowed=True,
            pesticide_use_requested=False,
            reason="일반 농업 질문",
        )

    # 5. 현재 농약 사용 추천
    if RECOMMENDATION.search(question):
        return InputDecision(
            intent="PESTICIDE_RECOMMENDATION",
            allowed=True,
            pesticide_use_requested=True,
            reason="현재 농약 사용 추천 요청",
        )

    # 6. 농약 등록정보 및 안전정보 조회
    if LOOKUP.search(question):
        return InputDecision(
            intent="PESTICIDE_LOOKUP",
            allowed=True,
            pesticide_use_requested=False,
            reason="농약 정보 조회 요청",
        )

    # 7. 농약 관련 의도가 불명확한 질문
    return InputDecision(
        intent="UNCERTAIN",
        allowed=False,
        pesticide_use_requested=None,
        reason="농약 요청 의도 불명확",
        response=UNCERTAIN_RESPONSE,
    )


def input_safety_node(state: dict[str, Any]) -> dict[str, Any]:
    """Supervisor가 검색 전에 호출하는 입력 레일 노드."""

    decision = classify_input(state.get("question", ""))

    update: dict[str, Any] = {
        "input_intent": decision.intent,
        "pesticide_use_requested": decision.pesticide_use_requested,
        "guardrail_reason": decision.reason,
    }

    if decision.allowed:
        return update

    is_emergency = decision.intent == "EXPOSURE"

    update.update({
        "guardrail_decision": (
            "BLOCK"
            if decision.intent in ("DANGEROUS_USE", "INJECTION")
            else "FALLBACK"
        ),
        "final_status": (
            "EMERGENCY" if is_emergency else
            "BLOCKED" if decision.intent in (
                "DANGEROUS_USE", "INJECTION"
            ) else "WITHHELD"
        ),
        "final_answer": decision.response,
    })

    return update


def after_input_safety(state: dict[str, Any]) -> str:
    """허용된 질문만 기존 Supervisor로 전달한다."""

    intent = state.get("input_intent", "")

    if intent in (
        "GENERAL",
        "PESTICIDE_LOOKUP",
        "PESTICIDE_RECOMMENDATION",
    ):
        return "route"

    return "end"
