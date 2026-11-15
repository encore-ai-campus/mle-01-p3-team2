"""조회와 추천의 필수 조건 분리에 대한 회귀 테스트.

정보 조회는 작물 + 병해충 기준 목록 검증,
추천은 제품명 + 작물 + 병해충 엄격 검증으로 분리됐는지 확인한다.
실제 농약 데이터가 아닌 가상 기록만 사용한다.
"""

import pytest

from src.guardrails.execution_adapter import execution_node
from src.guardrails.execution_rail import list_registered_usages
from src.guardrails.output_usage_validator import (
    validate_pesticide_output,
)
from src.guardrails.pesticide_identity import (
    extract_pesticide_identity,
)


CROP = "시험작물"
PEST = "시험해충"


def record(product, *, dilution="1,000배", use_count="3회"):
    """가상 등록 기록."""
    return {
        "product_name": f"{product} 품목",
        "brand": product,
        "crop": CROP,
        "target_pest": PEST,
        "dilution": dilution,
        "max_applications": use_count,
        "preharvest_interval": "수확 7일 전",
        "registration_status": "등록",
        "verification_status": "PRELIMINARY_PASS",
    }


RECORDS = [
    record("가상약제A"),
    record("가상약제B", dilution="2,000배", use_count="2회"),
]


def path(*nodes):
    """경로 노드만 담은 가상 검색 결과."""
    return {
        "path_nodes": [
            {
                "entity_id": f"e{index}",
                "name": name,
                "entity_type": kind,
            }
            for index, (name, kind) in enumerate(nodes)
        ]
    }


SCOPE_RESULTS = [
    path(
        (CROP, "Crop"),
        ("가상약제A", "PesticideUse"),
        (PEST, "Insect"),
    ),
    path(
        (CROP, "Crop"),
        ("가상약제B", "PesticideUse"),
        (PEST, "Insect"),
    ),
]

LIST_QUESTION = (
    f"{CROP}에서 {PEST} 방제에 등록된 농약 품목은 무엇인가요?"
)


def node(question, results, *, intent="PESTICIDE_LOOKUP", **kwargs):
    """Execution Rail 노드를 기본 상태로 실행한다."""
    state = {
        "question": question,
        "results": results,
        "evidence_status": "PASS",
        "input_intent": intent,
    }
    return execution_node(state, **kwargs)


def list_lookup(crop, target_pest):
    return [item.copy() for item in RECORDS]


def single_lookup(product_name, crop, target_pest):
    return [
        item.copy()
        for item in RECORDS
        if product_name in (item["product_name"], item["brand"])
        and item["crop"] == crop
        and item["target_pest"] == target_pest
    ]


# --- 1. 제품명 없는 작물 + 병해충 목록 조회 허용 ---

def test_identity_extracted_from_search_results():
    """검색 결과에서 작물·병해충을 구조적으로 추출한다."""
    identity = extract_pesticide_identity(
        LIST_QUESTION,
        SCOPE_RESULTS,
    )

    assert identity.crop == CROP
    assert identity.target_pest == PEST
    # 질문이 제품을 지목하지 않았으므로 후보만 전달한다.
    assert identity.product_name == ""
    assert identity.product_candidates == [
        "가상약제A",
        "가상약제B",
    ]


def test_lookup_without_product_name_allowed():
    """제품명이 없어도 등록 약제 목록 조회는 보류되지 않는다."""
    update = node(
        LIST_QUESTION,
        SCOPE_RESULTS,
        list_lookup_fn=list_lookup,
    )

    assert update["execution_decision"] == "WARN"
    assert "final_status" not in update
    assert len(update["execution_records"]) == 2
    assert update["crop"] == CROP
    assert update["target_pest"] == PEST
    assert update["warnings"]


def test_lookup_without_matching_scope_withheld():
    """작물·병해충이 일치하는 기록이 없을 때만 보류한다."""
    update = node(
        LIST_QUESTION,
        SCOPE_RESULTS,
        list_lookup_fn=lambda crop, target: [],
    )

    assert update["execution_decision"] == "FALLBACK"
    assert update["final_status"] == "WITHHELD"


def test_lookup_without_crop_or_pest_withheld():
    """작물·병해충을 확인할 수 없는 질문은 보류한다."""
    update = node(
        "등록된 농약 품목이 궁금합니다.",
        [path(("정체불명", "Chunk"))],
        list_lookup_fn=list_lookup,
    )

    assert update["execution_decision"] == "FALLBACK"
    assert "작물·병해충" in update["execution_reason"]


# --- 2. 특정 제품 조회는 3요소 정확 일치 ---

def test_specific_product_lookup_uses_strict_match():
    """질문이 제품을 지목하면 제품명까지 일치 검사한다."""
    update = node(
        f"{CROP}의 {PEST}에 가상약제A의 희석배수는 얼마인가요?",
        SCOPE_RESULTS,
        lookup_fn=single_lookup,
        list_lookup_fn=list_lookup,
    )

    assert update["execution_decision"] == "WARN"
    assert update["product_name"] == "가상약제A"
    assert update["execution_record"]["brand"] == "가상약제A"
    assert len(update["execution_records"]) == 1


def test_specific_product_without_registered_record_withheld():
    """지목한 제품의 등록 기록이 없으면 보류한다."""
    update = node(
        f"{CROP}의 {PEST}에 가상약제A를 쓸 때 희석배수는 얼마인가요?",
        SCOPE_RESULTS,
        lookup_fn=lambda *args: [],
        list_lookup_fn=list_lookup,
    )

    assert update["execution_decision"] == "FALLBACK"
    assert update["final_status"] == "WITHHELD"


# --- 3. 복수 제품 답변은 제품별로 사용조건 검증 ---

def multi_product_answer(first="1,000배", second="2,000배"):
    return (
        f"가상약제A: 희석배수 {first}, 최대 3회 사용.\n"
        f"가상약제B: 희석배수 {second}, 최대 2회 사용."
    )


def test_multi_product_answer_validated_per_product():
    """제품별 수치가 각 기록과 맞으면 통과한다."""
    result = validate_pesticide_output(
        multi_product_answer(),
        None,
        records=RECORDS,
        pesticide_use_requested=True,
    )

    assert result["allowed"] is True
    assert result["decision"] == "PASS"


def test_swapped_product_values_blocked():
    """제품끼리 수치를 뒤바꾼 답변은 차단한다."""
    result = validate_pesticide_output(
        multi_product_answer(first="2,000배", second="1,000배"),
        None,
        records=RECORDS,
        pesticide_use_requested=True,
    )

    assert result["allowed"] is False
    assert result["decision"] == "BLOCK"


# --- 4. 등록정보와 다른 수치는 차단 ---

def test_unregistered_value_blocked():
    """어느 기록에도 없는 수치는 차단한다."""
    result = validate_pesticide_output(
        "가상약제A: 희석배수 500배로 사용하세요.",
        None,
        records=RECORDS,
        pesticide_use_requested=True,
    )

    assert result["allowed"] is False
    assert result["decision"] == "BLOCK"


def test_use_count_alias_compared():
    """max_applications로 저장된 사용횟수도 비교 대상이다."""
    result = validate_pesticide_output(
        "가상약제A는 최대 3회까지 사용할 수 있습니다.",
        None,
        records=[RECORDS[0]],
        pesticide_use_requested=True,
    )

    assert result["allowed"] is True

    wrong = validate_pesticide_output(
        "가상약제A는 최대 9회까지 사용할 수 있습니다.",
        None,
        records=[RECORDS[0]],
        pesticide_use_requested=True,
    )

    assert wrong["decision"] == "BLOCK"


# --- 5. 최신 공식 검증 없는 추천은 계속 차단 ---

def test_recommendation_still_withheld():
    """추천은 제품을 지목해도 최신 검증 전까지 보류한다."""
    update = node(
        f"{CROP}의 {PEST}에 가상약제A를 지금 사용해도 되나요?",
        SCOPE_RESULTS,
        intent="PESTICIDE_RECOMMENDATION",
        lookup_fn=single_lookup,
        list_lookup_fn=list_lookup,
    )

    assert update["execution_decision"] in ("FALLBACK", "BLOCK")
    assert update["final_status"] in ("WITHHELD", "BLOCKED")


def test_recommendation_without_product_not_listed():
    """추천 요청은 목록 조회 경로로 우회되지 않는다."""
    update = node(
        f"{CROP}의 {PEST}에 쓸 농약을 추천해 주세요.",
        SCOPE_RESULTS,
        intent="PESTICIDE_RECOMMENDATION",
        lookup_fn=single_lookup,
        list_lookup_fn=list_lookup,
    )

    assert update["execution_decision"] == "FALLBACK"
    assert "execution_records" not in update


def test_recommendation_output_still_withheld():
    """Output Rail의 추천 보류 정책은 그대로 유지된다."""
    result = validate_pesticide_output(
        multi_product_answer(),
        None,
        records=RECORDS,
        pesticide_use_requested=True,
        recommendation_requested=True,
    )

    assert result["allowed"] is False
    assert result["decision"] == "WITHHELD"


# --- 6. 위험 요청은 계속 차단 ---

@pytest.mark.parametrize(
    "answer",
    [
        "가상약제A 희석액을 더 진하게 만들어 살포하세요.",
        "사용 횟수를 초과해 더 많이 살포해도 됩니다.",
        "등록 상태가 미확인이지만 바로 사용 가능합니다.",
    ],
)
def test_unsafe_answers_blocked(answer):
    """목록 검증 경로에서도 위험 표현은 차단한다."""
    result = validate_pesticide_output(
        answer,
        None,
        records=RECORDS,
        pesticide_use_requested=True,
    )

    assert result["allowed"] is False
    assert result["decision"] == "BLOCK"


# --- 7. 조회 경로의 기록 충돌 처리 ---

def test_conflicting_product_excluded_from_list():
    """사용조건이 충돌하는 제품만 목록에서 제외한다."""
    conflicting = record("가상약제B", dilution="4,000배")

    result = list_registered_usages(
        crop=CROP,
        target_pest=PEST,
        list_lookup_fn=lambda crop, target: [
            *[item.copy() for item in RECORDS],
            conflicting,
        ],
    )

    assert result.allowed is True
    brands = {item["brand"] for item in result.records}
    assert brands == {"가상약제A"}
    assert any("제외" in warning for warning in result.warnings)


def test_duplicate_records_deduplicated():
    """같은 내용의 기록은 한 건으로 합친다."""
    result = list_registered_usages(
        crop=CROP,
        target_pest=PEST,
        list_lookup_fn=lambda crop, target: [
            RECORDS[0].copy(),
            RECORDS[0].copy(),
        ],
    )

    assert len(result.records) == 1


# --- 8. GS012·GS014·GS020·GS061 유형의 불필요한 보류 방지 ---

def test_list_answer_reaches_answered(monkeypatch):
    """목록 조회 답변이 Output Rail에서 보류되지 않는다."""
    from src.guardrails import supervisor_adapter
    from src.guardrails.contracts import ValidationResult

    monkeypatch.setattr(
        supervisor_adapter,
        "validate_request",
        lambda request: ValidationResult(
            allowed=True,
            decision="ALLOW",
            reason="테스트용 근거 검증 통과",
        ),
    )

    execution = node(
        LIST_QUESTION,
        SCOPE_RESULTS,
        list_lookup_fn=list_lookup,
    )

    state = {
        "question": LIST_QUESTION,
        "results": SCOPE_RESULTS,
        "answer": multi_product_answer(),
        "final_answer": multi_product_answer(),
        "input_intent": "PESTICIDE_LOOKUP",
        "pesticide_use_requested": False,
        **execution,
    }

    update = supervisor_adapter.guardrails_node(state)

    assert update["final_status"] == "ANSWERED"
    assert update["warnings"]
