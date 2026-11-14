from types import SimpleNamespace

from src.retrievers.router import JevQuestionRouter, QuestionRouter


def test_router_selects_graph_for_relationship_question():
    decision = QuestionRouter().decide("마늘 품종별 적합 지역은 어디인가?")
    assert decision.route == "GRAPH"
    assert decision.crop == "마늘"


def test_router_selects_graph_for_variety_group_region_question():
    decision = QuestionRouter().decide(
        "중만생종 양파는 어느 지역에서 재배하기에 적합한가요?"
    )

    assert decision.route == "GRAPH"
    assert decision.crop == "양파"


def test_router_selects_hybrid_for_environment_and_management_question():
    decision = QuestionRouter().decide("저온다습한 환경에서 어떤 병해가 생기고 어떻게 예방하나요?")
    assert decision.route == "HYBRID"


def test_router_withholds_unsupported_numeric_pesticide_question():
    decision = QuestionRouter().decide("이 농약의 희석 배율은 얼마인가요?")
    assert decision.route == "GRAPH"
    assert decision.warnings


def test_router_returns_graph_decision_without_running_retriever():
    router = QuestionRouter()
    decision = router.route_query("마늘 품종별 적합 지역은 어디인가?")

    assert decision.route == "GRAPH"
    assert decision.crop == "마늘"


def test_jev_router_falls_back_to_rules_without_classifier():
    decision = JevQuestionRouter().route_query("마늘 품종별 적합 지역은 어디인가?")
    assert decision.route == "GRAPH"


def test_jev_router_uses_llm_only_for_graph_search_conditions():
    class Classifier:
        def invoke(self, *_args, **_kwargs):
            return SimpleNamespace(
                choices={"route": SimpleNamespace(choice="GRAPH")}
            )

    class Analyzer:
        def __init__(self):
            self.questions = []

        def analyze(self, question):
            self.questions.append(question)
            return SimpleNamespace(
                crop="양파",
                entities=["노균병"],
                relation_types=["AFFECTED_BY_DISEASE", "방제"],
                max_hops=2,
                needs_official_evidence=True,
            )

    analyzer = Analyzer()
    decision = JevQuestionRouter(Classifier(), analyzer).route_query(
        "양파 노균병에 등록된 약제는 무엇인가요?"
    )

    assert decision.route == "GRAPH"
    assert decision.crop == "양파"
    assert decision.relation_types == ["AFFECTED_BY_DISEASE"]
    assert "일부 관계 타입" in decision.warnings[0]
    assert analyzer.questions


def test_scope_gate_rejects_question_without_domain_signal():
    decision = QuestionRouter().decide("파이썬으로 퀵정렬 알고리즘을 구현해 주세요.")
    assert decision.route == "REJECT"
    assert decision.reason


def test_scope_gate_rejects_creative_request_mentioning_crop():
    decision = QuestionRouter().decide("마늘을 소재로 한 공포 영화의 시놉시스를 작성해 주세요.")
    assert decision.route == "REJECT"


def test_scope_gate_rejects_investment_request_mentioning_agriculture():
    decision = QuestionRouter().decide("농업 관련 주식 중 단기간에 수익이 날 종목을 추천해 주세요.")
    assert decision.route == "REJECT"


def test_scope_gate_rejects_marketing_copy_request_mentioning_crop():
    decision = QuestionRouter().decide("고추 사진을 이용해 SNS 광고 포스터 문구를 만들어 주세요.")
    assert decision.route == "REJECT"


def test_scope_gate_rejects_legal_question_with_explanation_verb():
    # "설명"은 VECTOR 신호지만 범위 게이트가 먼저 적용돼야 합니다.
    decision = QuestionRouter().decide("전세 계약을 해지할 때 필요한 법적 절차를 설명해 주세요.")
    assert decision.route == "REJECT"


def test_scope_gate_rejects_empty_question():
    assert QuestionRouter().decide("   ").route == "REJECT"


def test_scope_gate_passes_in_scope_question():
    assert QuestionRouter().scope_gate("마늘에 비료를 줄 때는 어떤 절차를 따르는 것이 바람직한가요?") is None


def test_scope_gate_keeps_in_scope_question_without_crop_name():
    decision = QuestionRouter().decide(
        "토양이 장기간 과습하거나 침수된 밭에서 발생하기 쉬운 병원균은 무엇인가요?"
    )
    assert decision.route != "REJECT"


def test_scope_gate_does_not_reject_in_scope_pesticide_question():
    decision = QuestionRouter().decide(
        "배추에 등록된 플루아지남 분제는 어떤 병해를 방제하는 데 사용되나요?"
    )
    assert decision.route != "REJECT"


def test_jev_router_rejects_out_of_scope_without_calling_classifier():
    class ExplodingClassifier:
        def invoke(self, *args, **kwargs):  # pragma: no cover - 호출되면 실패해야 합니다.
            raise AssertionError("범위 밖 질문은 분류기를 호출하지 않아야 합니다.")

    decision = JevQuestionRouter(ExplodingClassifier()).route_query(
        "서울에서 데이트하기 좋은 맛집을 추천해 주세요."
    )
    assert decision.route == "REJECT"


def test_scope_gate_keeps_soil_amendment_question_in_scope():
    decision = QuestionRouter().decide("우리 밭 흙이 산성인데 석회를 얼마나 넣어야 하나요?")
    assert decision.route != "REJECT"


def test_scope_gate_rejects_market_price_question():
    # 말뭉치에 시세 자료가 없어 작물 이름이 있어도 범위 밖입니다.
    decision = QuestionRouter().decide("배추 도매 가격이 얼마인가요?")
    assert decision.route == "REJECT"
