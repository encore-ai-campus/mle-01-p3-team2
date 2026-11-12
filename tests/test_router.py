from src.retrievers.router import JevQuestionRouter, QuestionRouter


def test_router_selects_graph_for_relationship_question():
    decision = QuestionRouter().decide("마늘 품종별 적합 지역은 어디인가?")
    assert decision.route == "GRAPH"
    assert decision.crop == "마늘"


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
