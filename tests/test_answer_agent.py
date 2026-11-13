from retrievers.hybrid_retriever import build_context
from search_result import SearchResult
from supervisor.answer_agent import ANSWER_MODEL, AnswerAgent


def test_answer_agent_fallback_preserves_graph_and_source_evidence():
    result = SearchResult(
        source="graph",
        text="마늘 → 대서마늘 → 남부 지역",
        evidence_ids=["EVD_1"],
        source_ids=["SRC_1"],
        chunk_ids=["CHK_1"],
        path=["마늘", "대서마늘", "남부 지역"],
        relation_types=["HAS_VARIETY", "SUITABLE_FOR_REGION"],
    )

    answer = AnswerAgent().generate("마늘의 적합 지역은?", [result])

    assert "EVD_1" in answer
    assert "SRC_1" in answer
    assert "CHK_1" in answer
    assert "마늘 → 대서마늘 → 남부 지역" in answer


def test_answer_agent_uses_injected_model():
    class FakeModel:
        def invoke(self, prompt):
            assert "EVD_1" in prompt
            return type("Response", (), {"content": "근거 기반 답변"})()

    result = SearchResult(source="vector", text="공식 문서", evidence_ids=["EVD_1"])
    answer = AnswerAgent(FakeModel()).generate("질문", [result])

    assert answer == "근거 기반 답변"


def test_answer_agent_uses_fixed_answer_model(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "langchain_openai.ChatOpenAI",
        FakeChatOpenAI,
    )

    AnswerAgent.from_env()

    assert captured["model"] == ANSWER_MODEL
