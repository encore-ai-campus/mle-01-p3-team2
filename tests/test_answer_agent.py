from src.retrievers.search_result import SearchResult
from src.retrievers.answer import (Citation, CitedAnswer, Claim, Evidence, render_cited_answer)
from src.supervisor.answer_agent import ANSWER_MODEL, AnswerAgent
from src.supervisor.citation_verifier import AnswerReview

def _evidence() -> Evidence:
    return Evidence(
        evidence_id="EVD_1",
        document_id="DOC_1",
        title="배추 농약 등록 정보",
        url="https://example.com/pesticide",
        quote=(
            "작물=배추; 병해충=무름병; "
            "사용적기=정식전; 최대 사용횟수=1회"
        ),
        relation_ids=["REL_1"],
        graph_path=["배추", "무름병", "등록약제"],
        source="graph",
        chunk_ids=["CHK_1"],
    )

def test_answer_agent_fallback_preserves_graph_and_source_evidence():
    result = SearchResult(
        source="graph",
        text="마늘 → 대서마늘 → 남부 지역",
        evidence_ids=["EVD_1", "EVD_2"],
        source_ids=["SRC_1", "SRC_2"],
        chunk_ids=["CHK_1", "CHK_2"],
        path=["마늘", "대서마늘", "남부 지역"],
        relation_types=[
            "HAS_VARIETY",
            "SUITABLE_FOR_REGION",
        ],
        hop_count=2,
        path_edges=[
            {
                "hop": 1,
                "relation_id": "REL_1",
                "relation_type": "HAS_VARIETY",
                "from_entity_id": "ENT_GARLIC",
                "to_entity_id": "ENT_VARIETY",
                "evidence_ids": ["EVD_1"],
                "chunk_ids": ["CHK_1"],
                "source_ids": ["SRC_1"],
                "original_chunks": [
                    {
                        "chunk_id": "CHK_1",
                        "evidence_id": "EVD_1",
                        "source_id": "SRC_1",
                        "text": "대서마늘은 마늘의 한 품종이다.",
                    }
                ],
            },
            {
                "hop": 2,
                "relation_id": "REL_2",
                "relation_type": "SUITABLE_FOR_REGION",
                "from_entity_id": "ENT_VARIETY",
                "to_entity_id": "ENT_SOUTH",
                "evidence_ids": ["EVD_2"],
                "chunk_ids": ["CHK_2"],
                "source_ids": ["SRC_2"],
                "original_chunks": [
                    {
                        "chunk_id": "CHK_2",
                        "evidence_id": "EVD_2",
                        "source_id": "SRC_2",
                        "text": "대서마늘은 남부 지역 재배에 적합하다.",
                    }
                ],
            },
        ],
    )

    answer = AnswerAgent().generate(
        "마늘의 적합 지역은?",
        [result],
    )

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

def test_answer_agent_generates_structured_cited_answer():
    expected = CitedAnswer(
        answer="해당 약제는 배추 무름병에 정식 전에 사용합니다.",
        claims=[
            Claim(
                text="해당 약제의 사용 적기는 정식 전이다.",
                citations=[
                    Citation(
                        evidence_id="EVD_1",
                        quote="사용적기=정식전",
                    )
                ],
            )
        ],
    )

    class FakeStructuredModel:
        def invoke(self, prompt):
            assert "evidence_id: EVD_1" in prompt
            assert "사용적기=정식전" in prompt
            return expected

    class FakeModel:
        def with_structured_output(self, schema):
            assert schema is CitedAnswer
            return FakeStructuredModel()

    result = AnswerAgent(FakeModel()).generate_cited(
        "배추 무름병 약제는 언제 사용하나요?",
        [_evidence()],
    )

    assert result == expected
    assert result.claims[0].citations[0].evidence_id == "EVD_1"


def test_answer_agent_reviews_claim_support_with_structured_output():
    expected = AnswerReview(
        supported=True,
        reason="인용 원문이 주장을 직접 뒷받침합니다.",
    )

    class FakeStructuredModel:
        def invoke(self, prompt):
            assert "주장: 중만생종은 중남부 내륙에 적합하다." in prompt
            assert "EVD_1" in prompt
            return expected

    class FakeModel:
        def with_structured_output(self, schema):
            assert schema is AnswerReview
            return FakeStructuredModel()

    review = AnswerAgent(FakeModel()).review_support(
        "중만생종은 중남부 내륙에 적합하다.",
        [{
            "evidence_id": "EVD_1",
            "citation_quote": "중만생종의 재배 가능 지역은 중남부 내륙이다.",
            "source_text": "중만생종의 재배 가능 지역은 중남부 내륙이다.",
        }],
    )

    assert review == expected

def test_answer_agent_returns_empty_claims_without_evidence():
    class FailIfCalledModel:
        def with_structured_output(self, schema):
            raise AssertionError("근거가 없으면 LLM을 호출하면 안 됩니다.")

    result = AnswerAgent(FailIfCalledModel()).generate_cited("질문", [])

    assert result.claims == []
    assert "근거가 없습니다" in result.answer

def test_answer_agent_without_model_does_not_create_claims():
    result = AnswerAgent().generate_cited("질문", [_evidence()])

    assert result.claims == []
    assert "답변을 보류" in result.answer

def test_render_cited_answer_includes_title_url_and_quote():
    answer = CitedAnswer(
        answer="해당 약제는 정식 전에 사용합니다.",
        claims=[Claim(
                text="사용 적기는 정식 전이다.",
                citations=[Citation(
                        evidence_id="EVD_1",
                        quote="사용적기=정식전",
                    )
                ],
            )
        ],
    )

    rendered = render_cited_answer(answer, [_evidence()])

    assert "[1] 배추 농약 등록 정보" in rendered
    assert "https://example.com/pesticide" in rendered
    assert "작물=배추; 병해충=무름병" in rendered
    assert "배추 → 무름병 → 등록약제" in rendered

def test_render_cited_answer_deduplicates_same_evidence_id():
    answer = CitedAnswer(
        answer="답변",
        claims=[
            Claim(
                text="주장 1",
                citations=[
                    Citation(
                        evidence_id="EVD_1",
                        quote="사용적기=정식전",
                    )
                ],
            ),
            Claim(
                text="주장 2",
                citations=[
                    Citation(
                        evidence_id="EVD_1",
                        quote="최대 사용횟수=1회",
                    )
                ],
            ),
        ],
    )

    rendered = render_cited_answer(answer, [_evidence()])

    assert rendered.count("[1] 배추 농약 등록 정보") == 1
