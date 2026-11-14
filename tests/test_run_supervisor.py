from __future__ import annotations

from scripts import run_supervisor


class _FakeAnswerAgent:
    """AnswerAgent.from_env() 대신 쓰는 가짜 에이전트입니다.

    build_app() 이 인용 답변 생성기와 주장 지원 판정기를 AnswerAgent 의 메서드로
    넘기는지만 보면 되므로, 실제 LLM 설정(.env)에는 의존하지 않습니다.
    """

    @classmethod
    def from_env(cls) -> "_FakeAnswerAgent":
        return cls()

    def generate_cited(self, *args, **kwargs):  # pragma: no cover - 배선만 확인
        raise AssertionError("테스트에서 호출하지 않습니다.")

    def review_support(self, *args, **kwargs):  # pragma: no cover - 배선만 확인
        raise AssertionError("테스트에서 호출하지 않습니다.")


class _FakeDriver:
    def __init__(self) -> None:
        self.verified = False

    def verify_connectivity(self) -> None:
        self.verified = True

    def close(self) -> None:
        pass


def test_build_app_wires_evidence_lookup_and_citation_checker(monkeypatch):
    driver = _FakeDriver()
    captured: dict = {}
    lookup_calls: list[tuple[object, list[str], str]] = []

    monkeypatch.setattr(
        run_supervisor,
        "load_config",
        lambda: {
            "uri": "bolt://test",
            "username": "neo4j",
            "password": "secret",
            "database": "farm",
        },
    )
    monkeypatch.setattr(
        run_supervisor.GraphDatabase,
        "driver",
        lambda *args, **kwargs: driver,
    )
    monkeypatch.setattr(
        run_supervisor,
        "GraphRetriever",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        run_supervisor,
        "VectorIndex",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(run_supervisor, "get_embedder", lambda: object())
    monkeypatch.setattr(run_supervisor, "AnswerAgent", _FakeAnswerAgent)

    def fake_lookup(bound_driver, ids, database="neo4j"):
        lookup_calls.append((bound_driver, ids, database))
        return {evidence_id: {"text": evidence_id} for evidence_id in ids}

    monkeypatch.setattr(
        run_supervisor,
        "neo4j_lookup",
        fake_lookup,
        raising=False,
    )

    def fake_build_supervisor(**kwargs):
        captured.update(kwargs)
        return "compiled-app"

    monkeypatch.setattr(run_supervisor, "build_supervisor", fake_build_supervisor)

    app, returned_driver = run_supervisor.build_app()

    assert app == "compiled-app"
    assert returned_driver is driver
    assert driver.verified is True
    checker = captured["citation_support_checker"]
    generator = captured["cited_answer_generator"]
    assert isinstance(checker.__self__, _FakeAnswerAgent)
    assert checker.__func__ is _FakeAnswerAgent.review_support
    assert generator.__self__ is checker.__self__
    assert generator.__func__ is _FakeAnswerAgent.generate_cited
    assert captured["evidence_lookup"](["EVD_1"]) == {"EVD_1": {"text": "EVD_1"}}
    assert lookup_calls == [(driver, ["EVD_1"], "farm")]

