from __future__ import annotations

from scripts import run_supervisor


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
    checker = object()
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
    monkeypatch.setattr(
        run_supervisor,
        "build_citation_support_checker",
        lambda: checker,
        raising=False,
    )

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
    assert captured["citation_support_checker"] is checker
    assert captured["evidence_lookup"](["EVD_1"]) == {"EVD_1": {"text": "EVD_1"}}
    assert lookup_calls == [(driver, ["EVD_1"], "farm")]

