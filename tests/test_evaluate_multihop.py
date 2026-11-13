import csv
import json

import pytest

from scripts.evaluation.evaluate_multihop import (
    FAILURE,
    FULL_SUCCESS,
    PARTIAL_SUCCESS,
    SKIPPED_REJECT,
    GraphRetrievalObserver,
    NullRetrievalObserver,
    OracleRetrievalObserver,
    RetrievalOutcome,
    evaluate_case,
    evaluate_multihop,
    expected_hop_of,
    format_report,
    load_relation_types,
    main,
    write_details,
    RetrievalOutcome,
    SupervisorRetrievalObserver,
    evaluate_case,
)
from src.retrievers.search_result import SearchResult


def _graph_case(case_id="A", *, strategy="graph", hops=2, question_type="multi_hop"):
    return {
        "case_id": case_id,
        "question": "배추 질문",
        "query_strategy": strategy,
        "question_type": question_type,
        "difficulty": "medium",
        "status": "READY",
        "expected_facts": ["사실"],
        "expected_entity_ids": [f"ENT_{i}" for i in range(hops + 1)],
        "expected_relation_ids": [f"REL_{i}" for i in range(hops)],
        "expected_evidence_ids": ["EVD_1"],
        "expected_source_chunk_ids": ["CHK_1"],
    }


def _vector_case(case_id="V"):
    return {
        "case_id": case_id,
        "question": "재배 방법 질문",
        "query_strategy": "vector",
        "question_type": "fact",
        "difficulty": "easy",
        "status": "READY",
        "expected_facts": ["사실"],
        "expected_entity_ids": [],
        "expected_relation_ids": [],
        "expected_evidence_ids": ["EVD_1"],
        "expected_source_chunk_ids": ["CHK_1"],
    }


def _reject_case(case_id="R"):
    return {
        "case_id": case_id,
        "question": "범위 밖 질문",
        "query_strategy": "reject",
        "question_type": "out_of_scope",
        "difficulty": "easy",
        "status": "READY",
        "expected_facts": [],
        "expected_entity_ids": [],
        "expected_relation_ids": [],
        "expected_evidence_ids": [],
        "expected_source_chunk_ids": [],
    }


def _path_result(*, node_ids, relation_ids, evidence_ids=(), chunk_ids=()):
    return SearchResult(
        source="graph",
        text="path",
        evidence_ids=list(evidence_ids),
        chunk_ids=list(chunk_ids),
        metadata={
            "node_ids": list(node_ids),
            "relation_ids": list(relation_ids),
            "hop_count": len(relation_ids),
        },
    )


class StubObserver:
    def __init__(self, outcome_by_case):
        self.outcome_by_case = outcome_by_case

    def retrieve(self, case):
        return self.outcome_by_case.get(case["case_id"], RetrievalOutcome())


def test_expected_hop_is_relation_id_count():
    assert expected_hop_of(_graph_case(hops=3)) == 3
    assert expected_hop_of(_vector_case()) == 0


def test_graph_case_is_full_success_when_everything_matches():
    case = _graph_case(hops=2)
    outcome = RetrievalOutcome(
        results=(
            _path_result(
                node_ids=["ENT_0", "ENT_1", "ENT_2"],
                relation_ids=["REL_0", "REL_1"],
                evidence_ids=["EVD_1"],
                chunk_ids=["CHK_1"],
            ),
        )
    )

    record = evaluate_case(case, outcome)

    assert record["verdict"] == FULL_SUCCESS
    assert record["expected_hop"] == 2
    assert record["actual_hop"] == 2
    assert record["hop_match"] is True
    assert record["node_coverage"] == 1.0
    assert record["relation_coverage"] == 1.0
    assert record["evidence_coverage"] == 1.0
    assert record["missing_relation_ids"] == []


def test_graph_case_is_partial_when_one_relation_missing():
    case = _graph_case(hops=2)
    outcome = RetrievalOutcome(
        results=(
            _path_result(
                node_ids=["ENT_0", "ENT_1"],
                relation_ids=["REL_0"],
                evidence_ids=["EVD_1"],
            ),
        )
    )

    record = evaluate_case(case, outcome)

    assert record["verdict"] == PARTIAL_SUCCESS
    assert record["relation_coverage"] == 0.5
    assert record["missing_relation_ids"] == ["REL_1"]
    assert record["actual_hop"] == 1
    assert record["hop_match"] is False


def test_graph_case_fails_when_nothing_retrieved():
    record = evaluate_case(_graph_case(), RetrievalOutcome())

    assert record["verdict"] == FAILURE
    assert record["actual_hop"] is None
    assert record["hop_match"] is None
    assert record["relation_coverage"] == 0.0


def test_hop_mismatch_alone_prevents_full_success():
    # 노드·관계·근거는 다 맞지만 더 긴 경로로 찾은 경우입니다.
    case = _graph_case(hops=2)
    outcome = RetrievalOutcome(
        results=(
            _path_result(
                node_ids=["ENT_0", "ENT_1", "ENT_2", "ENT_9"],
                relation_ids=["REL_0", "REL_1", "REL_9"],
                evidence_ids=["EVD_1"],
            ),
        )
    )

    record = evaluate_case(case, outcome)

    assert record["actual_hop"] == 3
    assert record["hop_match"] is False
    assert record["verdict"] == PARTIAL_SUCCESS


def test_best_path_prefers_the_one_covering_expected_relations():
    case = _graph_case(hops=2)
    outcome = RetrievalOutcome(
        results=(
            _path_result(node_ids=["X"], relation_ids=["REL_7", "REL_8", "REL_9"]),
            _path_result(
                node_ids=["ENT_0", "ENT_1", "ENT_2"],
                relation_ids=["REL_0", "REL_1"],
                evidence_ids=["EVD_1"],
            ),
        )
    )

    record = evaluate_case(case, outcome)

    assert record["actual_hop"] == 2
    assert record["verdict"] == FULL_SUCCESS


def test_vector_case_keeps_hop_and_graph_coverage_as_none():
    outcome = RetrievalOutcome(
        results=(
            SearchResult(
                source="vector", text="t", evidence_ids=["EVD_1"], chunk_ids=["CHK_1"]
            ),
        )
    )

    record = evaluate_case(_vector_case(), outcome)

    assert record["verdict"] == FULL_SUCCESS
    assert record["expected_hop"] is None
    assert record["actual_hop"] is None
    assert record["hop_match"] is None
    assert record["node_coverage"] is None
    assert record["relation_coverage"] is None
    assert record["evidence_coverage"] == 1.0
    assert record["chunk_coverage"] == 1.0


def test_vector_case_is_partial_when_only_evidence_found():
    outcome = RetrievalOutcome(
        results=(SearchResult(source="vector", text="t", evidence_ids=["EVD_1"]),)
    )

    record = evaluate_case(_vector_case(), outcome)

    assert record["verdict"] == PARTIAL_SUCCESS
    assert record["chunk_coverage"] == 0.0
    assert record["missing_chunk_ids"] == ["CHK_1"]


def test_reject_case_is_skipped():
    record = evaluate_case(_reject_case(), RetrievalOutcome())

    assert record["verdict"] == SKIPPED_REJECT
    assert record["expected_hop"] is None


def test_k_limits_the_results_considered():
    case = _graph_case(hops=1)
    outcome = RetrievalOutcome(
        results=tuple(
            _path_result(node_ids=["X"], relation_ids=["REL_9"]) for _ in range(3)
        )
        + (
            _path_result(
                node_ids=["ENT_0", "ENT_1"],
                relation_ids=["REL_0"],
                evidence_ids=["EVD_1"],
            ),
        )
    )

    assert evaluate_case(case, outcome, k=5)["verdict"] == FULL_SUCCESS
    assert evaluate_case(case, outcome, k=2)["verdict"] == FAILURE


def test_expected_relation_types_come_from_relation_lookup():
    record = evaluate_case(
        _graph_case(hops=2),
        RetrievalOutcome(),
        relation_types={"REL_0": "HAS_VARIETY", "REL_1": "SUITABLE_FOR_REGION"},
    )

    assert record["expected_relation_types"] == ["HAS_VARIETY", "SUITABLE_FOR_REGION"]


def test_load_relation_types_reads_relation_id_to_type(tmp_path):
    path = tmp_path / "relations.jsonl"
    path.write_text(
        json.dumps({"relation_id": "REL_0", "relation_type": "HAS_VARIETY"}) + "\n",
        encoding="utf-8",
    )

    assert load_relation_types(path) == {"REL_0": "HAS_VARIETY"}


def test_oracle_observer_reaches_full_success_on_every_graph_case():
    cases = [_graph_case("A", hops=1), _graph_case("B", hops=3), _vector_case("V")]

    evaluation = evaluate_multihop(cases, OracleRetrievalObserver())

    assert evaluation.summary["overall"]["full_success"] == 3
    assert evaluation.summary["hop_match"]["rate"] == 1.0


def test_null_observer_fails_every_case():
    cases = [_graph_case("A"), _vector_case("V")]

    evaluation = evaluate_multihop(cases, NullRetrievalObserver())

    assert evaluation.summary["overall"]["failure"] == 2
    assert evaluation.summary["overall"]["full_success_rate"] == 0.0


def test_reject_cases_are_excluded_from_denominators():
    cases = [_graph_case("A"), _reject_case("R1"), _reject_case("R2")]

    evaluation = evaluate_multihop(cases, OracleRetrievalObserver())
    summary = evaluation.summary

    assert summary["evaluated_cases"] == 1
    assert summary["skipped_reject_cases"] == 2
    assert "REJECT" not in summary["by_expected_route"]
    assert summary["verdict_counts"][SKIPPED_REJECT] == 2
    assert len(evaluation.details) == 3


def test_summary_buckets_by_route_type_and_hop():
    cases = [
        _graph_case("A", hops=1),
        _graph_case("B", strategy="hybrid", hops=2),
        _vector_case("V"),
    ]
    outcomes = {
        "A": RetrievalOutcome(
            results=(
                _path_result(
                    node_ids=["ENT_0", "ENT_1"],
                    relation_ids=["REL_0"],
                    evidence_ids=["EVD_1"],
                ),
            )
        ),
    }

    summary = evaluate_multihop(cases, StubObserver(outcomes)).summary

    assert summary["by_expected_route"]["GRAPH"]["full_success"] == 1
    assert summary["by_expected_route"]["HYBRID"]["failure"] == 1
    assert summary["by_expected_route"]["VECTOR"]["failure"] == 1
    assert summary["by_expected_hop"]["1"]["full_success"] == 1
    assert summary["by_expected_hop"]["2"]["total"] == 1
    assert summary["by_question_type"]["multi_hop"]["total"] == 2
    assert summary["mean_coverage"]["relation_coverage"] == 0.5


def test_graph_observer_requests_expected_hop_depth():
    class FakeRetriever:
        def __init__(self):
            self.calls = []

        def search(self, question, *, crop=None, max_hops=2, limit=20):
            self.calls.append({"question": question, "crop": crop, "max_hops": max_hops})
            return [_path_result(node_ids=["ENT_0"], relation_ids=["REL_0"])]

    retriever = FakeRetriever()
    observer = GraphRetrievalObserver(retriever=retriever, crop_of=lambda q: "배추")

    outcome = observer.retrieve(_graph_case(hops=3))

    assert retriever.calls[0]["max_hops"] == 3
    assert retriever.calls[0]["crop"] == "배추"
    assert len(outcome.results) == 1


def test_graph_observer_asks_for_at_least_one_hop():
    class FakeRetriever:
        def __init__(self):
            self.max_hops = None

        def search(self, question, *, crop=None, max_hops=2, limit=20):
            self.max_hops = max_hops
            return []

    retriever = FakeRetriever()
    GraphRetrievalObserver(retriever=retriever).retrieve(_vector_case())

    assert retriever.max_hops == 1


def test_write_details_csv_flattens_lists(tmp_path):
    evaluation = evaluate_multihop([_graph_case("A")], NullRetrievalObserver())

    path = write_details(evaluation.details, tmp_path / "d.csv", detail_format="csv")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert rows[0]["case_id"] == "A"
    assert rows[0]["verdict"] == FAILURE
    assert rows[0]["missing_relation_ids"] == "REL_0 | REL_1"


def test_format_report_mentions_hop_and_skipped_counts():
    evaluation = evaluate_multihop(
        [_graph_case("A"), _reject_case("R")], OracleRetrievalObserver()
    )

    report = format_report(evaluation.summary)

    assert "REJECT 제외 1건" in report
    assert "hop 일치: 1/1" in report
    assert "2-hop" in report


def test_main_writes_reports(tmp_path, capsys):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in [_graph_case("A"), _vector_case("V"), _reject_case("R")]
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "reports"

    exit_code = main(
        [
            "--golden-set",
            str(golden),
            "--output-dir",
            str(out_dir),
            "--observer",
            "oracle",
            "--skip-relation-types",
            "--tag",
            "unit",
        ]
    )

    assert exit_code == 0
    summary = json.loads((out_dir / "multihop_summary_unit.json").read_text(encoding="utf-8"))
    assert summary["evaluated_cases"] == 2
    assert summary["skipped_reject_cases"] == 1
    assert summary["meta"]["observer"] == "oracle"
    assert summary["meta"]["relation_types_loaded"] is False
    details = (out_dir / "multihop_details_unit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(details) == 3
    assert "FULL_SUCCESS" in capsys.readouterr().out


def test_main_rejects_non_positive_k(tmp_path):
    with pytest.raises(ValueError, match="k must be positive"):
        main(["--k", "0"])


def test_irrelevant_path_of_matching_length_is_not_a_hop_match():
    # 기대 관계를 하나도 지나지 않은 경로는 깊이가 같아도 일치가 아닙니다.
    case = _graph_case(hops=1)
    outcome = RetrievalOutcome(
        results=(_path_result(node_ids=["X", "Y"], relation_ids=["REL_9"]),)
    )

    record = evaluate_case(case, outcome)

    assert record["actual_hop"] == 1
    assert record["hop_match"] is False
    assert record["verdict"] == FAILURE

def test_supervisor_observer_uses_retrieved_results_not_verified_results():
    class FakeApp:
        def invoke(self, payload):
            return {
                "route": "GRAPH",
                "route_reason": "테스트",
                "results": [],  # 검증에서 걸러진 상태
                "retrieved_results": [SearchResult(source="graph", text="x", evidence_ids=["EVD_1"])],
                "warnings": ["w"],
            }

    outcome = SupervisorRetrievalObserver(app=FakeApp()).retrieve(
        {"question": "마늘 품종별 적합 지역은 어디인가?", "query_strategy": "graph"}
    )

    assert outcome.route == "GRAPH"
    assert [r.evidence_ids for r in outcome.results] == [["EVD_1"]]


def test_supervisor_observer_skips_reject_cases_without_invoking():
    class ExplodingApp:
        def invoke(self, payload):  # pragma: no cover - 호출되면 실패합니다.
            raise AssertionError("REJECT 문항은 Supervisor를 호출하지 않아야 합니다.")

    outcome = SupervisorRetrievalObserver(app=ExplodingApp()).retrieve(
        {"question": "서울 맛집 추천해 주세요.", "query_strategy": "reject"}
    )

    assert outcome.route == "REJECT"
    assert outcome.results == ()