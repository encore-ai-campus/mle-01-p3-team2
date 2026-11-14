import csv
import json

import pytest

from scripts.evaluation.evaluate_routing import (
    RouteObservation,
    RouterObserver,
    evaluate_routing,
    expected_route_of,
    format_report,
    load_golden_cases,
    main,
    write_details,
    write_summary,
)
from src.retrievers.router import QuestionRouter


class StubObserver:
    """정해진 경로를 순서대로 돌려주는 테스트용 관찰자입니다."""

    def __init__(self, routes):
        self.routes = list(routes)
        self.questions = []

    def observe(self, question):
        self.questions.append(question)
        return RouteObservation(route=self.routes.pop(0), reason="stub", warnings=("w",))


def _case(case_id, strategy, question_type, *, status="READY", question="질문"):
    return {
        "case_id": case_id,
        "question": question,
        "query_strategy": strategy,
        "question_type": question_type,
        "status": status,
    }


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_load_golden_cases_keeps_only_ready_rows(tmp_path):
    path = _write_jsonl(
        tmp_path / "golden.jsonl",
        [
            _case("A", "graph", "relation"),
            _case("B", "vector", "fact", status="REJECTED"),
        ],
    )

    cases = load_golden_cases(path)

    assert [case["case_id"] for case in cases] == ["A"]


def test_load_golden_cases_can_include_non_ready_rows(tmp_path):
    path = _write_jsonl(
        tmp_path / "golden.jsonl",
        [
            _case("A", "graph", "relation"),
            _case("B", "vector", "fact", status="REJECTED"),
        ],
    )

    cases = load_golden_cases(path, ready_only=False)

    assert [case["case_id"] for case in cases] == ["A", "B"]


def test_load_golden_cases_rejects_invalid_json(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_golden_cases(path)


def test_load_golden_cases_rejects_empty_result(tmp_path):
    path = _write_jsonl(
        tmp_path / "golden.jsonl", [_case("A", "graph", "relation", status="REJECTED")]
    )

    with pytest.raises(ValueError, match="No READY golden cases"):
        load_golden_cases(path)


def test_expected_route_uses_uppercase_query_strategy():
    assert expected_route_of(_case("A", "hybrid", "fact")) == "HYBRID"


@pytest.mark.parametrize("strategy", [None, "", "unknown"])
def test_expected_route_rejects_bad_query_strategy(strategy):
    with pytest.raises(ValueError):
        expected_route_of({"case_id": "A", "query_strategy": strategy})


def test_evaluate_routing_computes_all_metrics():
    cases = [
        _case("A", "graph", "relation"),
        _case("B", "vector", "fact"),
        _case("C", "reject", "out_of_scope"),
        _case("D", "reject", "out_of_scope"),
    ]
    observer = StubObserver(["GRAPH", "GRAPH", "REJECT", "HYBRID"])

    evaluation = evaluate_routing(cases, observer)
    summary = evaluation.summary

    assert summary["total_cases"] == 4
    assert summary["overall"] == {"total": 4, "correct": 2, "accuracy": 0.5}
    assert summary["by_expected_route"]["VECTOR"]["accuracy"] == 0.0
    assert summary["by_expected_route"]["REJECT"] == {
        "total": 2,
        "correct": 1,
        "accuracy": 0.5,
    }
    assert summary["reject"] == summary["by_expected_route"]["REJECT"]
    assert summary["by_question_type"]["out_of_scope"]["total"] == 2
    assert summary["confusion_matrix"]["VECTOR"]["GRAPH"] == 1
    assert summary["confusion_matrix"]["REJECT"]["REJECT"] == 1
    assert summary["confusion_matrix"]["REJECT"]["HYBRID"] == 1


def test_evaluate_routing_records_required_detail_fields():
    evaluation = evaluate_routing(
        [_case("A", "graph", "relation", question="마늘 질문")], StubObserver(["GRAPH"])
    )

    detail = evaluation.details[0]
    assert detail == {
        "case_id": "A",
        "question_type": "relation",
        "expected_route": "GRAPH",
        "actual_route": "GRAPH",
        "route_correct": True,
        "reason": "stub",
        "warnings": ["w"],
        "initial_route": "GRAPH",
        "final_route": "GRAPH",
        "route_path": ["GRAPH"],
    }


def test_evaluate_routing_uses_final_route_when_fallback_recorded():
    class FallbackObserver:
        def observe(self, question):
            return RouteObservation(
                route="GRAPH",
                initial_route="GRAPH",
                final_route="VECTOR",
                route_path=("GRAPH", "VECTOR"),
            )

    evaluation = evaluate_routing(
        [_case("A", "vector", "fact")], FallbackObserver()
    )

    detail = evaluation.details[0]
    assert detail["actual_route"] == "VECTOR"
    assert detail["route_correct"] is True
    assert detail["initial_route"] == "GRAPH"
    assert detail["route_path"] == ["GRAPH", "VECTOR"]


def test_router_observer_wraps_question_router():
    observation = RouterObserver(QuestionRouter()).observe("마늘 품종별 적합 지역은 어디인가?")

    assert observation.route == "GRAPH"
    assert observation.reason
    assert observation.resolved_final_route == "GRAPH"


def test_write_details_jsonl_roundtrip(tmp_path):
    evaluation = evaluate_routing([_case("A", "graph", "relation")], StubObserver(["GRAPH"]))

    path = write_details(evaluation.details, tmp_path / "d.jsonl", detail_format="jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert rows == evaluation.details


def test_write_details_csv_flattens_list_columns(tmp_path):
    evaluation = evaluate_routing([_case("A", "graph", "relation")], StubObserver(["VECTOR"]))

    path = write_details(evaluation.details, tmp_path / "d.csv", detail_format="csv")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert rows[0]["case_id"] == "A"
    assert rows[0]["expected_route"] == "GRAPH"
    assert rows[0]["actual_route"] == "VECTOR"
    assert rows[0]["route_correct"] == "False"
    assert rows[0]["warnings"] == "w"
    assert rows[0]["route_path"] == "VECTOR"


def test_write_details_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="Unknown detail format"):
        write_details([], tmp_path / "d.txt", detail_format="txt")


def test_write_summary_is_valid_json(tmp_path):
    evaluation = evaluate_routing([_case("A", "reject", "out_of_scope")], StubObserver(["REJECT"]))

    path = write_summary(evaluation.summary, tmp_path / "s.json")

    assert json.loads(path.read_text(encoding="utf-8")) == evaluation.summary


def test_format_report_includes_headline_metrics():
    evaluation = evaluate_routing(
        [_case("A", "graph", "relation"), _case("B", "reject", "out_of_scope")],
        StubObserver(["GRAPH", "HYBRID"]),
    )

    report = format_report(evaluation.summary)

    assert "4-way 정확도: 1/2" in report
    assert "REJECT 정확도: 0/2" not in report
    assert "REJECT 정확도: 0/1" in report
    # 계약 2절: 어떤 지표를 인용할 수 있는지 리포트에 드러나야 합니다.
    assert "[공식]" in report and "[단독 인용 금지]" in report
    assert "Confusion matrix" in report


def test_main_writes_both_outputs(tmp_path, capsys):
    golden = _write_jsonl(
        tmp_path / "golden.jsonl",
        [
            _case("A", "graph", "relation", question="마늘 품종별 적합 지역은 어디인가?"),
            _case("B", "reject", "out_of_scope", question="파이썬 퀵정렬을 구현해 주세요."),
        ],
    )
    out_dir = tmp_path / "reports"

    exit_code = main(
        [
            "--golden-set",
            str(golden),
            "--output-dir",
            str(out_dir),
            "--tag",
            "unit",
        ]
    )

    assert exit_code == 0
    summary = json.loads((out_dir / "routing_summary_unit.json").read_text(encoding="utf-8"))
    assert summary["total_cases"] == 2
    assert summary["meta"]["router"] == "rule"
    details = (out_dir / "routing_details_unit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(details) == 2
    assert "REJECT 정확도" in capsys.readouterr().out


def test_main_rejects_unknown_router(tmp_path):
    with pytest.raises(SystemExit):
        main(["--router", "nope"])
