from __future__ import annotations

from scripts.evaluation.trace_question import (
    DEMO_QUESTIONS,
    export_markdown,
    export_csv,
    select_demo_questions,
)


def test_demo_questions_include_graph_and_configurable_conflict_scenarios():
    names = {name for name, _ in DEMO_QUESTIONS}

    assert "graph_relation" in names
    assert "source_conflict" in names


def test_select_demo_questions_can_run_only_one_scenario():
    rows = select_demo_questions("graph_relation")

    assert rows == [
        (
            "graph_relation",
            "양파의 어린 유묘에서 생육이 정지되고 지상부가 시들며, 인경 내부에 검은색 가루가 가득 보이는 병은 무엇인가요?",
        )
    ]


def test_export_csv_and_markdown_use_e2e_table_columns(tmp_path):
    rows = [
        {
            "scenario": "graph_relation",
            "route_path": ["GRAPH"],
            "visited_nodes": ["route", "retrieve", "finalize"],
            "final_status": "ANSWERED",
            "trace_id": "trace-1",
            "error": None,
            "injected_fault": None,
        }
    ]

    csv_path = tmp_path / "results.csv"
    md_path = tmp_path / "results.md"
    export_csv(rows, csv_path)
    export_markdown(rows, md_path)

    csv_text = csv_path.read_text(encoding="utf-8")
    md_text = md_path.read_text(encoding="utf-8")
    for column in ("scenario", "route_path", "visited_nodes", "final_status", "trace_id"):
        assert column in csv_text
    assert "| # | 유형 | 실제 route_path | 실제 visited_nodes | 실제 final_status | 통과 | trace_id | 오류/비고 |" in md_text
    assert "| 1 | Graph 관계 |" in md_text
    assert "GRAPH" in csv_text
