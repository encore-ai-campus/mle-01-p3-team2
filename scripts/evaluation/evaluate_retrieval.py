"""Evaluate initial routing and expected-route retrieval in one report.

Run with existing Top-10 data:
    .venv/Scripts/python.exe -m scripts.evaluation.evaluate_retrieval

Refresh real Neo4j results first:
    .venv/Scripts/python.exe -m scripts.evaluation.evaluate_retrieval --collect

Check Neo4j connectivity only:
    .venv/Scripts/python.exe -m scripts.evaluation.evaluate_retrieval --check-neo4j
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.evaluation.evaluate_multihop import (
    DEFAULT_GOLDEN_PATH,
    expected_route_of,
    load_golden_cases,
)
from scripts.evaluation.report_io import write_summary
from src.retrievers.router import QuestionRouter
from src.retrievers.search_result import SearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = DEFAULT_GOLDEN_PATH.parent / "reports"
ROUTES = ("VECTOR", "GRAPH", "HYBRID", "REJECT")
SEARCH_ROUTES = ROUTES[:3]
KS = (1, 3, 5, 10)


def _hit_row(rank: int, result: SearchResult) -> dict[str, Any]:
    return {
        "rank": rank,
        "source": result.source,
        "score": result.score,
        "evidence_ids": result.evidence_ids,
        "chunk_ids": result.chunk_ids,
        "relation_ids": [
            edge["relation_id"]
            for edge in result.path_edges or []
            if edge.get("relation_id")
        ],
        "hop_count": result.hop_count,
        "path_nodes": result.path_nodes or [],
        "path_edges": [
            {
                "hop": edge["hop"],
                "relation_id": edge["relation_id"],
                "relation_type": edge["relation_type"],
                "from_entity_id": edge["from_entity_id"],
                "to_entity_id": edge["to_entity_id"],
                "evidence_ids": edge["evidence_ids"],
                "chunk_ids": edge["chunk_ids"],
                "original_chunks": [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "evidence_id": chunk["evidence_id"],
                        "has_text": bool(chunk["text"].strip()),
                    }
                    for chunk in edge["original_chunks"]
                ],
            }
            for edge in result.path_edges or []
        ],
    }


def _open_neo4j_for_evaluation() -> Any:
    """Load this project's .env and verify Neo4j before model loading/search."""
    from dotenv import load_dotenv

    from src.retrievers.graph_retriever import GraphRetriever

    env_path = PROJECT_ROOT / ".env"
    dotenv_loaded = load_dotenv(env_path)
    print(
        f"Project .env loaded: {dotenv_loaded} "
        "(existing process environment variables take precedence).",
        flush=True,
    )
    print("Checking Neo4j connectivity before evaluation...", flush=True)
    try:
        graph = GraphRetriever.from_env()
    except Exception as exc:
        raise ConnectionError(
            f"Neo4j preflight failed ({type(exc).__name__}): {exc}"
        ) from exc
    print("Neo4j connectivity check passed.", flush=True)
    return graph


def collect_top10(
    cases: list[dict], path: Path, router: QuestionRouter, graph: Any
) -> None:
    """Run each non-REJECT case through its expected retrieval path."""
    from sentence_transformers import SentenceTransformer

    from src.retrievers.hybrid_retriever import hybrid_search
    from src.retrievers.vector_retriever import VectorIndex

    try:
        vector = VectorIndex(
            graph.driver,
            SentenceTransformer("BAAI/bge-m3"),
            database=graph.database,
        )
        vector.verify_ready()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for case in cases:
                route = expected_route_of(case)
                if route == "REJECT":
                    continue
                question = case["question"]
                decision = router.route_query(question)
                if route == "VECTOR":
                    results = vector.search(question, k=10)
                elif route == "GRAPH":
                    results = graph.search(
                        question, crop=decision.crop,
                        max_hops=decision.max_hops, limit=10,
                    )
                elif route == "HYBRID":
                    results = hybrid_search(
                        question, graph_retriever=graph, vector_index=vector,
                        crop=decision.crop, max_hops=decision.max_hops,
                        graph_limit=10, vector_limit=10, limit=10,
                    )
                else:
                    raise ValueError(f"Unknown expected route: {route}")
                row = {
                    "case_id": case["case_id"], "route": route, "k": 10,
                    "result_count": len(results),
                    "hits": [_hit_row(rank, result) for rank, result in enumerate(results, 1)],
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{case['case_id']} {route} {len(results)}/10", flush=True)
    finally:
        graph.close()


def _read_top10(path: Path, cases: list[dict]) -> dict[str, dict]:
    """Validate a cached Top-10 file without silently accepting duplicate cases."""
    expected = {
        case["case_id"]: route
        for case in cases
        if (route := expected_route_of(case)) in SEARCH_ROUTES
    }
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case_id in golden set")

    results: dict[str, dict] = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = row["case_id"]
            if case_id in results:
                raise ValueError(f"Duplicate Top-10 case_id: {case_id} (line {line_no})")
            if case_id not in expected:
                raise ValueError(f"Unexpected Top-10 case_id: {case_id}")
            if row.get("route") != expected[case_id] or row.get("k") != 10:
                raise ValueError(f"Route/K mismatch for {case_id}")
            hits = row.get("hits", [])
            if len(hits) > 10 or any(hit.get("rank") != rank for rank, hit in enumerate(hits, 1)):
                raise ValueError(f"Invalid ranks for {case_id}")
            results[case_id] = row

    missing = expected.keys() - results.keys()
    if missing:
        raise ValueError(f"Missing Top-10 cases: {sorted(missing)}")
    return results


def evaluate(cases: list[dict], top10: dict[str, dict], router: QuestionRouter) -> tuple[list[dict], dict]:
    """Evaluate initial routing and oracle-route Hit@K (not end-to-end Hit@K)."""
    details: list[dict] = []
    matrix = {route: {actual: 0 for actual in ROUTES} for route in ROUTES}
    grouped: dict[str, list[dict]] = {route: [] for route in ROUTES}

    for case in cases:
        expected = expected_route_of(case)
        if expected not in ROUTES:
            raise ValueError(f"Invalid expected route: {expected}")
        decision = router.route_query(case["question"])
        actual = decision.route
        if actual not in ROUTES:
            raise ValueError(f"Invalid actual route: {actual}")
        matrix[expected][actual] += 1
        row: dict[str, Any] = {
            "case_id": case["case_id"],
            "question_type": case.get("question_type"),
            "expected_route": expected,
            "actual_route": actual,
            "route_match": expected == actual,
            "route_reason": decision.reason,
            "retrieval_route": expected if expected in SEARCH_ROUTES else None,
        }
        evidence_ids = set(case.get("expected_evidence_ids") or ())
        chunk_ids = set(case.get("expected_source_chunk_ids") or ())
        hits = top10[case["case_id"]]["hits"] if expected in SEARCH_ROUTES else []
        found_evidence: set[str] = set()
        found_chunks: set[str] = set()
        checked = 0
        for k in KS:
            if expected == "REJECT":
                row[f"hit_at_{k}"] = None
                row[f"evidence_hit_at_{k}"] = None
                row[f"chunk_hit_at_{k}"] = None
                continue
            for hit in hits[checked:k]:
                found_evidence.update(hit.get("evidence_ids") or ())
                found_chunks.update(hit.get("chunk_ids") or ())
            checked = k
            evidence_hit = bool(evidence_ids & found_evidence)
            chunk_hit = bool(chunk_ids & found_chunks)
            row[f"evidence_hit_at_{k}"] = evidence_hit
            row[f"chunk_hit_at_{k}"] = chunk_hit
            row[f"hit_at_{k}"] = evidence_hit or chunk_hit
        details.append(row)
        grouped[expected].append(row)

    def metric(rows: list[dict], k: int) -> dict:
        count = sum(row[f"hit_at_{k}"] is True for row in rows)
        return {"hits": count, "rate": round(count / len(rows), 6) if rows else None}

    def route_counts(route: str) -> dict:
        selected = grouped[route]
        correct = sum(row["route_match"] for row in selected)
        result = {
            "questions": len(selected),
            "correct": correct,
            "incorrect": len(selected) - correct,
            "routing_accuracy": round(correct / len(selected), 6) if selected else None,
        }
        for k in KS:
            result[f"hit_at_{k}"] = metric(selected, k) if route in SEARCH_ROUTES else {"hits": 0, "rate": None}
        return result

    correct = sum(row["route_match"] for row in details)
    search_rows = [row for route in SEARCH_ROUTES for row in grouped[route]]
    summary = {
        "routing": {
            "questions": len(details),
            "correct": correct,
            "incorrect": len(details) - correct,
            "accuracy": round(correct / len(details), 6) if details else None,
            "confusion_matrix": matrix,
        },
        "by_expected_route": {route: route_counts(route) for route in ROUTES},
        "search_questions": len(search_rows),
        "hit_at_k": {str(k): metric(search_rows, k) for k in KS},
        "hit_at_k_mode": "expected_route_oracle",
    }
    return details, summary


def _ordered_nodes(required: list[str], actual: list[str]) -> bool:
    position = 0
    for node_id in actual:
        if position < len(required) and node_id == required[position]:
            position += 1
    return position == len(required)


def _edge_has_original(edge: dict) -> bool:
    chunk_ids = set(edge.get("chunk_ids") or ())
    evidence_ids = set(edge.get("evidence_ids") or ())
    return any(
        chunk.get("has_text")
        and chunk.get("chunk_id") in chunk_ids
        and chunk.get("evidence_id") in evidence_ids
        for chunk in edge.get("original_chunks") or ()
    )


def _path_matches(expected_relations: list[str], edges: list[dict]) -> bool:
    """Require one contiguous, directed sequence of expected relation IDs."""
    if not expected_relations:
        return False
    for start in range(len(edges) - len(expected_relations) + 1):
        subpath = edges[start:start + len(expected_relations)]
        if [edge.get("relation_id") for edge in subpath] != expected_relations:
            continue
        if all(
            left.get("to_entity_id") == right.get("from_entity_id")
            for left, right in zip(subpath, subpath[1:])
        ):
            return True
    return False


def evaluate_multihop_paths(
    cases: list[dict], top10: dict[str, dict], route_details: list[dict],
    relation_path: Path,
) -> tuple[list[dict], dict]:
    """Score required path parts on the expected-route Top-10 graph hits."""
    multi_cases = [case for case in cases if case.get("question_type") == "multi_hop"]
    required_relation_ids = {
        item for case in multi_cases for item in case.get("expected_relation_ids") or []
    }
    relation_types: dict[str, str] = {}
    with relation_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                relation = json.loads(line)
                relation_id = relation.get("relation_id")
                if relation_id in required_relation_ids:
                    relation_types[relation_id] = relation.get("relation_type") or ""
    missing_types = required_relation_ids - set(relation_types)
    if missing_types:
        raise ValueError(f"Missing relation types for {len(missing_types)} golden relation IDs")

    routes_by_id = {row["case_id"]: row for row in route_details}
    details: list[dict] = []
    for case in multi_cases:
        case_id = case["case_id"]
        expected_route = expected_route_of(case)
        expected_relations = list(case.get("expected_relation_ids") or [])
        expected_nodes = list(case.get("expected_entity_ids") or [])
        expected_types = [relation_types[item] for item in expected_relations]
        record: dict[str, Any] = {
            "case_id": case_id,
            "expected_route": expected_route,
            "actual_route": routes_by_id[case_id]["actual_route"],
            "route_match": expected_route == routes_by_id[case_id]["actual_route"],
            "expected_hop": len(expected_relations) if expected_relations else None,
            "expected_nodes": expected_nodes,
            "expected_relation_ids": expected_relations,
            "expected_relation_types": expected_types,
        }
        if not expected_relations or expected_route not in ("GRAPH", "HYBRID"):
            record.update({"status": "NOT_APPLICABLE", "reason": "No expected graph path"})
            details.append(record)
            continue

        hits = top10[case_id]["hits"]
        graph_hits = [hit for hit in hits if hit["source"] == "graph"]

        expected_relation_set = set(expected_relations)
        expected_node_set = set(expected_nodes)
        expected_type_counts = Counter(expected_types)

        def path_score(hit: dict) -> tuple[int, int, int, int, int]:
            edges = hit.get("path_edges") or []
            ids = {edge.get("relation_id") for edge in edges}
            nodes = {node.get("entity_id") for node in hit.get("path_nodes") or []}
            types = Counter(edge.get("relation_type") for edge in edges)
            return (
                int(_path_matches(expected_relations, edges)),
                len(ids & expected_relation_set),
                len(nodes & expected_node_set),
                sum((expected_type_counts & types).values()),
                -hit.get("rank", 10),
            )

        best = max(graph_hits, key=path_score) if graph_hits else None
        nodes = [node["entity_id"] for node in best["path_nodes"]] if best else []
        edges = best["path_edges"] if best else []
        actual_relation_ids = [edge["relation_id"] for edge in edges]
        actual_types = [edge["relation_type"] for edge in edges]
        actual_hop = best["hop_count"] if best else None
        node_match = _ordered_nodes(expected_nodes, nodes)
        relation_id_match = _path_matches(expected_relations, edges)
        relation_type_match = not (Counter(expected_types) - Counter(actual_types))
        hop_match = actual_hop is not None and actual_hop >= len(expected_relations) and len(edges) >= len(expected_relations)
        # Every required edge must be connected to its own original source.
        source_link_match = relation_id_match and all(
            _edge_has_original(edge)
            for edge in edges
            if edge.get("relation_id") in expected_relation_set
        )
        # Full path success must use evidence linked to the selected graph path,
        # not evidence pulled independently from unrelated Top-10 results.
        found_evidence = {
            chunk.get("evidence_id")
            for edge in edges for chunk in edge.get("original_chunks") or ()
            if _edge_has_original({**edge, "original_chunks": [chunk]})
        }
        found_chunks = {
            chunk.get("chunk_id")
            for edge in edges for chunk in edge.get("original_chunks") or ()
            if _edge_has_original({**edge, "original_chunks": [chunk]})
        }
        required_evidence_match = (
            set(case.get("expected_evidence_ids") or []) <= found_evidence
            and set(case.get("expected_source_chunk_ids") or []) <= found_chunks
        )
        route_match = expected_route == record["actual_route"]
        full_success = all((
            route_match, node_match, relation_id_match, relation_type_match,
            hop_match, source_link_match, required_evidence_match,
        ))
        failure_reasons = []
        if not route_match:
            failure_reasons.append("routing_mismatch")
        if not node_match:
            failure_reasons.append("required_nodes_missing")
        if not (relation_id_match and relation_type_match and hop_match):
            failure_reasons.append("relation_or_hop_missing")
        if not source_link_match:
            failure_reasons.append("edge_original_link_missing")
        if not required_evidence_match:
            failure_reasons.append("expected_evidence_missing")

        record.update({
            "status": "FULL_SUCCESS" if full_success else "FAILURE",
            "actual_hop": actual_hop,
            "actual_path_rank": best["rank"] if best else None,
            "actual_nodes": nodes,
            "actual_relation_ids": actual_relation_ids,
            "actual_relation_types": actual_types,
            "actual_path_edges": [
                {
                    "relation_id": edge["relation_id"],
                    "relation_type": edge["relation_type"],
                    "from_entity_id": edge["from_entity_id"],
                    "to_entity_id": edge["to_entity_id"],
                    "original_chunk_ids": [
                        chunk["chunk_id"] for chunk in edge["original_chunks"]
                        if chunk["has_text"]
                    ],
                    "source_linked": _edge_has_original(edge),
                }
                for edge in edges
            ],
            "missing_node_ids": [item for item in expected_nodes if item not in nodes],
            "missing_relation_ids": [item for item in expected_relations if item not in actual_relation_ids],
            "graph_hit_count": len(graph_hits),
            "node_match": node_match,
            "relation_id_match": relation_id_match,
            "relation_type_match": relation_type_match,
            "hop_match": hop_match,
            "source_link_match": source_link_match,
            "required_evidence_match": required_evidence_match,
            "full_success": full_success,
            "failure_reasons": failure_reasons,
        })
        details.append(record)

    applicable = [row for row in details if row["status"] != "NOT_APPLICABLE"]
    success = sum(row["full_success"] for row in applicable)
    components = (
        "node_match", "relation_id_match", "relation_type_match", "hop_match",
        "source_link_match", "required_evidence_match",
    )
    summary = {
        "multi_hop_questions": len(details),
        "graph_path_questions": len(applicable),
        "not_applicable": len(details) - len(applicable),
        "full_success": success,
        "full_success_rate": round(success / len(applicable), 6) if applicable else None,
        "component_pass_counts": {
            component: sum(row[component] for row in applicable) for component in components
        },
        "component_pass_rates": {
            component: round(sum(row[component] for row in applicable) / len(applicable), 6)
            if applicable else None for component in components
        },
        "failure_reason_counts": dict(Counter(
            reason for row in applicable for reason in row["failure_reasons"]
        )),
    }
    return details, summary


def _fraction(count: int, total: int, rate: float | None) -> str:
    return "N/A" if rate is None else f"{count}/{total} ({rate * 100:.2f}%)"


def write_reports(
    details: list[dict], summary: dict,
    multihop_details: list[dict], multihop_summary: dict, output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = (
        "case_id", "question_type", "expected_route", "actual_route", "route_match",
        "route_reason", "retrieval_route",
        *(field for k in KS for field in (
            f"evidence_hit_at_{k}", f"chunk_hit_at_{k}", f"hit_at_{k}"
        )),
    )
    csv_path = output_dir / "retrieval_case_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(details)
    report_summary = {**summary, "multihop": multihop_summary}
    write_summary(report_summary, output_dir / "retrieval_evaluation_summary.json")
    with (output_dir / "multihop_case_results.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for row in multihop_details:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# 검색 경로·Hit@K 평가",
        "",
        "라우팅은 Supervisor 기본 라우터의 최초 선택 경로입니다. Hit@K는 평가셋의 기대 경로로 검색한 Top-10 순위에서 정답 evidence ID 또는 chunk ID가 하나 이상 발견된 문항 비율입니다. REJECT 문항은 Hit@K에서 제외합니다.",
        "",
        "| 기대 경로 | 문항 | 라우팅 정답 | 라우팅 오답 | 라우팅 정확도 | Hit@1 | Hit@3 | Hit@5 | Hit@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for route in ROUTES:
        group = summary["by_expected_route"][route]
        values = [
            route, str(group["questions"]), str(group["correct"]),
            str(group["incorrect"]),
            "N/A" if group["routing_accuracy"] is None else f"{group['routing_accuracy'] * 100:.2f}%",
            *(_fraction(group[f"hit_at_{k}"]["hits"], group["questions"], group[f"hit_at_{k}"]["rate"]) for k in KS),
        ]
        lines.append("| " + " | ".join(values) + " |")
    routing = summary["routing"]
    overall_values = [
        "전체", str(routing["questions"]), str(routing["correct"]),
        str(routing["incorrect"]), f"{routing['accuracy'] * 100:.2f}%",
        *(_fraction(summary["hit_at_k"][str(k)]["hits"], summary["search_questions"], summary["hit_at_k"][str(k)]["rate"]) for k in KS),
    ]
    lines.extend([
        "| " + " | ".join(overall_values) + " |", "",
        "## 라우팅 혼동 행렬", "",
        "행은 기대 경로, 열은 실제 경로입니다.", "",
        "| 기대 \\ 실제 | VECTOR | GRAPH | HYBRID | REJECT |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for route in ROUTES:
        lines.append("| " + " | ".join([
            route, *(str(summary["routing"]["confusion_matrix"][route][actual]) for actual in ROUTES)
        ]) + " |")
    lines.extend([
        "", "## Multi-hop 근거 충족", "",
        f"기대 그래프 경로가 있는 {multihop_summary['graph_path_questions']}문항을 평가합니다. 기대 경로의 Top-10 그래프 결과 중 관계의 방향·순서 및 노드가 가장 잘 일치하는 단일 경로를 선택합니다. 원문 근거도 해당 경로의 관계에서만 확인합니다. 최초 라우팅도 일치해야 완전 성공이며, 적용 대상이 아닌 {multihop_summary['not_applicable']}문항은 제외합니다.",
        "",
        "| 항목 | 통과 / 대상 |",
        "| --- | ---: |",
    ])
    total = multihop_summary["graph_path_questions"]
    labels = (
        ("node_match", "필수 노드 연결"),
        ("relation_id_match", "필수 관계 ID"),
        ("relation_type_match", "필수 관계 타입"),
        ("hop_match", "최소 hop"),
        ("source_link_match", "관계별 원문 청크 연결"),
        ("required_evidence_match", "기대 원문 근거"),
    )
    for key, label in labels:
        lines.append(f"| {label} | {multihop_summary['component_pass_counts'][key]}/{total} |")
    lines.extend([
        f"| **완전 성공** | **{_fraction(multihop_summary['full_success'], total, multihop_summary['full_success_rate'])}** |",
        "", "### 실패 원인별 문항 수", "",
        "실패 관측 항목은 중복 집계합니다. 필수 노드가 경로에 없다는 사실만으로 엔티티 추출 단계의 오류라고 단정하지 않습니다.", "",
        "| 원인 | 문항 수 |", "| --- | ---: |",
    ])
    for key, label in (
        ("routing_mismatch", "라우팅 불일치"),
        ("required_nodes_missing", "필수 노드 경로 누락"),
        ("relation_or_hop_missing", "관계 또는 hop 부족"),
        ("edge_original_link_missing", "관계별 원문 연결 누락"),
        ("expected_evidence_missing", "기대 원문 근거 누락"),
    ):
        lines.append(f"| {label} | {multihop_summary['failure_reason_counts'].get(key, 0)} |")
    (output_dir / "retrieval_evaluation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--top10", type=Path, default=REPORT_DIR / "top10_results.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--relations", type=Path, default=PROJECT_ROOT / "data" / "processed" / "relations.jsonl")
    parser.add_argument("--collect", action="store_true", help="Refresh Top-10 from Neo4j")
    parser.add_argument(
        "--check-neo4j", action="store_true",
        help="Check Neo4j connectivity using this project's .env, then exit",
    )
    args = parser.parse_args()
    cases = load_golden_cases(args.golden_set)
    router = QuestionRouter()
    graph = None
    if args.collect or args.check_neo4j:
        print(f"Python executable: {sys.executable}", flush=True)
        graph = _open_neo4j_for_evaluation()
        if args.check_neo4j and not args.collect:
            graph.close()
            print("Neo4j preflight passed.")
            return
    if args.collect:
        collect_top10(cases, args.top10, router, graph)
    top10 = _read_top10(args.top10, cases)
    details, summary = evaluate(cases, top10, router)
    multihop_details, multihop_summary = evaluate_multihop_paths(
        cases, top10, details, args.relations
    )
    write_reports(details, summary, multihop_details, multihop_summary, args.output_dir)
    print(f"Wrote {len(details)} questions to {args.output_dir}")
    print(f"Overall routing accuracy: {summary['routing']['accuracy']:.2%}")
    for route in SEARCH_ROUTES:
        route_summary = summary["by_expected_route"][route]
        hits = ", ".join(
            f"Hit@{k}={route_summary[f'hit_at_{k}']['hits']}/{route_summary['questions']}"
            for k in KS
        )
        accuracy = route_summary["routing_accuracy"]
        print(
            f"{route}: routing accuracy="
            f"{'N/A' if accuracy is None else f'{accuracy:.2%}'}, {hits}"
        )
    print(
        "Multi-hop full success: "
        f"{multihop_summary['full_success']}/{multihop_summary['graph_path_questions']} "
        f"({multihop_summary['full_success_rate']:.2%})"
    )


if __name__ == "__main__":
    try:
        main()
    except ConnectionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
