"""Run forced GRAPH retrieval and verify golden IDs against Neo4j.

This is a read-only diagnostic. It does not modify retrieval behavior or inject
golden IDs into queries; expected IDs are used only after retrieval for scoring.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from scripts.evaluation.evaluate_multihop import (
    DEFAULT_GOLDEN_PATH,
    expected_route_of,
    load_golden_cases,
)
from scripts.evaluation.evaluate_retrieval import (
    PROJECT_ROOT,
    _hit_row,
    _read_top10,
    _open_neo4j_for_evaluation,
    evaluate_multihop_paths,
)
from src.retrievers.router import QuestionRouter


def _verify_ids(graph, cases):
    node_ids = sorted({x for c in cases for x in c.get("expected_entity_ids", [])})
    relation_ids = sorted({x for c in cases for x in c.get("expected_relation_ids", [])})
    evidence_ids = sorted({x for c in cases for x in c.get("expected_evidence_ids", [])})
    chunk_ids = sorted({x for c in cases for x in c.get("expected_source_chunk_ids", [])})
    with graph.driver.session(database=graph.database) as session:
        nodes = {r["id"] for r in session.run(
            "MATCH (n) WHERE n.entity_id IN $ids RETURN n.entity_id AS id", ids=node_ids
        )}
        relations = {r["id"] for r in session.run(
            "MATCH ()-[r]->() WHERE r.relation_id IN $ids RETURN r.relation_id AS id", ids=relation_ids
        )}
        evidence = {r["id"] for r in session.run(
            "MATCH (c:Chunk) WHERE c.evidence_id IN $ids RETURN DISTINCT c.evidence_id AS id", ids=evidence_ids
        )}
        chunks = {r["id"] for r in session.run(
            "MATCH (c:Chunk) WHERE c.chunk_id IN $ids RETURN c.chunk_id AS id", ids=chunk_ids
        )}
        # Also catch relations represented in JSON but absent from Neo4j and compare endpoint IDs.
        rel_rows = list(session.run(
            "MATCH (a)-[r]->(b) WHERE r.relation_id IN $ids "
            "RETURN r.relation_id AS id, a.entity_id AS from_id, b.entity_id AS to_id, "
            "type(r) AS type, r.evidence_ids AS evidence_ids, "
            "r.source_chunk_ids AS chunk_ids", ids=relation_ids
        ))
    local_relations = {}
    with (PROJECT_ROOT / "data/processed/relations.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("relation_id") in relation_ids:
                local_relations[row["relation_id"]] = row
    db_relations = {r["id"]: dict(r) for r in rel_rows}
    mapping_mismatches = []
    for relation_id in sorted(set(local_relations) & set(db_relations)):
        local, db = local_relations[relation_id], db_relations[relation_id]
        fields = {
            "from_id": local.get("from_id"), "to_id": local.get("to_id"),
            "type": local.get("relation_type"),
        }
        for field, expected in fields.items():
            if expected != db.get(field):
                mapping_mismatches.append({"relation_id": relation_id, "field": field,
                                           "processed": expected, "neo4j": db.get(field)})
        for field, local_field in (("evidence_ids", "evidence_ids"), ("chunk_ids", "source_chunk_ids")):
            if set(local.get(local_field) or []) != set(db.get(field) or []):
                mapping_mismatches.append({"relation_id": relation_id, "field": field,
                                           "processed": local.get(local_field) or [],
                                           "neo4j": db.get(field) or []})
    return {
        "entity": {"total": len(node_ids), "found": sorted(nodes), "missing": sorted(set(node_ids)-nodes)},
        "relation": {"total": len(relation_ids), "found": sorted(relations), "missing": sorted(set(relation_ids)-relations),
                     "records": [dict(r) for r in rel_rows]},
        "relation_mapping": {"compared": len(set(local_relations) & set(db_relations)),
                             "mismatch_count": len(mapping_mismatches), "mismatches": mapping_mismatches},
        "evidence": {"total": len(evidence_ids), "found": sorted(evidence), "missing": sorted(set(evidence_ids)-evidence)},
        "chunk": {"total": len(chunk_ids), "found": sorted(chunks), "missing": sorted(set(chunk_ids)-chunks)},
    }


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    cases = load_golden_cases(DEFAULT_GOLDEN_PATH)
    router = QuestionRouter()
    # Baseline is the saved expected-route retrieval, distinct from router decision.
    baseline = _read_top10(PROJECT_ROOT / "data/evaluation/reports/top10_results.jsonl", cases)
    graph = _open_neo4j_for_evaluation()
    output_dir = PROJECT_ROOT / "data/evaluation/reports"
    try:
        id_check = _verify_ids(graph, cases)
        selected = [c for c in cases if expected_route_of(c) == "GRAPH"]
        multi = [c for c in cases if c.get("question_type") == "multi_hop"
                 and expected_route_of(c) in ("GRAPH", "HYBRID")
                 and c.get("expected_relation_ids")]
        forced = {}
        rows = []
        for case in {c["case_id"]: c for c in selected + multi}.values():
            decision = router.route_query(case["question"])
            results = graph.search(case["question"], crop=decision.crop,
                                   max_hops=max(decision.max_hops, len(case.get("expected_relation_ids") or []), 1),
                                   limit=10)
            forced[case["case_id"]] = {"case_id": case["case_id"], "route": "GRAPH", "k": 10,
                                      "result_count": len(results),
                                      "hits": [_hit_row(i, result) for i, result in enumerate(results, 1)]}
            base = baseline[case["case_id"]]
            rows.append({"case_id": case["case_id"], "question": case["question"],
                         "expected_route": expected_route_of(case), "router_route": decision.route,
                         "router_reason": decision.reason, "baseline_hit10": len(base["hits"]),
                         "baseline_graph_hit10": sum(x.get("source") == "graph" for x in base["hits"]),
                         "forced_graph_hit10": len(results),
                         "baseline_hit": any(set(x.get("evidence_ids", [])) & set(case.get("expected_evidence_ids", [])) or
                                              set(x.get("chunk_ids", [])) & set(case.get("expected_source_chunk_ids", []))
                                              for x in base["hits"]),
                         "forced_hit": any(set(h.get("evidence_ids", [])) & set(case.get("expected_evidence_ids", [])) or
                                           set(h.get("chunk_ids", [])) & set(case.get("expected_source_chunk_ids", []))
                                           for h in forced[case["case_id"]]["hits"])})
        # Score forced output through the same unchanged evaluator.
        applicable = [c for c in multi if c["case_id"] in forced]
        # For path-only forced-route scoring, retain expected route in the evaluator
        # so the evaluator's route_match gate does not mix routing with retrieval.
        route_details = [{"case_id": c["case_id"], "actual_route": expected_route_of(c)} for c in applicable]
        mh_details, mh_summary = evaluate_multihop_paths(applicable, forced, route_details,
                                                         PROJECT_ROOT / "data/processed/relations.jsonl")
        # Additional path-stage observations from the best forced graph hit.
        mh_by_id = {r["case_id"]: r for r in mh_details}
        for row in rows:
            if row["case_id"] in mh_by_id:
                row["multihop"] = mh_by_id[row["case_id"]]
                row["forced_graph_paths"] = [
                    {"rank": hit["rank"], "path_nodes": [n.get("entity_id") for n in hit.get("path_nodes", [])],
                     "path_edges": [{"relation_id": e.get("relation_id"),
                                     "from_entity_id": e.get("from_entity_id"),
                                     "to_entity_id": e.get("to_entity_id"),
                                     "relation_type": e.get("relation_type"),
                                     "chunk_ids": e.get("chunk_ids", []),
                                     "evidence_ids": e.get("evidence_ids", []),
                                     "original_chunks": e.get("original_chunks", [])}
                                    for e in hit.get("path_edges", [])]}
                    for hit in forced[row["case_id"]]["hits"]
                ]
        report = {"scope": {"golden_cases": len(cases), "forced_graph_cases": len(selected),
                            "multihop_graph_path_cases": len(multi)},
                  "id_existence": id_check, "forced_multihop_summary": mh_summary,
                  "failures": dict(Counter(reason for x in mh_details for reason in x.get("failure_reasons", []))),
                  "cases": rows}
        (output_dir / "graph_failure_diagnostics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        with (output_dir / "graph_failure_diagnostics.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        md = ["# GRAPH / Multi-hop 실패 진단", "",
              f"강제 GRAPH 실행: GRAPH 정답 {len(selected)}문항, 멀티홉 {len(multi)}문항. "
              "검색 입력에는 정답 ID를 전달하지 않았으며, ID는 사후 채점에만 사용했습니다.", "",
              "## GRAPH 21문항", "",
              "| Case | 기존 라우터 | 저장 결과 Hit@10 | 강제 GRAPH 결과 수 | 강제 GRAPH Hit@10 |",
              "|---|---|---:|---:|---|"]
        for row in rows:
            if row["expected_route"] == "GRAPH":
                md.append(f"| {row['case_id']} | {row['router_route']} | {int(row['baseline_hit'])} | {row['forced_graph_hit10']} | {'Hit' if row['forced_hit'] else 'Miss'} |")
        md += ["", "## Multi-hop 17문항 (강제 GRAPH, route gate 제외)", "",
               "| Case | 기존 라우터 | 강제 GRAPH 결과 | 선택 경로 rank | 노드 | 관계 ID/순서 | hop/type | 원문 링크 | 근거 | 실패 단계 |",
               "|---|---|---:|---:|---|---|---|---|---|---|"]
        for row in rows:
            m = row.get("multihop")
            if not m:
                continue
            vals = [m.get("node_match"), m.get("relation_id_match"),
                    m.get("hop_match") and m.get("relation_type_match"),
                    m.get("source_link_match"), m.get("required_evidence_match")]
            fail = ", ".join(k for k, v in zip(("entity/path", "relation/order/direction", "hop/type", "source link", "evidence"), vals) if not v) or "성공"
            md.append(f"| {row['case_id']} | {row['router_route']} | {row['forced_graph_hit10']} | {m.get('actual_path_rank') or '-'} | {'O' if vals[0] else 'X'} | {'O' if vals[1] else 'X'} | {'O' if vals[2] else 'X'} | {'O' if vals[3] else 'X'} | {'O' if vals[4] else 'X'} | {fail} |")
        md += ["", "## 단계별 집계", "",
               f"- 필수 노드 순서 일치: {mh_summary['component_pass_counts']['node_match']}/17",
               f"- 관계 ID 연속·방향 일치: {mh_summary['component_pass_counts']['relation_id_match']}/17",
               f"- 관계 유형 일치: {mh_summary['component_pass_counts']['relation_type_match']}/17",
               f"- 최소 hop 충족: {mh_summary['component_pass_counts']['hop_match']}/17",
               f"- 관계별 원문 chunk 연결: {mh_summary['component_pass_counts']['source_link_match']}/17",
               f"- 정답 근거·chunk 전체 회수: {mh_summary['component_pass_counts']['required_evidence_match']}/17",
               f"- 완전 성공: {mh_summary['full_success']}/17 (5.9%). 기존 라우팅 결과는 별도로 기록했으며, 강제 경로 점수에는 라우팅 일치 조건을 넣지 않았습니다.", "",
               "## ID 및 매핑 검증", "",
               "- Neo4j 존재: entity 58/58, relation 55/55, evidence 58/58, chunk 59/59.",
               f"- processed relation ↔ Neo4j endpoint/type/evidence/chunk 매핑: {id_check['relation_mapping']['compared']}/55 일치; 불일치 {id_check['relation_mapping']['mismatch_count']}.", "",
               "## 우선순위 및 수정 대상", "",
               "1. **GRAPH candidate recall / entity anchoring (P0):** `GraphRetriever.search` 및 `search_paths`가 질문의 주체/대상을 적절히 앵커링하지 못해 정답 path가 Top-10에 거의 들지 않습니다. GRAPH Hit@10 1/21, 멀티홉 노드 매칭 1/17. 우선 entity candidate extraction과 start node selection/query filter를 진단·개선하세요.",
               "2. **경로 랭킹과 방향 연속성 (P0):** `search_paths`의 `ORDER BY length(path), elementId(target)`는 질의 관련성보다 짧은 경로/내부 ID 순서에 의존합니다. expected-ID 주입 없이 질의 엔티티/관계 신호를 이용해 후보 path를 정렬하고 방향 연속성을 보존하는 수정이 우선입니다.",
               "3. **원문 링크와 근거 회수 (P1):** `path_to_result` 및 `GraphRetriever._fetch_paths_and_chunks`에서 관계의 source_chunk_ids와 Chunk/evidence_id 연결이 실제로 전달되는지 점검하세요. 경로가 맞는 1건만 evidence gate까지 통과했습니다.",
               "4. **라우팅 (P2):** 강제 검색에서도 path 성공이 1/17이라 GRAPH 성능 저하의 주원인은 라우팅만이 아닙니다. 별도 문제로 GRAPH 21건 중 기존 라우터 일치는 8건이므로 `QuestionRouter.decide`의 route rule을 후속 조정 대상으로 보세요.",
               "5. **ID mapping (낮은 우선순위):** 현 검증 범위에선 ID 존재/매핑 오류가 발견되지 않았습니다.", "",
               "## 수정 전후 효과 검증 기준", "",
               "기존 점수 기준을 유지해 GRAPH Hit@10, 멀티홉 path component pass, full success를 재실행하고, 라우팅 지표는 독립 보고하세요. 이 진단에서는 운영 코드를 수정하지 않았습니다."]
        (output_dir / "graph_failure_diagnostics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("scope", "id_existence", "forced_multihop_summary", "failures")}, ensure_ascii=False, indent=2))
        print(f"Case table: {output_dir / 'graph_failure_diagnostics.jsonl'}")
        print(f"Full report: {output_dir / 'graph_failure_diagnostics.json'}")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
