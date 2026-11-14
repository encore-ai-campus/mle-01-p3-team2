"""Render the fixed-criteria diagnosis for the seven GRAPH multi-hop misses."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluation.evaluate_retrieval import PROJECT_ROOT


REPORTS = PROJECT_ROOT / "data/evaluation/reports"
TARGET_IDS = ("GS012", "GS018", "GS026", "GS027", "GS036", "GS066", "GS067")


def exact_path_rank(row: dict, case: dict) -> int | None:
    for hit in row["paths"]:
        if hit.get("relation_ids") != case.get("expected_relation_ids"):
            continue
        if [node.get("entity_id") for node in hit.get("path_nodes") or []] != case.get("expected_entity_ids"):
            continue
        if not set(case.get("expected_evidence_ids") or []) & set(hit.get("evidence_ids") or []):
            continue
        if not set(case.get("expected_source_chunk_ids") or []) & set(hit.get("chunk_ids") or []):
            continue
        return int(hit["rank"])
    return None


def main() -> None:
    cases = {
        case["case_id"]: case for case in (
            json.loads(line) for line in (PROJECT_ROOT / "data/evaluation/golden_set_final.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()
        )
    }
    neo4j = {row["case_id"]: row for row in json.loads(
        (REPORTS / "graph_miss7_neo4j_path_check.json").read_text(encoding="utf-8")
    )}
    baseline = json.loads((REPORTS / "graph_candidate_sweep_miss7_diagnostic.json")
                          .read_text(encoding="utf-8"))
    changed = json.loads((REPORTS / "graph_candidate_sweep_miss7_after_wide_prefetch.json")
                         .read_text(encoding="utf-8"))
    final = json.loads((REPORTS / "graph_miss7_final_full_recheck.json")
                       .read_text(encoding="utf-8"))

    by_size = {
        size: {row["case_id"]: row for row in baseline["cases"]
               if row["candidate_pool_size"] == size}
        for size in (30, 50, 100)
    }
    lines = [
        "# GRAPH Multi-hop 7개 미회수 문항 진단", "",
        "정답 ID는 평가·사후 진단에만 사용했습니다. 검색 코드는 변경하지 않았고, 점수·정답 기준은 그대로 유지했습니다.", "",
        "## 문항별 진단", "",
        "| ID | 앵커 매칭 | Neo4j 정방향 2-hop | 관계 타입 | 대상 경로 수 | Gold 순위(대상 고정) | 후보30 | 후보50 | 후보100 | 판정 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for case_id in TARGET_IDS:
        case = cases[case_id]
        graph_case = by_size[30][case_id]
        path_check = neo4j[case_id]
        ranks = [exact_path_rank(by_size[size][case_id], case) for size in (30, 50, 100)]
        relations = " → ".join((path_check.get("neo4j_path") or {}).get("relation_types") or []) or "미확인"
        outcome = (
            "Top-10 컷오프" if ranks[0] and ranks[0] > 10 else
            "Cypher 사전 후보 컷오프" if ranks[0] is None and any(rank for rank in ranks[1:]) else
            "점수 동률·순위 민감" if any(rank for rank in ranks) else
            "경로 조건 누락"
        )
        lines.append(
            f"| {case_id} | {graph_case['entity_match_count']} ({', '.join(graph_case['matched_start_entity_ids'])}) "
            f"| {'확인' if path_check['gold_nodes_exist_in_directed_path'] else '없음'} | {relations} "
            f"| {path_check['target_specific_valid_path_count']} "
            f"| {path_check['gold_rank_within_target_specific_paths']} "
            f"| {ranks[0] or '미회수'} | {ranks[1] or '미회수'} | {ranks[2] or '미회수'} | {outcome} |"
        )

    lines.extend([
        "", "모든 문항에서 시작 작물 앵커가 1개 매칭됐습니다. Gold 노드·관계는 Neo4j에서 정확한 방향으로 연결되고, 관계 ID에 연결된 정답 evidence와 chunk도 확인됐습니다.",
        "‘대상 경로 수’는 해당 작물과 병해충 대상 사이의 등록 약제 2-hop 경로 조합 수입니다. 각 질문에는 79~257개가 존재하며, 질문에는 특정 제품명이나 속성값이 없습니다.",
        "Gold가 Top-10 밖에 있는 경우는 경로 방향/hop 생성 실패가 아니라 후보 사전 제한 또는 동점 경로 순서에 따른 순위 탈락입니다. Gold 순위는 현재 점수 함수로 대상 노드를 고정해 후보군을 직접 정렬한 진단 순위이며 검색에는 사용되지 않습니다.",
        "", "## 30·50·100 후보 스윕", "",
        "| 후보 수 | GRAPH Hit@10 | Multi-hop 후보 완전 경로 | GRAPH Multi-hop 완전 성공@10 | 평균 검색(ms) |",
        "|---:|---:|---:|---:|---:|",
    ])
    for item in baseline["summaries"]:
        lines.append(
            f"| {item['candidate_pool_size']} | {item['graph_hit_at_10']}/{item['graph_questions']} "
            f"| {item['multihop_candidate_full_path_recall']}/{item['multihop_questions']} "
            f"| {item['multihop_top10_full_success']}/{item['multihop_questions']} "
            f"| {item['mean_search_ms']:.1f} |"
        )
    lines.extend([
        "", "## 후보 확대 실험과 최종 결과", "",
        "| 설정 | GRAPH 21 Hit@10 | VECTOR 19 Hit@10 | HYBRID 17 Hit@10 | Multi-hop 완전 성공@10 | 평균 GRAPH/HYBRID 검색(ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    final_summary = final["summary"]
    lines.append(
        f"| 기본 후보30 (최종 재검증) | {final_summary['GRAPH']['hit_at_10']}/21 "
        f"| {final_summary['VECTOR']['hit_at_10']}/19 | {final_summary['HYBRID']['hit_at_10']}/17 "
        f"| {final['multihop_integrated']['full_success']}/17 "
        f"| {final_summary['GRAPH']['mean_elapsed_ms']:.1f}/{final_summary['HYBRID']['mean_elapsed_ms']:.1f} |"
    )
    lines.append(
        "| 확대 사전 탐색30 (폐기) | 9/21 | 15/19 | 7/17 | 7/17 | 920.6/1357.4 |"
    )
    lines.extend([
        "", "확대 사전 탐색 후보 수별 GRAPH 단독 결과:", "",
        "| 후보 수 | GRAPH Hit@10 | Multi-hop 후보 완전 경로 | GRAPH Multi-hop 완전 성공@10 | 평균 검색(ms) |",
        "|---:|---:|---:|---:|---:|",
    ])
    for item in changed["summaries"]:
        lines.append(
            f"| {item['candidate_pool_size']} | {item['graph_hit_at_10']}/21 "
            f"| {item['multihop_candidate_full_path_recall']}/17 "
            f"| {item['multihop_top10_full_success']}/17 | {item['mean_search_ms']:.1f} |"
        )
    lines.extend([
        "", "후보 사전 탐색을 넓히고 근거 사용 조건 점수를 보강한 실험은 GRAPH 후보 회수를 10/17→12/17, 14/17, 15/17로 높였지만, 후보30/50/100 GRAPH Top-10 완전 성공은 7/17, 6/17, 6/17이었습니다. 평균 검색 시간은 957/1,154/1,433ms였고, 기본 후보30 실제 재평가에서 GRAPH 9/21, VECTOR 15/19, HYBRID Hit@10 7/17, 통합 Multi-hop 7/17을 기록해 기존 통합 성공 9/17보다 낮았습니다.",
        "이 실험은 기존 성공을 보존하지 못해 검색 코드에는 남기지 않았습니다. 현재 기본 동작을 재검증해 GRAPH 9/21, VECTOR 15/19, HYBRID 9/17, 통합 Multi-hop 9/17을 확인했습니다.",
        "", "## 우선순위", "",
        "1. P0: Cypher의 `LIMIT` 이전 사전 정렬이 node.name 표면 일치 중심인 구조를 개선해야 합니다. Python의 richer relevance 점수는 이미 잘린 경로에는 적용되지 않습니다.",
        "2. P0: 동일한 작물·대상·관계의 약제 경로가 수백 개입니다. 질문이 특정 제품·사용값을 정하지 않는 문항은 gold 제품을 자연어 근거 없이 식별할 수 없습니다. 검색은 다수 후보를 주되 단일 Gold ID 우선순위는 임의화하지 않아야 합니다.",
        "3. P1: 질문에 제품명/상표/사용 시기 조건이 포함된 경우에만 속성·원문 일치 점수를 강화하고, 고정 평가셋 외 질문으로 퇴행을 점검해야 합니다.",
        "", "### 생성 근거 파일", "",
        "- `graph_miss7_neo4j_path_check.json`: Neo4j 방향 경로, evidence/chunk 연결, 후보 규모와 점수 순위",
        "- `graph_candidate_sweep_miss7_diagnostic.json`: 변경 전 후보 수별 30/50/100 스윕",
        "- `graph_candidate_sweep_miss7_after_wide_prefetch.json`: 폐기한 확대 후보 실험",
        "- `graph_miss7_final_full_recheck.json`: 되돌린 기본 설정의 GRAPH/VECTOR/HYBRID 통합 재평가",
        "",
    ])
    target = REPORTS / "graph_miss7_diagnosis.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
