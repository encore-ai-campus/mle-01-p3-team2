"""Read-only exact-match and alternative-path audit for the 17 multi-hop cases."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scripts.evaluation.evaluate_multihop import expected_route_of, load_golden_cases
from scripts.evaluation.evaluate_retrieval import PROJECT_ROOT
from src.retrievers.graph_retriever import GraphRetriever


REPORTS = PROJECT_ROOT / "data/evaluation/reports"
GOLDEN_PATH = PROJECT_ROOT / "data/evaluation/golden_set_final.jsonl"
RELATIONS_PATH = PROJECT_ROOT / "data/processed/relations.jsonl"
MULTIHOP_IDS = {
    "GS011", "GS012", "GS014", "GS018", "GS020", "GS023", "GS026", "GS027",
    "GS035", "GS036", "GS052", "GS054", "GS059", "GS061", "GS062", "GS066", "GS067",
}

FIELD_KEYS = {
    "제품/품목": ("product", "brand", "active_ingredient"),
    "사용 시기": ("use_timing", "application_method"),
    "수확 전 간격": ("use_timing",),
    "희석 배수": ("dilution",),
    "최대 사용 횟수": ("use_count",),
    "살포 간격": ("application_method", "use_timing"),
    "상표": ("brand",),
    "등록번호": ("registration_no",),
}


def parse_attributes(node: Any) -> dict[str, Any]:
    raw = node.get("attributes")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def requested_fields(question: str) -> list[str]:
    q = re.sub(r"\s+", "", question).lower()
    fields = ["제품/품목"]
    if any(cue in q for cue in ("언제", "시기", "적기", "발생", "발병", "살포", "처리", "정식", "애벌레")):
        fields.append("사용 시기")
    if "수확" in q and any(cue in q for cue in ("전", "까지", "시기")):
        fields.append("수확 전 간격")
    if any(cue in q for cue in ("희석", "배수")):
        fields.append("희석 배수")
    if any(cue in q for cue in ("횟수", "몇 회", "회까지")):
        fields.append("최대 사용 횟수")
    if "간격" in q:
        fields.append("살포 간격")
    if "상표" in q or "브랜드" in q:
        fields.append("상표")
    if "등록번호" in q:
        fields.append("등록번호")
    return list(dict.fromkeys(fields))


def has_value(attrs: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return any(
        attrs.get(key) is not None and str(attrs.get(key)).strip() not in {"", "-", "없음"}
        for key in keys
    )


def answer_terms(attrs: dict[str, Any], name: str) -> set[str]:
    values = [name]
    for key in ("product", "brand", "active_ingredient", "formulation", "registration_no", "aliases", "synonyms"):
        value = attrs.get(key)
        if isinstance(value, (list, tuple, set)):
            values.extend(str(item) for item in value)
        elif value is not None:
            values.append(str(value))
    return {re.sub(r"\s+", "", value).casefold() for value in values if len(value.strip()) >= 2}


def question_names_product(question: str, attrs: dict[str, Any], name: str,
                           crop_name: str, target_name: str) -> bool:
    q = re.sub(r"\s+", "", question).casefold()
    # Crop and target names occur in many composite pesticide nodes, so they
    # cannot count as product-identifying clues by themselves.
    for generic in (crop_name, target_name):
        normalized = re.sub(r"\s+", "", generic).casefold()
        if normalized:
            q = q.replace(normalized, "")
    terms = [(re.sub(r"\s+", "", name.split(" - ", 1)[0]).casefold(), 4)]
    for key in ("product", "brand", "active_ingredient", "registration_no", "상표", "품목명", "등록번호"):
        value = attrs.get(key)
        if value is not None:
            # Korean brand names are often only two or three characters. Keep
            # the threshold low for explicit brand/registration fields while
            # requiring longer strings for generic product/ingredient terms.
            minimum = 2 if key in ("brand", "상표", "registration_no", "등록번호") else 4
            terms.append((re.sub(r"\s+", "", str(value).split(" - ", 1)[0]).casefold(), minimum))
    return any(len(term) >= minimum and term in q for term, minimum in terms)


def path_to_row(path: Any, chunks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    nodes = list(path.nodes)
    rels = list(path.relationships)
    edges = []
    for index, rel in enumerate(rels):
        chunk_ids = [str(x) for x in (rel.get("source_chunk_ids") or [])]
        evidence_ids = [str(x) for x in (rel.get("evidence_ids") or [])]
        edges.append({
            "relation_id": str(rel.get("relation_id") or ""),
            "relation_type": str(rel.type),
            "from_entity_id": str(nodes[index].get("entity_id") or ""),
            "to_entity_id": str(nodes[index + 1].get("entity_id") or ""),
            "evidence_ids": evidence_ids,
            "chunk_ids": chunk_ids,
            "chunks": [chunks[cid] for cid in chunk_ids if cid in chunks],
        })
    middle = nodes[1] if len(nodes) > 2 else {}
    middle_attrs = parse_attributes(middle)
    all_chunks = [chunk for edge in edges for chunk in edge["chunks"]]
    return {
        "node_ids": [str(node.get("entity_id") or "") for node in nodes],
        "node_names": [str(node.get("name") or "") for node in nodes],
        "node_types": [str(node.get("entity_type") or next(iter(node.labels), "")) for node in nodes],
        "relation_ids": [edge["relation_id"] for edge in edges],
        "relation_types": [edge["relation_type"] for edge in edges],
        "evidence_ids": sorted({value for edge in edges for value in edge["evidence_ids"]}),
        "chunk_ids": sorted({value for edge in edges for value in edge["chunk_ids"]}),
        "all_edges_have_source_text": bool(edges) and all(
            any(str(chunk.get("text") or "").strip() for chunk in edge["chunks"])
            for edge in edges
        ),
        "middle_entity_id": str(middle.get("entity_id") or ""),
        "middle_name": str(middle.get("name") or ""),
        "middle_type": str(middle.get("entity_type") or next(iter(middle.labels), "")),
        "crop_name": str(nodes[0].get("name") or ""),
        "target_name": str(nodes[-1].get("name") or ""),
        "middle_attributes": middle_attrs,
        "edges": edges,
        "path_signature": (
            tuple(str(node.get("entity_id") or "") for node in nodes),
            tuple(str(rel.get("relation_id") or "") for rel in rels),
        ),
    }


def hit_exact_path(hit: dict[str, Any], case: dict[str, Any]) -> bool:
    if [node.get("entity_id") for node in hit.get("path_nodes") or []] != case.get("expected_entity_ids"):
        return False
    rel_ids = [edge.get("relation_id") for edge in hit.get("path_edges") or []]
    if rel_ids != case.get("expected_relation_ids"):
        return False
    if not set(case.get("expected_evidence_ids") or []) & set(hit.get("evidence_ids") or []):
        return False
    return bool(set(case.get("expected_source_chunk_ids") or []) & set(hit.get("chunk_ids") or []))


def main() -> None:
    cases = [case for case in load_golden_cases(GOLDEN_PATH)
             if case["case_id"] in MULTIHOP_IDS and case.get("expected_relation_ids")]
    relations = [json.loads(line) for line in RELATIONS_PATH.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    relation_type = {str(row.get("relation_id")): str(row.get("relation_type") or "")
                     for row in relations}
    baseline_sweep = json.loads((REPORTS / "graph_candidate_sweep_miss7_diagnostic.json")
                                .read_text(encoding="utf-8"))
    capture_path = REPORTS / "graph_miss7_final_full_recheck.json"
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    candidate_cases = {
        size: {row["case_id"]: row for row in baseline_sweep["cases"]
               if row["candidate_pool_size"] == size and row["case_id"] in MULTIHOP_IDS}
        for size in (30, 50, 100)
    }

    load_dotenv(PROJECT_ROOT / ".env")
    graph = GraphRetriever.from_env()
    cache: dict[tuple[str, str, tuple[str, ...]], list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    try:
        session_options = {"database": graph.database} if graph.database != "neo4j" else {}
        with graph.driver.session(**session_options) as session:
            for case in cases:
                case_id = case["case_id"]
                expected_nodes = list(case.get("expected_entity_ids") or [])
                expected_relations = list(case.get("expected_relation_ids") or [])
                expected_types = tuple(relation_type.get(rel, "") for rel in expected_relations)
                if len(expected_nodes) != 3 or len(expected_types) != 2 or not all(expected_types):
                    rows.append({"case_id": case_id, "diagnostic_error": "expected path is not a typed 2-hop path"})
                    continue
                if not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value) for value in expected_types):
                    raise ValueError(f"Unsafe relationship type for {case_id}")
                key = (expected_nodes[0], expected_nodes[2], expected_types)
                if key not in cache:
                    query = (
                        f"MATCH p=(s)-[:`{expected_types[0]}`]->(m)-[:`{expected_types[1]}`]->(t) "
                        "WHERE s.entity_id=$start_id AND t.entity_id=$target_id "
                        "RETURN p AS path LIMIT $path_limit"
                    )
                    params = {"start_id": key[0], "target_id": key[1], "path_limit": 2000}
                    paths, chunks = graph._fetch_paths_and_chunks(session, query, params)
                    cache[key] = [path_to_row(path, chunks) for path in paths]
                alternatives = cache[key]
                gold_signature = (tuple(expected_nodes), tuple(expected_relations))
                gold_rows = [row for row in alternatives if row["path_signature"] == gold_signature]
                gold = gold_rows[0] if gold_rows else None
                question = case["question"]
                requested = requested_fields(question)
                candidates = []
                for alternative in alternatives:
                    attrs = alternative["middle_attributes"]
                    fields_present = [field for field in requested
                                      if has_value(attrs, FIELD_KEYS[field])]
                    explicit_product = question_names_product(
                        question, attrs, alternative["middle_name"],
                        alternative["crop_name"], alternative["target_name"],
                    )
                    candidate = {
                        "middle_entity_id": alternative["middle_entity_id"],
                        "middle_name": alternative["middle_name"],
                        "middle_type": alternative["middle_type"],
                        "distinct_from_gold_product": bool(gold and alternative["middle_entity_id"] != gold["middle_entity_id"]),
                        "requested_fields_present": fields_present,
                        "requested_field_coverage": len(fields_present) / max(len(requested), 1),
                        "explicit_product_mentioned": explicit_product,
                        "source_complete": alternative["all_edges_have_source_text"],
                        "registration_status_field": attrs.get("registration_status"),
                        "recommendation_eligible_field": attrs.get("recommendation_eligible"),
                        "evidence_ids": alternative["evidence_ids"],
                        "chunk_ids": alternative["chunk_ids"],
                        "path_signature": alternative["path_signature"],
                    }
                    candidates.append(candidate)
                unique_by_middle: dict[str, dict[str, Any]] = {}
                for candidate in candidates:
                    unique_by_middle.setdefault(candidate["middle_entity_id"], candidate)
                gold_candidate = unique_by_middle.get(gold["middle_entity_id"]) if gold else None
                explicit_middle_ids = {
                    candidate["middle_entity_id"] for candidate in unique_by_middle.values()
                    if candidate["explicit_product_mentioned"]
                }
                gold_explicitly_named = bool(
                    gold_candidate and gold_candidate["middle_entity_id"] in explicit_middle_ids
                )
                question_disambiguates_gold = gold_explicitly_named and len(explicit_middle_ids) == 1
                plausible_alternatives = [
                    candidate for candidate in unique_by_middle.values()
                    if candidate["distinct_from_gold_product"]
                    and candidate["source_complete"]
                    and candidate["requested_field_coverage"] == 1.0
                    and (not explicit_middle_ids or candidate["middle_entity_id"] in explicit_middle_ids)
                ]
                query_mentions_gold = gold_explicitly_named
                exact_match = any(hit_exact_path(hit, case)
                                  for hit in capture["cases"][expected_route_of(case)][case_id]["merged_hits"])
                graph_top10 = capture["cases"][expected_route_of(case)][case_id]["graph_hits"]
                graph_rank = next((hit["rank"] for hit in graph_top10 if hit_exact_path(hit, case)), None)
                hybrid_hits = capture["cases"][expected_route_of(case)][case_id]["merged_hits"]
                hybrid_rank = next((hit["rank"] for hit in hybrid_hits if hit_exact_path(hit, case)), None)
                candidate_ranks: dict[str, int | None] = {}
                for size in (30, 50, 100):
                    candidate_row = candidate_cases[size].get(case_id)
                    rank_found = None
                    if candidate_row:
                        for hit in candidate_row.get("paths") or []:
                            if (hit.get("relation_ids") == expected_relations
                                    and [node.get("entity_id") for node in hit.get("path_nodes") or []] == expected_nodes
                                    and set(case.get("expected_evidence_ids") or []) & set(hit.get("evidence_ids") or [])
                                    and set(case.get("expected_source_chunk_ids") or []) & set(hit.get("chunk_ids") or [])):
                                rank_found = int(hit["rank"])
                                break
                    candidate_ranks[str(size)] = rank_found
                candidate30_rank = candidate_ranks["30"]
                path_present_candidate30 = candidate30_rank is not None
                if exact_match:
                    retrieval_issue = "검색 성공"
                elif path_present_candidate30 and candidate30_rank and candidate30_rank <= 10:
                    retrieval_issue = "병합/평가 노출 순위 확인 필요"
                elif path_present_candidate30:
                    retrieval_issue = "랭킹 실패"
                else:
                    retrieval_issue = "검색 후보 컷오프/미회수"
                reasons = [retrieval_issue]
                if not question_disambiguates_gold and len(plausible_alternatives) > 0:
                    reasons.extend(("질문 정보 부족", "복수 정답 가능"))
                if not gold or not gold["all_edges_have_source_text"]:
                    reasons.append("근거 불충분")
                gold_node = next((node for node in alternatives
                                  if node["middle_entity_id"] == (gold or {}).get("middle_entity_id")), None)
                plausible_alternative_ids = {
                    candidate["middle_entity_id"] for candidate in plausible_alternatives
                }
                semantic_alternative_rank = None
                expected_type_pair = list(expected_types)
                for hit in hybrid_hits:
                    hit_nodes = hit.get("path_nodes") or []
                    hit_edges = hit.get("path_edges") or []
                    hit_types = [edge.get("relation_type") for edge in hit_edges]
                    if (hit.get("source") == "graph" and len(hit_nodes) == 3
                            and hit_nodes[0].get("entity_id") == expected_nodes[0]
                            and hit_nodes[-1].get("entity_id") == expected_nodes[-1]
                            and hit_types == expected_type_pair
                            and hit_nodes[1].get("entity_id") in plausible_alternative_ids
                            and set(hit.get("chunk_ids") or [])):
                        semantic_alternative_rank = int(hit["rank"])
                        break
                rows.append({
                    "case_id": case_id,
                    "expected_route": expected_route_of(case),
                    "question": question,
                    "expected_facts": case.get("expected_facts") or [],
                    "requested_fields": requested,
                    "expected_nodes": expected_nodes,
                    "expected_relation_ids": expected_relations,
                    "expected_relation_types": list(expected_types),
                    "gold_path_exists_in_neo4j_with_source": bool(gold and gold["all_edges_have_source_text"]),
                    "gold_product_name": gold["middle_name"] if gold else None,
                    "gold_product_attributes": gold["middle_attributes"] if gold else None,
                    "gold_explicitly_named_in_question": query_mentions_gold,
                    "question_uniquely_disambiguates_gold": question_disambiguates_gold,
                    "explicitly_mentioned_middle_entity_count": len(explicit_middle_ids),
                    "gold_middle_node_exists": gold is not None,
                    "same_type_endpoint_path_count": len(alternatives),
                    "distinct_middle_entity_count": len(unique_by_middle),
                    "same_middle_duplicate_path_count": max(len(alternatives) - len(unique_by_middle), 0),
                    "different_middle_alternative_count": len(plausible_alternatives),
                    "plausible_alternatives": plausible_alternatives[:20],
                    "candidate_exact_ranks": candidate_ranks,
                    "candidate30_exact_rank": candidate30_rank,
                    "graph_top10_exact_rank": graph_rank,
                    "hybrid_top10_exact_rank": hybrid_rank,
                    "hybrid_exact_success": exact_match,
                    "hybrid_semantic_alternative_rank": semantic_alternative_rank,
                    "hybrid_semantic_aux_success": exact_match or semantic_alternative_rank is not None,
                    "reason_labels": list(dict.fromkeys(reasons)),
                    "assessment_note": (
                        "대체 후보는 Neo4j 관계 유형과 요청 속성/원문 연결을 만족하는 데이터상 후보입니다. "
                        "공식 등록 효력 또는 실제 사용 가능성을 의미하지 않습니다."
                    ),
                })
    finally:
        graph.close()

    payload = {
        "mode": "read-only diagnostic; exact-match labels unchanged; no search code or golden data changed",
        "multi_hop_question_count": len(rows),
        "exact_integrated_success_count": sum(bool(row.get("hybrid_exact_success")) for row in rows),
        "question_alone_distinguishes_gold_count": sum(bool(row.get("question_uniquely_disambiguates_gold")) for row in rows),
        "questions_with_distinct_plausible_alternatives": sum(
            row.get("different_middle_alternative_count", 0) > 0 for row in rows
        ),
        "semantic_aux_success_count_top10": sum(
            bool(row.get("hybrid_semantic_aux_success")) for row in rows
        ),
        "semantic_alternative_recovery_count_top10": sum(
            not row.get("hybrid_exact_success") and row.get("hybrid_semantic_alternative_rank") is not None
            for row in rows
        ),
        "rows": rows,
    }
    out_json = REPORTS / "multihop_path_uniqueness.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, out_json.with_suffix(".md"))
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"},
                     ensure_ascii=False, indent=2))
    print(f"Reports: {out_json}, {out_json.with_suffix('.md')}")


def write_markdown(payload: dict[str, Any], target: Path) -> None:
    lines = [
        "# Multi-hop Gold 경로 유일성 및 동등 후보 진단", "",
        "이 보고서는 Neo4j와 고정 평가 데이터를 읽어 만든 진단입니다. 검색 코드·라우터·평가셋·exact-match 기준은 변경하지 않았습니다. 후보 대체 가능성은 데이터상 경로와 속성/원문 필드 완성도를 뜻하며, 공식 등록 또는 실제 사용 가능성을 판정하지 않습니다.", "",
        f"- 문항: {payload['multi_hop_question_count']}개",
        f"- 기존 exact-match 통합 성공: {payload['exact_integrated_success_count']}/17",
        f"- 질문에 Gold 제품/상표가 명시된 문항: {payload['question_alone_distinguishes_gold_count']}/17",
        f"- 구별되지 않는 조건에서 원문·요청 필드를 갖춘 서로 다른 대체 노드가 있는 문항: {payload['questions_with_distinct_plausible_alternatives']}/17",
        f"- 의미 보조 기준 Top-10 충족(기존 Gold 또는 대체 경로): {payload['semantic_aux_success_count_top10']}/17",
        f"- Gold는 없지만 의미 대체 경로가 HYBRID Top-10에 든 문항: {payload['semantic_alternative_recovery_count_top10']}/17",
        "",
        "| ID | 질문 및 Gold 정보 | 경로 수 / 중간 노드 / 중복 | Gold 식별 단서 | 후보30/50/100 | GRAPH Top10 | HYBRID Gold/의미대체 | 분류 |",
        "|---|---|---:|---|---|---:|---|---|",
    ]
    for row in payload["rows"]:
        facts = "；".join(row["expected_facts"])
        labels = ", ".join(row["reason_labels"])
        lines.append(
            f"| {row['case_id']} | {row['question']}<br>Gold: {facts} "
            f"| {row['same_type_endpoint_path_count']} / {row['distinct_middle_entity_count']} / {row['same_middle_duplicate_path_count']} "
            f"| {'유일' if row['question_uniquely_disambiguates_gold'] else ('부분' if row['gold_explicitly_named_in_question'] else '없음')} "
            f"| {'/'.join(str(row['candidate_exact_ranks'].get(size) or '—') for size in ('30','50','100'))} "
            f"| {row['graph_top10_exact_rank'] or '미회수'} "
            f"| Gold {row['hybrid_top10_exact_rank'] or '—'} / 대체 {row['hybrid_semantic_alternative_rank'] or '—'} "
            f"| {labels} |"
        )
    lines.extend([
        "", "## 해석", "",
        "- `검색 후보 컷오프/미회수`: 현재 후보 30에 정확한 Gold path가 없었습니다. 50/100 결과와 Neo4j 직접 경로 확인을 함께 보면 Cypher 사전 후보 제한인지 관계 구성 누락인지 구분합니다.",
        "- `랭킹 실패`: Gold path가 후보 30에는 있었지만 Graph/Hybrid Top-10에서 빠졌습니다.",
        "- `질문 정보 부족`: 질문에 제품명/상표 등 Gold 중간 노드를 특정하는 구절이 없고, 같은 작물·대상·관계 유형에 다른 제품 노드가 있습니다.",
        "- `복수 정답 가능`: Gold와 다른 제품 노드의 경로가 같은 질문 관계를 만족하며 요청된 속성 필드와 원문 chunk를 갖습니다. 이 보조 판단은 exact-match 점수를 변경하지 않습니다.",
        "- `근거 불충분`: Gold 관계의 연결 원문이 없거나 요청된 값이 출처로 뒷받침되지 않는 경우입니다.",
        "", "## 보조 평가와 exact-match 차이", "",
        f"Exact-match 통합 성공은 기존 그대로 {payload['exact_integrated_success_count']}/17입니다. 의미 보조 Top-10은 {payload['semantic_aux_success_count_top10']}/17이며, 그중 {payload['semantic_alternative_recovery_count_top10']}건은 Gold가 아니라 대체 경로가 들어간 경우입니다. 의미 보조 평가는 동일 endpoint/관계 유형, 서로 다른 중간 엔티티, 요청 속성의 데이터 존재, 관계 원문 연결을 확인한 진단입니다. 이 결과를 exact-match 통계에 합산하지 않습니다.",
        "", "## 다음 단계 분류", "",
        "- 검색 개선 가능: Gold가 후보 30 밖이거나 후보 안에서 Top-10 밖인 문항. 쿼리 전 정렬과 관련 원문을 활용한 랭킹을 검토할 수 있습니다.",
        "- 평가 데이터 보완 필요: 질문에 Gold 제품을 식별할 제약이 없는데 서로 다른 동등 경로가 여러 개 있는 문항. 기준은 유지하되 질문에 제품/상표/사용 조건을 추가하거나, Gold 경로를 여러 개 등록하는 방향을 검토할 수 있습니다.",
        "- 법적 상태 확인: 이 진단은 Neo4j의 속성과 연결 문서만 요약합니다. 공식 등록 효력이나 현행 사용 가능성을 단정하지 않습니다.",
        "",
    ])
    target.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
