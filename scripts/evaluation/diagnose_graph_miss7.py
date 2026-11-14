"""Inspect exact GRAPH gold paths and ranking for the seven current misses."""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from scripts.evaluation.evaluate_multihop import load_golden_cases
from scripts.evaluation.evaluate_retrieval import PROJECT_ROOT
from src.retrievers.graph_retriever import (
    GraphRetriever, _path_relevance_score, path_to_result,
)


MISS_IDS = ("GS012", "GS018", "GS026", "GS027", "GS036", "GS066", "GS067")
SIZES = (30, 50, 100)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    cases = {case["case_id"]: case for case in load_golden_cases(
        PROJECT_ROOT / "data/evaluation/golden_set_final.jsonl"
    )}
    graph = GraphRetriever.from_env()
    rows = []
    try:
        with graph.driver.session(**({"database": graph.database}
                                     if graph.database != "neo4j" else {})) as session:
            for case_id in MISS_IDS:
                case = cases[case_id]
                nodes = case["expected_entity_ids"]
                relations = case["expected_relation_ids"]
                record = session.run(
                    "MATCH p=(a:Crop)-[r1]->(b)-[r2]->(c) "
                    "WHERE a.entity_id=$a AND b.entity_id=$b AND c.entity_id=$c "
                    "AND r1.relation_id=$r1 AND r2.relation_id=$r2 "
                    "RETURN [n IN nodes(p) | n.entity_id] AS node_ids, "
                    "[r IN relationships(p) | type(r)] AS relation_types, "
                    "[r IN relationships(p) | r.evidence_ids] AS evidence_ids, "
                    "[r IN relationships(p) | r.source_chunk_ids] AS chunk_ids",
                    {"a": nodes[0], "b": nodes[1], "c": nodes[2],
                     "r1": relations[0], "r2": relations[1]},
                ).single()
                broad_query = (
                    "MATCH p=(a:Crop)-[:HAS_REGISTERED_PESTICIDE_USE]->"
                    "(b)-[:CONTROLS_TARGET]->(c) "
                    "WHERE a.entity_id=$a AND c.entity_id=$c RETURN p AS path"
                )
                paths, chunks = graph._fetch_paths_and_chunks(
                    session, broad_query, {"a": nodes[0], "c": nodes[2]},
                )
                scored = []
                for path in paths:
                    result = path_to_result(path, chunks_by_id=chunks)
                    result.score = _path_relevance_score(
                        result.path_nodes or [], result.path_edges or [], case["question"],
                        [node.get("attributes") for node in path.nodes],
                    )
                    scored.append((result, path))
                scored.sort(key=lambda item: item[0].score or 0.0, reverse=True)
                gold_key = (tuple(nodes), tuple(relations))
                gold_ranks = [
                    rank for rank, (result, _) in enumerate(scored, 1)
                    if (tuple(node["entity_id"] for node in result.path_nodes or []),
                        tuple(edge["relation_id"] for edge in result.path_edges or [])) == gold_key
                ]
                ranked_examples = []
                gold_product_attributes = None
                for result, path in scored:
                    if (result.path_nodes or [{}, {}])[1].get("entity_id") == nodes[1]:
                        product = next((node for node in path.nodes
                                        if str(node.get("entity_type") or "") == "PesticideUse"), None)
                        attrs = product.get("attributes") if product else None
                        if isinstance(attrs, str):
                            try:
                                attrs = json.loads(attrs)
                            except json.JSONDecodeError:
                                attrs = {"raw": attrs[:250]}
                        gold_product_attributes = attrs
                        break
                for rank, (result, path) in enumerate(scored[:10], 1):
                    product = next((node for node in path.nodes
                                    if str(node.get("entity_type") or "") == "PesticideUse"), None)
                    attrs = product.get("attributes") if product else None
                    if isinstance(attrs, str):
                        try:
                            attrs = json.loads(attrs)
                        except json.JSONDecodeError:
                            attrs = {"raw": attrs[:250]}
                    product_id = (result.path_nodes or [{}, {}])[1].get("entity_id")
                    ranked_examples.append({
                        "rank": rank,
                        "product_entity_id": product_id,
                        "product_name": (result.path_nodes or [{}, {}])[1].get("name"),
                        "score": result.score,
                        "attribute_keys": sorted(attrs.keys()) if isinstance(attrs, dict) else [],
                        "registration_fields": ({key: attrs.get(key) for key in (
                            "recommendation_eligible", "registration_status", "registration_valid_until",
                            "audit_disposition", "use_timing", "use_count", "dilution",
                        )} if isinstance(attrs, dict) else {}),
                        "source_text": next((chunk.get("text", "")
                                             for edge in result.path_edges or []
                                             for chunk in edge.get("original_chunks") or []), "")[:500],
                    })
                rows.append({
                    "case_id": case_id,
                    "question": case["question"],
                    "gold_nodes_exist_in_directed_path": record is not None,
                    "neo4j_path": dict(record) if record else None,
                    "target_specific_valid_path_count": len(paths),
                    "gold_rank_within_target_specific_paths": gold_ranks[0] if gold_ranks else None,
                    "gold_query_relevance_score": (
                        scored[gold_ranks[0] - 1][0].score if gold_ranks else None
                    ),
                    "gold_product_attributes": gold_product_attributes,
                    "top10_target_specific_paths": ranked_examples,
                })
    finally:
        graph.close()

    target = PROJECT_ROOT / "data/evaluation/reports/graph_miss7_neo4j_path_check.json"
    target.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([
        {"case_id": row["case_id"],
         "gold_nodes_exist_in_directed_path": row["gold_nodes_exist_in_directed_path"],
         "relation_types": (row["neo4j_path"] or {}).get("relation_types"),
         "target_specific_valid_path_count": row["target_specific_valid_path_count"],
         "gold_rank_within_target_specific_paths": row["gold_rank_within_target_specific_paths"],
         "gold_query_relevance_score": row["gold_query_relevance_score"]}
        for row in rows
    ], ensure_ascii=False, indent=2))
    print(f"Report: {target}")


if __name__ == "__main__":
    main()
