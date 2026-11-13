"""Focused checks for the Multi-hop path verdict."""

import json
from types import SimpleNamespace

from scripts.evaluation.evaluate_retrieval import evaluate, evaluate_multihop_paths


def test_complete_two_hop_path_and_broken_original_link(tmp_path):
    relations = tmp_path / "relations.jsonl"
    relations.write_text(
        json.dumps({"relation_id": "R1", "relation_type": "HAS_PEST"}) + "\n"
        + json.dumps({"relation_id": "R2", "relation_type": "CAUSES"}) + "\n",
        encoding="utf-8",
    )
    case = {
        "case_id": "M1", "question_type": "multi_hop", "query_strategy": "hybrid",
        "expected_entity_ids": ["A", "B", "C"],
        "expected_relation_ids": ["R1", "R2"],
        "expected_evidence_ids": ["E1"],
        "expected_source_chunk_ids": ["C1"],
    }
    edges = [
        {
            "relation_id": relation, "relation_type": relation_type,
            "from_entity_id": source, "to_entity_id": target,
            "evidence_ids": ["E1"], "chunk_ids": ["C1"],
            "original_chunks": [{"chunk_id": "C1", "evidence_id": "E1", "has_text": True}],
        }
        for relation, relation_type, source, target in (
            ("R1", "HAS_PEST", "A", "B"), ("R2", "CAUSES", "B", "C")
        )
    ]
    hit = {
        "rank": 1, "source": "graph", "relation_ids": ["R1", "R2"], "hop_count": 2,
        "path_nodes": [{"entity_id": item} for item in ("A", "B", "C")],
        "path_edges": edges, "evidence_ids": ["E1"], "chunk_ids": ["C1"],
    }
    top10 = {"M1": {"hits": [hit]}}
    route_details = [{"case_id": "M1", "actual_route": "HYBRID"}]

    details, summary = evaluate_multihop_paths([case], top10, route_details, relations)
    assert details[0]["full_success"] is True
    assert summary["full_success"] == 1

    edges[1]["original_chunks"] = []
    details, summary = evaluate_multihop_paths([case], top10, route_details, relations)
    assert details[0]["source_link_match"] is False
    assert "edge_original_link_missing" in details[0]["failure_reasons"]
    assert summary["full_success"] == 0


def test_forced_and_actual_route_hits_are_reported_separately():
    case = {
        "case_id": "H1", "question": "test", "question_type": "fact",
        "query_strategy": "hybrid", "expected_evidence_ids": ["E1"],
        "expected_source_chunk_ids": ["C1"],
    }
    forced = {"H1": {"hits": [{"evidence_ids": ["E1"], "chunk_ids": ["C1"]}]}}
    actual = {"H1": {"hits": [{"evidence_ids": ["E2"], "chunk_ids": ["C2"]}]}}

    class GraphRouter:
        def route_query(self, question):
            return SimpleNamespace(route="GRAPH", reason="graph")

    details, summary = evaluate([case], forced, GraphRouter(), actual)
    assert details[0]["route_match"] is False
    assert details[0]["hit_at_1"] is True
    assert details[0]["actual_route_hit_at_1"] is False
    assert summary["hit_at_k"]["1"]["hits"] == 1
    assert summary["actual_route_hit_at_k"]["1"]["hits"] == 0
