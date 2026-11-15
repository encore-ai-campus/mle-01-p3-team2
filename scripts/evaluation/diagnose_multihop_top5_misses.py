"""Multi-hop Top-5 미스 16문항의 실패 지점을 후보 회수 / 랭킹으로 분리 진단합니다.

각 문항에 대해
  1) 파이프라인과 동일한 설정(limit=10, 기본 후보 풀)으로 재검색해 Gold 순위를 찾고,
  2) 후보 풀을 최대로 넓힌 설정으로 재검색해 Gold가 후보에 존재하는지 확인하고,
  3) Neo4j에서 Gold 2-hop 경로 존재와 동형 경쟁 경로 수를 직접 세어
실패를 CANDIDATE_MISS(회수 실패) / RANK_LOSS(순위 경쟁) 로 분류합니다.

Gold ID는 검색 입력이나 점수 계산에 쓰지 않고, 사후 진단에만 사용합니다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.retrievers.graph_retriever import GraphRetriever  # noqa: E402
from src.retrievers.hybrid_retriever import hybrid_search  # noqa: E402
from src.retrievers.router import JevQuestionRouter  # noqa: E402
from src.retrievers.vector_retriever import VectorIndex  # noqa: E402

DEFAULT_CASE_IDS = (
    "GS011", "GS012", "GS018", "GS020", "GS023", "GS026", "GS027", "GS035",
    "GS036", "GS054", "GS057", "GS059", "GS061", "GS062", "GS066", "GS067",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _rank_of(results: Sequence[Any], field: str, expected: Sequence[str]) -> int | None:
    wanted = set(expected)
    if not wanted:
        return None
    for rank, result in enumerate(results, 1):
        if wanted.intersection(getattr(result, field, None) or []):
            return rank
    return None


def _entity_ranks(results: Sequence[Any], expected: Sequence[str]) -> dict[str, int | None]:
    ranks: dict[str, int | None] = {entity: None for entity in expected}
    for rank, result in enumerate(results, 1):
        for node in result.path_nodes or []:
            entity_id = node.get("entity_id")
            if entity_id in ranks and ranks[entity_id] is None:
                ranks[entity_id] = rank
    return ranks


def _relation_ranks(results: Sequence[Any], expected: Sequence[str]) -> dict[str, int | None]:
    ranks: dict[str, int | None] = {relation: None for relation in expected}
    for rank, result in enumerate(results, 1):
        for edge in result.path_edges or []:
            relation_id = edge.get("relation_id")
            if relation_id in ranks and ranks[relation_id] is None:
                ranks[relation_id] = rank
    return ranks


def _full_path_rank(
    results: Sequence[Any], entities: Sequence[str], relations: Sequence[str]
) -> int | None:
    """Gold 엔티티 전체와 Gold 관계 전체를 한 경로에 담은 결과의 순위."""
    want_entities, want_relations = set(entities), set(relations)
    if not want_entities or not want_relations:
        return None
    for rank, result in enumerate(results, 1):
        nodes = {node.get("entity_id") for node in result.path_nodes or []}
        edges = {edge.get("relation_id") for edge in result.path_edges or []}
        if want_entities <= nodes and want_relations <= edges:
            return rank
    return None


def _alternative_path_rank(
    results: Sequence[Any],
    entities: Sequence[str],
    relation_types: Sequence[str] | None,
) -> dict[str, int | None]:
    """Gold 경로 대신 '같은 작물 → 다른 등록 약제 → 같은 방제 대상' 경로의 순위.

    질문만으로 특정 제품 하나를 구별할 수 없는 문항에서, 단일 Gold exact-match와
    별도로 "타당한 대체 경로가 Top-k에 있는지"를 측정한다. 두 지표는 합산하지 않는다.
    """
    ranks: dict[str, int | None] = {"same_target_rank": None, "alternative_path_rank": None}
    if len(entities) < 3 or not relation_types:
        return ranks
    start, gold_middle, target = entities[0], entities[1], entities[2]
    wanted_types = list(relation_types)
    for rank, result in enumerate(results, 1):
        nodes = [node.get("entity_id") for node in result.path_nodes or []]
        types = [edge.get("relation_type") for edge in result.path_edges or []]
        if len(nodes) < 3 or nodes[0] != start or nodes[-1] != target:
            continue
        if ranks["same_target_rank"] is None:
            ranks["same_target_rank"] = rank
        if types == wanted_types and nodes[1] != gold_middle:
            if ranks["alternative_path_rank"] is None:
                ranks["alternative_path_rank"] = rank
    return ranks


def _run_route(
    route: str,
    question: str,
    *,
    graph: GraphRetriever,
    vector: VectorIndex,
    decision: Any,
    limit: int,
    candidate_pool_size: int | None,
) -> list[Any]:
    crop = decision.crop
    max_hops = decision.max_hops
    relation_types = list(decision.relation_types) or None
    if route == "VECTOR":
        return list(vector.search(question, k=limit))
    if route == "GRAPH":
        kwargs: dict[str, Any] = {"crop": crop, "max_hops": max_hops, "limit": limit}
        if relation_types is not None:
            kwargs["relation_types"] = relation_types
        if candidate_pool_size is not None:
            kwargs["candidate_pool_size"] = candidate_pool_size
        return list(graph.search(question, **kwargs))
    if route == "HYBRID":
        def graph_fn(q: str, **kw: Any) -> list[Any]:
            if candidate_pool_size is not None:
                kw["candidate_pool_size"] = candidate_pool_size
            return graph.search(q, **kw)

        return list(hybrid_search(
            question,
            graph_search_fn=graph_fn,
            vector_search_fn=lambda q, k=limit: vector.search(q, k=k),
            crop=crop,
            max_hops=max_hops,
            relation_types=relation_types,
            graph_limit=limit,
            vector_limit=limit,
            limit=limit,
        ))
    return []


GOLD_PATH_QUERY = (
    "MATCH p=(a)-[r1]->(b)-[r2]->(c) "
    "WHERE a.entity_id=$a AND b.entity_id=$b AND c.entity_id=$c "
    "AND r1.relation_id=$r1 AND r2.relation_id=$r2 "
    "RETURN type(r1) AS t1, type(r2) AS t2 LIMIT 1"
)
SIBLING_QUERY = (
    "MATCH (a)-[r1]->(b)-[r2]->(c) "
    "WHERE a.entity_id=$a AND c.entity_id=$c AND type(r1)=$t1 AND type(r2)=$t2 "
    "RETURN count(*) AS sibling_count"
)
ANY_PATH_QUERY = (
    "MATCH (a)-[r1]->(b)-[r2]->(c) "
    "WHERE a.entity_id=$a AND b.entity_id=$b AND c.entity_id=$c "
    "RETURN collect(DISTINCT type(r1))[0..5] AS t1, "
    "collect(DISTINCT type(r2))[0..5] AS t2, count(*) AS path_count"
)


def _graph_facts(
    graph: GraphRetriever, entities: Sequence[str], relations: Sequence[str]
) -> dict[str, Any]:
    if len(entities) < 3 or len(relations) < 2:
        return {
            "gold_two_hop_path_exists": None,
            "gold_relation_types": None,
            "sibling_path_count": None,
        }
    options = {"database": graph.database} if graph.database != "neo4j" else {}
    with graph.driver.session(**options) as session:
        record = session.run(GOLD_PATH_QUERY, {
            "a": entities[0], "b": entities[1], "c": entities[2],
            "r1": relations[0], "r2": relations[1],
        }).single()
        if record is None:
            loose = session.run(ANY_PATH_QUERY, {
                "a": entities[0], "b": entities[1], "c": entities[2],
            }).single()
            return {
                "gold_two_hop_path_exists": False,
                "gold_relation_types": None,
                "sibling_path_count": None,
                "entity_chain_path_count": loose["path_count"] if loose else 0,
                "entity_chain_relation_types": (
                    [loose["t1"], loose["t2"]] if loose and loose["path_count"] else None
                ),
            }
        sibling = session.run(SIBLING_QUERY, {
            "a": entities[0], "c": entities[2],
            "t1": record["t1"], "t2": record["t2"],
        }).single()
    return {
        "gold_two_hop_path_exists": True,
        "gold_relation_types": [record["t1"], record["t2"]],
        "sibling_path_count": sibling["sibling_count"] if sibling else None,
    }


def classify(row: dict[str, Any]) -> str:
    deep = row["deep"]
    pipeline = row["pipeline"]
    if pipeline["chunk_rank"] is not None and pipeline["chunk_rank"] <= 5:
        return "TOP5_HIT_NOW"
    if pipeline["chunk_rank"] is not None:
        return "RANK_LOSS_NEAR"      # 파이프라인 Top-10에는 있으나 Top-5에서 밀림
    if deep["chunk_rank"] is not None:
        return "RANK_LOSS_DEEP"      # 후보에는 있으나 파이프라인 Top-10 밖
    if deep["full_path_rank"] is not None or any(
        value is not None for value in deep["relation_ranks"].values()
    ):
        return "PARTIAL_CANDIDATE"   # Gold 관계/경로는 후보에 있으나 Gold chunk 미회수
    return "CANDIDATE_MISS"          # 후보 풀 자체에 Gold 없음


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-ids", nargs="*", default=list(DEFAULT_CASE_IDS))
    parser.add_argument("--golden-set", type=Path,
                        default=PROJECT_ROOT / "data/evaluation/golden_set_final.jsonl")
    parser.add_argument("--runs", type=Path,
                        default=PROJECT_ROOT / "data/evaluation/ragas_result/ragas_results.jsonl")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "data/evaluation/multihop_top5_miss_diagnosis.jsonl")
    parser.add_argument("--pipeline-limit", type=int, default=10)
    parser.add_argument("--deep-limit", type=int, default=200)
    parser.add_argument("--deep-pool", type=int, default=166)
    args = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env", override=True)
    cases = {case["case_id"]: case for case in read_jsonl(args.golden_set)}
    runs = {run["case_id"]: run for run in read_jsonl(args.runs)}

    from sentence_transformers import SentenceTransformer

    graph = GraphRetriever.from_env()
    router = JevQuestionRouter.from_env()
    rows: list[dict[str, Any]] = []
    try:
        vector = VectorIndex(graph.driver, SentenceTransformer("BAAI/bge-m3"),
                             database=graph.database)
        vector.verify_ready()
        for case_id in args.case_ids:
            case = cases[case_id]
            run = runs.get(case_id, {})
            question = case["question"]
            decision = router.route_query(question)
            route = decision.route
            entities = list(case.get("expected_entity_ids") or [])
            relations = list(case.get("expected_relation_ids") or [])
            chunks = list(case.get("expected_source_chunk_ids") or [])
            evidence = list(case.get("expected_evidence_ids") or [])

            graph_facts = _graph_facts(graph, entities, relations)
            measurements: dict[str, dict[str, Any]] = {}
            for name, limit, pool in (
                ("pipeline", args.pipeline_limit, None),
                ("deep", args.deep_limit, args.deep_pool),
            ):
                results = _run_route(route, question, graph=graph, vector=vector,
                                     decision=decision, limit=limit,
                                     candidate_pool_size=pool)
                measurements[name] = {
                    "limit": limit,
                    "candidate_pool_size": pool,
                    "result_count": len(results),
                    "chunk_rank": _rank_of(results, "chunk_ids", chunks),
                    "evidence_rank": _rank_of(results, "evidence_ids", evidence),
                    "entity_ranks": _entity_ranks(results, entities),
                    "relation_ranks": _relation_ranks(results, relations),
                    "full_path_rank": _full_path_rank(results, entities, relations),
                    "graph_result_count": sum(1 for r in results if r.source == "graph"),
                    **_alternative_path_rank(
                        results, entities, graph_facts.get("gold_relation_types")
                    ),
                }
            row = {
                "case_id": case_id,
                "question": question,
                "question_type": case.get("question_type"),
                "query_strategy": case.get("query_strategy"),
                "recorded_route": run.get("actual_route"),
                "rerun_route": route,
                "route_matches_strategy": (
                    str(case.get("query_strategy") or "").upper() == route
                ),
                "expected_entity_ids": entities,
                "expected_relation_ids": relations,
                "expected_source_chunk_ids": chunks,
                "pipeline": measurements["pipeline"],
                "deep": measurements["deep"],
                **graph_facts,
            }
            row["verdict"] = classify(row)
            rows.append(row)
            print(
                f"{case_id}\t{route}\tpipe_chunk={row['pipeline']['chunk_rank']}"
                f"\tdeep_chunk={row['deep']['chunk_rank']}"
                f"\tdeep_path={row['deep']['full_path_rank']}"
                f"\talt_path={row['pipeline']['alternative_path_rank']}"
                f"\tsiblings={row.get('sibling_path_count')}\t{row['verdict']}",
                flush=True,
            )
    finally:
        graph.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
