"""골든셋 기준으로 multi-hop 정답률(검색 충족도)을 측정합니다.

평가 원칙
- 기대 hop 수는 `len(expected_relation_ids)` 입니다.
- GRAPH·HYBRID: 노드·관계·hop·근거가 모두 충족되면 FULL_SUCCESS,
  일부만 충족되면 PARTIAL_SUCCESS, 하나도 못 맞히면 FAILURE 입니다.
- VECTOR: 그래프 hop 평가 대상이 아닙니다. hop 과 노드·관계 충족률은 None(N/A)으로
  남기고 Evidence Hit@K·Chunk Hit@K 로 평가합니다.
- REJECT: Hit@K 와 multi-hop 평가 분모에서 제외합니다(SKIPPED_REJECT).

검색기 연결
`RetrievalObserver` 를 구현한 객체가 문항별 검색 결과(`SearchResult` 목록)를
돌려줍니다. Supervisor 실행 계층이 생기면 그 계층을 감싸는 observer 만 새로
만들면 되고, 채점·집계·보고 코드는 그대로 씁니다.

실행 예시:
    uv run python -m scripts.evaluation.evaluate_multihop --observer oracle
    uv run python -m scripts.evaluation.evaluate_multihop --observer null
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from scripts.evaluation.report_io import write_detail_rows, write_summary
from src.retrievers.search_result import SearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLDEN_PATH = PROJECT_ROOT / "data" / "evaluation" / "golden_set_final.jsonl"
DEFAULT_RELATION_PATH = PROJECT_ROOT / "data" / "processed" / "relations.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "reports"

READY_STATUS = "READY"
GRAPH_ROUTES = ("GRAPH", "HYBRID")

FULL_SUCCESS = "FULL_SUCCESS"
PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
FAILURE = "FAILURE"
SKIPPED_REJECT = "SKIPPED_REJECT"


@dataclass(frozen=True)
class RetrievalOutcome:
    """한 문항의 검색 결과입니다.

    `route` 는 Supervisor 가 실제로 실행한 경로를 기록하기 위한 확장 지점이며,
    multi-hop 채점 자체는 `results` 만 사용합니다.
    """

    results: tuple[SearchResult, ...] = ()
    route: str | None = None
    reason: str = ""
    warnings: tuple[str, ...] = ()


class RetrievalObserver(Protocol):
    """문항을 받아 검색 결과를 돌려주는 객체의 최소 인터페이스입니다."""

    def retrieve(self, case: dict[str, Any]) -> RetrievalOutcome: ...


@dataclass(frozen=True)
class NullRetrievalObserver:
    """검색을 실행하지 않습니다.

    검색 계층을 붙이기 전의 하한선(모든 문항 FAILURE)을 기록할 때 씁니다.
    """

    def retrieve(self, case: dict[str, Any]) -> RetrievalOutcome:
        return RetrievalOutcome(reason="검색 계층이 연결되지 않았습니다.")


@dataclass(frozen=True)
class OracleRetrievalObserver:
    """골든셋의 기대 ID 를 그대로 돌려줍니다.

    실제 검색기가 아니라 채점 로직의 상한선(모든 문항 FULL_SUCCESS)을 확인하는
    용도입니다. 기대 hop 수와 같은 길이의 경로 하나를 만들어 돌려줍니다.
    """

    def retrieve(self, case: dict[str, Any]) -> RetrievalOutcome:
        relation_ids = list(case.get("expected_relation_ids") or [])
        node_ids = list(case.get("expected_entity_ids") or [])
        result = SearchResult(
            source="graph" if relation_ids else "vector",
            text="oracle",
            evidence_ids=list(case.get("expected_evidence_ids") or []),
            chunk_ids=list(case.get("expected_source_chunk_ids") or []),
            path=node_ids,
            metadata={
                "node_ids": node_ids,
                "relation_ids": relation_ids,
                "hop_count": len(relation_ids),
            },
        )
        return RetrievalOutcome(results=(result,), reason="oracle")


@dataclass(frozen=True)
class GraphRetrievalObserver:
    """`GraphRetriever` 를 multi-hop 평가에 연결합니다.

    Neo4j 연결이 필요합니다. Supervisor 가 생기기 전까지는 경로 탐색만
    사용하므로 질문에서 추출한 작물명과 기대 hop 수를 그대로 넘깁니다.
    """

    retriever: Any
    limit: int = 20
    crop_of: Any = None

    def retrieve(self, case: dict[str, Any]) -> RetrievalOutcome:
        question = case.get("question") or ""
        crop = self.crop_of(question) if self.crop_of is not None else None
        expected_hop = len(case.get("expected_relation_ids") or [])
        results = self.retriever.search(
            question,
            crop=crop,
            max_hops=max(expected_hop, 1),
            limit=self.limit,
        )
        return RetrievalOutcome(results=tuple(results), reason="graph_retriever")

@dataclass(frozen=True)
class SupervisorRetrievalObserver:
    """Supervisor 실행 결과를 multi-hop 채점에 연결합니다."""

    app: Any

    def retrieve(self, case: dict[str, Any]) -> RetrievalOutcome:
        # REJECT 문항은 어차피 채점에서 제외되므로 Supervisor를 돌리지 않습니다.
        if (case.get("query_strategy") or "").upper() == "REJECT":
            return RetrievalOutcome(route="REJECT", reason="REJECT 문항은 검색하지 않습니다.")

        state = self.app.invoke({"question": case.get("question") or ""})
        return RetrievalOutcome(
            # 검증 전 원본을 써야 검색 성능과 검증 성능이 섞이지 않습니다.
            results=tuple(state.get("retrieved_results") or ()),
            route=state.get("route"),
            reason=state.get("route_reason", ""),
            warnings=tuple(state.get("warnings") or ()),
        )

def load_golden_cases(
    golden_path: str | Path, *, ready_only: bool = True
) -> list[dict[str, Any]]:
    """골든셋 JSONL 을 읽어 평가 대상 문항을 반환합니다."""
    path = Path(golden_path)
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Golden case at line {line_number} must be an object")
            if ready_only and row.get("status") != READY_STATUS:
                continue
            cases.append(row)
    if not cases:
        raise ValueError(f"No {READY_STATUS} golden cases found in {path}")
    return cases


def load_relation_types(relation_path: str | Path) -> dict[str, str]:
    """relation_id → relation_type 매핑을 읽습니다."""
    path = Path(relation_path)
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            relation_id = row.get("relation_id")
            if relation_id:
                mapping[relation_id] = row.get("relation_type") or row.get("type") or ""
    return mapping


def expected_route_of(case: dict[str, Any]) -> str:
    strategy = case.get("query_strategy")
    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError(f"Case {case.get('case_id')!r} has no query_strategy")
    return strategy.strip().upper()


def expected_hop_of(case: dict[str, Any]) -> int:
    """기대 hop 수입니다. 기대 관계 ID 개수로 정의합니다."""
    return len(case.get("expected_relation_ids") or [])


def _coverage(expected: Sequence[str], retrieved: Iterable[str]) -> tuple[float | None, list[str]]:
    """충족률과 못 찾은 ID 목록을 반환합니다. 기대가 없으면 None 입니다."""
    expected_set = list(dict.fromkeys(expected))
    if not expected_set:
        return None, []
    found = set(retrieved)
    missing = [item for item in expected_set if item not in found]
    return round((len(expected_set) - len(missing)) / len(expected_set), 4), missing


def _collect(results: Sequence[SearchResult]) -> dict[str, set[str]]:
    """검색 결과 전체에서 ID 를 모읍니다."""
    nodes: set[str] = set()
    relations: set[str] = set()
    evidence: set[str] = set()
    chunks: set[str] = set()
    for result in results:
        metadata = result.metadata or {}
        nodes.update(str(item) for item in (metadata.get("node_ids") or []))
        relations.update(str(item) for item in (metadata.get("relation_ids") or []))
        evidence.update(result.evidence_ids or [])
        chunks.update(result.chunk_ids or [])
    return {"nodes": nodes, "relations": relations, "evidence": evidence, "chunks": chunks}


def _best_path(
    results: Sequence[SearchResult], expected_relation_ids: Sequence[str]
) -> SearchResult | None:
    """기대 관계를 가장 많이 담은 경로를 고릅니다.

    그 경로의 hop 수를 `actual_hop` 으로 봅니다. 기대 관계를 하나도 담은 경로가
    없으면 hop 수가 가장 큰 경로를 돌려줍니다.
    """
    graph_results = [
        result for result in results if (result.metadata or {}).get("relation_ids") is not None
    ]
    if not graph_results:
        return None
    expected = set(expected_relation_ids)

    def key(result: SearchResult) -> tuple[int, int]:
        relation_ids = {str(item) for item in (result.metadata or {}).get("relation_ids") or []}
        return (len(relation_ids & expected), -abs(len(relation_ids) - len(expected)))

    return max(graph_results, key=key)


def _hop_count_of(result: SearchResult | None) -> int | None:
    if result is None:
        return None
    metadata = result.metadata or {}
    hop_count = metadata.get("hop_count")
    if hop_count is not None:
        return int(hop_count)
    return len(metadata.get("relation_ids") or [])


def _hop_match(
    best: SearchResult | None,
    actual_hop: int | None,
    expected_hop: int,
    expected_relation_ids: Sequence[str],
) -> bool | None:
    """hop 수 일치 여부입니다.

    기대 관계를 하나도 지나지 않은 경로는 깊이가 같아도 일치로 보지 않습니다.
    관계 없이 길이만 맞은 경로를 부분 성공으로 세면 지표가 부풀기 때문입니다.
    """
    if actual_hop is None:
        return None
    if not expected_relation_ids:
        return actual_hop == expected_hop
    found = {str(item) for item in ((best.metadata or {}).get("relation_ids") or [])}
    if not found & set(expected_relation_ids):
        return False
    return actual_hop == expected_hop


def _verdict(parts: Sequence[float | None]) -> str:
    """충족률 목록으로 성공 등급을 정합니다."""
    scored = [part for part in parts if part is not None]
    if not scored:
        return FAILURE
    if all(part >= 1.0 for part in scored):
        return FULL_SUCCESS
    if any(part > 0.0 for part in scored):
        return PARTIAL_SUCCESS
    return FAILURE


def evaluate_case(
    case: dict[str, Any],
    outcome: RetrievalOutcome,
    *,
    k: int = 5,
    relation_types: dict[str, str] | None = None,
) -> dict[str, Any]:
    """한 문항의 multi-hop 충족도를 채점합니다."""
    expected_route = expected_route_of(case)
    expected_relation_ids = list(case.get("expected_relation_ids") or [])
    expected_entity_ids = list(case.get("expected_entity_ids") or [])
    expected_evidence_ids = list(case.get("expected_evidence_ids") or [])
    expected_chunk_ids = list(case.get("expected_source_chunk_ids") or [])
    expected_hop = expected_hop_of(case)

    record: dict[str, Any] = {
        "case_id": case.get("case_id"),
        "question_type": case.get("question_type") or "unknown",
        "difficulty": case.get("difficulty"),
        "expected_route": expected_route,
        "retrieved_route": outcome.route,
        "expected_hop": expected_hop if expected_route in GRAPH_ROUTES else None,
        "actual_hop": None,
        "hop_match": None,
        "node_coverage": None,
        "relation_coverage": None,
        "evidence_coverage": None,
        "chunk_coverage": None,
        "expected_relation_types": [
            (relation_types or {}).get(relation_id, "") for relation_id in expected_relation_ids
        ],
        "missing_entity_ids": [],
        "missing_relation_ids": [],
        "missing_evidence_ids": [],
        "missing_chunk_ids": [],
        "retrieved_result_count": len(outcome.results),
        "verdict": FAILURE,
        "reason": outcome.reason,
        "warnings": list(outcome.warnings),
    }

    if expected_route == "REJECT":
        record["verdict"] = SKIPPED_REJECT
        record["reason"] = record["reason"] or "REJECT 문항은 multi-hop 분모에서 제외합니다."
        return record

    results = list(outcome.results)[: max(k, 1)]
    collected = _collect(results)

    evidence_coverage, missing_evidence = _coverage(expected_evidence_ids, collected["evidence"])
    chunk_coverage, missing_chunks = _coverage(expected_chunk_ids, collected["chunks"])
    record["evidence_coverage"] = evidence_coverage
    record["chunk_coverage"] = chunk_coverage
    record["missing_evidence_ids"] = missing_evidence
    record["missing_chunk_ids"] = missing_chunks

    if expected_route in GRAPH_ROUTES:
        node_coverage, missing_nodes = _coverage(expected_entity_ids, collected["nodes"])
        relation_coverage, missing_relations = _coverage(
            expected_relation_ids, collected["relations"]
        )
        best = _best_path(results, expected_relation_ids)
        actual_hop = _hop_count_of(best)
        record.update(
            {
                "node_coverage": node_coverage,
                "relation_coverage": relation_coverage,
                "missing_entity_ids": missing_nodes,
                "missing_relation_ids": missing_relations,
                "actual_hop": actual_hop,
                "hop_match": _hop_match(
                    best, actual_hop, expected_hop, expected_relation_ids
                ),
            }
        )
        hop_part = 1.0 if record["hop_match"] else 0.0
        record["verdict"] = _verdict(
            [node_coverage, relation_coverage, hop_part, evidence_coverage]
        )
        return record

    # VECTOR: 그래프 hop 평가 대상이 아니므로 hop·노드·관계는 N/A 로 둡니다.
    record["verdict"] = _verdict([evidence_coverage, chunk_coverage])
    return record


@dataclass
class _Bucket:
    total: int = 0
    full: int = 0
    partial: int = 0
    failure: int = 0

    def add(self, verdict: str) -> None:
        self.total += 1
        if verdict == FULL_SUCCESS:
            self.full += 1
        elif verdict == PARTIAL_SUCCESS:
            self.partial += 1
        else:
            self.failure += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "full_success": self.full,
            "partial_success": self.partial,
            "failure": self.failure,
            "full_success_rate": round(self.full / self.total, 4) if self.total else None,
            "partial_or_better_rate": (
                round((self.full + self.partial) / self.total, 4) if self.total else None
            ),
        }


@dataclass
class _MatchRate:
    total: int = 0
    matched: int = 0

    def add(self, is_match: bool) -> None:
        self.total += 1
        self.matched += int(is_match)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "matched": self.matched,
            "rate": round(self.matched / self.total, 4) if self.total else None,
        }


@dataclass
class MultiHopEvaluation:
    details: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def evaluate_multihop(
    cases: Iterable[dict[str, Any]],
    observer: RetrievalObserver,
    *,
    k: int = 5,
    relation_types: dict[str, str] | None = None,
) -> MultiHopEvaluation:
    """문항별 multi-hop 결과와 요약 지표를 계산합니다."""
    details: list[dict[str, Any]] = []
    overall = _Bucket()
    by_route: dict[str, _Bucket] = defaultdict(_Bucket)
    by_question_type: dict[str, _Bucket] = defaultdict(_Bucket)
    by_expected_hop: dict[int, _Bucket] = defaultdict(_Bucket)
    hop_match = _MatchRate()
    coverage_totals: dict[str, list[float]] = defaultdict(list)
    skipped = 0

    for case in cases:
        outcome = observer.retrieve(case)
        record = evaluate_case(case, outcome, k=k, relation_types=relation_types)
        details.append(record)

        if record["verdict"] == SKIPPED_REJECT:
            skipped += 1
            continue

        verdict = record["verdict"]
        overall.add(verdict)
        by_route[record["expected_route"]].add(verdict)
        by_question_type[record["question_type"]].add(verdict)
        if record["expected_hop"] is not None:
            by_expected_hop[record["expected_hop"]].add(verdict)
        if record["hop_match"] is not None:
            hop_match.add(bool(record["hop_match"]))
        for key in ("node_coverage", "relation_coverage", "evidence_coverage", "chunk_coverage"):
            if record[key] is not None:
                coverage_totals[key].append(float(record[key]))

    summary = {
        "evaluated_cases": overall.total,
        "skipped_reject_cases": skipped,
        "overall": overall.as_dict(),
        "by_expected_route": {key: bucket.as_dict() for key, bucket in sorted(by_route.items())},
        "by_question_type": {
            key: bucket.as_dict() for key, bucket in sorted(by_question_type.items())
        },
        "by_expected_hop": {
            str(key): bucket.as_dict() for key, bucket in sorted(by_expected_hop.items())
        },
        "hop_match": hop_match.as_dict(),
        "mean_coverage": {
            key: round(sum(values) / len(values), 4)
            for key, values in sorted(coverage_totals.items())
            if values
        },
        "verdict_counts": dict(Counter(record["verdict"] for record in details)),
    }
    return MultiHopEvaluation(details=details, summary=summary)


DETAIL_COLUMNS = (
    "case_id",
    "question_type",
    "difficulty",
    "expected_route",
    "retrieved_route",
    "expected_hop",
    "actual_hop",
    "hop_match",
    "node_coverage",
    "relation_coverage",
    "evidence_coverage",
    "chunk_coverage",
    "expected_relation_types",
    "missing_entity_ids",
    "missing_relation_ids",
    "missing_evidence_ids",
    "missing_chunk_ids",
    "retrieved_result_count",
    "verdict",
    "reason",
    "warnings",
)


def write_details(
    details: Sequence[dict[str, Any]], path: str | Path, *, detail_format: str = "jsonl"
) -> Path:
    return write_detail_rows(
        details, path, columns=DETAIL_COLUMNS, detail_format=detail_format
    )


def build_observer(name: str) -> RetrievalObserver:
    if name == "null":
        return NullRetrievalObserver()
    if name == "oracle":
        return OracleRetrievalObserver()
    if name == "graph":
        # Neo4j 연결이 필요합니다. 연결 실패는 그대로 올립니다.
        from src.retrievers.graph_retriever import GraphRetriever
        from src.retrievers.router import QuestionRouter

        router = QuestionRouter()
        return GraphRetrievalObserver(
            retriever=GraphRetriever.from_env(),
            crop_of=lambda question: router.decide(question).crop,
        )

    if name == "supervisor":
        # Neo4j·임베딩 모델 연결이 필요합니다. 연결 실패는 그대로 올립니다.
        from sentence_transformers import SentenceTransformer

        from src.retrievers.graph_retriever import GraphRetriever, graph_search
        from src.retrievers.answer import Citation, CitedAnswer, Claim, neo4j_lookup
        from src.retrievers.vector_retriever import VectorIndex
        from src.supervisor.citation_verifier import AnswerReview
        from src.supervisor.graph import build_supervisor

        retriever = GraphRetriever.from_env()
        index = VectorIndex(retriever.driver, SentenceTransformer("BAAI/bge-m3"))
        app = build_supervisor(
            graph_search_fn=lambda question, **kwargs: graph_search(
                question, retriever=retriever, **kwargs
            ),
            vector_search_fn=index.search,
            evidence_lookup=lambda ids: neo4j_lookup(
                retriever.driver,
                ids,
                database=retriever.database,
            ),
            # 검색 채점에는 생성 품질이 필요 없으므로 유효한 최소 인용 답변을
            # 만들어 Supervisor의 검색 경로만 관찰합니다.
            cited_answer_generator=lambda question, evidences: CitedAnswer(
                answer=evidences[0].quote if evidences else "",
                claims=[
                    Claim(
                        text=evidences[0].quote,
                        citations=[
                            Citation(
                                evidence_id=evidences[0].evidence_id,
                                quote=evidences[0].quote,
                            )
                        ],
                    )
                ] if evidences else [],
            ),
            citation_support_checker=lambda claim, contexts: AnswerReview(
                supported=bool(claim and contexts),
                reason="검색 경로 평가용 결정적 판정",
            ),
        )
        return SupervisorRetrievalObserver(app=app)
    
    raise ValueError(f"Unknown observer: {name}")


def format_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    overall = summary["overall"]
    lines.append(
        f"평가 문항: {summary['evaluated_cases']}"
        f" (REJECT 제외 {summary['skipped_reject_cases']}건)"
    )
    lines.append(
        f"FULL_SUCCESS: {overall['full_success']}/{overall['total']}"
        f" ({_percent(overall['full_success_rate'])})"
    )
    lines.append(
        f"PARTIAL 이상: {overall['full_success'] + overall['partial_success']}/{overall['total']}"
        f" ({_percent(overall['partial_or_better_rate'])})"
    )
    hop = summary["hop_match"]
    lines.append(f"hop 일치: {hop['matched']}/{hop['total']} ({_percent(hop['rate'])})")

    lines.append("")
    lines.append("기대 경로별")
    for route, bucket in summary["by_expected_route"].items():
        lines.append(
            f"  {route:<7} FULL {bucket['full_success']}/{bucket['total']}"
            f" ({_percent(bucket['full_success_rate'])})"
            f"  PARTIAL {bucket['partial_success']}  FAIL {bucket['failure']}"
        )

    lines.append("")
    lines.append("기대 hop 별 (GRAPH·HYBRID)")
    for hop_count, bucket in summary["by_expected_hop"].items():
        lines.append(
            f"  {hop_count}-hop  FULL {bucket['full_success']}/{bucket['total']}"
            f" ({_percent(bucket['full_success_rate'])})"
        )

    if summary["mean_coverage"]:
        lines.append("")
        lines.append("평균 충족률")
        for key, value in summary["mean_coverage"].items():
            lines.append(f"  {key:<19} {value:.4f}")
    return "\n".join(lines)


def _percent(rate: float | None) -> str:
    return "N/A" if rate is None else f"{rate * 100:.1f}%"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="골든셋 multi-hop 정답률 평가")
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--relations", type=Path, default=DEFAULT_RELATION_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--observer", choices=("null", "oracle", "graph", 'supervisor'), default="null")
    parser.add_argument("--detail-format", choices=("jsonl", "csv"), default="jsonl")
    parser.add_argument("--k", type=int, default=5, help="문항당 사용할 검색 결과 수입니다.")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--include-non-ready", action="store_true")
    parser.add_argument(
        "--skip-relation-types",
        action="store_true",
        help="relations.jsonl 조회를 건너뜁니다(대용량 파일을 읽지 않습니다).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.k < 1:
        raise ValueError("k must be positive")
    cases = load_golden_cases(args.golden_set, ready_only=not args.include_non_ready)
    relation_types = (
        None
        if args.skip_relation_types or not Path(args.relations).exists()
        else load_relation_types(args.relations)
    )
    observer = build_observer(args.observer)
    evaluation = evaluate_multihop(cases, observer, k=args.k, relation_types=relation_types)

    tag = args.tag or args.observer
    detail_path = write_details(
        evaluation.details,
        args.output_dir / f"multihop_details_{tag}.{args.detail_format}",
        detail_format=args.detail_format,
    )
    summary = dict(evaluation.summary)
    summary["meta"] = {
        "golden_set": str(args.golden_set),
        "observer": args.observer,
        "k": args.k,
        "ready_only": not args.include_non_ready,
        "relation_types_loaded": relation_types is not None,
        "detail_path": str(detail_path),
    }
    summary_path = write_summary(summary, args.output_dir / f"multihop_summary_{tag}.json")

    print(format_report(summary))
    print()
    print(f"문항별 결과: {detail_path}")
    print(f"요약 지표: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
