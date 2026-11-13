"""골든셋 기준으로 검색 경로(라우팅) 정확도를 측정합니다.

이 단계에서는 검색을 실행하지 않고 경로 판단만 평가합니다.
Supervisor 실행 계층이 생기면 `RouteObserver` 프로토콜을 구현한 객체를
`evaluate_routing(..., observer=...)` 로 넘겨 fallback 경로(initial/final)까지
같은 보고 형식으로 확장할 수 있습니다.

실행 예시:
    uv run python -m scripts.evaluation.evaluate_routing
    uv run python -m scripts.evaluation.evaluate_routing --router jev --detail-format csv
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from scripts.evaluation.report_io import write_detail_rows, write_summary
from src.retrievers.router import JevQuestionRouter, QuestionRouter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLDEN_PATH = PROJECT_ROOT / "data" / "evaluation" / "golden_set_final.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "reports"

VALID_ROUTES = ("GRAPH", "VECTOR", "HYBRID", "REJECT")
READY_STATUS = "READY"

# docs/golden-set-contract.md 2절: GRAPH 와 HYBRID 는 gold 구조가 같고 relation
# 개수(1 vs 2)만 다릅니다. 질문 텍스트에는 hop 수 단서가 없으므로 두 경로를 한
# 묶음으로 보는 3-way 지표를 함께 계산합니다.
ROUTE_FAMILIES = {
    "GRAPH": "GRAPH_FAMILY",
    "HYBRID": "GRAPH_FAMILY",
    "VECTOR": "VECTOR",
    "REJECT": "REJECT",
}

# 계약 2절: 어떤 지표를 인용할 수 있는지 리포트에 함께 남깁니다.
METRIC_POLICY = {
    "official": ["reject"],
    "reference": ["route_family_overall", "by_route_family"],
    "do_not_cite_alone": ["overall"],
    "not_a_metric": ["hop_discrimination"],
    "note": (
        "query_strategy 는 후보 생성기 분기명이므로 정답 경로가 아닙니다. "
        "overall(4-way)을 라우터 성능으로 단독 인용하지 마십시오. "
        "자세한 근거는 docs/golden-set-contract.md 2절을 보십시오."
    ),
}


@dataclass(frozen=True)
class RouteObservation:
    """한 문항에 대한 경로 판단 결과입니다.

    `initial_route` / `final_route` 는 Supervisor fallback 을 기록하기 위한
    확장 지점입니다. 현재 규칙 라우터는 한 번만 판단하므로 두 값이 모두
    `route` 와 같습니다.
    """

    route: str
    reason: str = ""
    warnings: tuple[str, ...] = ()
    initial_route: str | None = None
    final_route: str | None = None
    route_path: tuple[str, ...] = ()

    @property
    def resolved_initial_route(self) -> str:
        return self.initial_route or self.route

    @property
    def resolved_final_route(self) -> str:
        return self.final_route or self.route


class RouteObserver(Protocol):
    """경로 판단 결과를 제공하는 객체의 최소 인터페이스입니다."""

    def observe(self, question: str) -> RouteObservation: ...


@dataclass(frozen=True)
class RouterObserver:
    """`QuestionRouter` 계열 객체를 `RouteObserver` 로 감쌉니다."""

    router: Any

    def observe(self, question: str) -> RouteObservation:
        decision = self.router.route_query(question)
        return RouteObservation(
            route=decision.route,
            reason=decision.reason,
            warnings=tuple(decision.warnings or ()),
        )


@dataclass
class _Bucket:
    total: int = 0
    correct: int = 0

    def add(self, is_correct: bool) -> None:
        self.total += 1
        self.correct += int(is_correct)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.correct / self.total, 4) if self.total else None,
        }


@dataclass
class RoutingEvaluation:
    details: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


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
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Golden case at line {line_number} must be an object")
            if ready_only and row.get("status") != READY_STATUS:
                continue
            cases.append(row)
    if not cases:
        raise ValueError(f"No {READY_STATUS} golden cases found in {path}")
    return cases


def expected_route_of(case: dict[str, Any]) -> str:
    """`query_strategy` 를 기대 경로로 변환합니다."""
    strategy = case.get("query_strategy")
    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError(f"Case {case.get('case_id')!r} has no query_strategy")
    route = strategy.strip().upper()
    if route not in VALID_ROUTES:
        raise ValueError(
            f"Case {case.get('case_id')!r} has unknown query_strategy {strategy!r}"
        )
    return route


def evaluate_routing(
    cases: Iterable[dict[str, Any]], observer: RouteObserver
) -> RoutingEvaluation:
    """문항별 라우팅 결과와 요약 지표를 계산합니다."""
    details: list[dict[str, Any]] = []
    overall = _Bucket()
    by_question_type: dict[str, _Bucket] = defaultdict(_Bucket)
    by_expected_route: dict[str, _Bucket] = defaultdict(_Bucket)
    confusion: Counter[tuple[str, str]] = Counter()
    family_overall = _Bucket()
    by_route_family: dict[str, _Bucket] = defaultdict(_Bucket)
    hop_discrimination = _Bucket()

    for case in cases:
        expected_route = expected_route_of(case)
        observation = observer.observe(case.get("question") or "")
        actual_route = observation.resolved_final_route
        route_correct = actual_route == expected_route
        question_type = case.get("question_type") or "unknown"

        overall.add(route_correct)
        by_question_type[question_type].add(route_correct)
        by_expected_route[expected_route].add(route_correct)
        confusion[(expected_route, actual_route)] += 1

        expected_family = ROUTE_FAMILIES[expected_route]
        family_correct = expected_family == ROUTE_FAMILIES[actual_route]
        family_overall.add(family_correct)
        by_route_family[expected_family].add(family_correct)
        if expected_family == "GRAPH_FAMILY":
            # 그래프 계열 안에서 1-hop/2-hop 을 맞췄는지. 지표가 아니라 진단용입니다.
            hop_discrimination.add(route_correct)

        details.append(
            {
                "case_id": case.get("case_id"),
                "question_type": question_type,
                "expected_route": expected_route,
                "actual_route": actual_route,
                "route_correct": route_correct,
                "reason": observation.reason,
                "warnings": list(observation.warnings),
                # Supervisor fallback 확장 지점.
                "initial_route": observation.resolved_initial_route,
                "final_route": actual_route,
                "route_path": list(observation.route_path) or [actual_route],
            }
        )

    reject_bucket = by_expected_route.get("REJECT", _Bucket())
    summary = {
        "total_cases": overall.total,
        "overall": overall.as_dict(),
        "by_question_type": {
            key: bucket.as_dict() for key, bucket in sorted(by_question_type.items())
        },
        "by_expected_route": {
            key: bucket.as_dict() for key, bucket in sorted(by_expected_route.items())
        },
        "reject": reject_bucket.as_dict(),
        "route_family_overall": family_overall.as_dict(),
        "by_route_family": {
            key: bucket.as_dict() for key, bucket in sorted(by_route_family.items())
        },
        "hop_discrimination": hop_discrimination.as_dict(),
        "confusion_matrix": _confusion_matrix(confusion),
        "metric_policy": METRIC_POLICY,
    }
    return RoutingEvaluation(details=details, summary=summary)


def _confusion_matrix(confusion: Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """기대 경로 × 실제 경로 교차표를 반환합니다."""
    expected_routes = sorted({expected for expected, _ in confusion})
    actual_routes = sorted({actual for _, actual in confusion})
    return {
        expected: {actual: confusion.get((expected, actual), 0) for actual in actual_routes}
        for expected in expected_routes
    }


DETAIL_COLUMNS = (
    "case_id",
    "question_type",
    "expected_route",
    "actual_route",
    "route_correct",
    "reason",
    "warnings",
    "initial_route",
    "final_route",
    "route_path",
)


def write_details(
    details: list[dict[str, Any]], path: str | Path, *, detail_format: str = "jsonl"
) -> Path:
    """문항별 결과를 JSONL 또는 CSV 로 저장합니다."""
    return write_detail_rows(
        details,
        path,
        columns=DETAIL_COLUMNS,
        detail_format=detail_format,
        list_separators={"route_path": " > "},
    )


def build_observer(router_name: str) -> RouteObserver:
    if router_name == "rule":
        return RouterObserver(QuestionRouter())
    if router_name == "jev":
        return RouterObserver(JevQuestionRouter.from_env())
    raise ValueError(f"Unknown router: {router_name}")


def format_report(summary: dict[str, Any]) -> str:
    """요약 지표를 콘솔 출력용 문자열로 만듭니다."""
    lines: list[str] = []
    overall = summary["overall"]
    lines.append(f"평가 문항: {summary['total_cases']}")
    reject = summary["reject"]
    lines.append(
        f"[공식] REJECT 정확도: {reject['correct']}/{reject['total']}"
        f" ({_percent(reject['accuracy'])})"
    )
    family = summary.get("route_family_overall")
    if family:
        lines.append(
            f"[참고] 3-way 정확도(그래프계열/VECTOR/REJECT): "
            f"{family['correct']}/{family['total']} ({_percent(family['accuracy'])})"
        )
    lines.append(
        f"[단독 인용 금지] 4-way 정확도: {overall['correct']}/{overall['total']}"
        f" ({_percent(overall['accuracy'])})"
    )
    hop = summary.get("hop_discrimination")
    if hop and hop["total"]:
        lines.append(
            f"[지표 아님] 그래프계열 내 hop 구분: {hop['correct']}/{hop['total']}"
            f" ({_percent(hop['accuracy'])})"
            " — 질문에 hop 단서가 없어 라우터 결함이 아닙니다"
        )

    lines.append("")
    lines.append("기대 경로별 정확도")
    for route, bucket in summary["by_expected_route"].items():
        lines.append(
            f"  {route:<7} {bucket['correct']}/{bucket['total']} ({_percent(bucket['accuracy'])})"
        )

    lines.append("")
    lines.append("질문 유형별 정확도")
    for question_type, bucket in summary["by_question_type"].items():
        lines.append(
            f"  {question_type:<14} {bucket['correct']}/{bucket['total']}"
            f" ({_percent(bucket['accuracy'])})"
        )

    matrix = summary["confusion_matrix"]
    actual_routes = sorted({actual for row in matrix.values() for actual in row})
    lines.append("")
    lines.append("Confusion matrix (행: 기대, 열: 실제)")
    lines.append("  " + " " * 9 + "".join(f"{actual:>9}" for actual in actual_routes))
    for expected, row in matrix.items():
        cells = "".join(f"{row.get(actual, 0):>9}" for actual in actual_routes)
        lines.append(f"  {expected:<9}{cells}")
    return "\n".join(lines)


def _percent(accuracy: float | None) -> str:
    return "N/A" if accuracy is None else f"{accuracy * 100:.1f}%"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="골든셋 라우팅 정확도 평가")
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--router", choices=("rule", "jev"), default="rule")
    parser.add_argument("--detail-format", choices=("jsonl", "csv"), default="jsonl")
    parser.add_argument(
        "--tag",
        default=None,
        help="출력 파일 이름에 붙일 접미사입니다. 기본값은 --router 값입니다.",
    )
    parser.add_argument(
        "--include-non-ready",
        action="store_true",
        help="status 가 READY 가 아닌 문항도 평가에 포함합니다.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_golden_cases(args.golden_set, ready_only=not args.include_non_ready)
    observer = build_observer(args.router)
    evaluation = evaluate_routing(cases, observer)

    tag = args.tag or args.router
    detail_path = write_details(
        evaluation.details,
        args.output_dir / f"routing_details_{tag}.{args.detail_format}",
        detail_format=args.detail_format,
    )
    summary = dict(evaluation.summary)
    summary["meta"] = {
        "golden_set": str(args.golden_set),
        "router": args.router,
        "ready_only": not args.include_non_ready,
        "detail_path": str(detail_path),
    }
    summary_path = write_summary(summary, args.output_dir / f"routing_summary_{tag}.json")

    print(format_report(summary))
    print()
    print(f"문항별 결과: {detail_path}")
    print(f"요약 지표: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
