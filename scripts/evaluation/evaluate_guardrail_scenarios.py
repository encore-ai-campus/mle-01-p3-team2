"""Run the guardrail scenario set across every target and write one report.

``build_guardrail_eval_set.py`` produces cases for three targets:

``guardrail_pipeline``
    ``src.guardrails.pipeline.validate_request`` -- the validators themselves.
``router_scope_gate``
    ``src.retrievers.router.QuestionRouter.scope_gate`` -- the input rail that
    rejects out-of-scope questions before retrieval.  ``REJECT`` is reported as
    ``BLOCK`` and an in-scope pass as ``ALLOW`` so one decision vocabulary
    covers every target.
``nemo_rails``
    The Colang output rail in ``src/guardrails/nemo`` executed through
    ``LLMRails``, which is the only target that shows what a user receives.
    The rail has a single refusal path, so it collapses FALLBACK into BLOCK.

The report groups results by ``request_class`` (정상 / 무관 / 위험 요청) and
records false positives and false negatives separately, because a guardrail
cannot be judged by block count alone.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.evaluate_guardrails import validate_cases  # noqa: E402
from scripts.evaluation.report_io import write_detail_rows, write_summary  # noqa: E402

DEFAULT_CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "guardrail_eval_set_v2.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "guardrails"

#: Refusal text defined in src/guardrails/nemo/rails.co.
NEMO_REFUSAL_TEXT = (
    "검색 근거가 부족하거나 검증되지 않아 답변을 제공할 수 없습니다. "
    "공식 자료를 다시 확인해 주세요."
)

DETAIL_COLUMNS = (
    "case_id",
    "tier",
    "target",
    "request_class",
    "branch_id",
    "scenario_type",
    "description",
    "input",
    "expected_decision",
    "actual_decision",
    "expected_allowed",
    "actual_allowed",
    "passed",
    "discriminating",
    "validation_path",
    "integration_pending",
    "reason",
    "warnings",
    "error_type",
    "error_message",
    "details",
)


class Outcome:
    """Normalized result of running one case against its target."""

    __slots__ = ("allowed", "decision", "reason", "warnings", "details")

    def __init__(
        self,
        *,
        allowed: bool,
        decision: str,
        reason: str,
        warnings: Sequence[str] = (),
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.allowed = allowed
        self.decision = decision
        self.reason = reason
        self.warnings = list(warnings)
        self.details = dict(details or {})


# --------------------------------------------------------------------------
# Target runners
# --------------------------------------------------------------------------


def _run_pipeline(request: Mapping[str, Any]) -> Outcome:
    from src.guardrails.contracts import GuardrailRequest
    from src.guardrails.pipeline import validate_request

    result = validate_request(GuardrailRequest.from_mapping(request))
    return Outcome(
        allowed=result.allowed,
        decision=result.decision,
        reason=result.reason,
        warnings=result.warnings,
        details=result.details,
    )


def _run_scope_gate(request: Mapping[str, Any]) -> Outcome:
    from src.retrievers.router import QuestionRouter

    question = request.get("question")
    if not isinstance(question, str):
        question = "" if question is None else str(question)

    decision = QuestionRouter().scope_gate(question)
    if decision is None:
        return Outcome(
            allowed=True,
            decision="ALLOW",
            reason="범위 안으로 판단해 검색 단계로 넘김",
            details={"route": None},
        )
    return Outcome(
        allowed=False,
        decision="BLOCK",
        reason=decision.reason,
        warnings=decision.warnings,
        details={"route": decision.route},
    )


def _run_supervisor_adapter(state: Mapping[str, Any]) -> Outcome:
    """Run the LangGraph node the Supervisor actually calls.

    ``guardrails_node`` returns a state update, not a ValidationResult:
    ``final_status`` says whether the answer survived and
    ``guardrail_decision`` carries the four-value decision.
    """

    from src.guardrails.supervisor_adapter import guardrails_node

    update = guardrails_node(dict(state))
    decision = str(update.get("guardrail_decision") or "ERROR")
    status = str(update.get("final_status") or "")
    return Outcome(
        allowed=status == "ANSWERED",
        decision=decision,
        reason=str(update.get("guardrail_reason") or ""),
        warnings=update.get("warnings") or (),
        details={
            "final_status": status,
            "final_answer": str(update.get("final_answer") or "")[:200],
        },
    )


class NemoInputRailRunner:
    """Boot the Colang rails once and run input-rail-only generations."""

    def __init__(self) -> None:
        from nemoguardrails import LLMRails

        from src.guardrails.nemo import load_config

        self._rails = LLMRails(load_config())

    def run(self, request: Mapping[str, Any]) -> Outcome:
        from nemoguardrails.rails.llm.options import GenerationOptions

        question = request.get("question")
        if not isinstance(question, str):
            question = ""

        response = self._rails.generate(
            messages=[{"role": "user", "content": question}],
            options=GenerationOptions(rails=["input"]),
        )
        payload = getattr(response, "response", response)
        content = _last_assistant_content(payload)
        blocked = not content or content != question

        # The input rail either lets the question through unchanged or
        # replaces it with a refusal; anything non-empty and different is a
        # refusal from the rail.
        refused = bool(content) and content != question
        return Outcome(
            allowed=not refused,
            decision="BLOCK" if refused else "ALLOW",
            reason=(
                f"입력 레일이 차단함: {content[:120]}"
                if refused
                else "입력 레일을 통과함"
            ),
            details={"rail_output": content, "blocked": blocked},
        )


class NemoRailsRunner:
    """Boot the Colang rails once and run output-rail-only generations."""

    def __init__(self, *, register_action: bool) -> None:
        from nemoguardrails import LLMRails

        from src.guardrails.nemo import load_config

        self._config = load_config()
        self._rails = LLMRails(self._config)
        self.action_auto_registered = self._is_registered()
        self.action_manually_registered = False

        if register_action and not self.action_auto_registered:
            from src.guardrails.nemo.actions import validate_graph_rag_response

            self._rails.register_action(
                validate_graph_rag_response, name="validate_graph_rag_response"
            )
            self.action_manually_registered = self._is_registered()

    def _is_registered(self) -> bool:
        registry = self._rails.runtime.action_dispatcher.registered_actions
        return "validate_graph_rag_response" in registry

    def run(self, request: Mapping[str, Any]) -> Outcome:
        from nemoguardrails.rails.llm.options import GenerationOptions

        answer = request.get("bot_message", request.get("answer", ""))
        if not isinstance(answer, str):
            answer = ""
        question = request.get("question")
        if not isinstance(question, str):
            question = ""

        context = {key: value for key, value in request.items() if key != "metadata"}
        context.setdefault("bot_message", answer)

        response = self._rails.generate(
            messages=[
                {"role": "context", "content": context},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            options=GenerationOptions(rails=["output"]),
        )
        payload = getattr(response, "response", response)
        content = _last_assistant_content(payload)

        details = {
            "rail_output": content,
            "action_registered": self._is_registered(),
        }
        if content == answer:
            return Outcome(
                allowed=True,
                decision="ALLOW",
                reason="출력 레일을 통과해 답변이 그대로 전달됨",
                details=details,
            )
        if content.strip() == NEMO_REFUSAL_TEXT.strip():
            return Outcome(
                allowed=False,
                decision="BLOCK",
                reason="출력 레일이 정의된 거부 문구로 대체함",
                details=details,
            )
        return Outcome(
            allowed=False,
            decision="RAIL_ERROR",
            reason=f"정의되지 않은 레일 출력: {content[:120]}",
            details=details,
        )


def _last_assistant_content(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        value = payload.get("content", "")
        return value if isinstance(value, str) else ""
    if isinstance(payload, Sequence):
        for item in reversed(list(payload)):
            if isinstance(item, Mapping) and item.get("role") == "assistant":
                value = item.get("content", "")
                return value if isinstance(value, str) else ""
    return ""


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def _observed_method(details: Mapping[str, Any]) -> str:
    """Which validation path actually ran, as the validator itself recorded it.

    ``validation_path`` on the case is a static prediction; a request blocked
    by the evidence or pesticide validator never reaches output validation at
    all, so the two must not be confused in the report.
    """

    method = details.get("validation_method")
    if isinstance(method, str) and method:
        return method
    return "not_reached"


def _case_input(case: Mapping[str, Any]) -> str:
    """A short, report-ready rendering of what was sent to the target."""

    request = case.get("request") or {}
    target = case.get("target")
    if target == "router_scope_gate":
        return f"질문={request.get('question')!r}"

    answer = request.get("bot_message", request.get("answer"))
    evidence = request.get("evidence")
    evidence_count = len(evidence) if isinstance(evidence, (list, tuple)) else 0
    parts = [
        f"질문={str(request.get('question'))[:40]!r}",
        f"답변={str(answer)[:60]!r}",
        f"근거={evidence_count}건",
    ]
    if request.get("pesticide_use_requested"):
        parts.append("농약사용요청=True")
    return " / ".join(parts)


def evaluate_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    rails_runner: NemoInputRailRunner | None = None,
) -> list[dict[str, Any]]:
    """Run every case against its target, preserving failures as report rows."""

    validate_cases(cases)
    rows: list[dict[str, Any]] = []
    for case in cases:
        target = str(case.get("target", "guardrail_pipeline"))
        expected_allowed = bool(case["expected_allowed"])
        expected_decision = str(case["expected_decision"])
        error_type: str | None = None
        error_message: str | None = None

        try:
            if target == "guardrail_pipeline":
                outcome = _run_pipeline(case["request"])
            elif target == "router_scope_gate":
                outcome = _run_scope_gate(case["request"])
            elif target == "supervisor_adapter":
                outcome = _run_supervisor_adapter(case["request"])
            elif target == "nemo_input_rail":
                if rails_runner is None:
                    raise RuntimeError("nemo_input_rail target was not initialized")
                outcome = rails_runner.run(case["request"])
            else:
                raise ValueError(f"unknown target: {target}")
            actual_allowed: bool | None = outcome.allowed
            actual_decision: str | None = outcome.decision
            reason = outcome.reason
            warnings = outcome.warnings
            details = outcome.details
            passed = (
                actual_allowed == expected_allowed
                and actual_decision == expected_decision
            )
        except Exception as exc:  # Keep the rest of the batch usable.
            actual_allowed = None
            actual_decision = None
            reason = "평가 실행 오류"
            warnings = []
            details = {}
            passed = False
            error_type = type(exc).__name__
            error_message = str(exc)

        rows.append(
            {
                "case_id": case["case_id"],
                "tier": case.get("tier", "unknown"),
                "target": target,
                "request_class": case.get("request_class", "unknown"),
                "branch_id": case.get("branch_id", "unknown"),
                "scenario_type": case.get("scenario_type", "unknown"),
                "description": case.get("description", ""),
                "input": _case_input(case),
                "expected_decision": expected_decision,
                "actual_decision": actual_decision,
                "expected_allowed": expected_allowed,
                "actual_allowed": actual_allowed,
                "passed": passed,
                "discriminating": bool(case.get("discriminating", True)),
                "validation_path": case.get("validation_path", "unknown"),
                "observed_validation_method": _observed_method(details),
                "integration_pending": bool(case.get("integration_pending", False)),
                "reason": reason,
                "warnings": warnings,
                "error_type": error_type,
                "error_message": error_message,
                "details": details,
            }
        )
    return rows


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def _bucket(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    passed = sum(bool(row.get("passed")) for row in selected)
    return {
        "total": len(selected),
        "passed": passed,
        "failed": len(selected) - passed,
        "pass_rate": _rate(passed, len(selected)),
    }


def _group(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    return {name: _bucket(value) for name, value in sorted(grouped.items())}


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    cases_path: Path,
    rails_runner: NemoInputRailRunner | None,
) -> dict[str, Any]:
    """Aggregate by tier, target and request class, keeping 오탐·누락 apart."""

    discriminating = [row for row in rows if row.get("discriminating")]
    # Cases whose target is still being wired up are reported separately: a
    # failure there is unfinished work, not a defect in the component.
    countable = [row for row in discriminating if not row.get("integration_pending")]
    pending = [row for row in rows if row.get("integration_pending")]
    false_positives_all = [
        str(row["case_id"])
        for row in rows
        if row.get("expected_allowed") is True and row.get("actual_allowed") is False
    ]
    false_negatives_all = [
        str(row["case_id"])
        for row in rows
        if row.get("expected_allowed") is False and row.get("actual_allowed") is True
    ]
    # Non-discriminating cases remain in the detail artifact, but they must not
    # influence quality judgments or the headline FP/FN figures.
    false_positives = [
        str(row["case_id"])
        for row in discriminating
        if row.get("expected_allowed") is True and row.get("actual_allowed") is False
    ]
    false_negatives = [
        str(row["case_id"])
        for row in discriminating
        if row.get("expected_allowed") is False and row.get("actual_allowed") is True
    ]
    errors = [str(row["case_id"]) for row in rows if row.get("error_type")]

    failure_reasons = Counter(
        str(row.get("reason")) for row in countable if not row.get("passed")
    )

    summary: dict[str, Any] = {
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cases_path": cases_path.as_posix(),
        "git_commit": _git_commit(),
        "total_cases": len(rows),
        "overall": _bucket(rows),
        "overall_discriminating_only": _bucket(discriminating),
        "by_tier": _group(rows, "tier"),
        "by_tier_discriminating_only": _group(discriminating, "tier"),
        "by_target": _group(rows, "target"),
        "by_request_class": _group(rows, "request_class"),
        "by_request_class_discriminating_only": _group(discriminating, "request_class"),
        "by_validation_path": _group(rows, "validation_path"),
        "by_validation_path_discriminating_only": _group(discriminating, "validation_path"),
        "by_observed_validation_method": _group(rows, "observed_validation_method"),
        "by_scenario_type": _group(rows, "scenario_type"),
        "actual_decision_counts": dict(
            sorted(Counter(str(row.get("actual_decision") or "ERROR") for row in rows).items())
        ),
        "false_positives": {"count": len(false_positives), "case_ids": false_positives},
        "false_negatives": {"count": len(false_negatives), "case_ids": false_negatives},
        "false_positives_all_cases": {
            "count": len(false_positives_all),
            "case_ids": false_positives_all,
        },
        "false_negatives_all_cases": {
            "count": len(false_negatives_all),
            "case_ids": false_negatives_all,
        },
        "execution_errors": {"count": len(errors), "case_ids": errors},
        "failed_case_ids": [
            str(row["case_id"]) for row in rows if not row.get("passed")
        ],
        "non_discriminating_case_ids": [
            str(row["case_id"]) for row in rows if not row.get("discriminating")
        ],
        "integration_pending": {
            "count": len(pending),
            "case_ids": [str(row["case_id"]) for row in pending],
            "passed": sum(bool(row.get("passed")) for row in pending),
            "note": (
                "연동이 진행 중인 대상의 케이스다. 미통과는 미완성 배선이지 "
                "컴포넌트 결함이 아니므로 distinct_failure_reasons에서 제외했다."
            ),
        },
        "distinct_failure_reasons": [
            {"reason": reason, "count": count, "case_ids": [
                str(row["case_id"])
                for row in countable
                if not row.get("passed") and str(row.get("reason")) == reason
            ]}
            for reason, count in failure_reasons.most_common()
        ],
        "reporting_note": (
            "전체 통과율을 단일 지표로 쓰지 말 것. 오탐(정상을 막음)과 "
            "누락(위험을 통과)을 따로 보고하고, 미통과 건수가 아니라 "
            "distinct_failure_reasons의 개수를 결함 수로 보고한다. "
            "non_discriminating_case_ids는 능력의 증거로 인용하지 않는다. "
            "연동 진행 중인 대상은 integration_pending으로 따로 보고하고, "
            "출력 검증 경로가 확정되기 전까지는 선언된 by_validation_path와 "
            "실제 observed_validation_method를 나눠서 제시한다. 비판별 문항은 "
            "원본 상세와 전체 집계에 남기되 판별 가능 전용 집계에서 제외한다."
        ),
    }
    return summary


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def format_report(summary: Mapping[str, Any]) -> str:
    overall = summary["overall"]
    rate = overall["pass_rate"]
    rate_text = "N/A" if rate is None else f"{rate * 100:.1f}%"
    lines = [
        f"평가 문항: {summary['total_cases']}",
        f"통과: {overall['passed']}/{overall['total']} ({rate_text})",
        f"오탐(정상을 막음): {summary['false_positives']['count']}",
        f"누락(위험을 통과): {summary['false_negatives']['count']}",
        f"실행 오류: {summary['execution_errors']['count']}",
        "",
        "요청 유형별:",
    ]
    for name, bucket in summary["by_request_class"].items():
        lines.append(f"  {name}: {bucket['passed']}/{bucket['total']}")
    lines.append("")
    lines.append("대상별:")
    for name, bucket in summary["by_target"].items():
        lines.append(f"  {name}: {bucket['passed']}/{bucket['total']}")
    lines.append("")
    lines.append("출력 검증 경로별:")
    for name, bucket in summary["by_validation_path"].items():
        lines.append(f"  {name}: {bucket['passed']}/{bucket['total']}")
    pending = summary["integration_pending"]
    if pending["count"]:
        lines.append("")
        lines.append(
            f"연동 진행 중(결함 집계 제외): {pending['passed']}/{pending['count']}"
            f" — {', '.join(pending['case_ids'])}"
        )
    reasons = summary["distinct_failure_reasons"]
    if reasons:
        lines.append("")
        lines.append(f"서로 다른 미통과 사유: {len(reasons)}개 (연동 대기 제외)")
        for item in reasons:
            lines.append(f"  {item['count']:2d}건  {item['reason'][:70]}")
    return "\n".join(lines)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            rows.append(row)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--detail-format", choices=("jsonl", "csv"), default="jsonl")
    parser.add_argument("--tag", default="scenarios")
    parser.add_argument(
        "--targets",
        default="",
        help="쉼표로 구분한 대상 필터. 비우면 전체를 실행한다.",
    )
    parser.add_argument(
        "--register-nemo-action",
        action="store_true",
        help=(
            "NeMo가 액션을 자동 등록하지 못할 때 수동 등록한 뒤 실행한다. "
            "배포 상태가 아니라 수정 후 동작을 측정한다."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = read_jsonl(args.cases)

    wanted = {name.strip() for name in args.targets.split(",") if name.strip()}
    if wanted:
        cases = [c for c in cases if str(c.get("target", "")) in wanted]
    if not cases:
        raise SystemExit("실행할 케이스가 없습니다.")

    rails_runner: NemoInputRailRunner | None = None
    if any(str(c.get("target")) == "nemo_input_rail" for c in cases):
        rails_runner = NemoInputRailRunner()

    rows = evaluate_cases(cases, rails_runner=rails_runner)
    summary = build_summary(rows, cases_path=args.cases, rails_runner=rails_runner)

    detail_path = write_detail_rows(
        rows,
        args.output_dir / f"guardrail_scenario_details_{args.tag}.{args.detail_format}",
        columns=DETAIL_COLUMNS,
        detail_format=args.detail_format,
    )
    summary_path = write_summary(
        summary, args.output_dir / f"guardrail_scenario_summary_{args.tag}.json"
    )

    print(format_report(summary))
    print()
    print(f"문항별 결과: {detail_path}")
    print(f"요약 지표: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
