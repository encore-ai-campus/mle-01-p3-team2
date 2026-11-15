"""답변의 농약 사용조건이 공식 등록정보로 실제 대조되는 비율을 센다.

LLM-as-Judge에서 수치 대조를 떼어내고 결정론 검증에 맡겼으므로,
그 검증이 실제로 수치를 대조하고 있는지 수치로 확인해야 한다.

Supervisor 실행 결과(supervisor_runs.jsonl)의 draft_answer를 Neo4j
등록정보와 대조해 다음을 센다.

    대조   : 답변의 수치를 등록정보와 비교한 항목 수
    미대조 : 답변이 가리킨 제품의 등록값이 없어 비교하지 못한 항목 수

Neo4j만 읽으므로 LLM 호출 비용은 들지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.replay_usage_rail_offline import (  # noqa: E402
    parse_records,
)
from src.guardrails.execution_rail import (  # noqa: E402
    list_registered_usages,
)
from src.guardrails.neo4j_pesticide_lookup import (  # noqa: E402
    make_neo4j_pesticide_usage_list,
)
from src.guardrails.output_usage_validator import (  # noqa: E402
    PATTERNS,
    _answer_values,
    _records_for_row,
    _reference_values,
    _rows,
    validate_pesticide_output,
)

DEFAULT_RUNS = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "ragas_result"
    / "subset"
    / "supervisor_runs.jsonl"
)


def count_coverage(
    answer: str,
    records: list[dict[str, Any]],
) -> tuple[int, int]:
    """대조한 항목 수와 대조하지 못한 항목 수를 센다."""

    checked = 0
    unverified = 0

    for row in _rows(answer):
        scoped = _records_for_row(row, records)

        for field, pattern in PATTERNS.items():
            if not _answer_values(row, field):
                continue

            reference: set[int] = set()

            for item in scoped:
                reference |= _reference_values(item, field, pattern)

            if reference:
                checked += 1
            else:
                unverified += 1

    return checked, unverified


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cases = [
        json.loads(line)
        for line in args.runs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    lookup = make_neo4j_pesticide_usage_list()

    total_checked = 0
    total_unverified = 0
    conflicts: list[str] = []

    for case in sorted(cases, key=lambda item: item.get("case_id") or ""):
        case_id = case.get("case_id")
        evidence = parse_records(case.get("retrieved_contexts"))
        draft = case.get("draft_answer") or ""

        if not evidence or not draft:
            print(f"{case_id}: 등록정보 근거 또는 답변 없음")
            continue

        result = list_registered_usages(
            crop=evidence[0]["crop"],
            target_pest=evidence[0]["target_pest"],
            list_lookup_fn=lookup,
        )

        records = list(result.records) if result.allowed else []
        verdict = validate_pesticide_output(
            draft,
            records=records,
            pesticide_use_requested=True,
        )

        checked, unverified = count_coverage(draft, records)
        total_checked += checked
        total_unverified += unverified

        if not verdict["allowed"]:
            conflicts.append(f"{case_id}: {verdict['reason']}")

        print(
            f"{case_id}: {verdict['decision']:5} | "
            f"대조 {checked}항목 / 미대조 {unverified}항목 | "
            f"등록정보 {len(records)}건"
        )

    total = total_checked + total_unverified
    ratio = total_checked / total if total else 0.0

    print(
        f"\n합계: 대조 {total_checked}항목 / 미대조 {total_unverified}항목 "
        f"({ratio:.1%} 검증)"
    )

    if conflicts:
        print("\n불일치로 판정된 답변:")
        for line in conflicts:
            print(f"  {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
