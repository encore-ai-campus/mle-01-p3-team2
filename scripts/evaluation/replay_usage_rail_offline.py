"""저장된 평가 결과만으로 농약 사용조건 Output Rail을 재실행한다.

RAGAS 평가는 답변 생성과 채점에 모두 LLM을 쓰므로 반복 실행 비용이
크다. 이 스크립트는 이미 저장된 검색 근거(retrieved_contexts)를
등록정보 레코드로 되돌려, Output Rail의 숫자 대조만 API 없이 다시
돌린다.

확인하는 것
    1. 근거에 적힌 값을 그대로 인용한 답변이 통과하는지
       (v3에서 보류된 "희석배수 비교 불가" 오탐의 재발 여부)
    2. 제품 간 수치를 섞은 답변이 여전히 차단되는지
    3. 등록 상한을 넘는 사용횟수가 여전히 차단되는지

확인할 수 없는 것
    답변 생성 프롬프트와 LLM-as-Judge 판정은 LLM이 필요하므로
    이 스크립트 범위가 아니다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.guardrails.output_usage_validator import (  # noqa: E402
    validate_pesticide_output,
)

DEFAULT_RESULTS = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "ragas_result"
    / "ragas_results_v3.jsonl"
)

# 근거 청크의 한글 키를 Execution Rail 표준 필드로 되돌린다.
FIELD_MAP = {
    "작물": "crop",
    "병해충": "target_pest",
    "품목": "product_name",
    "상표": "brand",
    "등록번호": "registration_no",
    "사용적기": "application_timing",
    "수확 전 사용시기": "preharvest_interval",
    "최대 사용횟수": "max_applications",
    "희석배수": "dilution",
    "검증상태": "verification_status",
    "검토사유": "review_reason",
}

REQUIRED_FIELDS = ("product_name", "crop", "target_pest")


def parse_record(context: str) -> dict[str, Any] | None:
    """`키=값; 키=값` 형식의 근거 한 줄을 등록정보 레코드로 바꾼다."""

    if not isinstance(context, str) or "=" not in context:
        return None

    record: dict[str, Any] = {}

    for part in context.split(";"):
        if "=" not in part:
            continue

        key, _, value = part.partition("=")
        field = FIELD_MAP.get(key.strip())

        if field:
            record[field] = value.strip()

    if not all(record.get(field) for field in REQUIRED_FIELDS):
        return None

    return record


def parse_records(contexts: Iterable[Any]) -> list[dict[str, Any]]:
    """한 케이스의 근거 전체를 레코드 목록으로 바꾼다."""

    records = []

    for context in contexts or []:
        record = parse_record(context)

        if record is not None:
            records.append(record)

    return records


def faithful_answer(record: dict[str, Any]) -> str:
    """등록정보의 값을 그대로 인용한 답변 한 줄을 만든다."""

    brand = record.get("brand") or "상표 미확인"
    label = f"{record['product_name']}({brand}"

    # 등록번호까지 적어 상표-등록번호 교차 검사도 함께 재실행한다.
    if record.get("registration_no"):
        label += f", 등록번호 {record['registration_no']}"

    parts = [label + ")은"]

    if record.get("dilution"):
        parts.append(f"{record['dilution']}배로 희석해")

    if record.get("preharvest_interval"):
        parts.append(str(record["preharvest_interval"]))

    if record.get("max_applications"):
        parts.append(str(record["max_applications"]))

    parts.append("살포합니다.")

    return " ".join(parts)


def _first_number(value: Any) -> int | None:
    """문자열에서 처음 나오는 정수를 읽는다."""

    digits = ""

    for char in str(value or ""):
        if char.isdigit():
            digits += char
        elif digits:
            break

    return int(digits) if digits else None


def cross_product_answer(
    records: list[dict[str, Any]],
) -> str | None:
    """한 제품 이름에 다른 제품의 희석배수를 붙인 답변을 만든다."""

    with_dilution = [
        record
        for record in records
        if _first_number(record.get("dilution")) is not None
    ]

    for first in with_dilution:
        for second in with_dilution:
            if _first_number(first["dilution"]) == _first_number(
                second["dilution"]
            ):
                continue

            brand = first.get("brand") or "상표 미확인"

            return (
                f"{first['product_name']}({brand})은 "
                f"{second['dilution']}배로 희석해 살포합니다."
            )

    return None


def over_limit_answer(
    records: list[dict[str, Any]],
) -> str | None:
    """등록 상한보다 한 번 많은 사용횟수를 쓴 답변을 만든다."""

    for record in records:
        limit = _first_number(record.get("max_applications"))

        if limit is None:
            continue

        brand = record.get("brand") or "상표 미확인"

        return (
            f"{record['product_name']}({brand})은 "
            f"최대 {limit + 1}회까지 살포할 수 있습니다."
        )

    return None


def cross_brand_answer(
    records: list[dict[str, Any]],
) -> str | None:
    """한 제품의 등록번호에 다른 제품의 상표를 붙인 답변을 만든다."""

    named = [
        record
        for record in records
        if record.get("brand") and record.get("registration_no")
    ]

    for first in named:
        for second in named:
            if first["brand"] == second["brand"]:
                continue

            return (
                f"{second['brand']}"
                f"(등록번호 {first['registration_no']})은 "
                "등록된 약제입니다."
            )

    return None


def _decision(
    answer: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Output Rail의 농약 사용조건 검증만 실행한다."""

    return validate_pesticide_output(
        answer,
        records=records,
        pesticide_use_requested=True,
    )


def replay_case(case: dict[str, Any]) -> dict[str, Any] | None:
    """한 케이스의 근거로 Output Rail을 다시 돌린다."""

    records = parse_records(case.get("retrieved_contexts"))

    if not records:
        return None

    checks: list[dict[str, Any]] = []

    for record in records:
        result = _decision(faithful_answer(record), records)

        checks.append({
            "check": "근거 그대로 인용",
            "expected": "ALLOW",
            "decision": result["decision"],
            "ok": result["allowed"],
            "reason": result["reason"],
        })

    for name, builder in (
        ("제품 간 수치 혼입", cross_product_answer),
        ("사용횟수 상한 초과", over_limit_answer),
        ("상표-등록번호 불일치", cross_brand_answer),
    ):
        answer = builder(records)

        if answer is None:
            continue

        result = _decision(answer, records)

        checks.append({
            "check": name,
            "expected": "BLOCK",
            "decision": result["decision"],
            "ok": result["decision"] == "BLOCK",
            "reason": result["reason"],
        })

    return {
        "case_id": case.get("case_id"),
        "recorded_status": case.get("final_status"),
        "recorded_reason": case.get("guardrail_reason"),
        "record_count": len(records),
        "checks": checks,
        "passed": all(check["ok"] for check in checks),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="재실행할 평가 결과 JSONL (기본: v3)",
    )
    parser.add_argument(
        "--case-ids",
        default="",
        help="쉼표로 구분한 케이스 ID. 비우면 근거가 있는 전체 케이스",
    )
    parser.add_argument(
        "--only-withheld",
        action="store_true",
        help="저장된 결과에서 보류·차단된 케이스만 재실행",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cases = [
        json.loads(line)
        for line in args.results.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    wanted = {
        case_id.strip()
        for case_id in args.case_ids.split(",")
        if case_id.strip()
    }

    reports = []

    for case in cases:
        if wanted and case.get("case_id") not in wanted:
            continue

        if args.only_withheld and case.get("final_status") not in (
            "WITHHELD",
            "BLOCKED",
        ):
            continue

        report = replay_case(case)

        if report is not None:
            reports.append(report)

    failed = [report for report in reports if not report["passed"]]

    print(f"재실행 케이스: {len(reports)}건")
    print(f"기대와 다른 케이스: {len(failed)}건\n")

    for report in reports:
        status = "OK" if report["passed"] else "FAIL"

        print(
            f"[{status}] {report['case_id']} "
            f"(저장된 상태={report['recorded_status']}, "
            f"근거 {report['record_count']}건)"
        )

        if report["recorded_reason"]:
            print(f"    저장된 사유: {report['recorded_reason'][:70]}")

        grouped: dict[tuple[str, str, bool], int] = {}

        for check in report["checks"]:
            key = (check["check"], check["decision"], check["ok"])
            grouped[key] = grouped.get(key, 0) + 1

        for (name, decision, ok), count in grouped.items():
            mark = "o" if ok else "x"
            print(f"    {mark} {name}: {decision} x{count}")

        print()

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
