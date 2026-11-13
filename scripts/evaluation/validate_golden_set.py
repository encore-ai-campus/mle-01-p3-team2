"""골든셋이 `docs/golden-set-contract.md` 의 규칙을 지키는지 검사합니다.

문서만 두면 규칙이 썩기 때문에, 계약을 실행 가능한 검사로 고정합니다.
`tests/test_golden_set_contract.py` 가 이 검증기를 CI 에서 돌립니다.

실행:
    python -m scripts.evaluation.validate_golden_set
    python -m scripts.evaluation.validate_golden_set --golden-set <경로> --skip-corpus
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLDEN_PATH = PROJECT_ROOT / "data" / "evaluation" / "golden_set_final.jsonl"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LIST_FIELDS = (
    "expected_facts",
    "expected_entity_ids",
    "expected_relation_ids",
    "expected_evidence_ids",
    "expected_source_chunk_ids",
    "safety_assertions",
)
STR_FIELDS = (
    "case_id",
    "question",
    "reference",
    "question_type",
    "difficulty",
    "query_strategy",
    "expected_behavior",
    "status",
)
REQUIRED_FIELDS = tuple(sorted(LIST_FIELDS + STR_FIELDS))

VALID_STRATEGIES = {"graph", "vector", "hybrid", "reject"}
VALID_BEHAVIORS = {"ANSWER", "ANSWER_WITH_WARNING", "REJECT"}
READY_STATUS = "READY"

# 계약 3절: 선행어 없는 지시어를 금지합니다. 같은 문장에서 대상을 밝힌 경우는
# 아래 ANTECEDENT_HINTS 로 예외 처리합니다.
DEMONSTRATIVE_PATTERNS = (
    "이 병",
    "이 해충",
    "이 약제",
    "이 살균제",
    "이 살충제",
    "이 농약",
    "이 품종",
    "이 증상",
    "그 병",
)
# 지시어 앞에 대상이 명시됐는지 보는 단서입니다(병해충 이름은 보통 이 접미사로 끝납니다).
ANTECEDENT_HINTS = (
    "병",
    "나방",
    "벌레",
    "응애",
    "진딧물",
    "나비",
    "균",
)

# 계약 4절: 경고가 반드시 필요한 트리거입니다.
WARNING_TRIGGERS: tuple[tuple[str, str], ...] = (
    (r"희석|수확 \d+일 전|수확 전 사용|사용 횟수|사용횟수|살포", "농약 사용 조건"),
    (r"kg/10a", "비료 표준사용량"),
    (r"PRELIMINARY|등록 유효성|등록 여부를 확인|검증상태", "검증상태 미확정"),
    (r"CR 표기|표기만으로", "품종 표기 저항성 추정"),
)

CORPUS_FIELDS = (
    ("expected_entity_ids", "entities.jsonl", "entity_id"),
    ("expected_relation_ids", "relations.jsonl", "relation_id"),
    ("expected_evidence_ids", "evidence.jsonl", "evidence_id"),
    ("expected_source_chunk_ids", "chunks.jsonl", "chunk_id"),
)


@dataclass(frozen=True)
class Violation:
    case_id: str
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.case_id}: {self.detail}"


def load_rows(golden_path: str | Path) -> list[dict[str, Any]]:
    """골든셋 JSONL 을 읽습니다. 필터 없이 전부 돌려줍니다."""
    path = Path(golden_path)
    rows: list[dict[str, Any]] = []
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
            rows.append(row)
    if not rows:
        raise ValueError(f"No golden cases found in {path}")
    return rows


def warning_triggers(row: dict[str, Any]) -> list[str]:
    """계약 4절 트리거 중 해당하는 사유를 돌려줍니다."""
    blob = str(row.get("reference", "")) + " " + " ".join(row.get("expected_facts") or [])
    return [label for pattern, label in WARNING_TRIGGERS if re.search(pattern, blob)]


def _has_antecedent(question: str, marker: str) -> bool:
    """지시어 앞쪽에 대상이 밝혀져 있는지 봅니다."""
    head = question.split(marker)[0]
    return any(hint in head for hint in ANTECEDENT_HINTS)


def check_schema(rows: Iterable[dict[str, Any]]) -> list[Violation]:
    out: list[Violation] = []
    for row in rows:
        cid = str(row.get("case_id", "<no case_id>"))
        missing = [f for f in REQUIRED_FIELDS if f not in row]
        if missing:
            out.append(Violation(cid, "schema", f"필드 누락: {missing}"))
        extra = [f for f in row if f not in REQUIRED_FIELDS]
        if extra:
            out.append(Violation(cid, "schema", f"정의되지 않은 필드: {extra}"))
        for field in LIST_FIELDS:
            if field in row and not isinstance(row[field], list):
                out.append(Violation(cid, "schema", f"{field} 는 목록이어야 합니다"))
        for field in STR_FIELDS:
            if field in row and not isinstance(row[field], str):
                out.append(Violation(cid, "schema", f"{field} 는 문자열이어야 합니다"))
        if row.get("status") != READY_STATUS:
            out.append(Violation(cid, "status", f"status 가 {row.get('status')!r} 입니다"))
        if row.get("query_strategy") not in VALID_STRATEGIES:
            out.append(
                Violation(cid, "strategy", f"알 수 없는 값: {row.get('query_strategy')!r}")
            )
        if row.get("expected_behavior") not in VALID_BEHAVIORS:
            out.append(
                Violation(cid, "behavior", f"알 수 없는 값: {row.get('expected_behavior')!r}")
            )
    return out


def check_unique_case_ids(rows: Iterable[dict[str, Any]]) -> list[Violation]:
    counts = Counter(str(row.get("case_id")) for row in rows)
    return [
        Violation(cid, "case_id", f"{count}번 중복됩니다")
        for cid, count in counts.items()
        if count > 1
    ]


def check_reject_rows(rows: Iterable[dict[str, Any]]) -> list[Violation]:
    """계약 6절: REJECT 문항의 필드는 전부 비어 있어야 합니다."""
    out: list[Violation] = []
    for row in rows:
        cid = str(row.get("case_id"))
        is_reject_strategy = row.get("query_strategy") == "reject"
        is_reject_behavior = row.get("expected_behavior") == "REJECT"
        if is_reject_strategy != is_reject_behavior:
            out.append(
                Violation(
                    cid,
                    "reject",
                    "query_strategy='reject' 와 expected_behavior='REJECT' 가 어긋납니다",
                )
            )
        if not is_reject_strategy:
            continue
        if str(row.get("reference", "")).strip():
            out.append(Violation(cid, "reject", "reference 가 비어 있지 않습니다"))
        nonempty = [f for f in LIST_FIELDS if row.get(f)]
        if nonempty:
            out.append(Violation(cid, "reject", f"비어 있어야 하는 필드: {nonempty}"))
    return out


def check_behavior_rule(rows: Iterable[dict[str, Any]]) -> list[Violation]:
    """계약 4절: 트리거가 있으면 경고가 필수이고, 경고와 safety 는 함께 성립합니다."""
    out: list[Violation] = []
    for row in rows:
        cid = str(row.get("case_id"))
        warns = row.get("expected_behavior") == "ANSWER_WITH_WARNING"
        has_safety = bool(row.get("safety_assertions"))
        if warns != has_safety:
            out.append(
                Violation(
                    cid,
                    "behavior",
                    "ANSWER_WITH_WARNING 과 safety_assertions 가 함께 성립하지 않습니다",
                )
            )
        if row.get("query_strategy") == "reject":
            continue
        triggers = warning_triggers(row)
        if triggers and not warns:
            out.append(
                Violation(cid, "behavior", f"경고가 필요합니다: {triggers}")
            )
    return out


def check_questions(rows: Iterable[dict[str, Any]]) -> list[Violation]:
    """계약 3절: 선행어 없는 지시어를 금지합니다."""
    out: list[Violation] = []
    for row in rows:
        cid = str(row.get("case_id"))
        question = str(row.get("question", ""))
        if not question.strip():
            out.append(Violation(cid, "question", "question 이 비어 있습니다"))
            continue
        for marker in DEMONSTRATIVE_PATTERNS:
            if marker in question and not _has_antecedent(question, marker):
                out.append(
                    Violation(
                        cid,
                        "question",
                        f"선행어 없는 지시어 '{marker}' 가 있습니다",
                    )
                )
    return out


def check_duplicates(rows: Iterable[dict[str, Any]]) -> list[Violation]:
    """같은 relation+evidence 조합을 쓰는 문항이 둘 이상이면 중복입니다."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for row in rows:
        if row.get("query_strategy") == "reject":
            continue
        key = (
            tuple(sorted(row.get("expected_relation_ids") or [])),
            tuple(sorted(row.get("expected_evidence_ids") or [])),
        )
        groups[key].append(str(row.get("case_id")))
    return [
        Violation(ids[0], "duplicate", f"같은 정답 ID 조합을 쓰는 문항: {ids}")
        for ids in groups.values()
        if len(ids) > 1
    ]


def check_strategy_structure(rows: Iterable[dict[str, Any]]) -> list[Violation]:
    """계약 2·5절: 라벨과 gold 구조의 대응을 고정합니다."""
    out: list[Violation] = []
    for row in rows:
        cid = str(row.get("case_id"))
        strategy = row.get("query_strategy")
        hops = len(row.get("expected_relation_ids") or [])
        has_entities = bool(row.get("expected_entity_ids"))
        if strategy == "vector" and hops:
            out.append(Violation(cid, "strategy", "vector 인데 relation 이 있습니다"))
        if strategy in {"graph", "hybrid"} and not has_entities:
            out.append(
                Violation(cid, "strategy", f"{strategy} 인데 entity 가 없습니다")
            )
        if strategy == "graph" and hops != 1:
            out.append(Violation(cid, "strategy", f"graph 는 1-hop 이어야 합니다(현재 {hops})"))
        if strategy == "hybrid" and hops != 2:
            out.append(Violation(cid, "strategy", f"hybrid 는 2-hop 이어야 합니다(현재 {hops})"))
    return out


def check_id_duplicates(rows: Iterable[dict[str, Any]]) -> list[Violation]:
    out: list[Violation] = []
    for row in rows:
        cid = str(row.get("case_id"))
        for field, *_ in CORPUS_FIELDS:
            values = row.get(field) or []
            dupes = [v for v, c in Counter(values).items() if c > 1]
            if dupes:
                out.append(Violation(cid, "ids", f"{field} 안에 중복: {dupes}"))
    return out


def _corpus_ids(processed_dir: Path, filename: str, key: str) -> set[str]:
    path = processed_dir / filename
    ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                ids.add(json.loads(line)[key])
    return ids


def check_corpus_ids(
    rows: Iterable[dict[str, Any]], processed_dir: str | Path = PROCESSED_DIR
) -> list[Violation]:
    """expected ID 가 실제 코퍼스에 존재하는지 봅니다."""
    directory = Path(processed_dir)
    rows = list(rows)
    out: list[Violation] = []
    for field, filename, key in CORPUS_FIELDS:
        if not (directory / filename).exists():
            continue
        universe = _corpus_ids(directory, filename, key)
        for row in rows:
            missing = [i for i in (row.get(field) or []) if i not in universe]
            if missing:
                out.append(
                    Violation(str(row.get("case_id")), "ids", f"{field} 미존재: {missing}")
                )
    return out


def validate(
    rows: Iterable[dict[str, Any]],
    *,
    processed_dir: str | Path = PROCESSED_DIR,
    check_corpus: bool = True,
) -> list[Violation]:
    """모든 규칙을 적용해 위반 목록을 돌려줍니다. 빈 목록이면 통과입니다."""
    rows = list(rows)
    violations: list[Violation] = []
    violations += check_schema(rows)
    violations += check_unique_case_ids(rows)
    violations += check_reject_rows(rows)
    violations += check_behavior_rule(rows)
    violations += check_questions(rows)
    violations += check_duplicates(rows)
    violations += check_strategy_structure(rows)
    violations += check_id_duplicates(rows)
    if check_corpus:
        violations += check_corpus_ids(rows, processed_dir)
    return violations


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    return {
        "total": len(rows),
        "by_strategy": dict(Counter(r.get("query_strategy") for r in rows)),
        "by_question_type": dict(Counter(r.get("question_type") for r in rows)),
        "by_behavior": dict(Counter(r.get("expected_behavior") for r in rows)),
        "by_difficulty": dict(Counter(r.get("difficulty") for r in rows)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="골든셋 계약 검증")
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--skip-corpus",
        action="store_true",
        help="data/processed 참조 검사를 건너뜁니다(대용량 파일 없이 돌릴 때).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.golden_set)
    violations = validate(
        rows, processed_dir=args.processed_dir, check_corpus=not args.skip_corpus
    )

    stats = summarize(rows)
    print(f"대상: {args.golden_set}")
    print(f"문항 수: {stats['total']}")
    print(f"  경로 라벨: {stats['by_strategy']}")
    print(f"  질문 유형: {stats['by_question_type']}")
    print(f"  기대 동작: {stats['by_behavior']}")
    print(f"  난이도: {stats['by_difficulty']}")
    print()
    if not violations:
        print("계약 위반 없음 (docs/golden-set-contract.md)")
        return 0
    print(f"계약 위반 {len(violations)}건:")
    for violation in violations:
        print(f"  {violation}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
