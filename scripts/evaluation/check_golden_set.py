"""최종 골든셋의 무결성과 검수 원본과의 정합성을 검사합니다.

왜 필요한가
`golden_set_final.jsonl` 은 통합 노트북이 `reviewed_*.jsonl` 에서 생성하는
산출물입니다. 그런데 산출물만 직접 손으로 고치면 원본과 갈라지고, 다음에
노트북을 돌리는 사람이 그 수정을 조용히 잃습니다. 실제로 질문 구체화 2건,
안전 경고 추가 1건, 중복 제거 2건이 원본에 반영되지 않은 채 쌓여 있었습니다.

검사 항목
1. 파일 자체: JSON 유효성, 노트북 검증 규칙, 중복 질문·case_id
2. case_id 가 GS001 부터 빈 번호 없이 이어지는지
3. `reference` 가 `expected_facts`(+ ANSWER_WITH_WARNING 의 safety_assertions)와 일치하는지
4. 참조 ID 가 `data/processed` 에 실제로 있는지, 관계가 검색 대상인지
5. graph·hybrid 문항의 체인이 노드 순서대로 이어지는지
6. 검수 원본으로 재생성했을 때 결과가 같은지(= 손 편집 누락이 없는지)

실행:
    uv run python -m scripts.evaluation.check_golden_set
    uv run python -m scripts.evaluation.check_golden_set --skip-processed
문제가 있으면 종료 코드 1 입니다.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
DEFAULT_GOLDEN_PATH = EVALUATION_DIR / "golden_set_final.jsonl"
DEFAULT_TEAM_DIR = EVALUATION_DIR / "team_candidates"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 통합 노트북과 같은 값을 씁니다. 달라지면 이 검사는 의미가 없습니다.
REVIEWED_PATTERN = "reviewed_*.jsonl"
EXCLUDED_REVIEWED_FILES = {"reviewed_통합.jsonl"}
FINAL_KEYS = (
    "case_id",
    "question",
    "reference",
    "question_type",
    "difficulty",
    "query_strategy",
    "expected_behavior",
    "expected_facts",
    "expected_entity_ids",
    "expected_relation_ids",
    "expected_evidence_ids",
    "expected_source_chunk_ids",
    "safety_assertions",
    "status",
)
REQUIRED_KEYS = set(FINAL_KEYS) - {"question_type", "difficulty"}
VALID_QUERY_STRATEGIES = {"vector", "graph", "hybrid", "reject"}
VALID_EXPECTED_BEHAVIORS = {"ANSWER", "ANSWER_WITH_WARNING", "REJECT"}
REJECT_EMPTY_FIELDS = (
    "expected_facts",
    "expected_entity_ids",
    "expected_relation_ids",
    "expected_evidence_ids",
    "expected_source_chunk_ids",
    "safety_assertions",
)
GRAPH_ROUTES = {"graph", "hybrid"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return rows


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def add_terminal_punctuation(text: Any) -> str:
    text = str(text).strip()
    if text and text[-1] not in ".!?。！？":
        text += "."
    return text


def make_reference(row: dict[str, Any]) -> str:
    """통합 노트북의 reference 생성 규칙입니다."""
    sentences = [
        add_terminal_punctuation(fact)
        for fact in _ensure_list(row.get("expected_facts"))
        if str(fact).strip()
    ]
    if row.get("expected_behavior") == "ANSWER_WITH_WARNING":
        sentences.extend(
            add_terminal_punctuation(assertion)
            for assertion in _ensure_list(row.get("safety_assertions"))
            if str(assertion).strip()
        )
    return " ".join(dict.fromkeys(sentences))


def check_rules(rows: list[dict[str, Any]]) -> list[str]:
    """노트북 검증 규칙과 중복·연번을 확인합니다."""
    problems: list[str] = []
    for row in rows:
        case_id = row.get("case_id", "case_id 없음")
        missing = REQUIRED_KEYS - set(row)
        if missing:
            problems.append(f"{case_id}: 필수 키 누락 {sorted(missing)}")
            continue
        if row["status"] != "READY":
            problems.append(f"{case_id}: status 가 READY 가 아닙니다({row['status']}).")
        if not str(row["question"]).strip():
            problems.append(f"{case_id}: 질문이 비어 있습니다.")
        if row["query_strategy"] not in VALID_QUERY_STRATEGIES:
            problems.append(f"{case_id}: 잘못된 query_strategy {row['query_strategy']!r}")
        if row["expected_behavior"] not in VALID_EXPECTED_BEHAVIORS:
            problems.append(f"{case_id}: 잘못된 expected_behavior {row['expected_behavior']!r}")

        is_reject = row["query_strategy"] == "reject"
        if is_reject != (row["expected_behavior"] == "REJECT"):
            problems.append(f"{case_id}: reject 경로와 REJECT 동작이 일치하지 않습니다.")
        if is_reject:
            nonempty = [key for key in REJECT_EMPTY_FIELDS if row.get(key)]
            if nonempty:
                problems.append(f"{case_id}: REJECT 문항의 필드가 비어 있지 않습니다 {nonempty}")
            if str(row["reference"]).strip():
                problems.append(f"{case_id}: REJECT reference 가 비어 있지 않습니다.")
        else:
            if not row["expected_facts"]:
                problems.append(f"{case_id}: 기대 사실이 비어 있습니다.")
            if not str(row["reference"]).strip():
                problems.append(f"{case_id}: reference 가 비어 있습니다.")
            if not row["expected_evidence_ids"]:
                problems.append(f"{case_id}: 근거 ID 가 비어 있습니다.")

        expected_reference = make_reference(row)
        if row["reference"] != expected_reference:
            problems.append(
                f"{case_id}: reference 가 expected_facts 와 어긋납니다. "
                f"노트북 규칙상 {len(expected_reference)}자, 현재 {len(row['reference'])}자"
            )

    for question, count in sorted(Counter(r.get("question", "").strip() for r in rows).items()):
        if count > 1:
            problems.append(f"중복 질문({count}회): {question[:50]}")
    for case_id, count in sorted(Counter(r.get("case_id") for r in rows).items()):
        if count > 1:
            problems.append(f"중복 case_id({count}회): {case_id}")

    case_ids = [row.get("case_id") for row in rows]
    expected_ids = [f"GS{index:03d}" for index in range(1, len(rows) + 1)]
    if case_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(case_ids))
        extra = sorted(set(case_ids) - set(expected_ids))
        detail = []
        if missing:
            detail.append(f"빠진 번호 {missing}")
        if extra:
            detail.append(f"범위 밖 {extra}")
        problems.append(
            "case_id 가 GS001 부터 빈 번호 없이 이어지지 않습니다"
            + (f": {', '.join(detail)}" if detail else " (순서가 다릅니다).")
        )
    return problems


def check_processed_references(
    rows: list[dict[str, Any]], processed_dir: str | Path
) -> list[str]:
    """참조 ID 가 실제 데이터에 있는지, 체인이 이어지는지 확인합니다."""
    directory = Path(processed_dir)
    problems: list[str] = []

    entity_ids = {row["entity_id"] for row in read_jsonl(directory / "entities.jsonl")}
    relations = {row["relation_id"]: row for row in read_jsonl(directory / "relations.jsonl")}
    evidence_ids = {row["evidence_id"] for row in read_jsonl(directory / "evidence.jsonl")}
    chunk_ids = {row["chunk_id"] for row in read_jsonl(directory / "chunks.jsonl")}

    for row in rows:
        case_id = row.get("case_id")
        for entity_id in row.get("expected_entity_ids") or []:
            if entity_id not in entity_ids:
                problems.append(f"{case_id}: 없는 엔티티 ID {entity_id}")
        for relation_id in row.get("expected_relation_ids") or []:
            relation = relations.get(relation_id)
            if relation is None:
                problems.append(f"{case_id}: 없는 관계 ID {relation_id}")
            elif not relation.get("retrieval_eligible"):
                problems.append(f"{case_id}: 검색 대상이 아닌 관계 ID {relation_id}")
        for evidence_id in row.get("expected_evidence_ids") or []:
            if evidence_id not in evidence_ids:
                problems.append(f"{case_id}: 없는 근거 ID {evidence_id}")
        for chunk_id in row.get("expected_source_chunk_ids") or []:
            if chunk_id not in chunk_ids:
                problems.append(f"{case_id}: 없는 청크 ID {chunk_id}")

        if row.get("query_strategy") not in GRAPH_ROUTES:
            continue
        nodes = row.get("expected_entity_ids") or []
        relation_ids = row.get("expected_relation_ids") or []
        if len(nodes) != len(relation_ids) + 1:
            problems.append(
                f"{case_id}: 노드 수({len(nodes)})가 hop 수 + 1({len(relation_ids) + 1})이 아닙니다."
            )
            continue
        current = nodes[0]
        path = [current]
        for relation_id in relation_ids:
            relation = relations.get(relation_id)
            if relation is None or relation["from_id"] != current:
                problems.append(f"{case_id}: 관계 {relation_id} 에서 체인이 끊깁니다.")
                break
            current = relation["to_id"]
            path.append(current)
        else:
            if path != nodes:
                problems.append(f"{case_id}: 체인 노드 순서가 expected_entity_ids 와 다릅니다.")
    return problems


def regenerate_from_reviewed(team_dir: str | Path) -> list[dict[str, Any]]:
    """통합 노트북과 같은 방식으로 최종 골든셋을 다시 만듭니다."""
    directory = Path(team_dir)
    files = [
        path
        for path in sorted(directory.glob(REVIEWED_PATTERN))
        if path.name not in EXCLUDED_REVIEWED_FILES
    ]
    if not files:
        raise FileNotFoundError(f"{directory}에 {REVIEWED_PATTERN} 파일이 없습니다.")

    approved: list[dict[str, Any]] = []
    for path in files:
        for row in read_jsonl(path):
            status = row.get("status")
            if status not in {"READY", "REJECTED"}:
                raise ValueError(
                    f"검수가 끝나지 않은 문항이 있습니다: {path.name}:{row.get('case_id')}"
                )
            if status == "READY":
                approved.append(row)

    final: list[dict[str, Any]] = []
    for number, candidate in enumerate(approved, start=1):
        source = copy.deepcopy(candidate)
        source["reference"] = make_reference(source)
        built = {key: source[key] for key in FINAL_KEYS if key in source}
        built["case_id"] = f"GS{number:03d}"
        built["status"] = "READY"
        final.append(built)
    return final


def check_source_drift(
    rows: list[dict[str, Any]], team_dir: str | Path
) -> list[str]:
    """검수 원본으로 재생성한 결과와 다른 점을 찾습니다."""
    problems: list[str] = []
    regenerated = regenerate_from_reviewed(team_dir)

    by_question_now = {row["question"].strip(): row for row in rows}
    by_question_regen = {row["question"].strip(): row for row in regenerated}

    for question in sorted(set(by_question_now) - set(by_question_regen)):
        problems.append(
            "골든셋에만 있는 질문입니다(검수 원본에 반영되지 않았습니다): "
            f"{by_question_now[question]['case_id']} {question[:50]}"
        )
    for question in sorted(set(by_question_regen) - set(by_question_now)):
        problems.append(
            "검수 원본에만 있는 질문입니다(골든셋에서 지워졌거나 아직 통합되지 않았습니다): "
            f"{by_question_regen[question]['case_id']} {question[:50]}"
        )

    for question in sorted(set(by_question_now) & set(by_question_regen)):
        current = by_question_now[question]
        rebuilt = by_question_regen[question]
        for key in sorted(set(current) | set(rebuilt)):
            if key == "case_id":
                continue
            if current.get(key) != rebuilt.get(key):
                problems.append(
                    f"{current['case_id']}: {key} 가 검수 원본과 다릅니다"
                    f"(재생성하면 바뀝니다). {question[:40]}"
                )
    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="골든셋 무결성·정합성 검사")
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--team-dir", type=Path, default=DEFAULT_TEAM_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--skip-processed",
        action="store_true",
        help="data/processed 대조를 건너뜁니다(대용량 파일을 읽지 않습니다).",
    )
    parser.add_argument(
        "--skip-drift", action="store_true", help="검수 원본과의 정합성 검사를 건너뜁니다."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_jsonl(args.golden_set)
    print(f"검사 대상: {args.golden_set}")
    print(f"문항 수: {len(rows)}")

    sections: list[tuple[str, list[str]]] = [("파일 규칙", check_rules(rows))]
    if not args.skip_processed:
        sections.append(
            ("참조 ID·체인", check_processed_references(rows, args.processed_dir))
        )
    if not args.skip_drift:
        sections.append(("검수 원본 정합성", check_source_drift(rows, args.team_dir)))

    total = 0
    for title, problems in sections:
        print()
        print(f"== {title} ==")
        if not problems:
            print("  문제 없음")
            continue
        total += len(problems)
        for problem in problems:
            print("  -", problem)

    print()
    if total:
        print(f"문제 {total}건을 찾았습니다.")
        return 1
    print("모든 검사를 통과했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
