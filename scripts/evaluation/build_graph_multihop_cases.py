"""그래프 경로 단독 multi-hop 평가 문항 후보를 만듭니다.

왜 필요한가
기존 골든셋의 graph 21문항은 전부 1-hop 이고, 2-hop 은 hybrid 17문항뿐이며 그
17문항은 모두 `HAS_REGISTERED_PESTICIDE_USE → CONTROLS_TARGET` 한 가지
패턴입니다. 그래서 그래프를 2단 이상 타고 들어가는 경로를 측정할 문항이
없습니다. 이 스크립트는 `data/processed` 의 실제 그래프에서 체인을 찾아
검수용 후보 파일을 만듭니다.

ID 를 코드에 박지 않고 이름으로 그래프를 조회해 채우며, 체인이 실제로
이어지는지·근거와 청크가 있는지 모두 검증한 뒤에 저장합니다.

실행:
    uv run python -m scripts.evaluation.build_graph_multihop_cases
    uv run python -m scripts.evaluation.build_graph_multihop_cases --check
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TEAM_DIR = PROJECT_ROOT / "data" / "evaluation" / "team_candidates"
GOLDEN_PATH = PROJECT_ROOT / "data" / "evaluation" / "golden_set_final.jsonl"

# 통합 노트북은 reviewed_*.jsonl 만 읽습니다. 이 파일은 자동 생성 후보이므로
# candidates_ 로 두어 팀 검수를 거치기 전에는 골든셋에 섞이지 않게 합니다.
# 검수가 끝나면 reviewed_<검수자>.jsonl 로 옮기면 통합 대상이 됩니다.
DEFAULT_OUTPUT = TEAM_DIR / "candidates_graph_multihop.jsonl"

CANDIDATE_OWNER = "graph_multihop"


@dataclass(frozen=True)
class ChainSpec:
    """이름으로 지정한 2-hop 체인과 문항 내용입니다."""

    case_id: str
    crop: str
    middle: str
    tail: str
    relation_types: tuple[str, str]
    question: str
    facts: tuple[str, ...]
    question_type: str = "multi_hop"
    difficulty: str = "medium"
    expected_behavior: str = "ANSWER"
    safety_assertions: tuple[str, ...] = ()
    note: str = ""


# 패턴별로 고르게 뽑되, 기존 69문항과 질문이 겹치지 않도록 다른 품종·날짜를 씁니다.
CHAIN_SPECS: tuple[ChainSpec, ...] = (
    # 1) Crop -HAS_VARIETY-> Variety -SUITABLE_FOR_REGION-> Region
    ChainSpec(
        case_id="GMH_001",
        crop="마늘",
        middle="의성종",
        tail="경북 의성 및 영남 내륙",
        relation_types=("HAS_VARIETY", "SUITABLE_FOR_REGION"),
        question="마늘 재배지가 경북 의성과 영남 내륙일 때 적합한 재래종 품종은 무엇이며, 그 품종의 재배형과 저장성은 어떤가요?",
        facts=(
            "경북 의성 및 영남 내륙에 적합한 마늘 품종은 의성종이다",
            "의성종은 한지형 마늘이다",
            "의성종은 저장 중 열구 현상이 적고 껍질과 뿌리 부착력이 강해 저장성이 탁월하다",
        ),
    ),
    ChainSpec(
        case_id="GMH_002",
        crop="마늘",
        middle="서산종",
        tail="중부 해안",
        relation_types=("HAS_VARIETY", "SUITABLE_FOR_REGION"),
        question="중부 해안 지역에서 재배하기에 적응된 마늘 품종은 무엇이며, 그 품종의 추대 유형과 쪽수는 어떻게 되나요?",
        facts=(
            "중부 해안 재배에 적응한 마늘 품종은 서산종이다",
            "서산종은 완전추대형 재래종이다",
            "서산종의 쪽수는 6~8개이다",
        ),
    ),
    ChainSpec(
        case_id="GMH_003",
        crop="마늘",
        middle="단양종",
        tail="중부 내륙",
        relation_types=("HAS_VARIETY", "SUITABLE_FOR_REGION"),
        question="중부 내륙에 적응한 마늘 재래종 품종은 무엇이며, 그 품종의 파종기와 수확기는 언제인가요?",
        facts=(
            "중부 내륙 재배에 적응한 마늘 품종은 단양종이다",
            "단양종의 파종기는 10월 중하순이다",
            "단양종의 수확기는 6월 하순에서 7월 상순이다",
        ),
    ),
    ChainSpec(
        case_id="GMH_004",
        crop="상추",
        middle="치마상추 / 적상추",
        tail="전국",
        relation_types=("HAS_VARIETY", "SUITABLE_FOR_REGION"),
        question="전국에서 재배할 수 있는 상추 품종군은 무엇이며, 고온기에 어떤 문제가 생겨 어떤 관리가 필요한가요?",
        facts=(
            "전국에서 재배 가능한 상추 품종군은 치마상추 / 적상추이다",
            "치마상추 / 적상추는 고온기에 추대 현상과 쓴맛이 발생하기 쉽다",
            "고온기에는 차광이 필요하다",
        ),
        note=(
            "근거 두 건이 '차광 및 수온 관리'와 '차광 및 수분 관리'로 엇갈려 공통된 "
            "'차광'만 기대 사실로 넣었습니다. 원문 확인 후 수온/수분을 확정해 주세요."
        ),
    ),
    # 2) Crop -HAS_VARIETY-> Variety -HAS_SOWING_RECOMMENDATION-> SowingRecommendation
    ChainSpec(
        case_id="GMH_005",
        crop="마늘",
        middle="남도마늘",
        tail="남도마늘 파종 권고",
        relation_types=("HAS_VARIETY", "HAS_SOWING_RECOMMENDATION"),
        question="남부지방과 제주도에 적합한 난지형 마늘 품종의 권장 파종 시기는 언제이며, 파종 시 무엇을 지켜야 하나요?",
        facts=(
            "해당 난지형 마늘 품종은 남도마늘이다",
            "남도마늘의 권장 파종 시기는 9월 하순에서 10월 중순이다",
            "지온이 너무 내려가기 전에 파종을 완료해야 한다",
        ),
    ),
    ChainSpec(
        case_id="GMH_006",
        crop="배추",
        middle="가을배추 (김장배추)",
        tail="가을배추 (김장배추) 파종 권고",
        relation_types=("HAS_VARIETY", "HAS_SOWING_RECOMMENDATION"),
        question="김장용으로 노지에서 재배하는 배추 품종군의 권장 파종 시기와 정식 시기는 각각 언제인가요?",
        facts=(
            "해당 배추 품종군은 가을배추 (김장배추)이다",
            "권장 파종 시기는 8월 중순에서 8월 하순이다",
            "정식 시기는 9월 상순이며 약 30~35일간 육묘한 뒤 본밭에 정식한다",
        ),
    ),
    ChainSpec(
        case_id="GMH_007",
        crop="양파",
        middle="중만생종 양파",
        tail="중만생종 양파 파종 권고",
        relation_types=("HAS_VARIETY", "HAS_SOWING_RECOMMENDATION"),
        question="중부 및 남부 내륙 전역에서 재배하는 양파 품종군의 권장 파종 시기와 정식 시기는 언제인가요?",
        facts=(
            "해당 양파 품종군은 중만생종 양파이다",
            "권장 파종 시기는 9월 중순에서 9월 하순이다",
            "정식 시기는 10월 하순에서 11월 상순이다",
        ),
    ),
    # 3) Crop -HAS_SOWING_OBSERVATION-> SowingObservation -USES_VARIETY-> Variety
    ChainSpec(
        case_id="GMH_008",
        crop="고추",
        middle="고추 칼탄플러스 20231228 경상남도 진주시",
        tail="칼탄플러스",
        relation_types=("HAS_SOWING_OBSERVATION", "USES_VARIETY"),
        question="고추 파종 실측 기록 중 2023년 12월 28일 경상남도 진주시 기록에는 어떤 품종이 사용되었나요?",
        facts=(
            "2023년 12월 28일 경상남도 진주시에서 파종된 고추 품종은 칼탄플러스이다",
            "이 내용은 익명 농가·육묘장의 실측 기록이며 파종 권장 기준은 아니다",
        ),
    ),
    ChainSpec(
        case_id="GMH_009",
        crop="상추",
        middle="상추 다풍적축면 20231211 경상남도 밀양시",
        tail="다풍적축면",
        relation_types=("HAS_SOWING_OBSERVATION", "USES_VARIETY"),
        question="상추 파종 실측 기록 중 2023년 12월 11일 경상남도 밀양시 기록에 사용된 품종은 무엇인가요?",
        facts=(
            "2023년 12월 11일 경상남도 밀양시에서 파종된 상추 품종은 다풍적축면이다",
            "이 내용은 익명 농가·육묘장의 실측 기록이며 파종 권장 기준은 아니다",
        ),
    ),
    # 4) Crop -HAS_SOWING_OBSERVATION-> SowingObservation -OBSERVED_IN_REGION-> Region
    ChainSpec(
        case_id="GMH_010",
        crop="고추",
        middle="고추 신록풋 20231215 경상남도 밀양시",
        tail="경상남도 밀양시",
        relation_types=("HAS_SOWING_OBSERVATION", "OBSERVED_IN_REGION"),
        question="고추 품종 신록풋을 2023년 12월 15일에 파종한 실측 기록은 어느 지역의 농가·육묘장에서 작성되었나요?",
        facts=(
            "해당 파종 실측 기록의 지역은 경상남도 밀양시이다",
            "이 내용은 익명 농가·육묘장의 실측 기록이며 파종 권장 기준은 아니다",
        ),
    ),
)


@dataclass
class Graph:
    entities_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    relations: list[dict[str, Any]] = field(default_factory=list)
    out_edges: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def entity_id(self, name: str, *, entity_type: str | None = None) -> str:
        matches = [
            entity
            for entity in self.entities_by_id.values()
            if entity.get("name") == name
            and (entity_type is None or entity.get("entity_type") == entity_type)
        ]
        if not matches:
            raise LookupError(f"엔티티를 찾지 못했습니다: {name!r} ({entity_type})")
        if len({entity["entity_id"] for entity in matches}) > 1:
            raise LookupError(
                f"이름이 중복된 엔티티입니다: {name!r} -> "
                f"{[entity['entity_id'] for entity in matches]}"
            )
        return matches[0]["entity_id"]

    def relation(self, from_id: str, to_id: str, relation_type: str) -> dict[str, Any]:
        matches = [
            relation
            for relation in self.out_edges.get(from_id, [])
            if relation["to_id"] == to_id and relation["relation_type"] == relation_type
        ]
        if not matches:
            raise LookupError(f"관계를 찾지 못했습니다: {from_id} -{relation_type}-> {to_id}")
        return matches[0]

    def step(self, from_id: str, relation_type: str, to_name: str) -> dict[str, Any]:
        """출발 노드에서 관계 타입을 따라가 이름이 맞는 엣지를 고릅니다.

        같은 이름의 노드가 여럿 있을 때(예: 관측용 Region 과 기상용 Region)
        전역 이름 조회로는 가릴 수 없으므로 체인을 따라가며 찾습니다.
        """
        matches = [
            relation
            for relation in self.out_edges.get(from_id, [])
            if relation["relation_type"] == relation_type
            and (self.entities_by_id.get(relation["to_id"]) or {}).get("name") == to_name
        ]
        if not matches:
            raise LookupError(
                f"체인을 찾지 못했습니다: {from_id} -{relation_type}-> {to_name!r}"
            )
        if len({relation["to_id"] for relation in matches}) > 1:
            raise LookupError(
                f"체인이 모호합니다: {from_id} -{relation_type}-> {to_name!r} -> "
                f"{sorted({relation['to_id'] for relation in matches})}"
            )
        return matches[0]


def load_graph(processed_dir: str | Path = PROCESSED_DIR) -> Graph:
    """검색 대상(retrieval_eligible) 관계만 담은 그래프를 읽습니다."""
    directory = Path(processed_dir)
    graph = Graph()
    with (directory / "entities.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                entity = json.loads(line)
                graph.entities_by_id[entity["entity_id"]] = entity
    with (directory / "relations.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            relation = json.loads(line)
            if not relation.get("retrieval_eligible"):
                continue
            graph.relations.append(relation)
            graph.out_edges.setdefault(relation["from_id"], []).append(relation)
    return graph


def build_case(spec: ChainSpec, graph: Graph) -> dict[str, Any]:
    """체인 지정을 검수용 평가 문항 한 줄로 바꿉니다."""
    crop_id = graph.entity_id(spec.crop, entity_type="Crop")
    first = graph.step(crop_id, spec.relation_types[0], spec.middle)
    middle_id = first["to_id"]
    second = graph.step(middle_id, spec.relation_types[1], spec.tail)
    tail_id = second["to_id"]

    first_evidence = list(first.get("evidence_ids") or [])
    second_evidence = list(second.get("evidence_ids") or [])
    shared = [item for item in first_evidence if item in set(second_evidence)]
    evidence_ids = shared or sorted(set(first_evidence) | set(second_evidence))
    chunk_ids = sorted(
        set(first.get("source_chunk_ids") or []) | set(second.get("source_chunk_ids") or [])
    )
    if not evidence_ids:
        raise ValueError(f"{spec.case_id}: 근거 ID 가 없습니다.")
    if not chunk_ids:
        raise ValueError(f"{spec.case_id}: 청크 ID 가 없습니다.")

    return {
        "case_id": spec.case_id,
        "question": spec.question,
        "question_type": spec.question_type,
        "difficulty": spec.difficulty,
        "query_strategy": "graph",
        "expected_behavior": spec.expected_behavior,
        "expected_facts": list(spec.facts),
        "expected_entity_ids": [crop_id, middle_id, tail_id],
        "expected_relation_ids": [first["relation_id"], second["relation_id"]],
        "expected_evidence_ids": evidence_ids,
        "expected_source_chunk_ids": chunk_ids,
        "safety_assertions": list(spec.safety_assertions),
        "status": "READY",
        "review": {
            "question_ok": True,
            "facts_ok": True,
            "evidence_ok": True,
            "ids_ok": True,
            "reviewer_note": spec.note
            or "그래프 체인에서 자동 구성한 뒤 ID·근거 연결을 검증했습니다. 질문 표현은 팀 검수 필요.",
        },
        "normalization_warnings": [],
        "candidate_owner": CANDIDATE_OWNER,
        "original_case_id": spec.case_id,
        "generation_metadata": {
            "candidate_type": "graph_multihop_chain",
            "expected_hop": 2,
            "relation_types": list(spec.relation_types),
            "chain": [spec.crop, spec.middle, spec.tail],
        },
        "reference": "",
    }


def verify_cases(
    cases: list[dict[str, Any]], graph: Graph, *, golden_path: str | Path = GOLDEN_PATH
) -> list[str]:
    """저장 전 검증입니다. 문제를 모두 모아 문자열 목록으로 돌려줍니다."""
    problems: list[str] = []
    existing_questions: set[str] = set()
    golden = Path(golden_path)
    if golden.exists():
        with golden.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    existing_questions.add(json.loads(line)["question"].strip())

    seen_questions: set[str] = set()
    for case in cases:
        case_id = case["case_id"]
        question = case["question"].strip()
        if question in existing_questions:
            problems.append(f"{case_id}: 기존 골든셋과 질문이 중복됩니다.")
        if question in seen_questions:
            problems.append(f"{case_id}: 후보 안에서 질문이 중복됩니다.")
        seen_questions.add(question)

        entity_ids = case["expected_entity_ids"]
        relation_ids = case["expected_relation_ids"]
        if len(entity_ids) != len(relation_ids) + 1:
            problems.append(f"{case_id}: 노드 수가 hop 수 + 1 이 아닙니다.")
        if len(relation_ids) != 2:
            problems.append(f"{case_id}: 2-hop 문항이 아닙니다.")

        for entity_id in entity_ids:
            if entity_id not in graph.entities_by_id:
                problems.append(f"{case_id}: 없는 엔티티 ID {entity_id}")

        relations_by_id = {relation["relation_id"]: relation for relation in graph.relations}
        chain_nodes = [entity_ids[0]]
        for relation_id in relation_ids:
            relation = relations_by_id.get(relation_id)
            if relation is None:
                problems.append(f"{case_id}: 없는(또는 검색 제외된) 관계 ID {relation_id}")
                break
            if relation["from_id"] != chain_nodes[-1]:
                problems.append(f"{case_id}: 관계 {relation_id} 가 체인에서 끊깁니다.")
                break
            chain_nodes.append(relation["to_id"])
        else:
            if chain_nodes != entity_ids:
                problems.append(f"{case_id}: 체인 노드 순서가 expected_entity_ids 와 다릅니다.")

        if not case["expected_facts"]:
            problems.append(f"{case_id}: 기대 사실이 비어 있습니다.")
        if not case["expected_evidence_ids"]:
            problems.append(f"{case_id}: 근거 ID 가 비어 있습니다.")
        if case["query_strategy"] != "graph" or case["expected_behavior"] == "REJECT":
            problems.append(f"{case_id}: graph 경로 문항이 아닙니다.")
    return problems


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="그래프 단독 multi-hop 후보 생성")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--golden-set", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check", action="store_true", help="검증만 하고 파일을 쓰지 않습니다."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    graph = load_graph(args.processed_dir)
    cases = [build_case(spec, graph) for spec in CHAIN_SPECS]
    problems = verify_cases(cases, graph, golden_path=args.golden_set)
    if problems:
        print("검증 실패:")
        for problem in problems:
            print(" -", problem)
        return 1

    print(f"검증 통과: {len(cases)}문항 (모두 graph 경로 2-hop)")
    for case in cases:
        chain = " → ".join(case["generation_metadata"]["chain"])
        print(f"  {case['case_id']}  {'·'.join(case['generation_metadata']['relation_types'])}")
        print(f"      {chain}")
    if args.check:
        print("\n--check 이므로 파일을 쓰지 않았습니다.")
        return 0

    out_path = write_jsonl(cases, args.output)
    print(f"\n저장: {out_path}")
    print("팀 검수 후 reviewed_<검수자>.jsonl 로 옮기면 통합 노트북이 골든셋에 반영합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
