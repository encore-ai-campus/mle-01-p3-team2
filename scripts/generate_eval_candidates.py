from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(r"C:\Users\Playdata\Desktop\mle-01-p3-team2")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation"

EVIDENCE_PATH = PROCESSED_DIR / "evidence.jsonl"
ENTITY_PATH = PROCESSED_DIR / "entities.jsonl"
RELATION_PATH = PROCESSED_DIR / "relations.jsonl"

# 수치 관측 자료만 지나치게 많이 뽑히지 않도록 평가 가치가 높은 유형을 지정합니다.
VECTOR_EVIDENCE_TYPES = {
    "crop_management_reextracted",
    "crop_management_human_verified",
    "disease_record",
    "insect_record",
    "variety_record",
    "variety_group_record",
    "fertilizer_standard_record",
    "official_environment_fact",
    "external_environment_fact",
    "pesticide_use_validated_record",
    "smartfarm_korea_sowing_observation",
}

GRAPH_RELATION_TYPES = {
    "HAS_MANAGEMENT_FACT",
    "HAS_ENVIRONMENT_REQUIREMENT",
    "AFFECTED_BY_DISEASE",
    "AFFECTED_BY_INSECT",
    "HAS_VARIETY",
    "HAS_VARIETY_GROUP",
    "HAS_FERTILIZER_STANDARD",
    "HAS_SOWING_OBSERVATION",
    "HAS_SOWING_RECOMMENDATION",
    "HAS_REGISTERED_PESTICIDE_USE",
    "CONTROLS_TARGET",
    "SUITABLE_FOR_REGION",
    "USES_VARIETY",
}

# 아래 문자열이 포함된 품질 플래그는 자동 후보에서 제외합니다.
BLOCKED_FLAG_WORDS = {
    "LOW_CONFIDENCE",
    "BROKEN",
    "CORRUPTED",
    "EXCLUDED",
    "EMPTY",
    "UNVERIFIED_OCR",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path.name} {line_number}번째 줄의 JSON이 올바르지 않습니다."
                ) from error


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def unique_list(values: Iterable[str | None]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def is_good_evidence(row: dict[str, Any]) -> bool:
    text = str(row.get("text", "")).strip()

    # 너무 짧으면 질문을 만들 정보가 부족하고,
    # 너무 길면 한 평가 문항에 여러 사실이 섞일 가능성이 큽니다.
    if not 60 <= len(text) <= 1_500:
        return False

    flags = {str(flag).upper() for flag in row.get("quality_flags", [])}

    return not any(
        blocked_word in flag
        for flag in flags
        for blocked_word in BLOCKED_FLAG_WORDS
    )


def load_data():
    evidence_by_id: dict[str, dict[str, Any]] = {}

    for row in read_jsonl(EVIDENCE_PATH):
        if is_good_evidence(row):
            evidence_by_id[row["evidence_id"]] = {
                "evidence_id": row["evidence_id"],
                "evidence_type": row["evidence_type"],
                "text": row["text"].strip(),
                "chunk_ids": row.get("chunk_ids", []),
                "document_id": row.get("document_id"),
                "source_id": row.get("source_id"),
                "locator": row.get("locator", {}),
                "quality_flags": row.get("quality_flags", []),
            }

    entity_by_id: dict[str, dict[str, str]] = {}

    for row in read_jsonl(ENTITY_PATH):
        entity_by_id[row["entity_id"]] = {
            "entity_id": row["entity_id"],
            "entity_type": row.get("entity_type", ""),
            "name": str(row.get("name", "")).strip(),
        }

    relations: list[dict[str, Any]] = []

    for row in read_jsonl(RELATION_PATH):
        if not row.get("retrieval_eligible", False):
            continue

        valid_evidence_ids = [
            evidence_id
            for evidence_id in row.get("evidence_ids", [])
            if evidence_id in evidence_by_id
        ]

        if not valid_evidence_ids:
            continue

        if row.get("from_id") not in entity_by_id:
            continue

        if row.get("to_id") not in entity_by_id:
            continue

        relations.append(
            {
                "relation_id": row["relation_id"],
                "relation_type": row["relation_type"],
                "from_id": row["from_id"],
                "to_id": row["to_id"],
                "evidence_ids": valid_evidence_ids,
                "source_chunk_ids": row.get("source_chunk_ids", []),
                "confidence": row.get("confidence"),
            }
        )

    return evidence_by_id, entity_by_id, relations


def balanced_take(
    grouped_items: dict[str, list[dict[str, Any]]],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """각 유형에서 돌아가며 하나씩 뽑아 특정 유형의 과대표집을 막습니다."""

    groups = {}

    for group_name, items in grouped_items.items():
        copied = list(items)
        rng.shuffle(copied)

        if copied:
            groups[group_name] = copied

    selected: list[dict[str, Any]] = []

    while groups and len(selected) < count:
        group_names = list(groups)
        rng.shuffle(group_names)

        for group_name in group_names:
            if len(selected) >= count:
                break

            items = groups[group_name]

            if items:
                selected.append(items.pop())

            if not items:
                del groups[group_name]

    return selected


def make_vector_candidates(
    evidence_by_id: dict[str, dict[str, Any]],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for evidence in evidence_by_id.values():
        evidence_type = evidence["evidence_type"]

        if evidence_type in VECTOR_EVIDENCE_TYPES:
            grouped[evidence_type].append(evidence)

    selected = balanced_take(grouped, count, rng)

    return [
        {
            "candidate_type": "single_evidence",
            "query_strategy": "vector",
            "evidence_ids": [evidence["evidence_id"]],
            "entity_ids": [],
            "relation_ids": [],
            "graph_facts": [],
        }
        for evidence in selected
    ]


def relation_to_fact(
    relation: dict[str, Any],
    entity_by_id: dict[str, dict[str, str]],
) -> str:
    from_entity = entity_by_id[relation["from_id"]]
    to_entity = entity_by_id[relation["to_id"]]

    return (
        f"{from_entity['name']}({from_entity['entity_type']}) "
        f"--{relation['relation_type']}--> "
        f"{to_entity['name']}({to_entity['entity_type']})"
    )


def make_graph_candidates(
    relations: list[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for relation in relations:
        relation_type = relation["relation_type"]

        if relation_type in GRAPH_RELATION_TYPES:
            grouped[relation_type].append(relation)

    selected = balanced_take(grouped, count, rng)
    candidates = []

    for relation in selected:
        candidates.append(
            {
                "candidate_type": "single_relation",
                "query_strategy": "graph",
                "evidence_ids": relation["evidence_ids"],
                "entity_ids": [relation["from_id"], relation["to_id"]],
                "relation_ids": [relation["relation_id"]],
                "graph_facts": [relation],
            }
        )

    return candidates


def make_hybrid_candidates(
    relations: list[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    A → B → C 형태의 2-hop 경로를 우선 선택합니다.

    예:
    작물 → 등록 농약 → 방제 대상
    """

    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for relation in relations:
        if relation["relation_type"] in GRAPH_RELATION_TYPES:
            outgoing[relation["from_id"]].append(relation)

    path_candidates = []

    for first_relation in relations:
        if first_relation["relation_type"] not in GRAPH_RELATION_TYPES:
            continue

        next_relations = outgoing.get(first_relation["to_id"], [])
        shuffled_next = list(next_relations)
        rng.shuffle(shuffled_next)

        # 한 관계에서 지나치게 많은 경로가 생기는 것을 제한합니다.
        for second_relation in shuffled_next[:3]:
            node_ids = {
                first_relation["from_id"],
                first_relation["to_id"],
                second_relation["to_id"],
            }

            if len(node_ids) < 3:
                continue

            evidence_ids = unique_list(
                first_relation["evidence_ids"] + second_relation["evidence_ids"]
            )

            path_candidates.append(
                {
                    "candidate_type": "two_hop_path",
                    "query_strategy": "hybrid",
                    "evidence_ids": evidence_ids,
                    "entity_ids": [
                        first_relation["from_id"],
                        first_relation["to_id"],
                        second_relation["to_id"],
                    ],
                    "relation_ids": [
                        first_relation["relation_id"],
                        second_relation["relation_id"],
                    ],
                    "graph_facts": [first_relation, second_relation],
                }
            )

    rng.shuffle(path_candidates)
    selected = path_candidates[:count]

    # 2-hop 경로가 부족하면 같은 작물에 연결된 서로 다른 사실 두 개를 사용합니다.
    if len(selected) < count:
        by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for relation in relations:
            if relation["relation_type"] in GRAPH_RELATION_TYPES:
                by_anchor[relation["from_id"]].append(relation)

        fallback_candidates = []

        for anchor_id, anchor_relations in by_anchor.items():
            if len(anchor_relations) < 2:
                continue

            for _ in range(min(10, len(anchor_relations))):
                first_relation, second_relation = rng.sample(anchor_relations, 2)

                if first_relation["relation_type"] == second_relation["relation_type"]:
                    continue

                evidence_ids = unique_list(
                    first_relation["evidence_ids"] + second_relation["evidence_ids"]
                )

                if len(evidence_ids) < 2:
                    continue

                fallback_candidates.append(
                    {
                        "candidate_type": "shared_anchor_comparison",
                        "query_strategy": "hybrid",
                        "evidence_ids": evidence_ids,
                        "entity_ids": unique_list(
                            [
                                anchor_id,
                                first_relation["to_id"],
                                second_relation["to_id"],
                            ]
                        ),
                        "relation_ids": [
                            first_relation["relation_id"],
                            second_relation["relation_id"],
                        ],
                        "graph_facts": [first_relation, second_relation],
                    }
                )

        rng.shuffle(fallback_candidates)
        selected.extend(fallback_candidates[: count - len(selected)])

    return selected


def format_candidate_context(
    candidate: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
    entity_by_id: dict[str, dict[str, str]],
) -> str:
    evidence_lines = []

    for evidence_id in candidate["evidence_ids"]:
        evidence = evidence_by_id[evidence_id]
        evidence_lines.append(
            f"[{evidence_id} / {evidence['evidence_type']}]\n"
            f"{evidence['text']}"
        )

    graph_lines = [
        relation_to_fact(relation, entity_by_id)
        for relation in candidate["graph_facts"]
    ]

    graph_section = "\n".join(f"- {line}" for line in graph_lines)
    evidence_section = "\n\n".join(evidence_lines)

    return (
        f"후보 유형: {candidate['candidate_type']}\n"
        f"검색 전략: {candidate['query_strategy']}\n\n"
        f"그래프 사실:\n{graph_section or '- 없음'}\n\n"
        f"근거 원문:\n{evidence_section}"
    )


def parse_json_content(content: str) -> dict[str, Any]:
    content = content.strip()

    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```")
        content = content.removesuffix("```").strip()

    return json.loads(content)


def call_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    last_error: Exception | None = None

    for retry_number in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("모델이 빈 응답을 반환했습니다.")

            return parse_json_content(content)

        except Exception as error:
            last_error = error

            if retry_number + 1 < max_retries:
                time.sleep(2 ** retry_number)

    raise RuntimeError(f"LLM 호출에 실패했습니다: {last_error}") from last_error


GENERATOR_SYSTEM_PROMPT = """
당신은 농작물 재배 GraphRAG 평가셋을 만드는 데이터 작성자입니다.

제공된 근거와 그래프 관계만 사용하여 실제 농업인이 물어볼 법한
자연스러운 한국어 질문 1개를 만드세요.

규칙:
1. 제공된 근거만으로 답할 수 있어야 합니다.
2. 질문에 정답을 직접 노출하지 마세요.
3. evidence ID, entity ID, relation ID를 질문에 쓰지 마세요.
4. expected_facts는 답변에 반드시 들어가야 할 원자적 사실 목록입니다.
5. 멀티홉 후보라면 두 관계 또는 두 근거를 모두 사용해야 답할 수 있게 만드세요.
6. 농약, 안전 또는 불확실한 수치가 관련되면 경고가 필요한지 판단하세요.
7. 근거에 없는 내용을 추측하거나 일반 상식으로 보충하지 마세요.

다음 JSON 형식으로만 답하세요.
{
  "question": "질문",
  "expected_facts": ["핵심 사실 1", "핵심 사실 2"],
  "expected_behavior": "ANSWER 또는 ANSWER_WITH_WARNING",
  "safety_assertions": ["필요한 안전 조건"],
  "question_type": "fact 또는 relation 또는 multi_hop 또는 comparison",
  "difficulty": "easy 또는 medium 또는 hard"
}
""".strip()

CRITIC_SYSTEM_PROMPT = """
당신은 RAG 평가셋 검수자입니다.

제공된 질문이 주어진 근거만으로 답할 수 있는지 엄격하게 평가하세요.
다음 조건을 모두 확인하세요.

- 질문이 자연스럽고 의미가 명확한가
- 근거만으로 정답을 확정할 수 있는가
- 기대 사실이 모두 근거에 존재하는가
- 질문이 정답을 직접 노출하지 않는가
- 멀티홉 문항이면 여러 근거나 관계를 실제로 사용해야 하는가
- 농약이나 안전 관련 문항의 경고 조건이 적절한가

다음 JSON 형식으로만 답하세요.
{
  "answerable": true,
  "grounded": true,
  "natural": true,
  "not_leaky": true,
  "uses_required_evidence": true,
  "score": 0.0,
  "reason": "판정 이유"
}

score는 0에서 1 사이입니다.
""".strip()


def to_bool(value: Any) -> bool:
    if value is True:
        return True

    if isinstance(value, str):
        return value.strip().lower() == "true"

    return False


def critic_passed(result: dict[str, Any], threshold: float) -> bool:
    required_checks = [
        "answerable",
        "grounded",
        "natural",
        "not_leaky",
        "uses_required_evidence",
    ]

    checks_passed = all(to_bool(result.get(key)) for key in required_checks)
    score = float(result.get("score", 0))

    return checks_passed and score >= threshold


def generate_and_review(
    candidate: dict[str, Any],
    client: OpenAI,
    generation_model: str,
    critic_model: str,
    evidence_by_id: dict[str, dict[str, Any]],
    entity_by_id: dict[str, dict[str, str]],
    quality_threshold: float,
    max_quality_retries: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    context = format_candidate_context(
        candidate,
        evidence_by_id,
        entity_by_id,
    )
    previous_feedback = ""

    for _ in range(max_quality_retries + 1):
        generator_prompt = (
            f"{context}\n\n"
            f"이전 검수 의견:\n{previous_feedback or '없음'}"
        )

        generated = call_json(
            client=client,
            model=generation_model,
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            user_prompt=generator_prompt,
        )

        critic_prompt = (
            f"{context}\n\n"
            f"생성 결과:\n"
            f"{json.dumps(generated, ensure_ascii=False, indent=2)}"
        )

        critic_result = call_json(
            client=client,
            model=critic_model,
            system_prompt=CRITIC_SYSTEM_PROMPT,
            user_prompt=critic_prompt,
        )

        if critic_passed(critic_result, quality_threshold):
            return generated, critic_result

        previous_feedback = str(
            critic_result.get("reason", "검수 기준을 통과하지 못했습니다.")
        )

    return None, critic_result


def build_review_row(
    case_number: int,
    candidate: dict[str, Any],
    generated: dict[str, Any],
    critic_result: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    evidence_rows = [
        evidence_by_id[evidence_id]
        for evidence_id in candidate["evidence_ids"]
    ]

    expected_behavior = generated.get("expected_behavior", "ANSWER")

    if expected_behavior not in {"ANSWER", "ANSWER_WITH_WARNING"}:
        expected_behavior = "ANSWER"

    # 농약 근거는 실제 사용 전 공식 등록사항 확인이 필요하므로 경고형으로 고정합니다.
    if any(
        row["evidence_type"] == "pesticide_use_validated_record"
        for row in evidence_rows
    ):
        expected_behavior = "ANSWER_WITH_WARNING"

    return {
        "case_id": f"CAND_{case_number:04d}",
        "question": generated["question"].strip(),
        "question_type": generated.get("question_type", "fact"),
        "difficulty": generated.get("difficulty", "medium"),
        "query_strategy": candidate["query_strategy"],
        "expected_behavior": expected_behavior,
        "expected_facts": unique_list(generated.get("expected_facts", [])),
        "expected_entity_ids": unique_list(candidate["entity_ids"]),
        "expected_relation_ids": unique_list(candidate["relation_ids"]),
        "expected_evidence_ids": unique_list(candidate["evidence_ids"]),
        "expected_source_chunk_ids": unique_list(
            chunk_id
            for row in evidence_rows
            for chunk_id in row.get("chunk_ids", [])
        ),
        "safety_assertions": unique_list(
            generated.get("safety_assertions", [])
        ),
        "status": "PENDING_REVIEW",
        "review": {
            "question_ok": None,
            "facts_ok": None,
            "evidence_ok": None,
            "ids_ok": None,
            "reviewer_note": "",
        },
        "generation_metadata": {
            "candidate_type": candidate["candidate_type"],
            "critic_score": float(critic_result.get("score", 0)),
            "critic_reason": critic_result.get("reason", ""),
            "document_ids": unique_list(
                row.get("document_id") for row in evidence_rows
            ),
            "source_ids": unique_list(
                row.get("source_id") for row in evidence_rows
            ),
            "evidence_types": unique_list(
                row.get("evidence_type") for row in evidence_rows
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector", type=int, default=20)
    parser.add_argument("--graph", type=int, default=15)
    parser.add_argument("--hybrid", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality-threshold", type=float, default=0.8)
    parser.add_argument("--max-quality-retries", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "candidates_pending.jsonl",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    generation_model = os.getenv("OPENAI_CHAT_MODEL")
    critic_model = os.getenv("OPENAI_EVAL_MODEL")

    if not generation_model or not critic_model:
        raise ValueError(
            ".env에 OPENAI_CHAT_MODEL과 OPENAI_EVAL_MODEL을 지정해야 합니다."
        )

    rng = random.Random(args.seed)
    client = OpenAI()

    evidence_by_id, entity_by_id, relations = load_data()

    candidates = []
    candidates.extend(
        make_vector_candidates(evidence_by_id, args.vector, rng)
    )
    candidates.extend(
        make_graph_candidates(relations, args.graph, rng)
    )
    candidates.extend(
        make_hybrid_candidates(relations, args.hybrid, rng)
    )
    rng.shuffle(candidates)

    review_rows = []
    failed_candidates = []

    for index, candidate in enumerate(candidates, start=1):
        print(
            f"[{index}/{len(candidates)}] "
            f"{candidate['candidate_type']} 생성 중"
        )

        generated, critic_result = generate_and_review(
            candidate=candidate,
            client=client,
            generation_model=generation_model,
            critic_model=critic_model,
            evidence_by_id=evidence_by_id,
            entity_by_id=entity_by_id,
            quality_threshold=args.quality_threshold,
            max_quality_retries=args.max_quality_retries,
        )

        if generated is None:
            failed_candidates.append(
                {
                    "candidate": candidate,
                    "critic_result": critic_result,
                }
            )
            continue

        review_rows.append(
            build_review_row(
                case_number=len(review_rows) + 1,
                candidate=candidate,
                generated=generated,
                critic_result=critic_result,
                evidence_by_id=evidence_by_id,
            )
        )

    write_jsonl(args.output, review_rows)

    if failed_candidates:
        write_jsonl(
            args.output.with_name("candidates_failed.jsonl"),
            failed_candidates,
        )

    print()
    print(f"생성 후보: {len(candidates)}")
    print(f"검수 대기: {len(review_rows)}")
    print(f"자동 검수 탈락: {len(failed_candidates)}")
    print(f"저장 위치: {args.output}")


if __name__ == "__main__":
    main()