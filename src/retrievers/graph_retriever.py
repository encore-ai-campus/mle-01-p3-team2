
"""농업 GraphRAG 파이프라인을 위한 Neo4j 그래프 검색기입니다."""

from __future__ import annotations

import os
import re
import time
import json
from collections.abc import Mapping, Sequence
from typing import Any

from neo4j import Driver, GraphDatabase

from .search_result import (
    GraphPathEdge,
    GraphPathNode,
    OriginalChunk,
    SearchResult,
)


_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_KOREAN_SUFFIXES = ("에서는", "으로는", "에게는", "으로", "에서", "에게", "까지", "부터", "에는", "에게", "께서", "은", "는", "이", "가", "을", "를", "의", "와", "과", "도", "로")
_ENTITY_STOPWORDS = {
    "어떤", "무엇", "무엇을", "어떻게", "어느", "언제", "알려", "알려주세요",
    "사용", "사용법", "기준", "방법", "정보", "질문", "대해", "대해서",
    "등록", "등록된", "등록되어", "약제", "농약", "방제", "방제하는",
    "수확", "전", "까지", "최대", "횟수", "희석", "배수", "시기", "시기는",
    "작물", "병해", "해충", "발생", "재배", "적합", "사용할", "가능한",
}

_RELATION_CUES: dict[str, tuple[str, ...]] = {
    "HAS_REGISTERED_PESTICIDE_USE": ("등록", "농약", "약제", "사용할 수", "등록되어"),
    "CONTROLS_TARGET": ("방제", "방제용", "방제하는", "대상 해충", "병해충"),
    "AFFECTED_BY_DISEASE": ("병해", "병에", "병이", "질병", "발병"),
    "AFFECTED_BY_INSECT": ("해충", "벌레", "충해", "가해"),
    "HAS_VARIETY": ("품종", "계통", "품종은", "재배하는"),
    "SUITABLE_FOR_REGION": ("지역", "적합", "재배하기", "어디에서"),
    "HAS_FERTILIZER_STANDARD": ("비료", "시비", "성분량", "시비량"),
    "HAS_SOWING_OBSERVATION": ("파종", "파종기", "발아", "출현"),
}

GraphSearchResult = SearchResult


def _check_identifier(value: str, kind: str) -> str:
    """Neo4j 라벨 및 관계 타입을 검증합니다."""
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid Neo4j {kind}: {value!r}")
    return value


def _as_list(value: Any) -> list[str]:
    """Neo4j 속성을 문자열 리스트로 정규화합니다."""
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]

    return [str(value)]


def _unique_list(values: Sequence[str]) -> list[str]:
    """입력 순서를 유지하면서 중복을 제거합니다."""
    return list(dict.fromkeys(values))


def _query_terms(question: str) -> list[str]:
    """Extract useful surface forms without requiring a Korean tokenizer."""
    raw = re.findall(r"[가-힣A-Za-z0-9]+", question.lower())
    terms: list[str] = []
    for token in raw:
        variants = [token]
        for suffix in _KOREAN_SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 2:
                variants.append(token[:-len(suffix)])
                break
        for value in variants:
            if len(value) >= 2 and value not in terms:
                terms.append(value)
    # Longer phrases carry more identifying information; keep common short words
    # available for crop matching, but cap parameter growth for broad questions.
    return sorted(terms, key=lambda value: (-len(value), value))[:32]


_NON_WORD = re.compile(r"[^가-힣a-z0-9]")


def _normalize(text: Any) -> str:
    return _NON_WORD.sub("", str(text or "").lower())


# 질문에 이름이 직접 등장할 때만 쓰는 식별 속성입니다. 작물명은 같은 작물의
# 후보들을 구별하지 못하므로 가중치를 낮게 둡니다.
_IDENTITY_NAME_FIELDS = (
    "product", "brand", "active_ingredient_english", "source_variety_name",
    "target_name", "formulation", "company",
)
_ALIAS_NAME_FIELDS = ("aliases", "synonyms", "_variants", "classification_paths")
_CROP_NAME_FIELDS = ("crop", "canonical_crop")
# 완전 일치 > 별칭 일치 > 부분 일치. 후보를 제거하지 않고 재랭킹 보너스로만 씁니다.
_NAME_MATCH_EXACT = 6.0
_NAME_MATCH_ALIAS = 4.0
_NAME_MATCH_PARTIAL = 2.0
_NAME_MATCH_CROP = 2.0

# 질문이 요구한 농약 사용 속성별 가중치. 존재하면 가산, 없으면 감산합니다.
_USE_FIELD_WEIGHTS = {
    "use_timing": 2.0,
    "use_count": 2.0,
    "dilution": 2.0,
    "application_method": 1.5,
    "registration_no": 1.0,
    "product": 1.0,
    "brand": 1.0,
}
_USE_FIELD_MISSING_RATIO = 0.75
_USE_FIELD_COMPLETE_BONUS = 2.0
_EMPTY_ATTRIBUTE_VALUES = {"", "-", "없음", "해당 없음", "none", "null"}

# 질문에 실제 값이 적힌 경우에만 비교합니다. Gold 값은 쓰지 않습니다.
_TIMING_PHRASES = (
    "월동 후", "월동후", "발생 초기", "발생초기", "발병 초기", "발병초기",
    "생육기", "정식 전", "정식전", "파종 전", "파종전", "수확 전", "수확전",
    "경엽 처리", "경엽처리", "토양 혼화", "토양혼화",
)
_CONDITION_MATCH_BONUS = 3.0
_CONDITION_MISMATCH_PENALTY = -1.0


def _attribute_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return str(value).strip().lower() not in _EMPTY_ATTRIBUTE_VALUES


def _attribute_values(attributes: Mapping[str, Any], fields: Sequence[str]) -> list[str]:
    values: list[str] = []
    for field in fields:
        value = attributes.get(field)
        if isinstance(value, (list, tuple, set)):
            values.extend(str(item) for item in value if item is not None)
        elif _attribute_is_present(value):
            values.append(str(value))
    return values


def _name_match_bonus(
    question_norm: str,
    entity_terms: Sequence[str],
    node_name: str,
    attributes: Mapping[str, Any],
) -> float:
    """제품·상표·유효성분·품종명이 질문에 직접 등장할 때만 주는 재랭킹 보너스."""
    identity = [node_name, *_attribute_values(attributes, _IDENTITY_NAME_FIELDS)]
    aliases = _attribute_values(attributes, _ALIAS_NAME_FIELDS)
    term_norms = [
        normalized for normalized in (_normalize(term) for term in entity_terms)
        if len(normalized) >= 3
    ]

    best = 0.0
    for value in identity:
        value_norm = _normalize(value)
        if len(value_norm) < 3:
            continue
        if value_norm in question_norm:
            best = max(best, _NAME_MATCH_EXACT)
        elif any(term in value_norm for term in term_norms):
            best = max(best, _NAME_MATCH_PARTIAL)
    if best < _NAME_MATCH_EXACT:
        for value in aliases:
            value_norm = _normalize(value)
            if len(value_norm) >= 3 and value_norm in question_norm:
                best = max(best, _NAME_MATCH_ALIAS)

    for value in _attribute_values(attributes, _CROP_NAME_FIELDS):
        value_norm = _normalize(value)
        if len(value_norm) >= 2 and value_norm in question_norm:
            best += _NAME_MATCH_CROP
            break
    return best


def _requested_condition_values(question: str) -> dict[str, Any]:
    """질문에 적힌 사용 조건 값만 뽑습니다(Gold 값 사용 금지)."""
    requested: dict[str, Any] = {}
    dilution = re.search(r"(\d{2,5})\s*배", question)
    if dilution:
        requested["dilution"] = int(dilution.group(1))
    harvest = re.search(r"수확\s*(\d+)\s*일\s*전", question)
    if harvest:
        requested["harvest_days"] = int(harvest.group(1))
    interval = re.search(r"(\d+)\s*일\s*간격", question)
    if interval:
        requested["interval_days"] = int(interval.group(1))
    count = re.search(r"(\d+)\s*회", question)
    if count:
        requested["use_count"] = int(count.group(1))
    phrases = [phrase for phrase in _TIMING_PHRASES if phrase in question]
    if phrases:
        requested["timing_phrases"] = phrases
    return requested


def _condition_value_score(
    requested: Mapping[str, Any], attributes: Mapping[str, Any]
) -> float:
    """질문에 적힌 값과 경로의 사용 기준 속성값을 비교합니다."""
    if not requested:
        return 0.0
    score = 0.0
    timing = _normalize(attributes.get("use_timing"))
    method = _normalize(attributes.get("application_method"))

    def numeric(value: Any) -> int | None:
        digits = re.search(r"\d+", str(value or ""))
        return int(digits.group(0)) if digits else None

    for field in ("dilution", "use_count"):
        if field not in requested:
            continue
        actual = numeric(attributes.get(field))
        if actual is None:
            continue
        score += (
            _CONDITION_MATCH_BONUS if actual == requested[field]
            else _CONDITION_MISMATCH_PENALTY
        )
    if "harvest_days" in requested and timing:
        wanted = _normalize(f"수확{requested['harvest_days']}일전")
        score += (
            _CONDITION_MATCH_BONUS if wanted in timing
            else _CONDITION_MISMATCH_PENALTY
        )
    if "interval_days" in requested and method:
        wanted = _normalize(f"{requested['interval_days']}일간격")
        score += (
            _CONDITION_MATCH_BONUS if wanted in method
            else _CONDITION_MISMATCH_PENALTY
        )
    # 표현이 다양해 불일치 감산 없이 일치만 가산합니다.
    for phrase in requested.get("timing_phrases", ()):
        phrase_norm = _normalize(phrase)
        if phrase_norm and (phrase_norm in timing or phrase_norm in method):
            score += _CONDITION_MATCH_BONUS
            break
    return score


def _result_relevance(
    result: SearchResult,
    question: str,
    terms: Sequence[str],
    node_attributes: Sequence[Any] = (),
) -> float:
    """Score a path using query entities, relation intents, and path integrity."""
    return _path_relevance_score(
        result.path_nodes or [], result.path_edges or [], question, node_attributes
    )


def _path_relevance_score(
    path_nodes: Sequence[Mapping[str, Any]],
    path_edges: Sequence[Mapping[str, Any]],
    question: str,
    node_attributes: Sequence[Any] = (),
) -> float:
    """Query-driven score shared by live search and fixed-pool evaluation."""
    q = question.lower()
    terms = _query_terms(question)
    entity_terms = [
        term for term in terms
        if term not in _ENTITY_STOPWORDS and len(term) >= 2
    ]
    normalized = _normalize
    question_norm = normalized(question)
    score = 0.0
    attribute_keys = {
        "aliases", "synonyms", "brand", "product", "target_name", "crop",
        "canonical_crop", "active_ingredient_english", "source_variety_name",
        "_variants", "classification_paths", "cultivation_type", "formulation",
        "company", "region", "growth_stage", "use_timing", "application_method",
    }
    entity_coverage: set[str] = set()
    parsed_attributes: list[Mapping[str, Any]] = []
    name_match_bonus = 0.0
    for index, node in enumerate(path_nodes):
        name = normalized(str(node.get("name") or ""))
        attr = node_attributes[index] if index < len(node_attributes) else {}
        if isinstance(attr, str):
            try:
                attr = json.loads(attr)
            except (TypeError, json.JSONDecodeError):
                attr = {}
        attr_values: list[str] = []
        if isinstance(attr, Mapping):
            for key in attribute_keys:
                value = attr.get(key)
                if isinstance(value, (list, tuple, set)):
                    attr_values.extend(str(item).lower() for item in value if item is not None)
                elif value is not None:
                    attr_values.append(str(value).lower())
        attr_text = " ".join(attr_values)
        attr_norm = normalized(attr_text)
        parsed_attributes.append(attr if isinstance(attr, Mapping) else {})
        if isinstance(attr, Mapping):
            name_match_bonus = max(
                name_match_bonus,
                _name_match_bonus(
                    question_norm, entity_terms, str(node.get("name") or ""), attr
                ),
            )
        for term in entity_terms:
            term_norm = normalized(term)
            if not term_norm:
                continue
            if term_norm in name:
                entity_coverage.add(term_norm)
            elif term_norm in attr_norm:
                entity_coverage.add(term_norm)
    # Reward unique, informative query entities once; this avoids inflating a
    # path merely because generic words occur in several attributes.
    for term in entity_coverage:
        score += min(len(term), 12) * (1.25 if len(term) >= 4 else 0.6)

    # 제품명·상표명·유효성분·품종명이 질문에 직접 등장하면 그 경로를 올립니다.
    # 경로 전체에서 가장 강한 일치 한 번만 반영해 노드 수가 많은 경로가
    # 유리해지지 않도록 합니다.
    score += name_match_bonus

    # When a question asks for application details, prefer paths whose linked
    # use record actually contains those requested fields. This uses retrieved
    # metadata only and does not assume any particular product or gold answer.
    field_cues = {
        "dilution": ("희석", "배수"),
        "use_count": ("횟수", "몇 회", "회까지"),
        "use_timing": ("수확", "사용 시기", "사용적기", "언제"),
        "application_method": ("간격", "살포", "발생 초기", "발병 초", "사용 방법", "사용법"),
        "brand": ("상표", "브랜드"),
        "registration_no": ("등록번호",),
        "product": ("품목명", "제품명"),
    }
    requested_fields = {
        field for field, cues in field_cues.items() if any(cue in q for cue in cues)
    }
    if not requested_fields and any(cue in q for cue in ("어떻게", "방법", "사용할 수")):
        requested_fields.update(("use_timing", "application_method", "use_count", "dilution"))
    requested_values = _requested_condition_values(question)
    for index, node in enumerate(path_nodes):
        if node.get("entity_type") != "PesticideUse" or index >= len(parsed_attributes):
            continue
        attributes = parsed_attributes[index]
        # 필드별로 가중치를 달리 두고, 요구한 필드를 모두 갖춘 기록에 완전성 가산.
        present_count = 0
        for field in requested_fields:
            weight = _USE_FIELD_WEIGHTS.get(field, 1.5)
            if _attribute_is_present(attributes.get(field)):
                score += weight
                present_count += 1
            else:
                score -= weight * _USE_FIELD_MISSING_RATIO
        if requested_fields and present_count == len(requested_fields):
            score += _USE_FIELD_COMPLETE_BONUS
        # 질문에 적힌 사용 조건 값(희석배수·횟수·간격·시기)과의 일치.
        score += _condition_value_score(requested_values, attributes)

    query_intents: list[tuple[int, str]] = []
    for relation_type, cues in _RELATION_CUES.items():
        positions = [q.find(cue) for cue in cues if q.find(cue) >= 0]
        if positions:
            query_intents.append((min(positions), relation_type))
    query_intents.sort()
    intent_types = [kind for _, kind in query_intents]
    path_types = [str(edge.get("relation_type") or "") for edge in path_edges]
    matched_intents = {kind for kind in intent_types if kind in path_types}
    if intent_types:
        score += 5.0 * len(matched_intents) / len(set(intent_types))
    for relation_type in set(path_types):
        cues = _RELATION_CUES.get(relation_type, ())
        if any(cue in q for cue in cues):
            score += 2.5
    if len(intent_types) > 1:
        ordered_path_intents = [kind for kind in path_types if kind in intent_types]
        expected_order = [kind for _, kind in query_intents]
        if ordered_path_intents and ordered_path_intents == expected_order[:len(ordered_path_intents)]:
            score += 2.0

    # Continuity/evidence are intrinsic quality checks, not path-length rewards.
    continuous = all(
        left.get("to_entity_id") == right.get("from_entity_id")
        for left, right in zip(path_edges, path_edges[1:])
    )
    if path_edges:
        score += 1.0 if continuous else -8.0
        linked_edges = sum(
            any(chunk.get("has_text") for chunk in edge.get("original_chunks") or [])
            for edge in path_edges
        )
        score += 1.25 * linked_edges / len(path_edges)
        source_text = " ".join(
            str(chunk.get("text") or "")
            for edge in path_edges
            for chunk in edge.get("original_chunks") or []
            if chunk.get("has_text", bool(chunk.get("text")))
        )
        source_norm = normalized(source_text)
        source_terms = {
            normalized(term) for term in entity_terms
            if len(normalized(term)) >= 3 and normalized(term) in source_norm
        }
        score += sum(min(len(term), 10) * 0.25 for term in source_terms)

    requested_hops = re.search(r"\b([1-3])\s*(?:hop|홉)\b", q)
    if requested_hops:
        actual_hops = len(path_edges)
        score += 3.0 if actual_hops == int(requested_hops.group(1)) else -2.0 * abs(actual_hops - int(requested_hops.group(1)))
    return score


def path_to_result(
    path: Any,
    *,
    chunks_by_id: Mapping[str, OriginalChunk] | None = None,
) -> SearchResult:
    """Neo4j Path를 공통 SearchResult로 변환합니다."""

    nodes = list(path.nodes)
    relationships = list(path.relationships)

    # --------------------------------------------------
    # 1. 경로 노드 정보
    # --------------------------------------------------
    path_nodes: list[GraphPathNode] = [
        {
            "entity_id": str(node.get("entity_id") or ""),
            "name": str(
                node.get("name")
                or node.get("entity_id")
                or ""
            ),
            "entity_type": str(
                node.get("entity_type")
                or next(iter(sorted(node.labels)), "")
            ),
        }
        for node in nodes
    ]

    node_names = [node["name"] for node in path_nodes]
    node_ids = [node["entity_id"] for node in path_nodes]
    node_types = [node["entity_type"] for node in path_nodes]

    # --------------------------------------------------
    # 2. 관계 및 hop 정보
    # --------------------------------------------------
    path_edges: list[GraphPathEdge] = []

    for index, rel in enumerate(relationships):
        relation_id = rel.get("relation_id")

        path_edges.append(
            {
                "hop": index + 1,
                "relation_id": (
                    str(relation_id)
                    if relation_id is not None
                    else None
                ),
                "relation_type": str(rel.type),
                "from_entity_id": path_nodes[index]["entity_id"],
                "to_entity_id": path_nodes[index + 1]["entity_id"],
                "evidence_ids": _unique_list(
                    _as_list(rel.get("evidence_ids"))
                ),
                "chunk_ids": _unique_list(
                    _as_list(rel.get("source_chunk_ids"))
                ),
                "source_ids": _unique_list(
                    _as_list(rel.get("source_ids"))
                ),
                "original_chunks": [],
            }
        )

    # --------------------------------------------------
    # 3. 관계별 원문 Chunk 연결 및 검증
    # --------------------------------------------------
    warnings: list[str] = []

    if chunks_by_id is not None:
        for edge in path_edges:
            relation = (
                edge["relation_id"]
                or edge["relation_type"]
            )

            if not edge["chunk_ids"]:
                warnings.append(
                    f"Relation {relation} has no linked source chunks"
                )

            if not edge["evidence_ids"]:
                warnings.append(
                    f"Relation {relation} has no linked evidence IDs"
                )

            for chunk_id in edge["chunk_ids"]:
                chunk = chunks_by_id.get(chunk_id)

                if chunk is None:
                    warnings.append(
                        f"Relation {relation} references "
                        f"missing chunk {chunk_id}"
                    )
                    continue

                if not chunk["text"].strip():
                    warnings.append(
                        f"Relation {relation} has empty "
                        f"original text in {chunk_id}"
                    )
                    continue

                if (
                    edge["evidence_ids"]
                    and chunk["evidence_id"]
                    not in edge["evidence_ids"]
                ):
                    warnings.append(
                        f"Relation {relation} evidence "
                        f"does not match chunk {chunk_id}"
                    )
                    continue

                if (
                    edge["source_ids"]
                    and chunk["source_id"]
                    not in edge["source_ids"]
                ):
                    warnings.append(
                        f"Relation {relation} source "
                        f"does not match chunk {chunk_id}"
                    )
                    continue

                edge["original_chunks"].append(chunk)

        # 원문 근거 연결 상태
        if warnings and any(
            edge["original_chunks"]
            for edge in path_edges
        ):
            warnings.append(
                "Graph path has partial original evidence"
            )

        elif path_edges and not any(
            edge["original_chunks"]
            for edge in path_edges
        ):
            warnings.append(
                "Graph path has no original evidence"
            )

    # --------------------------------------------------
    # 4. 경로 전체 메타데이터
    # --------------------------------------------------
    relation_types = [
        edge["relation_type"]
        for edge in path_edges
    ]

    relation_ids = [
        edge["relation_id"]
        for edge in path_edges
    ]

    evidence_ids = sorted({
        item
        for edge in path_edges
        for item in edge["evidence_ids"]
    })

    source_ids = sorted({
        item
        for edge in path_edges
        for item in edge["source_ids"]
    })

    chunk_ids = sorted({
        item
        for edge in path_edges
        for item in edge["chunk_ids"]
    })

    confidences = [
        float(rel.get("confidence"))
        for rel in relationships
        if rel.get("confidence") is not None
    ]

    confidence = min(confidences) if confidences else None

    eligible = all(
        rel.get("retrieval_eligible", True) is True
        for rel in relationships
    )

    # 모든 관계에 검증된 원문이 있는지 확인
    has_complete_evidence = bool(path_edges) and all(
        bool(edge["original_chunks"])
        for edge in path_edges
    )

    return SearchResult(
        source="graph",
        text=" → ".join(node_names),
        score=None,
        path=node_names,
        path_nodes=path_nodes,
        path_edges=path_edges,
        relation_types=relation_types,
        hop_count=len(relationships),
        evidence_ids=evidence_ids,
        source_ids=source_ids,
        chunk_ids=chunk_ids,
        metadata={
            "node_ids": node_ids,
            "node_types": node_types,
            "relation_ids": relation_ids,
            "hop_count": len(relationships),
            "confidence": confidence,
            "retrieval_eligible": eligible,
            "has_complete_evidence": has_complete_evidence,
        },
        warnings=_unique_list(warnings),
    )


class GraphRetriever:
    """Neo4j 그래프에서 멀티홉 경로와 원문 근거를 검색합니다."""

    def __init__(
        self,
        driver: Driver,
        *,
        database: str = "neo4j",
    ) -> None:
        self.driver = driver
        self.database = database
        self.last_search_diagnostics: dict[str, Any] = {}
        self._crop_cache: list[tuple[str, str, str]] | None = None
        self._pest_name_cache: tuple[str, ...] | None = None
        self._last_crop_match_count = 0
        self._last_crop_anchor_strategy = "uninitialized"

    def _load_pest_names(self) -> tuple[str, ...]:
        """그래프에 있는 해충·병해 이름을 한 번만 읽어 둡니다."""

        options = {"database": self.database} if self.database != "neo4j" else {}
        try:
            with self.driver.session(**options) as session:
                rows = list(
                    session.run(
                        "MATCH (n:Insect|Disease) RETURN n.name AS name",
                        {},
                    )
                )
        except Exception:
            # 이름을 못 읽으면 분류어 검사만으로 판단하고 검색은 계속합니다.
            return ()

        names = set()
        for row in rows:
            name = row.get("name") if hasattr(row, "get") else None
            normalized = re.sub(r"\s+", "", str(name or "")).lower()
            if len(normalized) >= 2:
                names.add(normalized)
        return tuple(sorted(names, key=len, reverse=True))

    def _mentions_pest_or_disease(self, question: str) -> bool:
        """질문이 그래프에 있는 해충·병해를 가리키는지 봅니다.

        병해충 이름을 코드에 적어 두면 새 이름이 들어올 때마다 조용히 빗나가므로
        판정 근거를 그래프 데이터에 둡니다.
        """

        if self._pest_name_cache is None:
            self._pest_name_cache = self._load_pest_names()

        normalized_question = re.sub(r"\s+", "", question).lower()
        return any(
            name in normalized_question
            for name in self._pest_name_cache
        )

    def _crop_candidates(self, question: str, crop: str | None, limit: int = 8) -> list[str]:
        """Find crop anchors from names and serialized alias/attribute fields."""
        if self._crop_cache is None:
            options = {"database": self.database} if self.database != "neo4j" else {}
            with self.driver.session(**options) as session:
                self._crop_cache = [
                    (str(row["entity_id"]), str(row["name"] or ""), str(row["attributes"] or ""))
                    for row in session.run(
                        "MATCH (n:Crop) RETURN n.entity_id AS entity_id, "
                        "n.name AS name, n.attributes AS attributes", {},
                    )
                    if row.get("entity_id") is not None
                ]
        normalized_question = re.sub(r"\s+", "", question).lower()
        terms = _query_terms(question)
        ranked: list[tuple[float, str]] = []
        for entity_id, name, attributes in self._crop_cache:
            normalized_name = re.sub(r"\s+", "", name).lower()
            score = 0.0
            if crop and name == crop:
                score += 100.0
            if normalized_name and normalized_name in normalized_question:
                score += 40.0 + min(len(normalized_name), 12)
            for term in terms:
                if term in name.lower():
                    score += min(len(term), 10) * 2.0
                if term in attributes.lower() and len(term) >= 3:
                    score += min(len(term), 8) * 0.25
            if score > 0:
                ranked.append((score, entity_id))
        if not ranked and crop:
            ranked = [(1.0, entity_id) for entity_id, name, _ in self._crop_cache if name == crop]
        explicit_match_count = len(ranked)
        # This graph has a small, closed crop inventory. If the wording does not
        # expose a crop alias, explore all known crop anchors instead of treating
        # entity anchoring as a hard retrieval failure.
        if not ranked:
            ranked = [(0.0, entity_id) for entity_id, _, _ in self._crop_cache]
        self._last_crop_match_count = explicit_match_count
        self._last_crop_anchor_strategy = "question_or_crop_match" if explicit_match_count else "all_crop_fallback"
        ranked.sort(reverse=True)
        return [entity_id for _, entity_id in ranked[:limit]]

    @classmethod
    def from_env(cls) -> GraphRetriever:
        """환경변수를 사용하여 Neo4j에 연결합니다."""

        uri = os.getenv(
            "NEO4J_URI",
            "bolt://localhost:7687",
        )
        username = os.getenv(
            "NEO4J_USERNAME",
            "neo4j",
        )
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv(
            "NEO4J_DATABASE",
            "neo4j",
        )

        if not password:
            raise ValueError(
                "NEO4J_PASSWORD is required"
            )

        driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
        )

        try:
            driver.verify_connectivity()

            return cls(
                driver,
                database=database,
            )

        except Exception:
            driver.close()
            raise

    def close(self) -> None:
        """Neo4j 드라이버 연결을 종료합니다."""
        self.driver.close()

    @staticmethod
    def _fetch_paths_and_chunks(
        tx: Any,
        query: str,
        params: dict[str, Any],
    ) -> tuple[list[Any], dict[str, OriginalChunk]]:
        """하나의 읽기 트랜잭션에서 경로와 원문을 조회합니다."""

        paths = [
            record["path"]
            for record in tx.run(query, params)
        ]

        if not paths:
            return [], {}

        # 경로에 포함된 원문 Chunk ID 수집
        chunk_ids = sorted({
            chunk_id
            for path in paths
            for rel in path.relationships
            for chunk_id in _as_list(
                rel.get("source_chunk_ids")
            )
        })

        if not chunk_ids:
            return paths, {}

        # 원문 청크 일괄 조회
        chunk_query = (
            "MATCH (chunk:Chunk) WHERE chunk.chunk_id IN $chunk_ids "
            "RETURN chunk.chunk_id AS chunk_id, "
            "chunk.evidence_id AS evidence_id, "
            "chunk.source_id AS source_id, chunk.raw_text AS text"
        )

        chunks_by_id: dict[str, OriginalChunk] = {}

        for row in tx.run(
            chunk_query,
            {"chunk_ids": chunk_ids},
        ):
            chunk_id = row.get("chunk_id")

            if chunk_id is None:
                continue

            chunk_id = str(chunk_id)

            chunks_by_id[chunk_id] = {
                "chunk_id": chunk_id,
                "evidence_id": str(
                    row.get("evidence_id") or ""
                ),
                "source_id": str(
                    row.get("source_id") or ""
                ),
                "text": str(
                    row.get("text") or ""
                ),
            }

        return paths, chunks_by_id

    def search_paths(
        self,
        *,
        start_label: str = "Crop",
        start_name: str | None = None,
        min_hops: int = 1,
        max_hops: int = 2,
        relation_types: Sequence[str] | None = None,
        limit: int = 20,
        mentioned_in_question: str | None = None,
        start_entity_ids: Sequence[str] | None = None,
        question: str | None = None,
        candidate_limit: int | None = None,
    ) -> list[GraphSearchResult]:
        """지정된 엔티티를 시작점으로 멀티홉 경로를 검색합니다."""

        # --------------------------------------------------
        # 1. 입력값 검증
        # --------------------------------------------------
        _check_identifier(
            start_label,
            "label",
        )

        if min_hops < 1 or max_hops < min_hops:
            raise ValueError(
                "Require 1 <= min_hops <= max_hops"
            )

        if limit < 1:
            raise ValueError(
                "limit must be positive"
            )
        if candidate_limit is not None and candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")

        # 관계 타입 검증 및 중복 제거
        types = _unique_list(
            list(relation_types or [])
        )

        for relation_type in types:
            _check_identifier(
                relation_type,
                "relationship type",
            )

        rel_pattern = (
            ":" + "|".join(
                f"`{relation_type}`"
                for relation_type in types
            )
            if types
            else ""
        )

        # --------------------------------------------------
        # 2. Cypher 검색 쿼리
        # --------------------------------------------------
        conditions = []
        if start_entity_ids:
            conditions.append("start.entity_id IN $start_entity_ids")
        elif start_name is not None:
            conditions.append("start.name = $start_name")
        if mentioned_in_question is not None:
            conditions.append(
                "any(node IN nodes(path)[1..] WHERE "
                "node.name IS NOT NULL AND $question CONTAINS node.name)"
            )
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        fetch_limit = candidate_limit or limit
        if question and candidate_limit:
            fetch_limit = min(candidate_limit * 3, 500)
        query = f"""
        MATCH path = (start:`{start_label}`)-[{rel_pattern}*{min_hops}..{max_hops}]->(target)
        {where_clause}
        RETURN path
        ORDER BY reduce(score = 0.0, n IN nodes(path) |
          score + reduce(node_score = 0.0, term IN $terms |
            node_score + CASE WHEN toLower(coalesce(n.name, '')) CONTAINS term
              THEN toFloat(size(term)) ELSE 0.0 END
          )
        ) DESC, length(path) DESC
        LIMIT $candidate_limit
        """

        terms = _query_terms(question or mentioned_in_question or "")
        params = {
            "start_name": start_name,
            "start_entity_ids": list(start_entity_ids or []),
            "terms": terms,
            "candidate_limit": fetch_limit,
        }
        if mentioned_in_question is not None:
            params["question"] = mentioned_in_question

        # --------------------------------------------------
        # 3. Neo4j 읽기 트랜잭션
        # --------------------------------------------------
        session_factory = self.driver.session
        session_options = {"database": self.database} if self.database != "neo4j" else {}
        with session_factory(**session_options) as session:
            if hasattr(session, "execute_read"):
                paths, chunks_by_id = session.execute_read(
                    self._fetch_paths_and_chunks,
                    query,
                    params,
                )
            else:
                paths, chunks_by_id = self._fetch_paths_and_chunks(
                    session, query, params,
                )

        # --------------------------------------------------
        # 4. SearchResult 변환
        # --------------------------------------------------
        results = [
            path_to_result(
                path,
                chunks_by_id=chunks_by_id,
            )
            for path in paths
        ]
        # Apply the same query relevance ordering in Python so candidates with
        # equal Cypher scores have stable, semantically useful ordering.
        for path, result in zip(paths, results):
            result.score = _result_relevance(
                result,
                question or mentioned_in_question or "",
                terms,
                [node.get("attributes") for node in path.nodes],
            )
        deduplicated: dict[tuple[Any, ...], GraphSearchResult] = {}
        for result in results:
            key = (
                tuple(node["entity_id"] for node in result.path_nodes or []),
                tuple(edge["relation_id"] or edge["relation_type"] for edge in result.path_edges or []),
            )
            previous = deduplicated.get(key)
            if previous is None or (result.score or 0) > (previous.score or 0):
                deduplicated[key] = result
        # Preserve stable candidate order for score ties; hop count is not a
        # relevance signal and should not act as an implicit length bonus.
        ranked = sorted(deduplicated.values(), key=lambda item: item.score or 0.0, reverse=True)
        return ranked[:limit]

    def search(
        self,
        question: str,
        *,
        crop: str | None = None,
        max_hops: int = 2,
        limit: int = 20,
        relation_types: Sequence[str] | None = None,
        candidate_pool_size: int | None = None,
        entity_names: Sequence[str] | None = None,
    ) -> list[GraphSearchResult]:
        """질문과 Router의 검색 조건을 사용합니다."""

        if (
            not isinstance(question, str)
            or not question.strip()
        ):
            raise ValueError(
                "question must be a nonempty string"
            )

        if candidate_pool_size is not None and candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be positive")
        candidate_limit = candidate_pool_size or min(max(limit * 3, 30), 300)
        # Router 인터페이스를 유지합니다. 실제 엔티티 매칭은 질문 원문과
        # 그래프의 이름·속성을 사용해 후보 선정 및 재정렬 단계에서 수행합니다.
        started = time.perf_counter()
        start_ids = self._crop_candidates(question, crop)
        self.last_search_diagnostics = {
            "entity_match_count": self._last_crop_match_count,
            "anchored_start_count": len(start_ids) or int(bool(crop)),
            "anchor_strategy": self._last_crop_anchor_strategy,
            "matched_start_entity_ids": start_ids,
            "matched_start_name": crop if not start_ids else None,
            "path_search_attempted": False,
            "path_candidate_count": 0,
            "candidate_pool_size": candidate_limit,
        }
        anchor_kwargs = ({"start_entity_ids": start_ids} if start_ids else {"start_name": crop})

        # 구체적인 다중 관계 질문은 긴 경로를 먼저 찾습니다. 짧은 경로가
        # LIMIT을 전부 차지해 2-hop 근거가 사라지는 것을 막습니다.
        variety_region = (
            "지역" in question
            and (
                "품종" in question
                or re.search(r"[가-힣A-Za-z0-9]+종", question) is not None
            )
        )
        # "병해·해충" 같은 분류어를 쓰지 않고 실제 병해충 이름만 적은 질문
        # (예: "배추 파밤나방 방제 약제")도 약제 질문으로 봐야 합니다. 이름 목록은
        # 그래프에서 읽으므로 새 병해충이 들어와도 코드를 고치지 않아도 됩니다.
        pesticide_target = (
            ("방제" in question or "농약" in question or "약제" in question)
            and (
                "병해" in question
                or "해충" in question
                or self._mentions_pest_or_disease(question)
            )
        )
        explicit_multihop = "2홉" in question or "멀티홉" in question
        multi_hop_intent = explicit_multihop or variety_region or pesticide_target
        # 라우터가 약제·품종 질문에 max_hops=1을 주면 2-hop 분기에 진입조차 못 해
        # Gold 경로가 후보에서 사라진다. 질문 의도가 다중 관계면 최소 2홉을 보장한다.
        effective_max_hops = max(max_hops, 2) if multi_hop_intent else max_hops

        if effective_max_hops >= 2 and multi_hop_intent:
            # 관계 타입은 선호 신호이지 하드 필터가 아니다. 라우터 추정 → 도메인
            # 기본 조합 → 타입 무제한 순으로 완화하며 2-hop 후보를 확보한다.
            domain_types: list[str] | None = None
            if variety_region:
                domain_types = ["HAS_VARIETY", "SUITABLE_FOR_REGION"]
            elif pesticide_target:
                domain_types = ["HAS_REGISTERED_PESTICIDE_USE", "CONTROLS_TARGET"]

            attempts: list[tuple[str, Sequence[str] | None]] = []
            if relation_types is not None:
                attempts.append(("router_relation_types", relation_types))
            if domain_types is not None and list(domain_types) != list(relation_types or []):
                attempts.append(("domain_relation_types", domain_types))
            attempts.append(("unfiltered", None))

            for strategy, types in attempts:
                found = self.search_paths(
                    start_label="Crop", **anchor_kwargs,
                    min_hops=2, max_hops=effective_max_hops,
                    relation_types=types, limit=limit,
                    question=question, candidate_limit=candidate_limit,
                )
                if found:
                    self.last_search_diagnostics.update({
                        "path_search_attempted": True,
                        "path_candidate_count": len(found),
                        "path_search_strategy": strategy,
                        "max_hops_used": effective_max_hops,
                        "failure_stage": None,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                    })
                    return found[:limit]

        results = self.search_paths(
            start_label="Crop", **anchor_kwargs, min_hops=1,
            max_hops=effective_max_hops, relation_types=relation_types, limit=limit,
            question=question, candidate_limit=candidate_limit,
        )
        strategy = "one_hop_router_relation_types" if results else None
        # 시작 라벨에서 나갈 수 없는 관계 타입만 지정되면(예: Crop에 없는
        # CONTROLS_TARGET 단독) 결과가 공집합이 된다. 필터를 버리고 한 번 더 찾는다.
        if not results and relation_types is not None:
            results = self.search_paths(
                start_label="Crop", **anchor_kwargs, min_hops=1,
                max_hops=effective_max_hops, relation_types=None, limit=limit,
                question=question, candidate_limit=candidate_limit,
            )
            strategy = "one_hop_unfiltered" if results else None
        self.last_search_diagnostics.update({
            "path_search_attempted": True,
            "path_candidate_count": len(results),
            "path_search_strategy": strategy,
            "max_hops_used": effective_max_hops,
            "failure_stage": None if results else "path_search",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        })
        return results


def graph_search(
    question: str,
    *,
    retriever: GraphRetriever,
    crop: str | None = None,
    max_hops: int = 2,
    limit: int = 20,
    relation_types: Sequence[str] | None = None,
    candidate_pool_size: int | None = None,
    entity_names: Sequence[str] | None = None,
) -> list[SearchResult]:
    """Graph 검색 결과를 공통 SearchResult로 반환합니다."""

    kwargs: dict[str, Any] = {"crop": crop, "max_hops": max_hops, "limit": limit}
    if candidate_pool_size is not None:
        kwargs["candidate_pool_size"] = candidate_pool_size
    if relation_types is not None:
        kwargs["relation_types"] = relation_types
    if entity_names is not None:
        kwargs["entity_names"] = entity_names
    return retriever.search(question, **kwargs)
