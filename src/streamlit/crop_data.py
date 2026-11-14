"""작물 도감 화면이 사용하는 그래프 데이터 적재·집계 모듈입니다.

Neo4j 없이도 동작하도록 data/processed의 JSONL을 직접 읽습니다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENTITIES_PATH = PROJECT_ROOT / "data" / "processed" / "entities.jsonl"
DEFAULT_RELATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "relations.jsonl"

CROP_ORDER = ("상추", "양파", "마늘", "배추", "고추")

CROP_EMOJI = {
    "상추": "🥬",
    "양파": "🧅",
    "마늘": "🧄",
    "배추": "🥬",
    "고추": "🌶️",
}

ENV_CATEGORY_LABEL = {
    "temperature": "🌡️ 온도",
    "soil": "🪴 토양",
    "water": "💧 수분",
    "photoperiod": "🌗 일장",
    "light": "☀️ 광",
    "duration": "⏳ 기간",
    "environment": "🌤️ 환경",
    "기타": "📎 기타",
}

ENV_FIELD_LABEL = {
    "optimal_temperature": "생육 적온",
    "optimal_soil_ph": "적정 토양 pH",
    "optimal_annual_rainfall": "연간 적정 강수량",
    "optimal_soil_texture": "적합 토성",
    "soil_drainage": "배수 조건",
}

# category가 비어 있는 FAO EcoCrop 항목은 field 이름으로 범주를 정합니다.
ENV_FIELD_CATEGORY = {
    "optimal_temperature": "temperature",
    "optimal_soil_ph": "soil",
    "optimal_soil_texture": "soil",
    "soil_drainage": "soil",
    "optimal_annual_rainfall": "water",
}


@dataclass(frozen=True)
class Entity:
    """entities.jsonl의 한 줄입니다."""

    entity_id: str
    entity_type: str
    name: str
    attributes: dict[str, Any]


@dataclass(frozen=True)
class Relation:
    """relations.jsonl의 한 줄입니다."""

    from_id: str
    to_id: str
    relation_type: str


@dataclass
class Graph:
    """엔티티와 관계를 함께 들고 다니는 최소 그래프입니다."""

    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    _outgoing: dict[str, list[str]] = field(default_factory=dict, repr=False)
    _incoming: dict[str, list[str]] = field(default_factory=dict, repr=False)

    def index(self) -> None:
        """from_id / to_id 기준 인접 목록을 미리 만들어 둡니다."""

        self._outgoing = {}
        self._incoming = {}
        for relation in self.relations:
            self._outgoing.setdefault(relation.from_id, []).append(relation.to_id)
            self._incoming.setdefault(relation.to_id, []).append(relation.from_id)

    def neighbours(self, entity_id: str) -> list[Entity]:
        """해당 노드에서 나가는 관계의 상대 노드를 돌려줍니다."""

        if not self._outgoing and self.relations:
            self.index()

        found = []
        for target_id in self._outgoing.get(entity_id, []):
            target = self.entities.get(target_id)
            if target is not None:
                found.append(target)
        return found

    def neighbours_of_type(self, entity_id: str, entity_type: str) -> list[Entity]:
        """특정 종류의 이웃만 골라 돌려줍니다."""

        return [e for e in self.neighbours(entity_id) if e.entity_type == entity_type]

    def sources(self, entity_id: str) -> list[Entity]:
        """해당 노드로 들어오는 관계의 출발 노드를 돌려줍니다."""

        if not self._incoming and self.relations:
            self.index()

        found = []
        for source_id in self._incoming.get(entity_id, []):
            source = self.entities.get(source_id)
            if source is not None:
                found.append(source)
        return found

    def sources_of_type(self, entity_id: str, entity_type: str) -> list[Entity]:
        """들어오는 관계 중 특정 종류의 출발 노드만 골라 돌려줍니다."""

        return [e for e in self.sources(entity_id) if e.entity_type == entity_type]


@dataclass
class EnvCondition:
    """생육 환경 조건 한 줄입니다."""

    category: str
    label: str
    value: str
    note: str


@dataclass
class VarietyInfo:
    """품종과 그 품종에 달린 파종 권고·적합 지역입니다."""

    entity: Entity
    sowings: list[Entity] = field(default_factory=list)
    regions: list[Entity] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.entity.name

    @property
    def attributes(self) -> dict[str, Any]:
        return self.entity.attributes


@dataclass
class CropProfile:
    """작물 도감 한 쪽에 들어가는 내용입니다."""

    name: str
    emoji: str
    counts: dict[str, int] = field(default_factory=dict)
    environment: list[EnvCondition] = field(default_factory=list)
    varieties: list[VarietyInfo] = field(default_factory=list)
    variety_groups: list[Entity] = field(default_factory=list)
    diseases: list[Entity] = field(default_factory=list)
    insects: list[Entity] = field(default_factory=list)
    fertilizers: list[Entity] = field(default_factory=list)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_graph(
    entity_rows: Iterable[dict[str, Any]],
    relation_rows: Iterable[dict[str, Any]],
) -> Graph:
    """엔티티·관계 딕셔너리 묶음에서 Graph를 만듭니다."""

    graph = Graph()
    for row in entity_rows:
        entity = Entity(
            entity_id=row["entity_id"],
            entity_type=row["entity_type"],
            name=row.get("name", ""),
            attributes=row.get("attributes") or {},
        )
        graph.entities[entity.entity_id] = entity

    for row in relation_rows:
        graph.relations.append(
            Relation(
                from_id=row["from_id"],
                to_id=row["to_id"],
                relation_type=row["relation_type"],
            )
        )
    graph.index()
    return graph


def load_graph(
    entities_path: Path | str = DEFAULT_ENTITIES_PATH,
    relations_path: Path | str = DEFAULT_RELATIONS_PATH,
) -> Graph:
    """JSONL 두 개를 읽어 Graph를 만듭니다."""

    return build_graph(
        _read_jsonl(Path(entities_path)),
        _read_jsonl(Path(relations_path)),
    )


def _format_number(value: Any) -> str:
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return f"{value:g}"
    if isinstance(value, str):
        return value.strip()
    return ""


def _is_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    try:
        float(str(value).strip())
    except (TypeError, ValueError):
        return False
    return True


def format_env_value(attributes: dict[str, Any]) -> str:
    """min/max/unit 조합을 사람이 읽는 문자열로 만듭니다."""

    unit = (attributes.get("unit") or "").strip()
    low = _format_number(attributes.get("min_value"))
    high = _format_number(attributes.get("max_value"))

    if unit == "text" or (not low and not high):
        return ""

    unit_text = "" if unit in ("", "text") else f" {unit}"

    if low and high and low != high:
        return f"{low} ~ {high}{unit_text}"
    # 한쪽만 있는 값은 숫자일 때만 "이상/이하"로 읽습니다.
    if low and not high:
        suffix = " 이상" if _is_numeric(low) else ""
        return f"{low}{unit_text}{suffix}"
    if high and not low:
        suffix = " 이하" if _is_numeric(high) else ""
        return f"{high}{unit_text}{suffix}"
    return f"{low or high}{unit_text}"


def build_env_conditions(facts: Iterable[Entity]) -> list[EnvCondition]:
    """EnvironmentFact 목록을 화면용 조건 줄로 바꿉니다."""

    conditions: list[EnvCondition] = []
    for fact in facts:
        attributes = fact.attributes
        field_name = attributes.get("field") or ""
        category = (
            attributes.get("category")
            or ENV_FIELD_CATEGORY.get(field_name)
            or "기타"
        )
        label = (
            attributes.get("stage")
            or ENV_FIELD_LABEL.get(attributes.get("field") or "")
            or attributes.get("field")
            or fact.name
        )
        conditions.append(
            EnvCondition(
                category=category,
                label=str(label).strip(),
                value=format_env_value(attributes),
                note=str(attributes.get("evidence_summary") or attributes.get("notes") or "").strip(),
            )
        )

    order = list(ENV_CATEGORY_LABEL)
    conditions.sort(
        key=lambda c: (
            order.index(c.category) if c.category in order else len(order),
            c.label,
        )
    )
    return conditions


def build_profile(graph: Graph, crop: Entity) -> CropProfile:
    """작물 노드 하나에 대한 도감 내용을 모읍니다."""

    buckets: dict[str, list[Entity]] = {}
    for neighbour in graph.neighbours(crop.entity_id):
        buckets.setdefault(neighbour.entity_type, []).append(neighbour)

    def sorted_bucket(entity_type: str) -> list[Entity]:
        return sorted(buckets.get(entity_type, []), key=lambda e: e.name)

    # 파종 권고와 적합 지역은 작물이 아니라 품종에 달려 있습니다.
    varieties = [
        VarietyInfo(
            entity=variety,
            sowings=graph.neighbours_of_type(variety.entity_id, "SowingRecommendation"),
            regions=graph.neighbours_of_type(variety.entity_id, "Region"),
        )
        for variety in sorted_bucket("Variety")
    ]

    counts = {
        "병해": len(buckets.get("Disease", [])),
        "해충": len(buckets.get("Insect", [])),
        "품종": len(buckets.get("Variety", [])),
        "등록 농약": len(buckets.get("PesticideUse", [])),
        "재배 관리": len(buckets.get("ManagementFact", [])),
        "생육 관측": len(buckets.get("GrowthObservation", [])),
    }

    return CropProfile(
        name=crop.name,
        emoji=CROP_EMOJI.get(crop.name, "🌱"),
        counts=counts,
        environment=build_env_conditions(buckets.get("EnvironmentFact", [])),
        varieties=varieties,
        variety_groups=sorted_bucket("VarietyGroup"),
        diseases=sorted_bucket("Disease"),
        insects=sorted_bucket("Insect"),
        fertilizers=sorted_bucket("FertilizerStandard"),
    )


def build_profiles(graph: Graph) -> dict[str, CropProfile]:
    """모든 Crop 노드의 도감 내용을 작물 이름으로 묶어 돌려줍니다."""

    crops = [e for e in graph.entities.values() if e.entity_type == "Crop"]
    profiles = {crop.name: build_profile(graph, crop) for crop in crops}

    ordered: dict[str, CropProfile] = {}
    for name in CROP_ORDER:
        if name in profiles:
            ordered[name] = profiles.pop(name)
    ordered.update(sorted(profiles.items()))
    return ordered
