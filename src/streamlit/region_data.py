"""지역별 정보 화면이 사용하는 지역·기상 데이터 집계 모듈입니다.

Region 노드는 두 종류가 섞여 있습니다.
- 행정구역: province/sigungu 속성이 있고 월별 기상 관측이 달려 있습니다.
- 서술형: '남부지방', '전국'처럼 품종 적합 지역을 가리키는 이름뿐인 노드입니다.
두 가지는 쓰임이 달라서 따로 모읍니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from src.streamlit.crop_data import Entity, Graph


ANY = "전체"


@dataclass(frozen=True)
class WeatherMonth:
    """한 지역의 한 달 기상 관측입니다."""

    year_month: str
    max_temp: float | None
    min_temp: float | None
    humidity: float | None
    precipitation: float | None
    complete: bool

    @property
    def year(self) -> str:
        return self.year_month.split("-")[0] if "-" in self.year_month else ""

    @property
    def month(self) -> int:
        parts = self.year_month.split("-")
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0


@dataclass
class WeatherSummary:
    """월별 관측을 한 줄로 요약한 값입니다."""

    months: int = 0
    avg_max_temp: float | None = None
    avg_min_temp: float | None = None
    avg_humidity: float | None = None
    total_precipitation: float | None = None
    hottest: WeatherMonth | None = None
    coldest: WeatherMonth | None = None
    wettest: WeatherMonth | None = None


@dataclass
class SuitableVariety:
    """이 지역에 적합하다고 표시된 품종입니다."""

    name: str
    crop: str
    characteristics: str


@dataclass
class RegionProfile:
    """지역 한 곳에 대한 화면 내용입니다."""

    name: str
    province: str
    sigungu: str
    weather: list[WeatherMonth] = field(default_factory=list)
    varieties: list[SuitableVariety] = field(default_factory=list)
    observation_count: int = 0

    @property
    def is_administrative(self) -> bool:
        """행정구역(시군구) 노드인지 여부입니다."""

        return bool(self.province and self.sigungu)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def build_weather_month(attributes: dict[str, Any]) -> WeatherMonth:
    """WeatherObservation 속성에서 월별 관측을 만듭니다."""

    completeness = attributes.get("completeness")
    complete = (
        all(bool(flag) for flag in completeness.values())
        if isinstance(completeness, dict) and completeness
        else False
    )
    return WeatherMonth(
        year_month=_text(attributes.get("year_month")),
        max_temp=_number(attributes.get("monthly_max_temperature_c")),
        min_temp=_number(attributes.get("monthly_min_temperature_c")),
        humidity=_number(attributes.get("monthly_mean_humidity_pct")),
        precipitation=_number(attributes.get("monthly_precipitation_mm")),
        complete=complete,
    )


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 1) if values else None


def summarize_weather(months: Iterable[WeatherMonth]) -> WeatherSummary:
    """월별 관측을 평균·합계와 최고/최저 달로 요약합니다."""

    months = [month for month in months if month.year_month]
    if not months:
        return WeatherSummary()

    max_temps = [m.max_temp for m in months if m.max_temp is not None]
    min_temps = [m.min_temp for m in months if m.min_temp is not None]
    humidities = [m.humidity for m in months if m.humidity is not None]
    rainfalls = [m.precipitation for m in months if m.precipitation is not None]

    return WeatherSummary(
        months=len(months),
        avg_max_temp=_mean(max_temps),
        avg_min_temp=_mean(min_temps),
        avg_humidity=_mean(humidities),
        total_precipitation=round(sum(rainfalls), 1) if rainfalls else None,
        hottest=max(
            (m for m in months if m.max_temp is not None),
            key=lambda m: m.max_temp,
            default=None,
        ),
        coldest=min(
            (m for m in months if m.min_temp is not None),
            key=lambda m: m.min_temp,
            default=None,
        ),
        wettest=max(
            (m for m in months if m.precipitation is not None),
            key=lambda m: m.precipitation,
            default=None,
        ),
    )


def build_profile(graph: Graph, region: Entity) -> RegionProfile:
    """지역 노드 하나에 대한 화면 내용을 모읍니다."""

    weather = [
        build_weather_month(observation.attributes)
        for observation in graph.neighbours_of_type(
            region.entity_id, "WeatherObservation"
        )
    ]
    weather.sort(key=lambda m: m.year_month)

    varieties = [
        SuitableVariety(
            name=variety.name,
            crop=_text(variety.attributes.get("crop")),
            characteristics=_text(variety.attributes.get("characteristics")),
        )
        for variety in graph.sources_of_type(region.entity_id, "Variety")
    ]
    varieties.sort(key=lambda v: (v.crop, v.name))

    # 생육·파종 관측이 이 지역에서 몇 건 이루어졌는지입니다.
    observation_count = sum(
        1
        for source in graph.sources(region.entity_id)
        if source.entity_type in ("GrowthObservation", "SowingObservation")
    )

    return RegionProfile(
        name=region.name,
        province=_text(region.attributes.get("province")),
        sigungu=_text(region.attributes.get("sigungu")),
        weather=weather,
        varieties=varieties,
        observation_count=observation_count,
    )


def build_profiles(graph: Graph) -> list[RegionProfile]:
    """모든 지역의 화면 내용을 만듭니다."""

    profiles = [
        build_profile(graph, entity)
        for entity in graph.entities.values()
        if entity.entity_type == "Region"
    ]
    merge_observation_counts(profiles)
    profiles.sort(key=lambda p: (p.province, p.sigungu, p.name))
    return profiles


def merge_observation_counts(profiles: list[RegionProfile]) -> None:
    """같은 이름의 서술형 노드가 들고 있는 관측 건수를 시군구 쪽으로 합칩니다.

    '전라남도 무안군'처럼 기상이 달린 행정구역 노드와 생육 관측이 달린
    이름뿐인 노드가 따로 있어서, 화면에서는 한 지역으로 보이도록 더합니다.
    """

    extra: dict[str, int] = {}
    for profile in profiles:
        if not profile.is_administrative and profile.observation_count:
            extra[profile.name] = extra.get(profile.name, 0) + profile.observation_count

    for profile in profiles:
        if profile.is_administrative and profile.name in extra:
            profile.observation_count += extra[profile.name]


def administrative(profiles: Iterable[RegionProfile]) -> list[RegionProfile]:
    """시군구 노드만 골라 돌려줍니다."""

    return [profile for profile in profiles if profile.is_administrative]


def province_values(profiles: Iterable[RegionProfile]) -> list[str]:
    """시도 선택지를 돌려줍니다."""

    values = {profile.province for profile in administrative(profiles)}
    values.discard("")
    return sorted(values)


def sigungu_profiles(
    profiles: Iterable[RegionProfile], province: str
) -> list[RegionProfile]:
    """선택한 시도에 속한 시군구를 돌려줍니다."""

    return [
        profile
        for profile in administrative(profiles)
        if profile.province == province
    ]


def variety_regions(profiles: Iterable[RegionProfile]) -> list[RegionProfile]:
    """품종 적합 지역으로 쓰인 노드만 골라 돌려줍니다."""

    found = [profile for profile in profiles if profile.varieties]
    found.sort(key=lambda p: (-len(p.varieties), p.name))
    return found
