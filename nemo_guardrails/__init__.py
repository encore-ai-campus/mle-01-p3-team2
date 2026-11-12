"""NeMo Guardrails 0.23.0 adapter for the deterministic core validators."""

from pathlib import Path

from nemoguardrails import RailsConfig


def load_config() -> RailsConfig:
    """Load the local Colang 1.0 configuration without requiring an LLM."""

    return RailsConfig.from_path(str(Path(__file__).resolve().parent))


__all__ = ["load_config"]
