import pytest
from pathlib import Path

from src.guardrails.nemo import load_config
from src.guardrails.nemo.actions import validate_graph_rag_response


def _valid_payload() -> dict:
    return {
        "question": "마늘 잎마름병은 언제 심해지나요?",
        "answer": "4~5월 다습한 환경에서 심해집니다.",
        "evidence": [
            {
                "source": "vector",
                "text": "마늘 잎마름병은 4~5월 다습한 환경에서 심해진다.",
                "evidence_ids": ["EVD_18E55EAE2F72ED3A"],
                "source_ids": ["SRC_2DE42643BECD363B"],
                "chunk_ids": ["CHK_C6C71D52D933D8ED"],
                "metadata": {"evidence_type": "disease_record"},
            }
        ],
        "expected_facts": ["4~5월", "다습한 환경"],
    }


def test_nemo_config_and_colang_parse():
    config = load_config()

    assert config.colang_version == "1.0"
    assert len(config.flows) == 1


@pytest.mark.asyncio
async def test_action_uses_core_pipeline():
    result = await validate_graph_rag_response(context=_valid_payload())

    assert result["allowed"] is True
    assert result["decision"] == "ALLOW"


@pytest.mark.asyncio
async def test_action_uses_fail_closed_core_pipeline():
    payload = _valid_payload()
    payload["evidence"] = []

    result = await validate_graph_rag_response(context=payload)

    assert result["allowed"] is False
    assert result["decision"] == "BLOCK"


@pytest.mark.asyncio
async def test_action_validates_current_bot_message_when_both_fields_are_present():
    payload = _valid_payload()
    payload["bot_message"] = "9월 건조한 환경에서 심해집니다."

    result = await validate_graph_rag_response(context=payload)

    assert result["allowed"] is False
    assert result["decision"] == "FALLBACK"


def test_adapter_imports_with_installed_nemo_023():
    from nemoguardrails import RailsConfig
    from nemoguardrails.actions import action

    assert RailsConfig is not None
    assert action is not None


def test_existing_guardrails_practice_is_preserved():
    import guardrails_practice

    assert callable(guardrails_practice.check_answer)
    assert Path("guardrails_practice.py").exists()


def test_team_interface_note_documents_public_contract():
    note = Path("docs/guardrails-interface.md").read_text(encoding="utf-8")

    assert "GuardrailRequest" in note
    assert "validate_request" in note
    assert "VERIFIED_CURRENT" in note
    assert "fallback" in note.lower()
