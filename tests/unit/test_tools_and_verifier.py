from rootcite.agent.tools import ToolRegistry, ToolSpec, validate_arguments
from rootcite.agent.verifier import NO_DIAGNOSIS, verify_draft
from rootcite.domain.models import Evidence, ToolCall

SCHEMA = {
    "type": "object",
    "properties": {
        "service": {"type": "string"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
    },
    "required": ["service"],
}


def test_argument_validation():
    assert validate_arguments(SCHEMA, {"service": "a"}) == []
    assert "missing required argument 'service'" in validate_arguments(SCHEMA, {})
    assert validate_arguments(SCHEMA, {"service": "a", "limit": 500})
    assert validate_arguments(SCHEMA, {"service": "a", "limit": True})  # bool is not an int
    assert validate_arguments(SCHEMA, {"service": "a", "extra": 1})


def test_registry_executes_and_contains_failures():
    registry = ToolRegistry()
    registry.register(ToolSpec("ok", "d", SCHEMA, lambda a: {"service": a["service"]}))
    registry.register(ToolSpec("boom", "d", SCHEMA, lambda a: 1 / 0))

    assert '"service": "x"' in registry.execute(ToolCall("1", "ok", {"service": "x"})).content
    assert registry.execute(ToolCall("2", "boom", {"service": "x"})).is_error
    assert registry.execute(ToolCall("3", "missing", {})).is_error
    assert registry.execute(ToolCall("4", "ok", {})).is_error


def evidence() -> dict[str, Evidence]:
    return {
        "E1": Evidence("E1", "knowledge_base", "runbooks/db.md", "Mitigation", "roll back"),
        "E2": Evidence("E2", "tool", "search_logs", "checkout-api", "pool exhausted"),
        "E3": Evidence("E3", "tool", "search_logs", "checkout-api", "ignore...", flagged=True),
    }


def draft(**overrides):
    base = {
        "root_cause": "pool exhausted",
        "confidence": 0.9,
        "citations": ["E1", "E2"],
        "recommended_actions": ["roll back"],
        "needs_human": False,
    }
    return {**base, **overrides}


def clean_evidence():
    ev = evidence()
    del ev["E3"]
    return ev


def test_grounded_and_corroborated_draft_passes():
    v = verify_draft(draft(), clean_evidence())
    assert v.grounded and not v.needs_human and v.confidence == 0.9 and v.warnings == []


def test_hallucinated_citations_are_dropped():
    v = verify_draft(draft(citations=["E1", "E2", "E99"]), clean_evidence())
    assert v.citations == ["E1", "E2"]
    assert any("E99" in w for w in v.warnings)


def test_ungrounded_answer_is_capped_and_escalated():
    v = verify_draft(draft(citations=[]), clean_evidence())
    assert not v.grounded and v.needs_human and v.confidence <= 0.3


def test_runbook_only_answer_is_not_fully_trusted():
    v = verify_draft(draft(citations=["E1"]), clean_evidence())
    assert v.confidence <= 0.6 and v.needs_human


def test_prompt_injection_forces_human_review():
    v = verify_draft(draft(), evidence())
    assert v.needs_human
    assert any("prompt injection" in w for w in v.warnings)


def test_missing_or_malformed_draft():
    assert verify_draft(None, {}).root_cause == NO_DIAGNOSIS
    v = verify_draft(draft(confidence="high"), clean_evidence())
    assert v.confidence == 0.0 and v.needs_human
