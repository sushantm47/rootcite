"""Deterministic post-generation checks. The model proposes; code decides what to trust."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rootcite.domain.models import Evidence

UNGROUNDED_CONFIDENCE_CAP = 0.3
UNCORROBORATED_CONFIDENCE_CAP = 0.6
MAX_ACTIONS = 10

NO_DIAGNOSIS = "RootCite could not reach a diagnosis. Escalate to the service owner."


@dataclass
class Verification:
    root_cause: str
    confidence: float
    citations: list[str]
    recommended_actions: list[str]
    needs_human: bool
    grounded: bool
    warnings: list[str] = field(default_factory=list)


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def verify_draft(
    draft: dict[str, Any] | None,
    evidence: dict[str, Evidence],
    auto_confidence: float = 0.7,
) -> Verification:
    if draft is None:
        return Verification(
            NO_DIAGNOSIS, 0.0, [], ["Page the service owner."], True, False,
            ["model did not submit a diagnosis"],
        )

    warnings: list[str] = []
    root_cause = str(draft.get("root_cause", "")).strip() or NO_DIAGNOSIS
    try:
        confidence = min(max(float(draft.get("confidence", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.0
        warnings.append("confidence was not a number")

    requested = list(dict.fromkeys(c.upper() for c in _as_str_list(draft.get("citations"))))
    citations = [c for c in requested if c in evidence]
    dropped = [c for c in requested if c not in evidence]
    if dropped:
        warnings.append(f"dropped citations not present in evidence: {', '.join(dropped)}")

    actions = _as_str_list(draft.get("recommended_actions"))[:MAX_ACTIONS]
    needs_human = bool(draft.get("needs_human", True))
    grounded = bool(citations)

    if not grounded:
        confidence = min(confidence, UNGROUNDED_CONFIDENCE_CAP)
        needs_human = True
        warnings.append("diagnosis is not grounded in any evidence")
    elif not any(evidence[c].kind == "tool" for c in citations):
        confidence = min(confidence, UNCORROBORATED_CONFIDENCE_CAP)
        warnings.append("not corroborated by live telemetry; capped confidence")

    flagged_sources = sorted({e.source for e in evidence.values() if e.flagged})
    if flagged_sources:
        needs_human = True
        warnings.append(
            f"possible prompt injection in {', '.join(flagged_sources)} output; review manually"
        )

    if confidence < auto_confidence:
        needs_human = True

    return Verification(root_cause, confidence, citations, actions, needs_human, grounded, warnings)
