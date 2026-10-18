"""Test doubles and paths shared across the suite."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from rootcite.domain.models import LLMResponse, TokenUsage, ToolCall

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"
TELEMETRY_DIR = ROOT / "data" / "telemetry"
GOLDEN = ROOT / "evals" / "golden_alerts.jsonl"


class ScriptedLLM:
    """Returns pre-programmed responses in order and records every request."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.calls.append(
            {"messages": copy.deepcopy(messages), "tools": tools, "tool_choice": tool_choice}
        )
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of responses")
        return self._responses.pop(0)


_ids = iter(range(1, 10_000))


def tool_use(*calls: tuple[str, dict[str, Any]], tokens: int = 100) -> LLMResponse:
    parsed = [ToolCall(f"call-{next(_ids)}", name, args) for name, args in calls]
    return LLMResponse(
        text="",
        tool_calls=parsed,
        usage=TokenUsage(tokens, 20),
        stop_reason="tool_use",
        assistant_content=[
            {"type": "tool_use", "id": c.call_id, "name": c.name, "input": c.arguments}
            for c in parsed
        ],
    )


def submit(tokens: int = 100, **arguments: Any) -> LLMResponse:
    defaults: dict[str, Any] = {
        "root_cause": "db pool exhausted after deploy",
        "confidence": 0.9,
        "citations": [],
        "recommended_actions": ["Roll back v2.14.0"],
        "needs_human": False,
    }
    return tool_use(("submit_diagnosis", {**defaults, **arguments}), tokens=tokens)


def text(content: str) -> LLMResponse:
    return LLMResponse(
        text=content,
        tool_calls=[],
        usage=TokenUsage(50, 10),
        stop_reason="end_turn",
        assistant_content=[{"type": "text", "text": content}],
    )


class FlakyLLM:
    """Raises the given exceptions in order, then returns ``response``."""

    def __init__(self, errors: list[Exception], response: LLMResponse) -> None:
        self._errors = list(errors)
        self._response = response
        self.attempts = 0

    def complete(self, **_: Any) -> LLMResponse:
        self.attempts += 1
        if self._errors:
            raise self._errors.pop(0)
        return self._response
