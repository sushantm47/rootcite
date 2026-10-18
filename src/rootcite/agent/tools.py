"""Tool registry with JSON-schema specs, argument validation, and side-effect flags."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rootcite.domain.models import ToolCall, ToolResult

log = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Handler
    side_effect: bool = False  # True = changes production; never auto-executed

    def to_api(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_arguments(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """Minimal JSON-schema check: required keys, primitive types, integer bounds."""
    required: list[str] = schema.get("required", [])
    errors = [f"missing required argument '{r}'" for r in required if r not in args]
    properties: dict[str, Any] = schema.get("properties", {})
    for key, value in args.items():
        prop = properties.get(key)
        if prop is None:
            errors.append(f"unexpected argument '{key}'")
            continue
        expected = _JSON_TYPES.get(prop.get("type", ""))
        # bool is a subclass of int in Python, so reject it explicitly for numeric fields.
        wrong_bool = isinstance(value, bool) and prop.get("type") != "boolean"
        if expected and (not isinstance(value, expected) or wrong_bool):
            errors.append(f"argument '{key}' must be of type {prop['type']}")
            continue
        if prop.get("type") == "integer":
            if "minimum" in prop and value < prop["minimum"]:
                errors.append(f"argument '{key}' must be >= {prop['minimum']}")
            if "maximum" in prop and value > prop["maximum"]:
                errors.append(f"argument '{key}' must be <= {prop['maximum']}")
    return errors


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool '{spec.name}' already registered")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def api_schemas(self) -> list[dict[str, Any]]:
        return [spec.to_api() for spec in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        spec = self._tools.get(call.name)
        if spec is None:
            return ToolResult(f"unknown tool '{call.name}'", is_error=True)
        errors = validate_arguments(spec.input_schema, call.arguments)
        if errors:
            return ToolResult("; ".join(errors), is_error=True)
        try:
            output = spec.handler(call.arguments)
        except Exception as exc:  # a failing tool must not crash the agent loop
            log.exception("tool failed", extra={"tool": call.name})
            return ToolResult(f"tool '{call.name}' failed: {exc}", is_error=True)
        text = output if isinstance(output, str) else json.dumps(output, indent=1, default=str)
        return ToolResult(text)
