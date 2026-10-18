"""Immutable domain models shared by every layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """A retrievable slice of a knowledge-base document."""

    chunk_id: str
    doc_id: str
    source: str
    title: str
    text: str
    doc_type: str


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Alert:
    """An incoming page from the monitoring system."""

    alert_id: str
    service: str
    title: str
    description: str = ""
    severity: str = "SEV2"

    def as_query(self) -> str:
        return " ".join(part for part in (self.service, self.title, self.description) if part)


@dataclass(frozen=True)
class Evidence:
    """Anything the agent saw that a diagnosis may cite (E1, E2, ...)."""

    evidence_id: str
    kind: str  # knowledge_base | tool
    source: str  # document path or tool name
    section: str  # heading path or tool target
    content: str
    flagged: bool = False  # possible prompt injection


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens
        )


@dataclass(frozen=True)
class LLMResponse:
    """Vendor-neutral model response.

    ``assistant_content`` holds the raw content blocks so the conversation can be replayed
    to the model on the next turn.
    """

    text: str
    tool_calls: list[ToolCall]
    usage: TokenUsage
    stop_reason: str
    assistant_content: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Diagnosis:
    alert_id: str
    service: str
    root_cause: str
    confidence: float
    citations: list[str]
    recommended_actions: list[str]
    needs_human: bool
    grounded: bool
    warnings: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    blocked_actions: list[str] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    cached: bool = False
