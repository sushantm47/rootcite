"""Request/response contracts. Versioned under /v1; fields are only ever added, never changed."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from rootcite.domain.models import Alert, Diagnosis


class AlertRequest(BaseModel):
    alert_id: str = Field(min_length=1, max_length=128)
    service: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    title: str = Field(min_length=1, max_length=300)
    description: str = Field(default="", max_length=4000)
    severity: Literal["SEV1", "SEV2", "SEV3", "SEV4"] = "SEV2"

    def to_domain(self) -> Alert:
        return Alert(**self.model_dump())


class EvidenceOut(BaseModel):
    evidence_id: str
    kind: str
    source: str
    section: str
    flagged: bool


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int


class DiagnosisResponse(BaseModel):
    alert_id: str
    service: str
    root_cause: str
    confidence: float
    citations: list[str]
    recommended_actions: list[str]
    needs_human: bool
    grounded: bool
    warnings: list[str]
    evidence: list[EvidenceOut]
    tools_called: list[str]
    blocked_actions: list[str]
    usage: UsageOut
    latency_ms: float
    cached: bool

    @classmethod
    def from_domain(cls, d: Diagnosis) -> DiagnosisResponse:
        return cls(
            alert_id=d.alert_id,
            service=d.service,
            root_cause=d.root_cause,
            confidence=d.confidence,
            citations=d.citations,
            recommended_actions=d.recommended_actions,
            needs_human=d.needs_human,
            grounded=d.grounded,
            warnings=d.warnings,
            evidence=[
                EvidenceOut(
                    evidence_id=e.evidence_id,
                    kind=e.kind,
                    source=e.source,
                    section=e.section,
                    flagged=e.flagged,
                )
                for e in d.evidence
            ],
            tools_called=d.tools_called,
            blocked_actions=d.blocked_actions,
            usage=UsageOut(
                input_tokens=d.usage.input_tokens, output_tokens=d.usage.output_tokens
            ),
            latency_ms=d.latency_ms,
            cached=d.cached,
        )


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    indexed_total: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    stage: str
    indexed_chunks: int
    version: str
    llm_provider: str
    model: str
    public_demo: bool
    repo_url: str
    live_runs_remaining: int | None = None


class ScenarioOut(BaseModel):
    service: str
    title: str
    description: str
