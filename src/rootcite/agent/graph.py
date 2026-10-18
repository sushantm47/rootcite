"""Incident agent as an explicit state machine.

    RETRIEVE -> INVESTIGATE (tool loop) -> SYNTHESIZE (forced structured output) -> VERIFY

Each node is a method that mutates AgentState and returns the next node. Budgets (steps,
tool calls, tokens) are enforced in code, not left to the model, so cost and latency
have hard upper bounds.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from rootcite.agent.guardrails import screen_tool_output, wrap_untrusted
from rootcite.agent.prompts import FORCE_SUBMIT, SYSTEM_PROMPT
from rootcite.agent.toolkit import SUBMIT_DIAGNOSIS, SUBMIT_DIAGNOSIS_SCHEMA
from rootcite.agent.tools import ToolRegistry
from rootcite.agent.verifier import verify_draft
from rootcite.domain.models import Alert, Diagnosis, Evidence, TokenUsage, ToolCall
from rootcite.ports import LLMClient
from rootcite.retrieval.hybrid import HybridRetriever

log = logging.getLogger(__name__)


class Step(StrEnum):
    RETRIEVE = "retrieve"
    INVESTIGATE = "investigate"
    SYNTHESIZE = "synthesize"
    VERIFY = "verify"
    DONE = "done"


@dataclass(frozen=True)
class AgentConfig:
    retrieval_k: int = 5
    max_steps: int = 6
    max_tool_calls: int = 10
    token_budget: int = 60_000
    max_tokens_per_call: int = 1500
    auto_confidence: float = 0.7  # below this, a human must review before acting


@dataclass
class AgentState:
    alert: Alert
    messages: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    tools_called: list[str] = field(default_factory=list)
    blocked_actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    steps: int = 0
    tool_calls: int = 0
    draft: dict[str, Any] | None = None
    trace: list[Step] = field(default_factory=list)
    result: Diagnosis | None = None

    def add_evidence(
        self, kind: str, source: str, section: str, content: str, flagged: bool = False
    ) -> Evidence:
        evidence = Evidence(f"E{len(self.evidence) + 1}", kind, source, section, content, flagged)
        self.evidence[evidence.evidence_id] = evidence
        return evidence


def format_evidence_header(e: Evidence) -> str:
    return f"[{e.evidence_id}] kind={e.kind} source={e.source} section={e.section}"


def _tool_result(call_id: str, content: str, is_error: bool = False) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": call_id, "content": content, "is_error": is_error}


class IncidentAgent:
    def __init__(
        self,
        llm: LLMClient,
        retriever: HybridRetriever,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._tools = tools
        self._config = config or AgentConfig()
        self._clock = clock
        self._tool_schemas = [*tools.api_schemas(), SUBMIT_DIAGNOSIS_SCHEMA]
        self._nodes: dict[Step, Callable[[AgentState], Step]] = {
            Step.RETRIEVE: self._retrieve,
            Step.INVESTIGATE: self._investigate,
            Step.SYNTHESIZE: self._synthesize,
            Step.VERIFY: self._verify,
        }

    def run(self, alert: Alert) -> Diagnosis:
        started = self._clock()
        state = AgentState(alert)
        step = Step.RETRIEVE
        max_transitions = 2 * self._config.max_steps + 4  # defensive bound on the loop
        while step is not Step.DONE:
            if len(state.trace) >= max_transitions and step is not Step.VERIFY:
                state.warnings.append("agent exceeded transition limit")
                step = Step.VERIFY
            state.trace.append(step)
            log.debug("agent step", extra={"alert_id": alert.alert_id, "step": step.value})
            step = self._nodes[step](state)

        assert state.result is not None
        latency_ms = (self._clock() - started) * 1000
        log.info(
            "agent finished",
            extra={
                "alert_id": alert.alert_id,
                "trace": [s.value for s in state.trace],
                "tool_calls": state.tool_calls,
                "tokens": state.usage.total,
                "latency_ms": round(latency_ms, 1),
            },
        )
        return replace(state.result, latency_ms=round(latency_ms, 1))

    # ----- nodes ---------------------------------------------------------------------

    def _retrieve(self, state: AgentState) -> Step:
        alert = state.alert
        results = self._retriever.retrieve(alert.as_query(), self._config.retrieval_k)
        blocks = []
        for r in results:
            e = state.add_evidence("knowledge_base", r.chunk.source, r.chunk.title, r.chunk.text)
            blocks.append(f"{format_evidence_header(e)}\n{e.content}")
        kb = "\n\n".join(blocks) if blocks else "No matching runbooks or postmortems found."
        state.messages.append(
            {
                "role": "user",
                "content": (
                    "New alert\n"
                    f"alert_id: {alert.alert_id}\n"
                    f"service: {alert.service}\n"
                    f"severity: {alert.severity}\n"
                    f"title: {alert.title}\n"
                    f"description: {alert.description or '-'}\n\n"
                    f"Knowledge-base evidence:\n\n{kb}"
                ),
            }
        )
        return Step.INVESTIGATE

    def _investigate(self, state: AgentState) -> Step:
        if self._over_budget(state):
            return Step.SYNTHESIZE
        response = self._llm.complete(
            system=SYSTEM_PROMPT,
            messages=state.messages,
            tools=self._tool_schemas,
            max_tokens=self._config.max_tokens_per_call,
        )
        state.usage += response.usage
        state.steps += 1
        content = response.assistant_content or [
            {"type": "text", "text": response.text or "(no response)"}
        ]
        state.messages.append({"role": "assistant", "content": content})

        if not response.tool_calls:
            return Step.SYNTHESIZE

        submit = next((c for c in response.tool_calls if c.name == SUBMIT_DIAGNOSIS), None)
        results = [self._run_tool(state, c) for c in response.tool_calls if c is not submit]
        if submit is not None:
            state.draft = submit.arguments
            return Step.VERIFY
        state.messages.append({"role": "user", "content": results})
        return Step.INVESTIGATE

    def _synthesize(self, state: AgentState) -> Step:
        self._append_user_text(state, FORCE_SUBMIT)
        response = self._llm.complete(
            system=SYSTEM_PROMPT,
            messages=state.messages,
            tools=self._tool_schemas,
            tool_choice={"type": "tool", "name": SUBMIT_DIAGNOSIS},
            max_tokens=self._config.max_tokens_per_call,
        )
        state.usage += response.usage
        state.steps += 1
        submit = next((c for c in response.tool_calls if c.name == SUBMIT_DIAGNOSIS), None)
        if submit is not None:
            state.draft = submit.arguments
        return Step.VERIFY

    def _verify(self, state: AgentState) -> Step:
        v = verify_draft(state.draft, state.evidence, self._config.auto_confidence)
        state.result = Diagnosis(
            alert_id=state.alert.alert_id,
            service=state.alert.service,
            root_cause=v.root_cause,
            confidence=round(v.confidence, 3),
            citations=v.citations,
            recommended_actions=v.recommended_actions,
            needs_human=v.needs_human or bool(state.blocked_actions),
            grounded=v.grounded,
            warnings=[*state.warnings, *v.warnings],
            evidence=list(state.evidence.values()),
            tools_called=state.tools_called,
            blocked_actions=state.blocked_actions,
            usage=state.usage,
        )
        return Step.DONE

    # ----- helpers -------------------------------------------------------------------

    def _over_budget(self, state: AgentState) -> bool:
        c = self._config
        return (
            state.steps >= c.max_steps
            or state.tool_calls >= c.max_tool_calls
            or state.usage.total >= c.token_budget
        )

    def _run_tool(self, state: AgentState, call: ToolCall) -> dict[str, Any]:
        state.tool_calls += 1
        if state.tool_calls > self._config.max_tool_calls:
            return _tool_result(call.call_id, "Tool budget exhausted. Submit now.", is_error=True)

        spec = self._tools.get(call.name)
        if spec is not None and spec.side_effect:
            action = f"{call.name}({json.dumps(call.arguments, sort_keys=True)})"
            state.blocked_actions.append(action)
            state.warnings.append(f"blocked side-effect action pending approval: {action}")
            return _tool_result(
                call.call_id,
                "BLOCKED: this action changes production and needs human approval. "
                "Put it in recommended_actions instead.",
                is_error=True,
            )

        state.tools_called.append(call.name)
        result = self._tools.execute(call)
        if result.is_error:
            return _tool_result(call.call_id, result.content, is_error=True)

        screened = screen_tool_output(result.content)
        if screened.redactions:
            log.info("redacted tool output", extra={"tool": call.name, **screened.redactions})
        target = str(call.arguments.get("service") or call.arguments.get("query") or "-")
        e = state.add_evidence("tool", call.name, target, screened.text, screened.flagged)
        body = wrap_untrusted(call.name, screened.text, screened.flagged)
        return _tool_result(call.call_id, f"{format_evidence_header(e)}\n{body}")

    @staticmethod
    def _append_user_text(state: AgentState, text: str) -> None:
        last = state.messages[-1] if state.messages else None
        if last is not None and last["role"] == "user":
            if isinstance(last["content"], str):
                last["content"] = [{"type": "text", "text": last["content"]}]
            last["content"].append({"type": "text", "text": text})
        else:
            state.messages.append({"role": "user", "content": text})
