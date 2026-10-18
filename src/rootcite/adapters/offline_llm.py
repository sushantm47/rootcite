"""Deterministic stand-in for Claude so the full system runs with no API key.

It follows a fixed investigation plan (check deploys and error logs, then submit a
diagnosis citing the top runbook section). It is not intelligent: it exists for local
demos, CI end-to-end tests, and offline evals. Settings forbid it in gamma and prod.
"""

from __future__ import annotations

import re
from typing import Any

from rootcite.domain.models import LLMResponse, TokenUsage, ToolCall

_SERVICE = re.compile(r"^service: (\S+)", re.MULTILINE)
_EVIDENCE = re.compile(r"^\[(E\d+)\] kind=(\S+) source=(.*?) section=(.*)$", re.MULTILINE)
_STEP = re.compile(r"^\s*\d+\.\s+(.*\S)", re.MULTILINE)
_DEPLOY = re.compile(r'"version": "([^"]+)"')


def _evidence_blocks(transcript: str) -> list[tuple[str, str, str, str, str]]:
    """(id, kind, source, section, body) for every evidence block in the transcript."""
    matches = list(_EVIDENCE.finditer(transcript))
    blocks: list[tuple[str, str, str, str, str]] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(transcript)
        body = transcript[m.end() : end]
        blocks.append((m.group(1), m.group(2), m.group(3), m.group(4), body))
    return blocks


def _text_of(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        elif block.get("type") == "tool_result":
            parts.append(str(block.get("content", "")))
    return "\n".join(parts)


class OfflineLLM:
    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        transcript = "\n".join(_text_of(m) for m in messages if m["role"] == "user")
        turns = sum(1 for m in messages if m["role"] == "assistant")
        available = {t["name"] for t in tools}
        forced = bool(tool_choice and tool_choice.get("name") == "submit_diagnosis")

        if turns == 0 and not forced:
            match = _SERVICE.search(transcript)
            service = match.group(1) if match else "unknown"
            planned = [
                ("list_recent_deploys", {"service": service, "hours": 24}),
                ("search_logs", {"service": service, "query": "error warn", "limit": 20}),
            ]
            calls = [
                ToolCall(call_id=f"offline-{i}", name=name, arguments=args)
                for i, (name, args) in enumerate(planned)
                if name in available
            ]
            if calls:
                return self._respond(calls, transcript)

        return self._respond([self._diagnosis(transcript)], transcript)

    def _diagnosis(self, transcript: str) -> ToolCall:
        blocks = _evidence_blocks(transcript)
        kb = [b for b in blocks if b[1] == "knowledge_base"]
        tool_ids = [b[0] for b in blocks if b[1] == "tool"]
        runbooks = [b for b in kb if b[2].startswith("runbooks/")]
        top = runbooks[0] if runbooks else (kb[0] if kb else None)

        root_cause = f"Symptoms match '{top[3]}' ({top[2]})." if top else "No runbook match."
        deploy = _DEPLOY.search(transcript)
        if deploy:
            root_cause += f" Recent deploy {deploy.group(1)} is a likely trigger."

        # Mitigation steps: first numbered list found in the same document as the top match.
        same_doc = [b for b in kb if top and b[2] == top[2]]
        steps = next((s for s in (_STEP.findall(b[4])[:3] for b in same_doc) if s), [])
        if not steps and top:
            steps = [f"Follow the Mitigation section of {top[2]}."]
        other_kb = [b[0] for b in kb if top and b[0] != top[0]][:1]
        return ToolCall(
            call_id="offline-final",
            name="submit_diagnosis",
            arguments={
                "root_cause": root_cause,
                "confidence": 0.55,
                "citations": ([top[0]] if top else []) + other_kb + tool_ids[:2],
                "recommended_actions": steps or ["Escalate to the service owner."],
                "needs_human": True,
            },
        )

    @staticmethod
    def _respond(calls: list[ToolCall], transcript: str) -> LLMResponse:
        content = [
            {"type": "tool_use", "id": c.call_id, "name": c.name, "input": c.arguments}
            for c in calls
        ]
        usage = TokenUsage(input_tokens=len(transcript) // 4, output_tokens=60 * len(calls))
        return LLMResponse(
            text="", tool_calls=calls, usage=usage, stop_reason="tool_use",
            assistant_content=content,
        )
