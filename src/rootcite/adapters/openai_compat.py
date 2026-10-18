"""Adapter for any OpenAI-compatible chat API (Groq, OpenAI, Together, vLLM, Ollama, ...).

The agent speaks a provider-neutral, Anthropic-style message format: assistant turns hold
``tool_use`` blocks and user turns hold ``tool_result`` blocks. This adapter translates that
to the OpenAI wire format (``tool_calls`` on assistant messages, ``role: tool`` replies) and
back. It uses only the standard library, so it adds no dependencies.

Like the Claude adapter it does no retrying of its own: transient failures are raised as
RetryableError so the single retry policy in ResilientLLM handles them.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from rootcite.domain.errors import LLMRequestError, RetryableError
from rootcite.domain.models import LLMResponse, TokenUsage, ToolCall

# (url, headers, body, timeout) -> (status, response body)
Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, bytes]]

_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)


def urllib_transport(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RetryableError(f"connection error: {exc}") from exc


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


def to_openai_tool_choice(choice: dict[str, Any] | None) -> Any:
    if not choice:
        return None
    if choice.get("type") == "tool":
        return {"type": "function", "function": {"name": choice["name"]}}
    return choice.get("type", "auto")


def to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        content = message.get("content", "")
        if message["role"] == "assistant":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))},
                }
                for b in blocks
                if b.get("type") == "tool_use"
            ]
            entry: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
            continue

        if isinstance(content, str):
            out.append({"role": "user", "content": content})
            continue
        # Tool results must directly follow the assistant turn that requested them;
        # any plain text in the same turn goes after them as a user message.
        texts: list[str] = []
        for block in content:
            if block.get("type") == "tool_result":
                body = str(block.get("content", ""))
                if block.get("is_error"):
                    body = f"ERROR: {body}"
                out.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": body})
            elif block.get("type") == "text":
                texts.append(str(block.get("text", "")))
        if texts:
            out.append({"role": "user", "content": "\n\n".join(texts)})
    return out


def parse_arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string. Malformed JSON becomes a field the tool
    registry rejects, so the model gets an error back and can correct itself."""
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {"_unparseable_arguments": str(raw)}
    return value if isinstance(value, dict) else {"_unparseable_arguments": str(raw)}


def from_openai_response(data: dict[str, Any]) -> LLMResponse:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = _THINKING.sub("", message.get("content") or "").strip()
    calls = [
        ToolCall(
            call_id=str(c.get("id") or f"call-{i}"),
            name=str(c.get("function", {}).get("name", "")),
            arguments=parse_arguments(c.get("function", {}).get("arguments")),
        )
        for i, c in enumerate(message.get("tool_calls") or [])
    ]
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    content.extend(
        {"type": "tool_use", "id": c.call_id, "name": c.name, "input": c.arguments}
        for c in calls
    )
    usage = data.get("usage") or {}
    return LLMResponse(
        text=text,
        tool_calls=calls,
        usage=TokenUsage(
            int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        ),
        stop_reason=str(choice.get("finish_reason") or ""),
        assistant_content=content,
    )


class OpenAICompatLLM:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_seconds: float = 60.0,
        transport: Transport = urllib_transport,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._timeout = timeout_seconds
        self._transport = transport

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": to_openai_messages(system, messages),
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = to_openai_tools(tools)
            choice = to_openai_tool_choice(tool_choice)
            if choice:
                payload["tool_choice"] = choice

        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self._api_key}",
            "user-agent": "rootcite",
        }
        status, body = self._transport(
            self._url, headers, json.dumps(payload).encode(), self._timeout
        )
        if status == 429 or status >= 500:
            raise RetryableError(f"LLM provider returned {status}")
        if status >= 400:
            detail = body.decode(errors="replace")[:300]
            raise LLMRequestError(f"LLM provider rejected the request ({status}): {detail}")
        try:
            data = json.loads(body)
        except ValueError as exc:
            raise RetryableError("LLM provider returned invalid JSON") from exc
        return from_openai_response(data)
