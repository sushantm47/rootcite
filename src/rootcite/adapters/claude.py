"""Claude adapter for the Anthropic API and for Claude on Amazon Bedrock.

The SDK's own retries are disabled (max_retries=0) so there is exactly one retry policy
in the system: ResilientLLM (exponential backoff with full jitter + circuit breaker).
Stacked retries multiply load on a struggling dependency.
"""

from __future__ import annotations

from typing import Any, cast

from rootcite.domain.errors import RetryableError
from rootcite.domain.models import LLMResponse, TokenUsage, ToolCall


class ClaudeLLM:
    def __init__(
        self,
        model: str,
        provider: str = "anthropic",
        aws_region: str = "us-east-1",
        timeout_seconds: float = 60.0,
    ) -> None:
        import anthropic

        self._anthropic = anthropic
        self._model = model
        self._client: Any
        if provider == "bedrock":
            self._client = anthropic.AnthropicBedrock(
                aws_region=aws_region, max_retries=0, timeout=timeout_seconds
            )
        else:
            self._client = anthropic.Anthropic(max_retries=0, timeout=timeout_seconds)

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        try:
            response = self._client.messages.create(**kwargs)
        except self._anthropic.APIConnectionError as exc:  # includes timeouts
            raise RetryableError(f"connection error: {exc}") from exc
        except self._anthropic.APIStatusError as exc:
            if exc.status_code == 429 or exc.status_code >= 500:
                raise RetryableError(f"status {exc.status_code}: {exc}") from exc
            raise

        texts: list[str] = []
        calls: list[ToolCall] = []
        content: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                arguments = cast(dict[str, Any], block.input)
                calls.append(ToolCall(call_id=block.id, name=block.name, arguments=arguments))
                content.append(
                    {"type": "tool_use", "id": block.id, "name": block.name, "input": arguments}
                )

        return LLMResponse(
            text="".join(texts),
            tool_calls=calls,
            usage=TokenUsage(response.usage.input_tokens, response.usage.output_tokens),
            stop_reason=response.stop_reason or "",
            assistant_content=content,
        )
