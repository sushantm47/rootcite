"""Decorator that adds retries and a circuit breaker to any LLMClient."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rootcite.domain.models import LLMResponse
from rootcite.ports import LLMClient
from rootcite.resilience.circuit_breaker import CircuitBreaker
from rootcite.resilience.retry import retry_with_backoff


class ResilientLLM:
    def __init__(
        self,
        inner: LLMClient,
        breaker: CircuitBreaker,
        attempts: int = 3,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._inner = inner
        self._breaker = breaker
        self._attempts = attempts
        self._sleep = sleep

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        def attempt() -> LLMResponse:
            return self._inner.complete(
                system=system,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
            )

        def with_retries() -> LLMResponse:
            if self._sleep is None:
                return retry_with_backoff(attempt, attempts=self._attempts)
            return retry_with_backoff(attempt, attempts=self._attempts, sleep=self._sleep)

        # Breaker wraps the whole retry sequence: one "failure" = retries exhausted.
        return self._breaker.call(with_retries)
