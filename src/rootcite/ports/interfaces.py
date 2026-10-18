"""Protocol definitions. The core never imports a vendor SDK directly."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from rootcite.domain.models import Chunk, LLMResponse, ScoredChunk


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dimension) float32 array of L2-normalized vectors."""
        ...


class SearchIndex(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int: ...

    def dense_search(self, vector: np.ndarray, k: int) -> list[ScoredChunk]: ...

    def keyword_search(self, query: str, k: int) -> list[ScoredChunk]: ...

    def count(self) -> int: ...


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], k: int
    ) -> list[ScoredChunk]: ...


class LLMClient(Protocol):
    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...


class TelemetrySource(Protocol):
    """Read-only access to metrics, logs, and deployments (CloudWatch, Datadog, ...)."""

    def get_metric(self, service: str, metric: str, window_minutes: int) -> dict[str, Any]: ...

    def search_logs(self, service: str, query: str, limit: int) -> list[dict[str, Any]]: ...

    def recent_deploys(self, service: str, hours: int) -> list[dict[str, Any]]: ...


class Clock(Protocol):
    def __call__(self) -> float: ...
