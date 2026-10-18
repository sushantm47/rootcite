"""Thread-safe in-memory index: numpy cosine similarity + BM25. For local dev and tests."""

from __future__ import annotations

import threading
from collections.abc import Sequence

import numpy as np

from rootcite.domain.models import Chunk, ScoredChunk
from rootcite.retrieval.bm25 import BM25
from rootcite.retrieval.text import tokenize


class InMemoryIndex:
    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._lock = threading.RLock()
        self._ids: list[str] = []
        self._matrix = np.zeros((0, dimension), dtype=np.float32)
        self._bm25: BM25 | None = None

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if len(chunks) and vectors.shape[1] != self._dimension:
            raise ValueError(f"expected dimension {self._dimension}, got {vectors.shape[1]}")
        with self._lock:
            for chunk, vector in zip(chunks, vectors, strict=True):
                self._chunks[chunk.chunk_id] = chunk
                self._vectors[chunk.chunk_id] = np.asarray(vector, dtype=np.float32)
            self._rebuild()
        return len(chunks)

    def _rebuild(self) -> None:
        self._ids = sorted(self._chunks)
        self._matrix = (
            np.stack([self._vectors[i] for i in self._ids])
            if self._ids
            else np.zeros((0, self._dimension), dtype=np.float32)
        )
        self._bm25 = BM25([tokenize(self._chunks[i].text) for i in self._ids])

    def dense_search(self, vector: np.ndarray, k: int) -> list[ScoredChunk]:
        with self._lock:
            if not self._ids:
                return []
            sims = self._matrix @ np.asarray(vector, dtype=np.float32)
            top = np.argsort(-sims)[:k]
            return [ScoredChunk(self._chunks[self._ids[i]], float(sims[i])) for i in top]

    def keyword_search(self, query: str, k: int) -> list[ScoredChunk]:
        with self._lock:
            if not self._ids or self._bm25 is None:
                return []
            scores = self._bm25.scores(tokenize(query))
            top = [i for i in np.argsort(-scores)[:k] if scores[i] > 0]
            return [ScoredChunk(self._chunks[self._ids[i]], float(scores[i])) for i in top]

    def count(self) -> int:
        with self._lock:
            return len(self._chunks)
