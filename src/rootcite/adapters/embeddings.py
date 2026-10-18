"""Embedding adapters.

HashingEmbedder is deterministic and dependency-free (local dev, CI, offline evals).
SentenceTransformerEmbedder is the neural option for real deployments.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from rootcite.domain.models import ScoredChunk
from rootcite.retrieval.text import tokenize


class HashingEmbedder:
    """Feature hashing over unigrams + bigrams with the signed-hash trick."""

    def __init__(self, dimension: int = 512) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self._dimension, 1.0 if (value >> 63) & 1 else -1.0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            features = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)]
            for feature in features:
                idx, sign = self._bucket(feature)
                out[row, idx] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model: Any = SentenceTransformer(model_name)
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), normalize_embeddings=True, batch_size=64)
        return np.asarray(vectors, dtype=np.float32)


class CrossEncoderReranker:
    """Second-stage reranker: scores (query, passage) pairs jointly for higher precision."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self._model: Any = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        scores = self._model.predict([(query, c.chunk.text) for c in candidates])
        order = np.argsort(-np.asarray(scores))
        return [ScoredChunk(candidates[i].chunk, float(scores[i])) for i in order[:k]]
