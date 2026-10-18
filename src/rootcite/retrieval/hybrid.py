"""Hybrid retrieval: lexical + dense search fused with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from rootcite.domain.models import ScoredChunk
from rootcite.ports import Embedder, Reranker, SearchIndex

RRF_K = 60


class SearchMode(StrEnum):
    KEYWORD = "keyword"
    DENSE = "dense"
    HYBRID = "hybrid"


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], k: int = RRF_K
) -> list[ScoredChunk]:
    """score(d) = sum over rankings of 1 / (k + rank). Scale-free, so no score calibration."""
    scores: dict[str, float] = {}
    chunks: dict[str, ScoredChunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            cid = item.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunks.setdefault(cid, item)
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [ScoredChunk(chunks[cid].chunk, scores[cid]) for cid in ordered]


class HybridRetriever:
    def __init__(
        self,
        index: SearchIndex,
        embedder: Embedder,
        reranker: Reranker | None = None,
        candidates: int = 30,
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._reranker = reranker
        self._candidates = candidates

    def retrieve(
        self, query: str, k: int = 5, mode: SearchMode = SearchMode.HYBRID
    ) -> list[ScoredChunk]:
        if not query.strip():
            return []
        rankings: list[list[ScoredChunk]] = []
        if mode in (SearchMode.KEYWORD, SearchMode.HYBRID):
            rankings.append(self._index.keyword_search(query, self._candidates))
        if mode in (SearchMode.DENSE, SearchMode.HYBRID):
            vector = self._embedder.embed([query])[0]
            rankings.append(self._index.dense_search(vector, self._candidates))

        fused = rankings[0] if len(rankings) == 1 else reciprocal_rank_fusion(rankings)
        if self._reranker is not None:
            return self._reranker.rerank(query, fused[: self._candidates], k)
        return fused[:k]
