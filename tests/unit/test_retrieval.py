import numpy as np

from rootcite.adapters.embeddings import HashingEmbedder
from rootcite.adapters.memory_index import InMemoryIndex
from rootcite.domain.models import Chunk, ScoredChunk
from rootcite.retrieval.bm25 import BM25
from rootcite.retrieval.hybrid import HybridRetriever, SearchMode, reciprocal_rank_fusion
from rootcite.retrieval.text import tokenize


def chunk(cid: str, text: str) -> Chunk:
    return Chunk(cid, "doc", f"{cid}.md", cid, text, "runbook")


CHUNKS = [
    chunk("pool", "database connection pool exhausted timeout acquiring connection"),
    chunk("kafka", "kafka consumer lag rebalance max.poll.interval.ms exceeded"),
    chunk("tls", "tls certificate expired x509 handshake failure"),
]


def build_index() -> tuple[InMemoryIndex, HashingEmbedder]:
    embedder = HashingEmbedder()
    index = InMemoryIndex(embedder.dimension)
    index.upsert(CHUNKS, embedder.embed([c.text for c in CHUNKS]))
    return index, embedder


def test_tokenizer_keeps_identifiers():
    tokens = tokenize("The v2.14.0 deploy set db_pool to 5xx")
    assert tokens == ["v2.14.0", "deploy", "set", "db_pool", "5xx"]


def test_bm25_ranks_matching_doc_first():
    bm25 = BM25([tokenize(c.text) for c in CHUNKS])
    scores = bm25.scores(tokenize("certificate expired"))
    assert int(np.argmax(scores)) == 2


def test_hashing_embedder_is_deterministic_and_normalized():
    e = HashingEmbedder()
    a, b = e.embed(["pool exhausted"]), e.embed(["pool exhausted"])
    assert np.allclose(a, b)
    assert abs(float(np.linalg.norm(a[0])) - 1.0) < 1e-5


def test_rrf_rewards_agreement_between_rankings():
    a, b, c = (ScoredChunk(x, 1.0) for x in CHUNKS)
    fused = reciprocal_rank_fusion([[a, b, c], [b, a, c]])
    assert {fused[0].chunk.chunk_id, fused[1].chunk.chunk_id} == {"pool", "kafka"}
    assert fused[-1].chunk.chunk_id == "tls"


def test_upsert_is_idempotent():
    index, embedder = build_index()
    index.upsert(CHUNKS, embedder.embed([c.text for c in CHUNKS]))
    assert index.count() == 3


def test_every_mode_finds_the_right_chunk():
    index, embedder = build_index()
    retriever = HybridRetriever(index, embedder)
    for mode in SearchMode:
        top = retriever.retrieve("kafka consumer lag", k=1, mode=mode)
        assert top[0].chunk.chunk_id == "kafka", mode


def test_empty_query_returns_nothing():
    index, embedder = build_index()
    assert HybridRetriever(index, embedder).retrieve("   ") == []


class ReverseReranker:
    def rerank(self, query, candidates, k):
        return list(reversed(candidates))[:k]


def test_reranker_is_applied_after_fusion():
    index, embedder = build_index()
    retriever = HybridRetriever(index, embedder, reranker=ReverseReranker())
    plain = HybridRetriever(index, embedder).retrieve("kafka consumer lag", k=3)
    reranked = retriever.retrieve("kafka consumer lag", k=3)
    assert reranked[-1].chunk.chunk_id == plain[0].chunk.chunk_id
