"""PostgreSQL index: pgvector HNSW for dense search + tsvector/GIN for keyword search.

One database gives hybrid search, transactional upserts, and the operational maturity of
RDS/Aurora (backups, IAM, Multi-AZ) without running a separate vector database.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from rootcite.domain.models import Chunk, ScoredChunk

_COLUMNS = "chunk_id, doc_id, source, title, doc_type, text"


def _vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vector) + "]"


def _row_to_scored(row: Sequence[Any]) -> ScoredChunk:
    chunk_id, doc_id, source, title, doc_type, text, score = row
    return ScoredChunk(Chunk(chunk_id, doc_id, source, title, text, doc_type), float(score))


class PgVectorIndex:
    def __init__(self, dsn: str, dimension: int, pool_size: int = 10) -> None:
        from psycopg_pool import ConnectionPool

        self._dimension = int(dimension)
        self._pool: Any = ConnectionPool(dsn, min_size=1, max_size=pool_size, open=True)
        self._migrate()

    def _migrate(self) -> None:
        # Dimension is an int we control, so interpolating it into DDL is safe.
        ddl = f"""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id   TEXT PRIMARY KEY,
            doc_id     TEXT NOT NULL,
            source     TEXT NOT NULL,
            title      TEXT NOT NULL,
            doc_type   TEXT NOT NULL,
            text       TEXT NOT NULL,
            embedding  vector({self._dimension}) NOT NULL,
            tsv        tsvector GENERATED ALWAYS AS
                           (to_tsvector('english', title || ' ' || text)) STORED,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
            ON chunks USING hnsw (embedding vector_cosine_ops);
        CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
        """
        with self._pool.connection() as conn:
            conn.execute(ddl)

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        rows = [
            (c.chunk_id, c.doc_id, c.source, c.title, c.doc_type, c.text, _vector_literal(v))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        sql = f"""
        INSERT INTO chunks ({_COLUMNS}, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (chunk_id) DO UPDATE SET
            doc_id = EXCLUDED.doc_id, source = EXCLUDED.source, title = EXCLUDED.title,
            doc_type = EXCLUDED.doc_type, text = EXCLUDED.text,
            embedding = EXCLUDED.embedding, updated_at = now()
        """
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(sql, rows)
        return len(rows)

    def dense_search(self, vector: np.ndarray, k: int) -> list[ScoredChunk]:
        literal = _vector_literal(vector)
        sql = f"""
        SELECT {_COLUMNS}, 1 - (embedding <=> %s::vector) AS score
        FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s
        """
        with self._pool.connection() as conn:
            rows = conn.execute(sql, (literal, literal, k)).fetchall()
        return [_row_to_scored(r) for r in rows]

    def keyword_search(self, query: str, k: int) -> list[ScoredChunk]:
        # OR the sanitized terms: alert text is long, and AND semantics would match nothing.
        terms = re.findall(r"[a-z0-9]+", query.lower())[:32]
        if not terms:
            return []
        sql = f"""
        SELECT {_COLUMNS}, ts_rank_cd(tsv, q) AS score
        FROM chunks, to_tsquery('english', %s) AS q
        WHERE tsv @@ q ORDER BY score DESC LIMIT %s
        """
        with self._pool.connection() as conn:
            rows = conn.execute(sql, (" | ".join(terms), k)).fetchall()
        return [_row_to_scored(r) for r in rows]

    def count(self) -> int:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT count(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0
