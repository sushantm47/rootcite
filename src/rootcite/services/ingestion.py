"""Ingest markdown runbooks and postmortems into the search index (idempotent)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from rootcite.ports import Embedder, SearchIndex
from rootcite.retrieval.chunking import chunk_markdown

log = logging.getLogger(__name__)

_DOC_TYPES = {"runbooks": "runbook", "postmortems": "postmortem"}


@dataclass(frozen=True)
class IngestReport:
    documents: int
    chunks: int


class IngestionService:
    def __init__(self, index: SearchIndex, embedder: Embedder, batch_size: int = 64) -> None:
        self._index = index
        self._embedder = embedder
        self._batch_size = batch_size

    def ingest_directory(self, root: Path) -> IngestReport:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"knowledge directory not found: {root}")

        documents, total = 0, 0
        for path in sorted(root.rglob("*.md")):
            source = path.relative_to(root).as_posix()
            doc_type = _DOC_TYPES.get(path.parent.name, "document")
            text = path.read_text(encoding="utf-8")
            chunks = chunk_markdown(text, source=source, doc_type=doc_type)
            for i in range(0, len(chunks), self._batch_size):
                batch = chunks[i : i + self._batch_size]
                vectors = self._embedder.embed([c.text for c in batch])
                total += self._index.upsert(batch, vectors)
            documents += 1

        log.info("ingestion complete", extra={"documents": documents, "chunks": total})
        return IngestReport(documents, total)
