"""Markdown-aware chunking with contextual headers.

Each chunk is prefixed with its document title and heading path ("contextual chunk
headers"), so a chunk that only says "increase the pool size" still retrieves for a
query about database connection exhaustion.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from rootcite.domain.models import Chunk

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass(frozen=True)
class _Section:
    path: tuple[str, ...]
    body: str


def doc_id_for(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def _sections(markdown: str) -> list[_Section]:
    sections: list[_Section] = []
    path: list[str] = []
    body: list[str] = []
    in_code = False

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            sections.append(_Section(tuple(path), text))
        body.clear()

    for line in markdown.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            del path[level - 1 :]
            path.extend([""] * (level - 1 - len(path)))
            path.append(match.group(2))
        else:
            body.append(line)
    flush()
    return sections


def _windows(text: str, max_chars: int, overlap: int) -> list[str]:
    """Pack paragraphs up to max_chars; hard-split any paragraph that is still too long."""
    pieces: list[str] = []
    current = ""
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        if len(para) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            step = max_chars - overlap
            pieces.extend(para[i : i + max_chars] for i in range(0, len(para), step))
            continue
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > max_chars:
            pieces.append(current)
            current = para
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(
    markdown: str, *, source: str, doc_type: str, max_chars: int = 1200, overlap: int = 150
) -> list[Chunk]:
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be >= 0 and smaller than max_chars")
    doc_id = doc_id_for(source)
    sections = _sections(markdown)
    doc_title = next((s.path[0] for s in sections if s.path and s.path[0]), source)

    chunks: list[Chunk] = []
    for section in sections:
        heading = " > ".join(p for p in section.path if p) or doc_title
        for piece in _windows(section.body, max_chars, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-{len(chunks):03d}",
                    doc_id=doc_id,
                    source=source,
                    title=heading,
                    text=f"{heading}\n{piece}",
                    doc_type=doc_type,
                )
            )
    return chunks
