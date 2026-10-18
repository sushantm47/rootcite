"""Tokenization shared by BM25 and the hashing embedder."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was "
    "were will with when what which who how why do does did not no if then than so".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase tokens that keep identifiers like ``5xx``, ``db_pool`` and ``v2.14.0`` intact."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
