"""Okapi BM25 over pre-tokenized documents (used by the in-memory index)."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np


class BM25:
    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._tf = [Counter(doc) for doc in corpus]
        self._len = np.array([len(doc) for doc in corpus], dtype=np.float64)
        self._avg_len = float(self._len.mean()) if len(corpus) else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(corpus)
        # BM25+ style idf: always positive, so common terms never subtract score.
        self._idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: Sequence[str]) -> np.ndarray:
        out = np.zeros(len(self._tf), dtype=np.float64)
        if not self._tf:
            return out
        norm = self.k1 * (1 - self.b + self.b * self._len / max(self._avg_len, 1e-9))
        for term in set(query):
            idf = self._idf.get(term)
            if idf is None:
                continue
            tf = np.array([doc_tf.get(term, 0) for doc_tf in self._tf], dtype=np.float64)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out
