"""Semantic cache for diagnoses.

During an incident the same alert fires repeatedly (every pod, every region). If a new
alert is near-identical to a recent one for the SAME service, we return the cached
diagnosis instead of spending another agent run. Scoping by service prevents a
checkout-api answer from being served for payments-api.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rootcite.domain.models import Alert, Diagnosis
from rootcite.ports import Embedder


@dataclass
class _Entry:
    vector: np.ndarray
    diagnosis: Diagnosis
    stored_at: float


class SemanticCache:
    def __init__(
        self,
        embedder: Embedder,
        threshold: float = 0.95,
        ttl_seconds: float = 900,
        max_entries: int = 1000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._ttl = ttl_seconds
        self._max = max_entries
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, _Entry] = OrderedDict()  # key: service|alert_id

    @staticmethod
    def _text(alert: Alert) -> str:
        return f"{alert.title} {alert.description}"

    def get(self, alert: Alert) -> Diagnosis | None:
        vector = self._embedder.embed([self._text(alert)])[0]
        now = self._clock()
        with self._lock:
            best_key: str | None = None
            best_sim = -1.0
            for key, entry in list(self._entries.items()):
                if now - entry.stored_at > self._ttl:
                    del self._entries[key]
                    continue
                if entry.diagnosis.service != alert.service:
                    continue
                sim = float(entry.vector @ vector)
                if sim > best_sim:
                    best_key, best_sim = key, sim
            if best_key is None or best_sim < self._threshold:
                return None
            self._entries.move_to_end(best_key)
            return self._entries[best_key].diagnosis

    def put(self, alert: Alert, diagnosis: Diagnosis) -> None:
        vector = self._embedder.embed([self._text(alert)])[0]
        with self._lock:
            key = f"{alert.service}|{alert.alert_id}"
            self._entries[key] = _Entry(vector, diagnosis, self._clock())
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
