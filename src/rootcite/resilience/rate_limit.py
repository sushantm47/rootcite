"""Per-client sliding-window rate limiter for the public demo.

Keeps one deque of request times per client key (an IP address). Memory is bounded: idle
keys are pruned, and the number of tracked keys is capped. A single process is enough for
a demo; a multi-instance service would keep these counters in Redis instead.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable


class SlidingWindowLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self._limit = limit
        self._window = window_seconds
        self._max_keys = max_keys
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, float]:
        """Record a request. Returns (allowed, seconds until the next request is allowed)."""
        now = self._clock()
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                if len(self._hits) >= self._max_keys:
                    self._prune(now)
                hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                return False, max(0.0, self._window - (now - hits[0]))
            hits.append(now)
            return True, 0.0

    def _prune(self, now: float) -> None:
        idle = [k for k, h in self._hits.items() if not h or now - h[-1] >= self._window]
        for k in idle:
            del self._hits[k]
        # Still full (a flood of distinct keys): drop the oldest half rather than grow.
        if len(self._hits) >= self._max_keys:
            oldest = sorted(self._hits, key=lambda k: self._hits[k][-1])
            for k in oldest[: len(oldest) // 2]:
                del self._hits[k]

    def __len__(self) -> int:
        with self._lock:
            return len(self._hits)


class DailyBudget:
    """A global allowance of expensive operations per UTC day, such as live model runs."""

    def __init__(self, limit: int, clock: Callable[[], float] = time.time) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self._limit = limit
        self._clock = clock
        self._day = -1
        self._used = 0
        self._lock = threading.Lock()

    def _roll(self) -> None:
        day = int(self._clock() // 86_400)
        if day != self._day:
            self._day, self._used = day, 0

    def try_consume(self) -> bool:
        with self._lock:
            self._roll()
            if self._used >= self._limit:
                return False
            self._used += 1
            return True

    def remaining(self) -> int:
        with self._lock:
            self._roll()
            return self._limit - self._used

    def seconds_until_reset(self) -> float:
        return 86_400 - (self._clock() % 86_400)
