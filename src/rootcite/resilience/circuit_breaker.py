"""Circuit breaker: stop calling a dependency that keeps failing, then probe for recovery.

CLOSED --(failure_threshold consecutive failures)--> OPEN
OPEN   --(reset_timeout elapsed)--> HALF_OPEN (one trial call)
Only dependency failures (RetryableError by default) count toward opening the breaker.
HALF_OPEN --success--> CLOSED, --failure--> OPEN
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from enum import StrEnum
from typing import TypeVar

from rootcite.domain.errors import DependencyUnavailableError, RetryableError

T = TypeVar("T")


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        failure_on: tuple[type[BaseException], ...] = (RetryableError,),
    ) -> None:
        self.name = name
        self._failure_on = failure_on
        self._threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._clock = clock
        self._lock = threading.Lock()
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._trial_in_flight = False

    @property
    def state(self) -> BreakerState:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _maybe_half_open(self) -> None:
        if (
            self._state is BreakerState.OPEN
            and self._clock() - self._opened_at >= self._reset_timeout
        ):
            self._state = BreakerState.HALF_OPEN
            self._trial_in_flight = False

    def call(self, fn: Callable[[], T]) -> T:
        with self._lock:
            self._maybe_half_open()
            if self._state is BreakerState.OPEN or (
                self._state is BreakerState.HALF_OPEN and self._trial_in_flight
            ):
                remaining = self._reset_timeout - (self._clock() - self._opened_at)
                raise DependencyUnavailableError(self.name, max(remaining, 1.0))
            if self._state is BreakerState.HALF_OPEN:
                self._trial_in_flight = True

        try:
            result = fn()
        except self._failure_on:
            self._record_failure()
            raise
        except BaseException:
            # Caller errors (e.g. HTTP 400) say nothing about dependency health.
            self._release_trial()
            raise
        self._record_success()
        return result

    def _release_trial(self) -> None:
        with self._lock:
            self._trial_in_flight = False

    def _record_success(self) -> None:
        with self._lock:
            self._state = BreakerState.CLOSED
            self._failures = 0
            self._trial_in_flight = False

    def _record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state is BreakerState.HALF_OPEN or self._failures >= self._threshold:
                self._state = BreakerState.OPEN
                self._opened_at = self._clock()
                self._trial_in_flight = False
