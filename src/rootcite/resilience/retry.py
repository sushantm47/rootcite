"""Exponential backoff with full jitter.

Full jitter (sleep = random(0, min(cap, base * 2**attempt))) spreads retries out so many
clients recovering at once don't hammer a dependency in synchronized waves.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

from rootcite.domain.errors import RetryableError

T = TypeVar("T")
log = logging.getLogger(__name__)


def backoff_delay(
    attempt: int, base: float, cap: float, rand: Callable[[], float] = random.random
) -> float:
    return rand() * min(cap, base * (2**attempt))


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.25,
    max_delay: float = 8.0,
    retry_on: tuple[type[BaseException], ...] = (RetryableError,),
    sleep: Callable[[float], None] = time.sleep,
    rand: Callable[[], float] = random.random,
) -> T:
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    for attempt in range(attempts):
        try:
            return fn()
        except retry_on as exc:
            if attempt == attempts - 1:
                raise
            delay = backoff_delay(attempt, base_delay, max_delay, rand)
            log.warning(
                "retrying after transient error",
                extra={"attempt": attempt + 1, "delay_s": round(delay, 3), "error": str(exc)},
            )
            sleep(delay)
    raise AssertionError("unreachable")
