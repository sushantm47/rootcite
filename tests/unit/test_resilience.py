import pytest

from rootcite.domain.errors import DependencyUnavailableError, RetryableError
from rootcite.resilience.circuit_breaker import BreakerState, CircuitBreaker
from rootcite.resilience.resilient_llm import ResilientLLM
from rootcite.resilience.retry import backoff_delay, retry_with_backoff

from support import FlakyLLM, text


def raiser(exc: Exception):
    def fn() -> None:
        raise exc

    return fn


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_retry_succeeds_after_transient_errors():
    attempts = []
    sleeps: list[float] = []

    def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise RetryableError("throttled")
        return "ok"

    assert retry_with_backoff(flaky, attempts=3, sleep=sleeps.append, rand=lambda: 1.0) == "ok"
    assert sleeps == [0.25, 0.5]  # exponential growth; rand=1.0 shows the jitter ceiling


def test_retry_gives_up_and_reraises():
    def always_fails() -> None:
        raise RetryableError("down")

    with pytest.raises(RetryableError):
        retry_with_backoff(always_fails, attempts=2, sleep=lambda _: None)


def test_non_retryable_errors_are_not_retried():
    calls = []

    def bad_request() -> None:
        calls.append(1)
        raise ValueError("400")

    with pytest.raises(ValueError):
        retry_with_backoff(bad_request, attempts=5, sleep=lambda _: None)
    assert len(calls) == 1


def test_full_jitter_is_capped():
    assert backoff_delay(10, base=0.25, cap=8.0, rand=lambda: 1.0) == 8.0
    assert backoff_delay(3, base=0.25, cap=8.0, rand=lambda: 0.0) == 0.0


def test_breaker_opens_then_half_opens_then_closes():
    clock = FakeClock()
    breaker = CircuitBreaker("llm", failure_threshold=2, reset_timeout=10, clock=clock)

    def fail() -> None:
        raise RetryableError("5xx")

    for _ in range(2):
        with pytest.raises(RetryableError):
            breaker.call(fail)
    assert breaker.state is BreakerState.OPEN
    with pytest.raises(DependencyUnavailableError):
        breaker.call(lambda: "never called")

    clock.now = 10
    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.call(lambda: "probe ok") == "probe ok"
    assert breaker.state is BreakerState.CLOSED


def test_failed_probe_reopens_breaker():
    clock = FakeClock()
    breaker = CircuitBreaker("llm", failure_threshold=1, reset_timeout=5, clock=clock)
    with pytest.raises(RetryableError):
        breaker.call(raiser(RetryableError("x")))
    clock.now = 5
    with pytest.raises(RetryableError):
        breaker.call(raiser(RetryableError("still down")))
    assert breaker.state is BreakerState.OPEN


def test_caller_errors_do_not_open_breaker():
    breaker = CircuitBreaker("llm", failure_threshold=1)
    with pytest.raises(ValueError):
        breaker.call(raiser(ValueError("bad request")))
    assert breaker.state is BreakerState.CLOSED


def test_resilient_llm_retries_inside_breaker():
    inner = FlakyLLM([RetryableError("429"), RetryableError("529")], text("done"))
    llm = ResilientLLM(inner, CircuitBreaker("claude"), attempts=3, sleep=lambda _: None)
    response = llm.complete(system="s", messages=[], tools=[])
    assert response.text == "done"
    assert inner.attempts == 3
