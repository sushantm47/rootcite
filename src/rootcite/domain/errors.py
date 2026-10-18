"""Domain exceptions. Adapters translate vendor errors into these."""


class RootCiteError(Exception):
    """Base class for all RootCite errors."""


class InvalidRequestError(RootCiteError):
    """The caller sent something we cannot process. Maps to HTTP 400."""


class RetryableError(RootCiteError):
    """A transient dependency failure (throttling, timeout, 5xx). Safe to retry."""


class DependencyUnavailableError(RootCiteError):
    """A dependency is failing and its circuit breaker is open. Maps to HTTP 503."""

    def __init__(self, dependency: str, retry_after_seconds: float) -> None:
        super().__init__(f"{dependency} is unavailable; retry in {retry_after_seconds:.0f}s")
        self.dependency = dependency
        self.retry_after_seconds = retry_after_seconds


class ConfigurationError(RootCiteError):
    """Settings are invalid for the current stage."""


class LLMRequestError(RootCiteError):
    """The model provider rejected the request (bad model name, invalid key). Not retried."""


class QuotaExceededError(RootCiteError):
    """A usage budget is spent (for example the public demo's daily runs). Maps to HTTP 429."""

    def __init__(self, message: str, retry_after_seconds: float) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
