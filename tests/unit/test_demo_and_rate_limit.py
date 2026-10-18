import pytest

from rootcite.api.demo import SCENARIOS, client_key
from rootcite.config import Settings
from rootcite.domain.errors import ConfigurationError
from rootcite.resilience.rate_limit import SlidingWindowLimiter

from support import TELEMETRY_DIR


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_limiter_allows_up_to_the_limit_then_reports_wait():
    clock = Clock()
    limiter = SlidingWindowLimiter(3, window_seconds=60, clock=clock)
    assert [limiter.check("ip")[0] for _ in range(3)] == [True, True, True]
    allowed, wait = limiter.check("ip")
    assert not allowed and wait == pytest.approx(60)
    assert limiter.check("other-ip")[0], "limits are per visitor"
    clock.now = 60
    assert limiter.check("ip")[0], "the window slides"


def test_limiter_memory_is_bounded():
    clock = Clock()
    limiter = SlidingWindowLimiter(1, window_seconds=60, max_keys=100, clock=clock)
    for i in range(1000):
        clock.now = i * 0.01
        limiter.check(f"ip-{i}")
    assert len(limiter) <= 100


def test_client_key_prefers_the_forwarded_client():
    assert client_key({"x-forwarded-for": "1.2.3.4, 10.0.0.1"}, "10.0.0.1") == "1.2.3.4"
    assert client_key({}, "9.9.9.9") == "9.9.9.9"
    assert client_key({}, None) == "unknown"


def test_every_scenario_has_telemetry_to_investigate():
    for s in SCENARIOS:
        assert (TELEMETRY_DIR / f"{s['service']}.json").is_file(), s["service"]
        assert s["title"] and len(s["title"]) <= 300


def test_a_key_alone_selects_the_free_compatible_api_and_stays_out_of_logs():
    s = Settings.from_env({"ROOTCITE_LLM_API_KEY": "gsk_secret"})
    assert s.llm_provider == "openai" and s.model == "openai/gpt-oss-20b"
    assert "gsk_secret" not in repr(s)
    assert Settings.from_env({}).llm_provider == "offline"


def test_public_demo_defaults_to_a_rate_limit():
    assert Settings.from_env({"ROOTCITE_PUBLIC_DEMO": "true"}).rate_limit_per_minute == 10
    assert Settings.from_env({}).rate_limit_per_minute == 0
    with pytest.raises(ConfigurationError):
        Settings.from_env({"ROOTCITE_LLM_PROVIDER": "openai"})  # no key
