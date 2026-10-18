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


# ----- daily budget -----

import io  # noqa: E402

from rootcite.container import build_container  # noqa: E402
from rootcite.domain.errors import QuotaExceededError  # noqa: E402
from rootcite.domain.models import Alert  # noqa: E402
from rootcite.resilience.rate_limit import DailyBudget  # noqa: E402

from support import KNOWLEDGE_DIR  # noqa: E402


def test_daily_budget_resets_at_midnight_utc():
    clock = Clock()
    clock.now = 86_400 * 10 + 3_600  # 01:00 UTC
    budget = DailyBudget(2, clock=clock)
    assert budget.try_consume() and budget.try_consume()
    assert not budget.try_consume() and budget.remaining() == 0
    assert budget.seconds_until_reset() == pytest.approx(23 * 3_600)
    clock.now = 86_400 * 11 + 5
    assert budget.remaining() == 2 and budget.try_consume()


def test_cache_hits_do_not_spend_the_daily_budget():
    settings = Settings(knowledge_dir=KNOWLEDGE_DIR, telemetry_dir=TELEMETRY_DIR, daily_run_limit=1)
    container = build_container(settings, metrics_stream=io.StringIO())
    container.ingestion.ingest_directory(settings.knowledge_dir)
    alert = Alert("a1", "checkout-api", "p99 latency above 2s and 5xx errors rising")

    first = container.diagnosis.diagnose(alert)
    repeat = container.diagnosis.diagnose(Alert("a2", alert.service, alert.title))
    assert not first.cached and repeat.cached, "the repeat is served from cache"
    assert container.run_budget.remaining() == 0

    with pytest.raises(QuotaExceededError) as err:
        container.diagnosis.diagnose(Alert("a3", "auth-gateway", "TLS handshake failures"))
    assert err.value.retry_after_seconds > 0


def test_demo_settings_parse_origins_and_default_budget():
    s = Settings.from_env({
        "ROOTCITE_PUBLIC_DEMO": "true",
        "ROOTCITE_ALLOWED_ORIGINS": "https://sushantm47.github.io/, https://example.com",
    })
    assert s.allowed_origins == ("https://sushantm47.github.io", "https://example.com")
    assert s.daily_run_limit == 100
    assert Settings.from_env({}).daily_run_limit == 0
