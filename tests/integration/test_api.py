"""HTTP contract tests. Skipped automatically if FastAPI isn't installed."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from rootcite.api.app import create_app  # noqa: E402
from rootcite.domain.errors import DependencyUnavailableError  # noqa: E402

BODY = {
    "alert_id": "api-1",
    "service": "checkout-api",
    "title": "p99 latency above 2s and 5xx errors rising",
}


@pytest.fixture
def client(make_container):
    return TestClient(create_app(make_container()))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["indexed_chunks"] > 0


def test_diagnose_returns_grounded_answer_and_request_id(client):
    r = client.post("/v1/diagnoses", json=BODY, headers={"x-request-id": "req-123"})
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] and body["citations"]
    assert r.headers["x-request-id"] == "req-123"


@pytest.mark.parametrize(
    "patch",
    [{"service": "Checkout API!"}, {"title": ""}, {"severity": "SEV9"}],
)
def test_request_validation(client, patch):
    assert client.post("/v1/diagnoses", json={**BODY, **patch}).status_code == 422


def test_open_circuit_maps_to_503_with_retry_after(make_container):
    container = make_container()

    def unavailable(alert, use_cache=True):
        raise DependencyUnavailableError("claude", 12)

    container.diagnosis.diagnose = unavailable
    r = TestClient(create_app(container)).post("/v1/diagnoses", json=BODY)
    assert r.status_code == 503 and r.headers["retry-after"] == "12"


# ----- public demo -----

import io  # noqa: E402

from rootcite.config import Settings  # noqa: E402
from rootcite.container import build_container  # noqa: E402
from rootcite.domain.errors import RetryableError  # noqa: E402

from support import KNOWLEDGE_DIR, TELEMETRY_DIR  # noqa: E402


def demo_client(rate):
    settings = Settings(
        knowledge_dir=KNOWLEDGE_DIR, telemetry_dir=TELEMETRY_DIR,
        public_demo=True, rate_limit_per_minute=rate, repo_url="https://github.com/x/rootcite",
    )
    return TestClient(create_app(build_container(settings, metrics_stream=io.StringIO())))


def test_demo_page_and_scenarios_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "Run diagnosis" in page.text
    scenarios = client.get("/v1/scenarios").json()
    assert len(scenarios) == 5 and scenarios[0]["service"] == "checkout-api"


def test_health_describes_the_model_but_never_the_key(client):
    body = client.get("/health").json()
    assert body["llm_provider"] == "offline" and body["model"] == "offline planner"
    assert body["public_demo"] is False
    assert not any("key" in k for k in body)


def test_public_demo_blocks_ingestion_and_rate_limits_per_visitor():
    client = demo_client(rate=3)
    assert client.get("/health").json()["repo_url"] == "https://github.com/x/rootcite"
    assert client.post("/v1/ingest").status_code == 403  # counts as request 1
    assert client.post("/v1/diagnoses", json=BODY).status_code == 200  # 2
    assert client.post("/v1/diagnoses", json=BODY).status_code == 200  # 3
    limited = client.post("/v1/diagnoses", json=BODY)
    assert limited.status_code == 429 and int(limited.headers["retry-after"]) >= 1
    other = client.post("/v1/diagnoses", json=BODY, headers={"x-forwarded-for": "203.0.113.9"})
    assert other.status_code == 200, "another visitor has their own limit"
    assert client.get("/health").status_code == 200, "reads are never limited"


def test_transient_model_failures_return_503(make_container):
    container = make_container()

    def flaky(alert, use_cache=True):
        raise RetryableError("provider returned 429")

    container.diagnosis.diagnose = flaky
    r = TestClient(create_app(container)).post("/v1/diagnoses", json=BODY)
    assert r.status_code == 503 and r.headers["retry-after"] == "30"
