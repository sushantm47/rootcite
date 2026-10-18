"""Pure helpers for the public demo: sample incidents and client identification."""

from __future__ import annotations

from collections.abc import Mapping

# One incident per service in data/telemetry. Each matches a runbook and has live signals
# (deploys, metrics, logs) for the agent to investigate. checkout-api's logs also contain a
# planted prompt-injection attempt, to show the guardrails.
SCENARIOS: list[dict[str, str]] = [
    {
        "service": "checkout-api",
        "title": "p99 latency above 2s and 5xx errors rising",
        "description": "Latency and 5xx alarms fired within 10 minutes of each other.",
    },
    {
        "service": "payments-worker",
        "title": "Consumer lag growing on settlement-consumers",
        "description": "Lag has increased for 60 minutes and settlement is delayed.",
    },
    {
        "service": "search-service",
        "title": "Pods restarting repeatedly, OOMKilled",
        "description": "Restart count increasing; memory near limit.",
    },
    {
        "service": "auth-gateway",
        "title": "TLS handshake failures on public endpoint",
        "description": "Clients report certificate errors; error rate at 40%.",
    },
    {
        "service": "notification-service",
        "title": "Health checks failing, writes erroring",
        "description": "Errors mention no space left on device.",
    },
]


def client_key(headers: Mapping[str, str], peer: str | None) -> str:
    """The visitor's address for rate limiting. Behind a proxy (Hugging Face, a load
    balancer) the real client is the first X-Forwarded-For entry. This can be spoofed, which
    is acceptable for a demo limiter; production would trust only its own proxy's header."""
    forwarded = headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip()
    return first or peer or "unknown"
