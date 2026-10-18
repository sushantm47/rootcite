"""Diagnose-alert use case: semantic cache -> agent -> metrics."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace

from rootcite.agent.graph import IncidentAgent
from rootcite.domain.errors import InvalidRequestError, QuotaExceededError
from rootcite.domain.models import Alert, Diagnosis
from rootcite.observability.metrics import MetricsLogger
from rootcite.resilience.rate_limit import DailyBudget
from rootcite.resilience.semantic_cache import SemanticCache

log = logging.getLogger(__name__)

MetricsFactory = Callable[[str], MetricsLogger]


class DiagnosisService:
    def __init__(
        self,
        agent: IncidentAgent,
        cache: SemanticCache | None = None,
        metrics_factory: MetricsFactory = MetricsLogger,
        run_budget: DailyBudget | None = None,
    ) -> None:
        self._agent = agent
        self._cache = cache
        self._metrics_factory = metrics_factory
        self._run_budget = run_budget

    def diagnose(self, alert: Alert, use_cache: bool = True) -> Diagnosis:
        if not alert.title.strip():
            raise InvalidRequestError("alert title must not be empty")

        metrics = self._metrics_factory("Diagnose")
        metrics.set_property("alert_id", alert.alert_id)
        metrics.set_property("service", alert.service)
        try:
            # "is not None", not truthiness: an empty cache has len() == 0 and is falsy.
            cache = self._cache if use_cache else None
            cached = cache.get(alert) if cache is not None else None
            if cached is not None:
                metrics.put("CacheHit", 1)
                log.info("semantic cache hit", extra={"alert_id": alert.alert_id})
                return replace(cached, alert_id=alert.alert_id, cached=True, latency_ms=0.0)

            metrics.put("CacheHit", 0)
            # Cache hits are free; only live agent runs spend the daily budget.
            if self._run_budget is not None and not self._run_budget.try_consume():
                metrics.put("BudgetExhausted", 1)
                raise QuotaExceededError(
                    "The daily budget of live runs is spent.",
                    self._run_budget.seconds_until_reset(),
                )
            diagnosis = self._agent.run(alert)
            # Only cache answers we'd be comfortable serving again without a human.
            if cache is not None and diagnosis.grounded and not diagnosis.blocked_actions:
                cache.put(alert, diagnosis)

            metrics.put("Latency", diagnosis.latency_ms, "Milliseconds")
            metrics.put("InputTokens", diagnosis.usage.input_tokens)
            metrics.put("OutputTokens", diagnosis.usage.output_tokens)
            metrics.put("ToolCalls", len(diagnosis.tools_called))
            metrics.put("Grounded", int(diagnosis.grounded))
            metrics.put("NeedsHuman", int(diagnosis.needs_human))
            metrics.put("BlockedActions", len(diagnosis.blocked_actions))
            metrics.put("Confidence", diagnosis.confidence, "None")
            return diagnosis
        except Exception:
            metrics.put("Fault", 1)
            raise
        finally:
            metrics.flush()
