"""Evals run against a golden dataset of alerts with known answers.

Retrieval eval: hit@k and MRR for keyword, dense, and hybrid search.
Agent eval: root-cause accuracy, grounding rate, tool recall, injection detection,
human-escalation rate, latency percentiles, and token cost.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rootcite.domain.models import Alert
from rootcite.retrieval.hybrid import HybridRetriever, SearchMode
from rootcite.services.diagnosis import DiagnosisService


@dataclass(frozen=True)
class EvalCase:
    alert: Alert
    expected_sources: list[str]
    expected_tools: list[str]
    root_cause_keywords: list[str]
    expect_injection_flag: bool = False


def load_cases(path: Path) -> list[EvalCase]:
    cases = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        raw: dict[str, Any] = json.loads(line)
        cases.append(
            EvalCase(
                alert=Alert(**raw["alert"]),
                expected_sources=raw.get("expected_sources", []),
                expected_tools=raw.get("expected_tools", []),
                root_cause_keywords=[k.lower() for k in raw.get("root_cause_keywords", [])],
                expect_injection_flag=raw.get("expect_injection_flag", False),
            )
        )
    return cases


def run_retrieval_eval(
    retriever: HybridRetriever, cases: list[EvalCase], k: int = 5
) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    scored = [c for c in cases if c.expected_sources]
    if not scored:
        raise ValueError("no cases with expected_sources")
    for mode in SearchMode:
        hits, reciprocal = 0, 0.0
        for case in scored:
            results = retriever.retrieve(case.alert.as_query(), k, mode=mode)
            for rank, r in enumerate(results, start=1):
                if r.chunk.source in case.expected_sources:
                    hits += 1
                    reciprocal += 1 / rank
                    break
        report[mode.value] = {
            f"hit@{k}": round(hits / len(scored), 3),
            "mrr": round(reciprocal / len(scored), 3),
        }
    return report


@dataclass(frozen=True)
class AgentEvalReport:
    cases: int
    root_cause_accuracy: float
    grounded_rate: float
    tool_recall: float
    injection_detection_rate: float | None
    needs_human_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    avg_tokens: float
    est_cost_per_diagnosis_usd: float
    failures: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def run_agent_eval(
    service: DiagnosisService,
    cases: list[EvalCase],
    usd_per_mtok_in: float = 3.0,
    usd_per_mtok_out: float = 15.0,
) -> AgentEvalReport:
    correct, grounded, human = 0, 0, 0
    tool_recalls: list[float] = []
    injection_hits, injection_cases = 0, 0
    latencies: list[float] = []
    tokens: list[int] = []
    cost = 0.0
    failures: list[str] = []

    for case in cases:
        d = service.diagnose(case.alert, use_cache=False)
        text = d.root_cause.lower()
        ok = any(k in text for k in case.root_cause_keywords) if case.root_cause_keywords else True
        correct += ok
        grounded += d.grounded
        human += d.needs_human
        if not ok:
            failures.append(f"{case.alert.alert_id}: root cause missed ({d.root_cause[:80]})")
        if case.expected_tools:
            called = set(d.tools_called)
            tool_recalls.append(len(called & set(case.expected_tools)) / len(case.expected_tools))
        if case.expect_injection_flag:
            injection_cases += 1
            if any("prompt injection" in w for w in d.warnings):
                injection_hits += 1
            else:
                failures.append(f"{case.alert.alert_id}: injection not flagged")
        latencies.append(d.latency_ms)
        tokens.append(d.usage.total)
        cost += (
            d.usage.input_tokens * usd_per_mtok_in + d.usage.output_tokens * usd_per_mtok_out
        ) / 1_000_000

    n = max(len(cases), 1)
    return AgentEvalReport(
        cases=len(cases),
        root_cause_accuracy=round(correct / n, 3),
        grounded_rate=round(grounded / n, 3),
        tool_recall=round(statistics.fmean(tool_recalls), 3) if tool_recalls else 0.0,
        injection_detection_rate=(
            round(injection_hits / injection_cases, 3) if injection_cases else None
        ),
        needs_human_rate=round(human / n, 3),
        latency_p50_ms=round(_percentile(latencies, 50), 1),
        latency_p95_ms=round(_percentile(latencies, 95), 1),
        avg_tokens=round(statistics.fmean(tokens), 1) if tokens else 0.0,
        est_cost_per_diagnosis_usd=round(cost / n, 5),
        failures=failures,
    )


def retrieval_markdown(report: dict[str, dict[str, float]]) -> str:
    metrics = list(next(iter(report.values())))
    lines = ["| mode | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
    for mode, values in report.items():
        lines.append(f"| {mode} | " + " | ".join(f"{values[m]:.3f}" for m in metrics) + " |")
    return "\n".join(lines)
