"""Telemetry adapter backed by JSON fixtures (one file per service).

Production would implement the same TelemetrySource port with CloudWatch Metrics,
CloudWatch Logs Insights, and the deployment pipeline's API.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from rootcite.retrieval.text import tokenize


class FixtureTelemetry:
    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _load(self, service: str) -> dict[str, Any]:
        with self._lock:
            if service not in self._cache:
                path = (self._root / f"{service}.json").resolve()
                # Reject path traversal: the file must sit directly inside the fixture root.
                ok = path.is_file() and path.parent == self._root
                self._cache[service] = json.loads(path.read_text()) if ok else {}
            return self._cache[service]

    def get_metric(self, service: str, metric: str, window_minutes: int) -> dict[str, Any]:
        metrics = self._load(service).get("metrics", {})
        if metric not in metrics:
            return {"error": f"unknown metric '{metric}'", "available": sorted(metrics)}
        series = metrics[metric]
        points = [float(p) for p in series["points"]][-max(1, window_minutes) :]
        return {
            "service": service,
            "metric": metric,
            "unit": series.get("unit", ""),
            "window_minutes": len(points),
            "summary": {
                "min": min(points),
                "max": max(points),
                "avg": round(sum(points) / len(points), 3),
                "latest": points[-1],
                "baseline": series.get("baseline"),
            },
        }

    def search_logs(self, service: str, query: str, limit: int) -> list[dict[str, Any]]:
        terms = set(tokenize(query))
        logs: list[dict[str, Any]] = self._load(service).get("logs", [])
        matches = [
            line
            for line in logs
            if not terms
            or terms & set(tokenize(f"{line.get('level', '')} {line.get('message', '')}"))
        ]
        return matches[: max(1, limit)]

    def recent_deploys(self, service: str, hours: int) -> list[dict[str, Any]]:
        deploys: list[dict[str, Any]] = self._load(service).get("deploys", [])
        return [d for d in deploys if d.get("minutes_ago", 0) <= hours * 60]
