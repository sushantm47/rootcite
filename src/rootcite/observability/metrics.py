"""CloudWatch Embedded Metric Format (EMF).

Metrics are written as structured JSON log lines. On ECS/Lambda with the CloudWatch agent,
CloudWatch extracts them into real metrics asynchronously, so emitting a metric never
adds a network call to the request path.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, TextIO

NAMESPACE = "RootCite"


class MetricsLogger:
    def __init__(
        self,
        operation: str,
        stream: TextIO | None = None,
        namespace: str = NAMESPACE,
        dimensions: dict[str, str] | None = None,
    ) -> None:
        self._stream = stream
        self._namespace = namespace
        self._dimensions = {"Operation": operation, **(dimensions or {})}
        self._metrics: dict[str, tuple[float, str]] = {}
        self._properties: dict[str, Any] = {}

    def put(self, name: str, value: float, unit: str = "Count") -> None:
        self._metrics[name] = (float(value), unit)

    def set_property(self, key: str, value: Any) -> None:
        """High-cardinality context (alert_id, request_id) goes here, never in dimensions."""
        self._properties[key] = value

    def to_emf(self) -> dict[str, Any]:
        return {
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [
                    {
                        "Namespace": self._namespace,
                        "Dimensions": [sorted(self._dimensions)],
                        "Metrics": [
                            {"Name": name, "Unit": unit}
                            for name, (_, unit) in sorted(self._metrics.items())
                        ],
                    }
                ],
            },
            **self._dimensions,
            **self._properties,
            **{name: value for name, (value, _) in self._metrics.items()},
        }

    def flush(self) -> None:
        if not self._metrics:
            return
        stream = self._stream or sys.stdout
        stream.write(json.dumps(self.to_emf(), default=str) + "\n")
        stream.flush()
        self._metrics.clear()
