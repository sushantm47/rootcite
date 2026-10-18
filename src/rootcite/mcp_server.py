"""Model Context Protocol (MCP) server.

Exposes RootCite's read-only tools and the diagnose use case to any MCP client
(Claude Desktop, Claude Code, IDE agents). Run:  python -m rootcite.mcp_server
"""

from __future__ import annotations

import json
from typing import Any

from rootcite.config import Settings
from rootcite.container import Container, build_container
from rootcite.domain.models import Alert


def build_server(container: Container) -> Any:
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("rootcite")

    @server.tool()
    def search_runbooks(query: str, k: int = 5) -> str:
        """Search runbooks and postmortems with hybrid (keyword + semantic) retrieval."""
        results = container.retriever.retrieve(query, min(max(k, 1), 10))
        return json.dumps(
            [{"source": r.chunk.source, "section": r.chunk.title, "text": r.chunk.text}
             for r in results],
            indent=1,
        )

    @server.tool()
    def get_metric(service: str, metric: str, window_minutes: int = 60) -> str:
        """Summarize a service metric over the last N minutes."""
        return json.dumps(container.telemetry.get_metric(service, metric, window_minutes))

    @server.tool()
    def list_recent_deploys(service: str, hours: int = 24) -> str:
        """List deployments to a service in the last N hours."""
        return json.dumps(container.telemetry.recent_deploys(service, hours))

    @server.tool()
    def diagnose_alert(service: str, title: str, description: str = "") -> str:
        """Run the full RootCite investigation for an alert and return the diagnosis."""
        alert = Alert(alert_id=f"mcp-{service}", service=service, title=title,
                      description=description)
        d = container.diagnosis.diagnose(alert)
        return json.dumps(
            {
                "root_cause": d.root_cause,
                "confidence": d.confidence,
                "citations": d.citations,
                "recommended_actions": d.recommended_actions,
                "needs_human": d.needs_human,
                "warnings": d.warnings,
            },
            indent=1,
        )

    return server


def main() -> None:
    settings = Settings.from_env()
    container = build_container(settings)
    if container.index.count() == 0:
        container.ingestion.ingest_directory(settings.knowledge_dir)
    build_server(container).run()


if __name__ == "__main__":
    main()
