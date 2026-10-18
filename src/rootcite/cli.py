"""Command-line entry point: ingest, diagnose, eval, serve."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from rootcite.config import Settings
from rootcite.container import build_container
from rootcite.domain.models import Alert
from rootcite.evals.runner import (
    load_cases,
    retrieval_markdown,
    run_agent_eval,
    run_retrieval_eval,
)
from rootcite.observability.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rootcite")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="index the knowledge base")

    diag = sub.add_parser("diagnose", help="diagnose one alert")
    diag.add_argument("--service", required=True)
    diag.add_argument("--title", required=True)
    diag.add_argument("--description", default="")
    diag.add_argument("--alert-id", default="cli-alert")

    ev = sub.add_parser("eval", help="run offline evals against the golden dataset")
    ev.add_argument("suite", choices=["retrieval", "agent"])
    ev.add_argument("--cases", type=Path, default=Path("evals/golden_alerts.jsonl"))
    ev.add_argument("--k", type=int, default=5)
    ev.add_argument("--min-hit-rate", type=float, default=0.0, help="fail below this (CI gate)")
    ev.add_argument("--out", type=Path, help="write the JSON report here")

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    settings = Settings.from_env()
    configure_logging(settings.log_level)

    if args.command == "serve":
        import uvicorn

        uvicorn.run("rootcite.api.app:create_app", factory=True, host="0.0.0.0", port=args.port)
        return 0

    container = build_container(settings, metrics_stream=sys.stderr)
    report = container.ingestion.ingest_directory(settings.knowledge_dir)
    if args.command == "ingest":
        print(json.dumps(asdict(report)))
        return 0

    if args.command == "diagnose":
        alert = Alert(args.alert_id, args.service, args.title, args.description)
        diagnosis = container.diagnosis.diagnose(alert)
        print(json.dumps(asdict(diagnosis), indent=2, default=str))
        return 0

    cases = load_cases(args.cases)
    if args.suite == "retrieval":
        result = run_retrieval_eval(container.retriever, cases, args.k)
        print(retrieval_markdown(result))
        _write(args.out, result)
        hybrid_hit = result["hybrid"][f"hit@{args.k}"]
        if hybrid_hit < args.min_hit_rate:
            print(f"FAIL: hybrid hit@{args.k}={hybrid_hit} < {args.min_hit_rate}", file=sys.stderr)
            return 1
        return 0

    agent_report = run_agent_eval(container.diagnosis, cases)
    print(json.dumps(agent_report.to_dict(), indent=2))
    _write(args.out, agent_report.to_dict())
    return 0


def _write(path: Path | None, data: object) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
