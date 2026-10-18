"""End-to-end tests over the real sample knowledge base and telemetry, fully offline."""

import io
import json

from rootcite.cli import main as cli_main
from rootcite.domain.models import Alert
from rootcite.evals.runner import load_cases, run_agent_eval, run_retrieval_eval
from rootcite.observability.metrics import MetricsLogger
from rootcite.services.diagnosis import DiagnosisService

from support import GOLDEN, ROOT, ScriptedLLM, submit

ALERT = Alert("e2e-1", "checkout-api", "p99 latency above 2s and 5xx errors rising")


def test_ingestion_is_idempotent(make_container):
    container = make_container()
    before = container.index.count()
    report = container.ingestion.ingest_directory(container.settings.knowledge_dir)
    assert report.documents == 8
    assert container.index.count() == before


def test_offline_pipeline_diagnoses_pool_exhaustion(make_container):
    d = make_container().diagnosis.diagnose(ALERT)
    assert "pool" in d.root_cause.lower()
    assert "v2.14.0" in d.root_cause
    assert d.grounded
    assert d.needs_human  # offline mode is never trusted to act alone


def test_semantic_cache_serves_repeat_alerts(make_container):
    llm = ScriptedLLM(
        [
            submit(citations=["E1"]),  # first run; second identical alert must not call the LLM
        ]
    )
    container = make_container(llm)
    first = container.diagnosis.diagnose(ALERT)
    repeat = container.diagnosis.diagnose(Alert("e2e-2", ALERT.service, ALERT.title))
    assert not first.cached and repeat.cached
    assert repeat.alert_id == "e2e-2"
    assert len(llm.calls) == 1


def test_metrics_are_emitted_per_request(make_container):
    container = make_container()
    stream = io.StringIO()
    service = DiagnosisService(
        container.agent, metrics_factory=lambda op: MetricsLogger(op, stream=stream)
    )
    service.diagnose(ALERT)
    record = json.loads(stream.getvalue())
    assert record["CacheHit"] == 0 and record["Grounded"] == 1 and record["ToolCalls"] == 2


def test_retrieval_eval_meets_quality_bar(make_container):
    report = run_retrieval_eval(make_container().retriever, load_cases(GOLDEN), k=5)
    assert report["hybrid"]["hit@5"] >= 0.9
    assert report["hybrid"]["mrr"] >= report["dense"]["mrr"]


def test_agent_eval_offline(make_container):
    report = run_agent_eval(make_container().diagnosis, load_cases(GOLDEN))
    assert report.cases == 11
    assert report.grounded_rate == 1.0
    assert report.injection_detection_rate == 1.0


def test_cli_eval_gate_fails_below_threshold(monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    out = tmp_path / "r.json"
    assert cli_main(["eval", "retrieval", "--out", str(out), "--min-hit-rate", "0.9"]) == 0
    assert "hybrid" in json.loads(out.read_text())
    assert cli_main(["eval", "retrieval", "--min-hit-rate", "1.01"]) == 1
