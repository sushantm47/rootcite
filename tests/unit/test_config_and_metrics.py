import io
import json

import pytest

from rootcite.config import Settings, Stage
from rootcite.domain.errors import ConfigurationError
from rootcite.observability.metrics import MetricsLogger


def test_settings_from_env():
    s = Settings.from_env({"ROOTCITE_STAGE": "beta", "ROOTCITE_RETRIEVAL_K": "7"})
    assert s.stage is Stage.BETA and s.retrieval_k == 7


def test_dsn_built_from_ecs_secret_parts():
    s = Settings.from_env(
        {"ROOTCITE_INDEX": "postgres", "DB_HOST": "db.internal", "DB_PASSWORD": "pw"}
    )
    assert s.database_url == "postgresql://rootcite:pw@db.internal:5432/rootcite"


@pytest.mark.parametrize(
    "env",
    [
        {"ROOTCITE_STAGE": "prod"},  # offline LLM and memory index are not allowed in prod
        {"ROOTCITE_INDEX": "postgres"},  # missing DATABASE_URL
        {"ROOTCITE_LLM_PROVIDER": "gpt"},
    ],
)
def test_invalid_settings_fail_fast(env):
    with pytest.raises(ConfigurationError):
        Settings.from_env(env)


def test_emf_format():
    stream = io.StringIO()
    m = MetricsLogger("Diagnose", stream=stream, dimensions={"Stage": "beta"})
    m.put("Latency", 12.5, "Milliseconds")
    m.set_property("alert_id", "a1")
    m.flush()
    record = json.loads(stream.getvalue())
    directive = record["_aws"]["CloudWatchMetrics"][0]
    assert directive["Namespace"] == "RootCite"
    assert directive["Dimensions"] == [["Operation", "Stage"]]
    assert directive["Metrics"] == [{"Name": "Latency", "Unit": "Milliseconds"}]
    assert record["Latency"] == 12.5 and record["alert_id"] == "a1"


def test_flush_without_metrics_writes_nothing():
    stream = io.StringIO()
    MetricsLogger("Diagnose", stream=stream).flush()
    assert stream.getvalue() == ""
