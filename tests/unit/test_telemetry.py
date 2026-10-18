from rootcite.adapters.telemetry import FixtureTelemetry

from support import TELEMETRY_DIR


def test_metric_summary():
    t = FixtureTelemetry(TELEMETRY_DIR)
    m = t.get_metric("checkout-api", "db_connections_in_use", 10)
    assert m["summary"]["latest"] == 5 and m["window_minutes"] == 10


def test_unknown_metric_lists_alternatives():
    m = FixtureTelemetry(TELEMETRY_DIR).get_metric("checkout-api", "nope", 10)
    assert "error" in m and "p99_latency_ms" in m["available"]


def test_log_search_and_deploy_window():
    t = FixtureTelemetry(TELEMETRY_DIR)
    assert all("pool" in line["message"] for line in t.search_logs("checkout-api", "pool", 10))
    assert [d["version"] for d in t.recent_deploys("checkout-api", 1)] == ["v2.14.0"]


def test_path_traversal_is_rejected():
    assert FixtureTelemetry(TELEMETRY_DIR).search_logs("../../etc/passwd", "root", 5) == []
