from rootcite.adapters.embeddings import HashingEmbedder
from rootcite.domain.models import Alert, Diagnosis
from rootcite.resilience.semantic_cache import SemanticCache


class FakeClock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


def diagnosis(service: str) -> Diagnosis:
    return Diagnosis("a1", service, "pool exhausted", 0.9, ["E1"], [], False, True)


ALERT = Alert("a1", "checkout-api", "p99 latency above 2s", "5xx rising")


def test_hit_for_near_identical_alert_same_service():
    cache = SemanticCache(HashingEmbedder(), threshold=0.9)
    cache.put(ALERT, diagnosis("checkout-api"))
    again = Alert("a2", "checkout-api", "p99 latency above 2s", "5xx rising")
    assert cache.get(again) is not None


def test_never_shares_answers_across_services():
    cache = SemanticCache(HashingEmbedder(), threshold=0.5)
    cache.put(ALERT, diagnosis("checkout-api"))
    assert cache.get(Alert("a3", "payments-api", ALERT.title, ALERT.description)) is None


def test_miss_below_threshold():
    cache = SemanticCache(HashingEmbedder(), threshold=0.95)
    cache.put(ALERT, diagnosis("checkout-api"))
    assert cache.get(Alert("a4", "checkout-api", "disk full on log volume")) is None


def test_entries_expire():
    clock = FakeClock()
    cache = SemanticCache(HashingEmbedder(), ttl_seconds=60, clock=clock)
    cache.put(ALERT, diagnosis("checkout-api"))
    clock.now = 61
    assert cache.get(ALERT) is None
    assert len(cache) == 0


def test_lru_eviction():
    cache = SemanticCache(HashingEmbedder(), max_entries=2)
    for i in range(3):
        cache.put(Alert(f"a{i}", "checkout-api", f"alert {i}"), diagnosis("checkout-api"))
    assert len(cache) == 2
