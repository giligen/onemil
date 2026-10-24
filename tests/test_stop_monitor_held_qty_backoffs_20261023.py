"""2026-10-22 SNDQ: held-for-orders retries must span the paper OMS cancel propagation (>= 10 s)."""
from trading.stop_monitor import StopMonitor


def test_backoffs_span_at_least_ten_seconds():
    assert sum(StopMonitor._HELD_QTY_RETRY_BACKOFFS_S) >= 10.0
    assert len(StopMonitor._HELD_QTY_RETRY_BACKOFFS_S) >= 4
