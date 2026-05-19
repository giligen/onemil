"""
Unit tests for scanner._stop_monitor_circuit_breaker_state.

Pure state-machine that decides whether a brief StopMonitor unhealthy
window should trip the scanner's force-close + exit circuit breaker.

Context: today (2026-05-19) the scanner exited after only 8 seconds of
unhealthy StopMonitor (a transient WS disconnect that was actively
reconnecting). The force-close + exit cascade left 4 ORB bracket-held
positions stuck in `held_for_orders` on restart. The state machine now
tolerates sustained unhealthy state up to a configurable threshold
(_SM_DEAD_THRESHOLD_S, default 180s).
"""

from scanner.realtime_scanner import _stop_monitor_circuit_breaker_state


THRESHOLD = 180.0


class TestHealthyPath:
    """No-action paths when the monitor is healthy."""

    def test_healthy_clean_returns_healthy(self):
        new_ts, action = _stop_monitor_circuit_breaker_state(
            is_healthy=True, unhealthy_since=None,
            now=1000.0, threshold_s=THRESHOLD,
        )
        assert new_ts is None
        assert action == 'healthy'

    def test_recovery_from_unhealthy_returns_recovered(self):
        """Was unhealthy at t=1000, healthy at t=1010 → recovered."""
        new_ts, action = _stop_monitor_circuit_breaker_state(
            is_healthy=True, unhealthy_since=1000.0,
            now=1010.0, threshold_s=THRESHOLD,
        )
        assert new_ts is None
        assert action == 'recovered'


class TestUnhealthyGracePeriod:
    """Unhealthy but within tolerance — record + log, do NOT trip."""

    def test_first_unhealthy_records_timestamp(self):
        new_ts, action = _stop_monitor_circuit_breaker_state(
            is_healthy=False, unhealthy_since=None,
            now=1000.0, threshold_s=THRESHOLD,
        )
        assert new_ts == 1000.0
        assert action == 'unhealthy_start'

    def test_unhealthy_within_grace_preserves_timestamp(self):
        """Tick 2 of an unhealthy window: same timestamp, no action."""
        new_ts, action = _stop_monitor_circuit_breaker_state(
            is_healthy=False, unhealthy_since=1000.0,
            now=1060.0, threshold_s=THRESHOLD,
        )
        assert new_ts == 1000.0
        assert action == 'unhealthy_grace'

    def test_unhealthy_just_below_threshold_still_grace(self):
        new_ts, action = _stop_monitor_circuit_breaker_state(
            is_healthy=False, unhealthy_since=1000.0,
            now=1000.0 + THRESHOLD - 0.5,
            threshold_s=THRESHOLD,
        )
        assert action == 'unhealthy_grace'


class TestTripped:
    """Sustained unhealthy past threshold — trip the circuit breaker."""

    def test_unhealthy_at_threshold_trips(self):
        new_ts, action = _stop_monitor_circuit_breaker_state(
            is_healthy=False, unhealthy_since=1000.0,
            now=1000.0 + THRESHOLD,
            threshold_s=THRESHOLD,
        )
        assert action == 'tripped'
        # timestamp preserved so caller can log the duration
        assert new_ts == 1000.0

    def test_unhealthy_well_past_threshold_trips(self):
        new_ts, action = _stop_monitor_circuit_breaker_state(
            is_healthy=False, unhealthy_since=1000.0,
            now=1000.0 + THRESHOLD * 10,
            threshold_s=THRESHOLD,
        )
        assert action == 'tripped'


class TestIncidentReplay:
    """Replay today's actual incident: WS disconnect at 13:45:02,
    declared dead at 13:45:10 (only 8 seconds). New state machine must NOT
    trip at the 8-second mark."""

    def test_today_8s_unhealthy_does_not_trip(self):
        new_ts, action = _stop_monitor_circuit_breaker_state(
            is_healthy=False,
            unhealthy_since=1000.0,           # 13:45:02
            now=1008.0,                       # 13:45:10
            threshold_s=THRESHOLD,
        )
        assert action == 'unhealthy_grace'    # tolerate, don't exit

    def test_today_recovery_within_60s_clean(self):
        """If WS reconnects in <threshold, no trip."""
        new_ts, action = _stop_monitor_circuit_breaker_state(
            is_healthy=True, unhealthy_since=1000.0,
            now=1060.0,                       # reconnected after 60s
            threshold_s=THRESHOLD,
        )
        assert action == 'recovered'
        assert new_ts is None


class TestStateMachineProperties:
    """Algebraic properties that lock the contract."""

    def test_idempotent_on_no_change(self):
        """Two consecutive healthy ticks: state unchanged."""
        s1 = _stop_monitor_circuit_breaker_state(
            is_healthy=True, unhealthy_since=None, now=1.0,
            threshold_s=THRESHOLD,
        )
        s2 = _stop_monitor_circuit_breaker_state(
            is_healthy=True, unhealthy_since=s1[0], now=2.0,
            threshold_s=THRESHOLD,
        )
        assert s1 == s2 == (None, 'healthy')

    def test_state_carries_across_calls(self):
        """Standard state-machine sweep: start healthy → unhealthy →
        unhealthy → recovered."""
        # T=0 healthy
        ts, a = _stop_monitor_circuit_breaker_state(
            is_healthy=True, unhealthy_since=None, now=0.0,
            threshold_s=THRESHOLD,
        )
        assert a == 'healthy' and ts is None
        # T=10 first unhealthy
        ts, a = _stop_monitor_circuit_breaker_state(
            is_healthy=False, unhealthy_since=ts, now=10.0,
            threshold_s=THRESHOLD,
        )
        assert a == 'unhealthy_start' and ts == 10.0
        # T=20 still unhealthy
        ts, a = _stop_monitor_circuit_breaker_state(
            is_healthy=False, unhealthy_since=ts, now=20.0,
            threshold_s=THRESHOLD,
        )
        assert a == 'unhealthy_grace' and ts == 10.0
        # T=30 recovered
        ts, a = _stop_monitor_circuit_breaker_state(
            is_healthy=True, unhealthy_since=ts, now=30.0,
            threshold_s=THRESHOLD,
        )
        assert a == 'recovered' and ts is None
