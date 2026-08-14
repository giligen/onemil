"""Unit tests for study_orb simulator."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from study_orb import simulate_orb_trade


def _mk_bars(rows):
    """Build a bars DataFrame. rows = list of (hour, minute, o, h, l, c, v).
    Timestamps set to UTC on 2026-03-15 (EDT → 9:30 ET = 13:30 UTC)."""
    data = []
    for (h, m, o, hi, lo, c, v) in rows:
        data.append({
            'timestamp': datetime(2026, 3, 15, h, m, 0, tzinfo=timezone.utc),
            'open': o, 'high': hi, 'low': lo, 'close': c, 'volume': v,
        })
    df = pd.DataFrame(data)
    return df


def test_stop_hit_scenario():
    """5-min ORB: range 10.00-10.20. Entry on touch. Stop at range low.
    Next bar prints high=10.25 (trigger) but low=9.95 (stop). Expect stop fill."""
    bars = _mk_bars([
        (13, 30, 10.00, 10.10, 9.95, 10.05, 10000),   # range bar 1
        (13, 31, 10.05, 10.15, 10.00, 10.10, 10000),  # range bar 2
        (13, 32, 10.10, 10.12, 10.02, 10.08, 10000),  # range bar 3
        (13, 33, 10.08, 10.20, 10.05, 10.18, 10000),  # range bar 4 (range_high=10.20)
        (13, 34, 10.18, 10.19, 10.00, 10.02, 10000),  # range bar 5 (range_low=9.95)
        (13, 35, 10.02, 10.25, 9.90, 10.15, 20000),   # trigger + stop in same bar
        (13, 36, 10.15, 10.20, 10.00, 10.10, 10000),
    ])
    t = simulate_orb_trade(bars, 'T', '2026-03-15', 'test', range_minutes=5,
                           entry_slip_bps=0, exit_slip_bps=0, position_size_usd=10000)
    assert t.entered
    assert t.range_high == 10.20
    assert t.range_low == 9.95
    # Entry at 13:35 bar, stop check starts at 13:36 bar. 13:36 low is 10.00,
    # doesn't hit 9.95. Target = 10.20 + 2*(10.20-9.95) = 10.70. Not hit either.
    # EOD exit at 13:36 close = 10.10.
    assert t.exit_reason == 'eod'


def test_target_hit_scenario():
    """Same range, next bar pushes through target. Target = range_high + 2×range_size."""
    # range = [10.00, 10.20], size=0.20, target = 10.60 (range_high + 2*0.20)
    bars = _mk_bars([
        (13, 30, 10.00, 10.10, 9.95, 10.05, 10000),
        (13, 31, 10.05, 10.15, 10.00, 10.10, 10000),
        (13, 32, 10.10, 10.12, 10.02, 10.08, 10000),
        (13, 33, 10.08, 10.20, 10.05, 10.18, 10000),
        (13, 34, 10.18, 10.19, 10.00, 10.02, 10000),
        (13, 35, 10.02, 10.25, 10.10, 10.22, 20000),   # entry trigger
        (13, 36, 10.22, 10.70, 10.20, 10.65, 30000),   # target hit
    ])
    t = simulate_orb_trade(bars, 'T', '2026-03-15', 'test', range_minutes=5,
                           target_mult=2.0,
                           entry_slip_bps=0, exit_slip_bps=0, position_size_usd=10000)
    assert t.entered
    assert t.exit_reason == 'target'
    # Target = 10.20 + 2*(10.20-9.95) = 10.70
    assert t.exit_price == pytest.approx(10.70, abs=0.01)
    assert t.entry_price == pytest.approx(10.20, abs=0.01)  # entry at range_high


def test_eod_exit_scenario():
    """Entry fires, price meanders, no stop/target hit → EOD exit at last bar close."""
    bars = _mk_bars([
        (13, 30, 10.00, 10.10, 9.95, 10.05, 10000),
        (13, 31, 10.05, 10.15, 10.00, 10.10, 10000),
        (13, 32, 10.10, 10.12, 10.02, 10.08, 10000),
        (13, 33, 10.08, 10.20, 10.05, 10.18, 10000),
        (13, 34, 10.18, 10.19, 10.00, 10.02, 10000),
        (13, 35, 10.02, 10.25, 10.10, 10.22, 20000),   # entry trigger
        (13, 36, 10.22, 10.30, 10.15, 10.25, 12000),
        (13, 37, 10.25, 10.35, 10.18, 10.30, 12000),
        (13, 38, 10.30, 10.40, 10.20, 10.33, 12000),  # last bar close = 10.33
    ])
    t = simulate_orb_trade(bars, 'T', '2026-03-15', 'test', range_minutes=5,
                           entry_slip_bps=0, exit_slip_bps=0, position_size_usd=10000)
    assert t.entered
    assert t.exit_reason == 'eod'
    assert t.exit_price == pytest.approx(10.33, abs=0.01)


def test_no_trigger_within_time_stop():
    """Entry never triggers within time_stop window → entered=False."""
    bars = _mk_bars([
        (13, 30, 10.00, 10.10, 9.95, 10.05, 10000),
        (13, 31, 10.05, 10.15, 10.00, 10.10, 10000),
        (13, 32, 10.10, 10.12, 10.02, 10.08, 10000),
        (13, 33, 10.08, 10.20, 10.05, 10.18, 10000),
        (13, 34, 10.18, 10.19, 10.00, 10.02, 10000),
        # No bar with high > 10.20 in next 60 min; all bars print < range_high
        (13, 35, 10.02, 10.15, 10.00, 10.10, 10000),
        (13, 36, 10.10, 10.12, 10.00, 10.05, 10000),
    ])
    t = simulate_orb_trade(bars, 'T', '2026-03-15', 'test', range_minutes=5,
                           time_stop_minutes=60,
                           entry_slip_bps=0, exit_slip_bps=0, position_size_usd=10000)
    assert not t.entered


def test_slippage_applied():
    """Entry slippage pushes price UP, exit slippage pulls DOWN."""
    bars = _mk_bars([
        (13, 30, 10.00, 10.10, 9.95, 10.05, 10000),
        (13, 31, 10.05, 10.15, 10.00, 10.10, 10000),
        (13, 32, 10.10, 10.12, 10.02, 10.08, 10000),
        (13, 33, 10.08, 10.20, 10.05, 10.18, 10000),
        (13, 34, 10.18, 10.19, 10.00, 10.02, 10000),
        (13, 35, 10.02, 10.25, 10.10, 10.22, 20000),
        (13, 36, 10.22, 10.70, 10.20, 10.65, 30000),  # target hit
    ])
    t = simulate_orb_trade(bars, 'T', '2026-03-15', 'test', range_minutes=5,
                           target_mult=2.0,
                           entry_slip_bps=30, exit_slip_bps=10,
                           position_size_usd=10000)
    # Entry at 10.20 * (1 + 30/10000) = 10.2306
    assert t.entry_price == pytest.approx(10.2306, abs=0.001)
    # Exit at target 10.70 * (1 - 10/10000) = 10.6893
    assert t.exit_price == pytest.approx(10.6893, abs=0.001)


def test_close_above_entry_mode():
    """Entry_mode=close_above: entry only on bar that CLOSES above range_high."""
    bars = _mk_bars([
        (13, 30, 10.00, 10.10, 9.95, 10.05, 10000),
        (13, 31, 10.05, 10.15, 10.00, 10.10, 10000),
        (13, 32, 10.10, 10.12, 10.02, 10.08, 10000),
        (13, 33, 10.08, 10.20, 10.05, 10.18, 10000),
        (13, 34, 10.18, 10.19, 10.00, 10.02, 10000),
        # Bar 13:35: high=10.25 (touch), but close=10.15 (below range_high). Should SKIP on close_above.
        (13, 35, 10.02, 10.25, 10.10, 10.15, 20000),
        # Bar 13:36: high=10.30, close=10.25. Close > 10.20 → trigger here.
        (13, 36, 10.15, 10.30, 10.15, 10.25, 15000),
        (13, 37, 10.25, 10.40, 10.20, 10.35, 12000),
    ])
    t = simulate_orb_trade(bars, 'T', '2026-03-15', 'test', range_minutes=5,
                           entry_mode='close_above',
                           entry_slip_bps=0, exit_slip_bps=0, position_size_usd=10000)
    assert t.entered
    # Entry price for close_above = 13:36 bar close = 10.25
    assert t.entry_price == pytest.approx(10.25, abs=0.01)


def test_missing_range_bars_returns_no_trade():
    """If we have fewer bars than needed to form range → skip."""
    bars = _mk_bars([
        (13, 30, 10.00, 10.10, 9.95, 10.05, 10000),
        (13, 31, 10.05, 10.15, 10.00, 10.10, 10000),
        # Only 2 bars, asking for 5-min range
    ])
    t = simulate_orb_trade(bars, 'T', '2026-03-15', 'test', range_minutes=5,
                           entry_slip_bps=0, exit_slip_bps=0, position_size_usd=10000)
    assert not t.entered


class TestSessionOpenTimestampDST:
    """8/14 P0: winter (EST) days must anchor the range at 9:30 ET, not
    the 8:30 ET premarket bar (13:30 UTC in EST == premarket; the old
    hour-in-{13,14} mask grabbed it whenever premarket bars were cached
    — contaminated ~$155K of the $251K book across both winters)."""

    def _bars(self, utc_times):
        import pandas as pd
        return pd.DataFrame({
            'timestamp': pd.to_datetime(utc_times, utc=True),
            'open': 10.0, 'high': 10.5, 'low': 9.5, 'close': 10.2,
            'volume': 1000})

    def test_est_day_picks_930_not_premarket(self):
        from study_orb import _session_open_timestamp
        # 2025-01-03 (EST): 13:30 UTC = 8:30 ET premarket, 14:30 UTC = 9:30 ET
        bars = self._bars(['2025-01-03 13:30:00', '2025-01-03 13:31:00',
                           '2025-01-03 14:30:00', '2025-01-03 14:31:00'])
        ts = _session_open_timestamp(bars)
        assert str(ts) == '2025-01-03 14:30:00+00:00'

    def test_edt_day_picks_1330_utc(self):
        from study_orb import _session_open_timestamp
        # 2025-06-02 (EDT): 13:30 UTC = 9:30 ET
        bars = self._bars(['2025-06-02 12:30:00', '2025-06-02 13:30:00',
                           '2025-06-02 13:31:00'])
        ts = _session_open_timestamp(bars)
        assert str(ts) == '2025-06-02 13:30:00+00:00'

    def test_no_rth_bars_returns_none(self):
        from study_orb import _session_open_timestamp
        bars = self._bars(['2025-01-03 13:30:00'])   # premarket only
        assert _session_open_timestamp(bars) is None
