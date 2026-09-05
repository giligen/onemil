"""Signal-study exit hooks inside the production walkers (2026-09-05).

Pins on a synthetic tape:
  * flags OFF -> simulate_static_lock / simulate_winner_stack unchanged
    (the same tape gives the same exit as before the hooks existed);
  * ORB_EXP_MID_KILL -> the first CLOSED bar below the range midpoint before
    +0.5R exits at the NEXT bar's open, reason 'mid_kill'; not after +0.5R.
"""
from dataclasses import replace

import pandas as pd
import pytest

import study_orb_pipeline_static_lock as P


def _tape(closes, opens=None, highs=None, lows=None, start='2026-03-02 14:30'):
    """1-min bars from 9:30 ET (14:30 UTC in March); entry bar is index 5."""
    n = len(closes)
    ts = pd.date_range(start, periods=n, freq='1min', tz='UTC')
    opens = opens or closes
    highs = highs or [max(o, c) + 0.02 for o, c in zip(opens, closes)]
    lows = lows or [min(o, c) - 0.02 for o, c in zip(opens, closes)]
    return pd.DataFrame({'timestamp': ts, 'open': opens, 'high': highs, 'low': lows,
                         'close': closes, 'volume': [1000] * n})


RH, RL = 10.0, 9.0           # range 9:30-9:34, R = 1.0, mid = 9.5
ENTRY = 10.03


@pytest.fixture
def flags_off(monkeypatch):
    monkeypatch.setattr(P, 'EXP', replace(P.EXP, mid_kill=False, rearm=False))


@pytest.fixture
def mid_kill_on(monkeypatch):
    monkeypatch.setattr(P, 'EXP', replace(P.EXP, mid_kill=True, rearm=False))


def _bars_fade_below_mid():
    # range bars, then breakout bar (closes strong: no Rule M), bar1 mild (no Rule D),
    # then a close at 9.40 (< mid 9.5) before ever reaching +0.5R (10.53), then drift.
    closes = [9.5, 9.6, 9.7, 9.8, 9.9,   10.20, 10.05, 9.40, 9.45, 9.50, 9.55, 9.60]
    opens = [9.4, 9.5, 9.6, 9.7, 9.8,    10.02, 10.20, 10.00, 9.42, 9.45, 9.50, 9.55]
    highs = [9.6, 9.7, 9.8, 9.9, 10.0,   10.25, 10.22, 10.01, 9.48, 9.52, 9.58, 9.62]
    lows = [9.3, 9.4, 9.5, 9.6, 9.7,     10.00, 10.00, 9.38, 9.40, 9.43, 9.48, 9.53]
    return _tape(closes, opens, highs, lows)


def test_flags_off_static_lock_unchanged(flags_off):
    b = _bars_fade_below_mid()
    px, reason = P.simulate_static_lock(b, ENTRY, RH, RL, b['timestamp'].iloc[5])
    assert reason == 'eod'                      # never hit range_low, never armed


def test_mid_kill_exits_next_open(mid_kill_on):
    b = _bars_fade_below_mid()
    px, reason = P.simulate_static_lock(b, ENTRY, RH, RL, b['timestamp'].iloc[5])
    assert reason == 'mid_kill'
    assert px == pytest.approx(9.42 * (1 - P.EXIT_SLIP_BPS / 10000))   # open of bar after the 9.40 close


def test_mid_kill_not_after_half_r(mid_kill_on):
    b = _bars_fade_below_mid()
    b.loc[6, 'high'] = 10.60                    # +0.5R (10.53) touched on bar 1
    px, reason = P.simulate_static_lock(b, ENTRY, RH, RL, b['timestamp'].iloc[5])
    assert reason == 'eod'


def test_mid_kill_in_winner_stack_matches(mid_kill_on):
    b = _bars_fade_below_mid()
    px, reason = P.simulate_winner_stack(b, ENTRY, RH, RL, b['timestamp'].iloc[5], shares=100,
                                         atr14=None, atr_floor_enabled=False, scale_enabled=True)
    assert reason == 'mid_kill'
    assert px == pytest.approx(9.42 * (1 - P.EXIT_SLIP_BPS / 10000))


def test_winner_stack_flags_off_unchanged(flags_off):
    b = _bars_fade_below_mid()
    px, reason = P.simulate_winner_stack(b, ENTRY, RH, RL, b['timestamp'].iloc[5], shares=100,
                                         atr14=None, atr_floor_enabled=False, scale_enabled=True)
    assert reason == 'eod'
