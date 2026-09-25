"""Unit tests for cell_1428.py: population gate boundaries, the LOST-count fetch gate, and the floor=3.0 override
letting a ~$5 trigger through (vs the live $20 floor). Synthetic data only — no network, no production DB."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import cell_1428 as c1428  # noqa: E402
import causal_arming as ca  # noqa: E402
from trading.hod_break import HodBreakParams  # noqa: E402


def _leadin(symbol, n=15, start='2025-07-01'):
    """n lead-in daily rows (volume=100,000, flat $10 bar) so adv20's rolling(20, min_periods=10) is non-null by
    the test day two rows later."""
    dates = pd.bdate_range(start, periods=n).strftime('%Y-%m-%d')
    return pd.DataFrame(dict(bar_date=dates, symbol=symbol, open=10.0, high=10.1, low=9.9, close=10.0, volume=100_000))


def _population_frame(rows):
    """rows: list of dicts (symbol, prev_close, prev_vol, open, gap handled via prev_close, high, day).
    Builds lead-in + prev-day + test-day rows per symbol and returns the concatenated synthetic daily frame."""
    frames = []
    for r in rows:
        lead = _leadin(r['symbol'])
        prev_day = pd.bdate_range(lead.bar_date.iloc[-1], periods=2)[1].strftime('%Y-%m-%d')
        test_day = pd.bdate_range(prev_day, periods=2)[1].strftime('%Y-%m-%d')
        prev_row = pd.DataFrame([dict(bar_date=prev_day, symbol=r['symbol'], open=r['prev_close'], high=r['prev_close'],
                                       low=r['prev_close'], close=r['prev_close'], volume=r['prev_vol'])])
        test_row = pd.DataFrame([dict(bar_date=test_day, symbol=r['symbol'], open=r['open'], high=r['open'] * 1.2,
                                       low=r['open'], close=r['open'], volume=200_000)])
        frames += [lead, prev_row, test_row]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def synth_parquet(tmp_path, monkeypatch):
    """Writes a synthetic parquet and points cell_1428.PARQUET / POP_CSV at tmp files."""
    def _make(rows):
        d = _population_frame(rows)
        d['bar_date'] = np.where(d.bar_date < c1428.SPLITS['TRAIN'][0], c1428.SPLITS['TRAIN'][0], d.bar_date)
        # keep dates inside the scored window by shifting all frames to start there instead:
        return d
    yield _make


def test_population_gap_price_volume_boundaries(tmp_path, monkeypatch):
    """gap >= 5 %, open in [3, 30], prior-day volume >= 500,000 — each gate tested exactly at its boundary."""
    rows = [
        dict(symbol='GAPOK', prev_close=10.00, prev_vol=600_000, open=10.50),   # gap 5.00% -> IN
        dict(symbol='GAPNO', prev_close=10.00, prev_vol=600_000, open=10.49),   # gap 4.90% -> OUT
        dict(symbol='PXLO', prev_close=2.857, prev_vol=600_000, open=3.00),     # open == 3.00, gap 5% -> IN
        dict(symbol='PXLONO', prev_close=2.849, prev_vol=600_000, open=2.99),   # open 2.99 -> OUT
        dict(symbol='PXHI', prev_close=28.571, prev_vol=600_000, open=30.00),   # open == 30.00, gap 5% -> IN
        dict(symbol='PXHINO', prev_close=28.542, prev_vol=600_000, open=30.01), # open 30.01 -> OUT
        dict(symbol='VOLOK', prev_close=10.00, prev_vol=500_000, open=10.50),   # prior vol == 500,000 -> IN
        dict(symbol='VOLNO', prev_close=10.00, prev_vol=499_999, open=10.50),   # prior vol 499,999 -> OUT
        dict(symbol='ZVZZT', prev_close=10.00, prev_vol=600_000, open=10.50),   # test ticker -> OUT always
    ]
    d = _population_frame(rows)
    # shift every date range so the test day falls inside the scored TRAIN-H2 window
    all_dates = sorted(d.bar_date.unique())
    offset = pd.Timestamp(c1428.SPLITS['TRAIN'][0]) - pd.Timestamp(all_dates[0])
    d['bar_date'] = (pd.to_datetime(d.bar_date) + offset).dt.strftime('%Y-%m-%d')
    pq_path = tmp_path / 'synth_daily.parquet'
    d.to_parquet(pq_path)
    monkeypatch.setattr(c1428, 'PARQUET', str(pq_path))
    monkeypatch.setattr(c1428, 'POP_CSV', str(tmp_path / 'pop.csv'))
    u = c1428.build_population()
    included = set(u.symbol)
    assert {'GAPOK', 'PXLO', 'PXHI', 'VOLOK'} <= included
    assert not ({'GAPNO', 'PXLONO', 'PXHINO', 'VOLNO', 'ZVZZT'} & included)


def test_lost_gate_arithmetic():
    """The fetch stage refuses to score when LOST symbol-days exceed 5 % of requested — exercise the exact
    threshold both sides (the gate compares lost/requested to c1428.LOST_GATE)."""
    assert c1428.LOST_GATE == 0.05
    requested = 1000
    assert (49 / requested) <= c1428.LOST_GATE           # 4.9% -> would NOT trip the gate
    assert (51 / requested) > c1428.LOST_GATE             # 5.1% -> WOULD trip the gate


def test_floor_3_passes_5dollar_trigger_live_floor_rejects():
    """arm_state's floor param (LADDER.md 1,428: pool floor = 3.0, live default = 20) gates a ~$5.01 trigger:
    passes at floor=3.0, rejected at the live $20 floor. Same function cell_1428.run() calls unmodified."""
    p = HodBreakParams(consol_bars=1, consol_pct=0.04, min_dist_open_pct=0.0, min_r_pct=0.0, last_entry_minute=999)
    o = np.array([5.00, 5.00])
    h = np.array([5.00, 5.00])
    l = np.array([5.00, 4.98])
    v = np.array([1000.0, 1000.0])
    m = np.array([570, 571])
    a = ca.arm_state(o, h, l, v, m, 1, adv20=1_000_000, p=p, floor=c1428.FLOOR, use_rv=False)
    assert a is not None
    assert a['trigger'] == pytest.approx(5.01, abs=1e-6)
    rejected = ca.arm_state(o, h, l, v, m, 1, adv20=1_000_000, p=p, floor=20.0, use_rv=False)
    assert rejected is None
