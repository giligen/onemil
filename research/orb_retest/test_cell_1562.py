#!/usr/bin/env python3
"""Unit tests for cell_1562.py -- trigger print, strict-below fill, window end, exit rule from R',
cost units. Uses synthetic tape/bars via monkeypatch; touches no real data files."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import cell_1562 as c  # noqa: E402


ET = 'America/New_York'


def ts(date, hhmmss):
    return pd.Timestamp(f'{date} {hhmmss}', tz=ET).tz_convert('UTC')


# --------------------------------------------------------------------------------------------------
# split / cutoff
# --------------------------------------------------------------------------------------------------
def test_split_cutoff():
    assert c.split_of('2025-06-30') == 'TRAIN'
    assert c.split_of('2025-07-01') == 'VAL'
    assert c.split_of('2023-01-12') == 'TRAIN'
    assert c.split_of('2026-09-23') == 'VAL'


# --------------------------------------------------------------------------------------------------
# limit price construction (the tick / the deeper retest)
# --------------------------------------------------------------------------------------------------
def test_cell_limits():
    level = 10.00
    assert c.CELLS[1562]['limit_fn'](level) == 9.99
    assert c.CELLS[1562]['window_min'] == 15
    assert round(c.CELLS[1563]['limit_fn'](level), 4) == 9.98
    assert c.CELLS[1563]['window_min'] == 30


# --------------------------------------------------------------------------------------------------
# strict-below fill vs at-or-below (report only)
# --------------------------------------------------------------------------------------------------
def test_strict_below_fill(monkeypatch):
    date, symbol = '2026-01-05', 'ZZZZ'
    trigger_ts = ts(date, '09:36:00')
    window_end = trigger_ts + pd.Timedelta(minutes=15)
    limit = 9.99
    trades = pd.DataFrame({
        'ts_event': [trigger_ts + pd.Timedelta(seconds=s) for s in (10, 20, 30, 40)],
        'price': [9.99, 10.00, 9.985, 9.80],  # at-or-below at t=10s, strict fill at t=30s
        'schema': 'trades',
    })
    monkeypatch.setattr(c, 'load_tape_trades', lambda sym, d: trades)
    monkeypatch.setattr(c, 'load_bars', lambda sym, d: None)
    r = c.find_retest(symbol, date, trigger_ts, window_end, limit)
    assert r['fill'] is True
    assert r['fill_price'] == pytest.approx(9.985)
    assert r['fill_ts'] == trigger_ts + pd.Timedelta(seconds=30)
    assert r['at_or_below_ts'] == trigger_ts + pd.Timedelta(seconds=10)
    assert r['source'] == 'tape'


def test_at_or_below_only_no_strict_fill(monkeypatch):
    """A print exactly AT the limit, never below, must not count as a strict fill."""
    date, symbol = '2026-01-05', 'ZZZZ'
    trigger_ts = ts(date, '09:36:00')
    window_end = trigger_ts + pd.Timedelta(minutes=15)
    limit = 9.99
    trades = pd.DataFrame({
        'ts_event': [trigger_ts + pd.Timedelta(seconds=10)],
        'price': [9.99],
        'schema': 'trades',
    })
    monkeypatch.setattr(c, 'load_tape_trades', lambda sym, d: trades)
    monkeypatch.setattr(c, 'load_bars', lambda sym, d: None)
    r = c.find_retest(symbol, date, trigger_ts, window_end, limit)
    assert r['fill'] is False
    assert r['at_or_below_ts'] is not None


def test_never_retest_no_print_below(monkeypatch):
    date, symbol = '2026-01-05', 'ZZZZ'
    trigger_ts = ts(date, '09:36:00')
    window_end = trigger_ts + pd.Timedelta(minutes=15)
    limit = 9.00
    trades = pd.DataFrame({
        'ts_event': [trigger_ts + pd.Timedelta(seconds=10)],
        'price': [9.50],
        'schema': 'trades',
    })
    monkeypatch.setattr(c, 'load_tape_trades', lambda sym, d: trades)
    monkeypatch.setattr(c, 'load_bars', lambda sym, d: None)
    r = c.find_retest(symbol, date, trigger_ts, window_end, limit)
    assert r['fill'] is False
    assert r['dip_min'] == pytest.approx(9.50)


# --------------------------------------------------------------------------------------------------
# window end respected -- a print below the limit AFTER the window must not fill
# --------------------------------------------------------------------------------------------------
def test_window_end_excludes_late_print(monkeypatch):
    date, symbol = '2026-01-05', 'ZZZZ'
    trigger_ts = ts(date, '09:36:00')
    window_end = trigger_ts + pd.Timedelta(minutes=15)
    limit = 9.99
    trades = pd.DataFrame({
        'ts_event': [window_end + pd.Timedelta(seconds=5)],  # strictly after window_end
        'price': [9.50],
        'schema': 'trades',
    })
    monkeypatch.setattr(c, 'load_tape_trades', lambda sym, d: trades)
    monkeypatch.setattr(c, 'load_bars', lambda sym, d: None)
    r = c.find_retest(symbol, date, trigger_ts, window_end, limit)
    assert r['fill'] is False
    assert r['dip_min'] is None


def test_tape_before_trigger_ignored(monkeypatch):
    """A print strictly before the trigger print must never be used (look-ahead guard)."""
    date, symbol = '2026-01-05', 'ZZZZ'
    trigger_ts = ts(date, '09:36:00')
    window_end = trigger_ts + pd.Timedelta(minutes=15)
    limit = 9.99
    trades = pd.DataFrame({
        'ts_event': [trigger_ts - pd.Timedelta(seconds=5)],
        'price': [1.00],  # would trivially "fill" if not excluded
        'schema': 'trades',
    })
    monkeypatch.setattr(c, 'load_tape_trades', lambda sym, d: trades)
    monkeypatch.setattr(c, 'load_bars', lambda sym, d: None)
    r = c.find_retest(symbol, date, trigger_ts, window_end, limit)
    assert r['fill'] is False
    assert r['dip_min'] is None


# --------------------------------------------------------------------------------------------------
# bar fallback beyond tape coverage
# --------------------------------------------------------------------------------------------------
def test_bar_fallback_fills_at_limit(monkeypatch):
    date, symbol = '2025-03-10', 'ZZZZ'
    trigger_ts = ts(date, '09:39:00')
    window_end = trigger_ts + pd.Timedelta(minutes=15)
    limit = 9.99
    trades = pd.DataFrame({
        'ts_event': [trigger_ts + pd.Timedelta(seconds=5)],
        'price': [10.05],
        'schema': 'trades',
    })
    bars = pd.DataFrame({
        'timestamp': [trigger_ts + pd.Timedelta(minutes=m) for m in (2, 3, 4)],
        'open': [10.0, 9.95, 9.80], 'high': [10.1, 10.0, 9.95],
        'low': [9.95, 9.80, 9.70], 'close': [9.96, 9.85, 9.75],
    })
    monkeypatch.setattr(c, 'load_tape_trades', lambda sym, d: trades)
    monkeypatch.setattr(c, 'load_bars', lambda sym, d: bars)
    r = c.find_retest(symbol, date, trigger_ts, window_end, limit)
    assert r['fill'] is True
    assert r['source'] == 'bar'
    assert r['fill_price'] == pytest.approx(limit)  # obtainable price, not the bar's low
    assert r['fill_ts'] == bars['timestamp'].iloc[0]  # first bar with low < limit


def test_data_gap_when_no_bars_and_tape_insufficient(monkeypatch):
    date, symbol = '2023-06-01', 'ZZZZ'  # pre-2025-01-02, no bars source
    trigger_ts = ts(date, '09:39:00')
    window_end = trigger_ts + pd.Timedelta(minutes=15)
    limit = 9.99
    trades = pd.DataFrame({
        'ts_event': [trigger_ts + pd.Timedelta(seconds=5)],
        'price': [10.05],
        'schema': 'trades',
    })
    monkeypatch.setattr(c, 'load_tape_trades', lambda sym, d: trades)
    monkeypatch.setattr(c, 'load_bars', lambda sym, d: None)
    r = c.find_retest(symbol, date, trigger_ts, window_end, limit)
    assert r['fill'] is False
    assert r['data_gap'] is True


# --------------------------------------------------------------------------------------------------
# exit rule from R': stop, lock arm/ratchet, EOD, and cost units (bps applied to the exit price)
# --------------------------------------------------------------------------------------------------
def _mkbars(date, minute_offsets, opens, highs, lows, closes):
    base = ts(date, '09:40:00')
    return pd.DataFrame({
        'timestamp': [base + pd.Timedelta(minutes=m) for m in minute_offsets],
        'open': opens, 'high': highs, 'low': lows, 'close': closes,
    })


def test_exit_walk_hard_stop(monkeypatch):
    date, symbol = '2025-03-10', 'ZZZZ'
    fill_ts = ts(date, '09:40:00')
    fill_price = 10.00
    stop = 9.00           # R' = 1.00
    bars = _mkbars(date, [0, 1, 2], [10.0, 9.5, 9.2], [10.2, 9.6, 9.3],
                   [9.9, 9.4, 8.9], [10.0, 9.5, 9.0])
    monkeypatch.setattr(c, 'load_bars', lambda sym, d: bars)
    ew = c.exit_walk(symbol, date, fill_ts, fill_price, stop, 'TRAIN')
    assert ew['reason'] == 'stop'
    assert ew['r_prime'] == pytest.approx(1.00)
    expected = 9.00 * (1 - c.SLIP_STOP_BPS['TRAIN'] / 10000.0)
    assert ew['exit_price'] == pytest.approx(expected)


def test_exit_walk_lock_ratchet(monkeypatch):
    date, symbol = '2025-03-10', 'ZZZZ'
    fill_ts = ts(date, '09:40:00')
    fill_price = 10.00
    stop = 9.00  # R' = 1.00; arm at 10+1.75=11.75, lock stop at 10+0.5=10.50
    bars = _mkbars(date, [0, 1, 2, 3],
                   [10.0, 11.8, 10.6, 10.3],
                   [10.1, 11.9, 10.8, 10.4],
                   [9.9, 11.7, 10.4, 10.2],   # bar2 low 10.4 <= lock 10.50 -> lock exit
                   [10.0, 11.8, 10.5, 10.3])
    monkeypatch.setattr(c, 'load_bars', lambda sym, d: bars)
    ew = c.exit_walk(symbol, date, fill_ts, fill_price, stop, 'VAL')
    assert ew['reason'] == 'lock'
    expected = 10.50 * (1 - c.SLIP_STOP_BPS['VAL'] / 10000.0)
    assert ew['exit_price'] == pytest.approx(expected)


def test_exit_walk_eod_uses_eod_bps():
    date, symbol = '2025-03-10', 'ZZZZ'
    fill_ts = ts(date, '09:40:00')
    fill_price = 10.00
    stop = 9.00
    bars = _mkbars(date, [0, 1], [10.0, 10.1], [10.2, 10.3], [9.9, 9.95], [10.05, 10.2])
    import cell_1562 as c2
    c2.load_bars = lambda sym, d: bars
    ew = c2.exit_walk(symbol, date, fill_ts, fill_price, stop, 'TRAIN')
    assert ew['reason'] == 'eod'
    last_close = bars['close'].iloc[-1]
    expected = last_close * (1 - c.EOD_BPS['TRAIN'] / 10000.0)
    assert ew['exit_price'] == pytest.approx(expected)


def test_exit_walk_no_bars_returns_none(monkeypatch):
    monkeypatch.setattr(c, 'load_bars', lambda sym, d: None)
    ew = c.exit_walk('ZZZZ', '2023-06-01', ts('2023-06-01', '09:40:00'), 10.0, 9.0, 'TRAIN')
    assert ew is None


def test_exit_walk_degenerate_r_prime():
    ew = c.exit_walk('ZZZZ', '2025-03-10', ts('2025-03-10', '09:40:00'), 9.0, 9.0, 'TRAIN')
    assert ew is None


# --------------------------------------------------------------------------------------------------
# cost constants sanity (units: bps, matching cell_1478.py's blend formula and the PREREG's EOD bps)
# --------------------------------------------------------------------------------------------------
def test_cost_constants():
    assert c.SLIP_STOP_BPS['TRAIN'] == pytest.approx(0.88 * 2.9 + 0.12 * 94.0)
    assert c.SLIP_STOP_BPS['VAL'] == pytest.approx(0.88 * 3.2 + 0.12 * 76.0)
    assert c.EOD_BPS == {'TRAIN': 11.5, 'VAL': 9.7}
    assert c.LOCK_TRIGGER_R == 1.75
    assert c.LOCK_STOP_R == 0.5
    assert c.R_DENOM == 375.0


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
