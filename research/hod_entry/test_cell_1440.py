"""Unit tests for cell_1440 (stop-distance floor/cap), synthetic — no DB access."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cell_1440 as c1440  # noqa: E402


def bars(rows):
    """rows: list of (m, o, h, l, c, v)."""
    return pd.DataFrame(rows, columns=['m', 'o', 'h', 'l', 'c', 'v'])


def make_row(fill, stop, R=None, cost_R=0.1, fill_min=100.4):
    return pd.Series(dict(fill=fill, stop=stop, R=(R if R is not None else fill - stop),
                           cost_R=cost_R, fill_min=fill_min, day='2025-07-01', symbol='TEST'))


def test_floor_raises_R():
    """d_old = 0.002 (< 0.008 floor) -> d_new = 0.008, new_stop closer to fill is WRONG direction;
    floor RAISES the stop distance so new_stop is FARTHER from fill -> R_new > R_old."""
    fill = 100.0
    stop = fill * (1 - 0.002)  # d_old 0.2%
    row = make_row(fill, stop)
    b = bars([(100, 100.0, 100.1, 99.9, 100.0, 1000),
              (101, 100.0, 100.2, 99.7, 100.1, 1000),
              (955, 101.0, 101.0, 101.0, 101.0, 1000)])
    d_new = c1440.new_stop_distance(np.array([0.002]))[0]
    assert d_new == pytest.approx(0.008)
    r = c1440.recompute_one(row, b, d_new)
    assert r is not None
    R_old = fill - stop
    assert r['R'] > R_old
    assert r['stop'] == pytest.approx(fill * (1 - 0.008))


def test_cap_lowers_R():
    """d_old = 0.05 (> 0.03 cap) -> d_new = 0.03 -> stop moves closer to fill -> R_new < R_old."""
    fill = 100.0
    stop = fill * (1 - 0.05)
    row = make_row(fill, stop)
    b = bars([(100, 100.0, 100.1, 99.9, 100.0, 1000),
              (101, 100.0, 100.2, 96.0, 100.1, 1000),  # below both old and new stop
              (955, 96.5, 96.5, 96.5, 96.5, 1000)])
    d_new = c1440.new_stop_distance(np.array([0.05]), cap_pct=0.03)[0]
    assert d_new == pytest.approx(0.03)
    r = c1440.recompute_one(row, b, d_new)
    R_old = fill - stop
    assert r['R'] < R_old
    assert r['stop'] == pytest.approx(fill * (1 - 0.03))


def test_unchanged_stop_zero_delta():
    """d_old already inside [floor, cap] -> d_new == d_old (no recompute triggered upstream)."""
    d_old = 0.015
    d_new_a = c1440.new_stop_distance(np.array([d_old]))[0]
    d_new_b = c1440.new_stop_distance(np.array([d_old]), cap_pct=0.03)[0]
    assert d_new_a == pytest.approx(d_old)
    assert d_new_b == pytest.approx(d_old)
    assert np.isclose(d_new_a, d_old, atol=1e-12)


def test_stopped_inside_fill_bar():
    """Fill bar's own low breaches the new (tighter) stop -> stopped inside the fill bar,
    why='stop_infill', raw R == -1 exactly."""
    fill = 100.0
    stop = fill * (1 - 0.05)  # will be capped to 0.03 -> new_stop = 97.0
    row = make_row(fill, stop, fill_min=100.7)
    b = bars([(100, 100.0, 100.2, 96.5, 96.6, 1000),  # fill bar m=100, low 96.5 <= new_stop 97.0
              (101, 96.6, 97.0, 96.0, 96.5, 1000),
              (955, 96.5, 96.5, 96.5, 96.5, 1000)])
    d_new = c1440.new_stop_distance(np.array([0.05]), cap_pct=0.03)[0]
    r = c1440.recompute_one(row, b, d_new)
    assert r['why'] == 'stop_infill'
    assert r['stopped_infill'] is True
    assert r['raw_R'] == pytest.approx(-1.0)
    assert r['exit_m'] == 100


def test_cost_dollar_held_fixed():
    """cost_$ = cost_R_old * R_old must be preserved: cost_R_new * R_new == cost_R_old * R_old."""
    fill, stop, cost_R = 100.0, 95.0, 0.2  # R_old = 5, cost_$ = 1.0
    row = make_row(fill, stop, cost_R=cost_R)
    b = bars([(100, 100.0, 100.1, 99.9, 100.0, 1000),
              (101, 100.0, 100.2, 99.0, 100.1, 1000),
              (955, 99.5, 99.5, 99.5, 99.5, 1000)])
    d_new = c1440.new_stop_distance(np.array([0.05]), cap_pct=0.03)[0]  # 0.05 -> capped 0.03
    r = c1440.recompute_one(row, b, d_new)
    cost_dollar_old = cost_R * (fill - stop)
    cost_dollar_new = r['cost_R'] * r['R']
    assert cost_dollar_new == pytest.approx(cost_dollar_old)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
