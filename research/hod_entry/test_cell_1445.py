#!/usr/bin/env python3
"""Unit tests for research/hod_entry/cell_1445.py (PREREG_1445.md).

Synthetic-data tests only (no live DB/parquet reads) covering the causality-critical pieces named
in the PREREG's execution plan: corrected-cost recovery for both exit_half_src cases, arm-bar
gap/range replay using only bars <= j, bar-density share, prior-session/20-session-high excluding
the current day, and the placebo's full-day (look-ahead, by design) computation.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell_1445 as c1445


# --------------------------------------------------------------------------------------------
# Corrected cost
# --------------------------------------------------------------------------------------------

def test_corrected_cost_fill_instant_recovers_half_entry():
    """exit_half_src == 'fill_instant': exit_half == half_entry, so half_entry =
    (cost_R*R - 0.0002*exit_price)/2 -- construct a row from a KNOWN half_entry and check round-trip."""
    R, exit_price, true_half_entry = 2.0, 50.0, 0.05
    exit_half = true_half_entry
    cost_R = (true_half_entry + exit_half + 0.0002 * exit_price) / R
    fills = pd.DataFrame({
        'R': [R], 'cost_R': [cost_R], 'exit_price': [exit_price], 'net_R': [1.9],
        'exit_half_src': ['fill_instant'], 'day': ['2025-07-01'], 'symbol': ['AAA'],
    })
    half_entry, net_R_costfix, fallback = c1445.corrected_cost(fills, nbbo_lookup={})
    assert half_entry[0] == pytest.approx(true_half_entry, abs=1e-9)
    assert not fallback[0]
    assert net_R_costfix[0] == pytest.approx(1.9 + true_half_entry / R, abs=1e-9)


def test_corrected_cost_nbbo_recovers_half_entry():
    """exit_half_src == 'nbbo': exit_half = spread_mean/2 from nbbo.csv; half_entry =
    cost_R*R - exit_half - 0.0002*exit_price."""
    R, exit_price, true_half_entry, spread_mean = 1.5, 30.0, 0.08, 0.20
    exit_half = spread_mean / 2.0
    cost_R = (true_half_entry + exit_half + 0.0002 * exit_price) / R
    fills = pd.DataFrame({
        'R': [R], 'cost_R': [cost_R], 'exit_price': [exit_price], 'net_R': [1.0],
        'exit_half_src': ['nbbo'], 'day': ['2025-07-02'], 'symbol': ['BBB'],
    })
    nbbo_lookup = {('2025-07-02', 'BBB'): spread_mean}
    half_entry, net_R_costfix, fallback = c1445.corrected_cost(fills, nbbo_lookup)
    assert half_entry[0] == pytest.approx(true_half_entry, abs=1e-9)
    assert not fallback[0]


def test_corrected_cost_nbbo_join_failure_falls_back_and_is_counted():
    """A 'nbbo'-src row with no (day,symbol) match in nbbo.csv falls back to the fill_instant
    formula and is counted in `fallback`, never silently dropped."""
    R, exit_price = 1.0, 20.0
    cost_R = 0.10
    fills = pd.DataFrame({
        'R': [R], 'cost_R': [cost_R], 'exit_price': [exit_price], 'net_R': [0.5],
        'exit_half_src': ['nbbo'], 'day': ['2025-07-03'], 'symbol': ['NOJOIN'],
    })
    half_entry, net_R_costfix, fallback = c1445.corrected_cost(fills, nbbo_lookup={})
    expected = (cost_R * R - 0.0002 * exit_price) / 2.0
    assert half_entry[0] == pytest.approx(expected, abs=1e-9)
    assert fallback[0]


# --------------------------------------------------------------------------------------------
# Arm-bar features: causal (bars <= j only)
# --------------------------------------------------------------------------------------------

def _bars(rows):
    return pd.DataFrame(rows, columns=['m', 'o', 'h', 'l', 'c', 'v'])


def test_arm_bar_features_ignores_bars_at_or_after_fill_min():
    """A bar AT or AFTER fill_min must never move gap/range/close_j -- the classic look-ahead bug."""
    bars = _bars([
        (570, 10, 10.5, 9.8, 10.2, 1000),   # 09:30
        (571, 10.2, 10.3, 10.1, 10.25, 500),  # 09:31 -- last bar strictly before fill_min=572
        (572, 10.25, 50.0, 10.2, 49.0, 999999),  # 09:32 -- AFTER fill_min, huge spike; must be ignored
    ])
    feat = c1445.arm_bar_features(bars, fill_min=572.0)
    assert feat['close_j'] == pytest.approx(10.25)
    assert feat['running_high_j'] == pytest.approx(10.5)   # max(bar0.h, bar1.h) -- NOT 50.0
    assert feat['running_low_j'] == pytest.approx(9.8)
    assert feat['arm_m'] == 571


def test_arm_bar_features_no_bar_before_fill_min_returns_none():
    bars = _bars([(600, 10, 10.5, 9.8, 10.2, 1000)])
    assert c1445.arm_bar_features(bars, fill_min=580.0) is None


def test_bar_density_share():
    """bar_density = (# distinct RTH minutes with a bar through j) / (minutes 09:30..j inclusive)."""
    # 09:30 (570) through 09:34 (574) = 5 possible minutes; only 3 bars present.
    bars = _bars([(570, 1, 1, 1, 1, 1), (571, 1, 1, 1, 1, 1), (574, 1, 1, 1, 1, 1)])
    feat = c1445.arm_bar_features(bars, fill_min=575.0)
    assert feat['n_bars_j'] == 3
    assert feat['bar_density_j'] == pytest.approx(3 / 5)


# --------------------------------------------------------------------------------------------
# Multi-day-high terms: prior session / 20-session high exclude the CURRENT day
# --------------------------------------------------------------------------------------------

def test_prev_close_and_prev_high_exclude_current_day():
    dates = pd.date_range('2025-01-01', periods=25, freq='B')
    df = pd.DataFrame({
        'instrument_id': [1] * 25,
        'bar_date': dates,
        'symbol': ['ZZZ'] * 25,
        'open': np.arange(25) + 1.0,
        'high': np.arange(25) + 2.0,     # day i's high = i+2, strictly increasing
        'low': np.arange(25) + 0.5,
        'close': np.arange(25) + 1.5,
        'volume': [1000] * 25,
    })
    tmp_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(tmp_dir, '_test_daily_panel.parquet')
    df.to_parquet(parquet_path)
    try:
        c1445.DAILY_PARQUET = parquet_path
        panel = c1445.build_daily_panel({1})
        row24 = panel[panel.bar_date == dates[24]].iloc[0]
        # day 24's high is 26 (24+2); prev_high must be day 23's high = 25, NEVER 26.
        assert row24.prev_high == pytest.approx(23 + 2.0)
        assert row24.prev_close == pytest.approx(23 + 1.5)
        # high20 = max high of the 20 sessions strictly before day 24 = days 4..23 -> high = 25 (day23)
        assert row24.high20 == pytest.approx(23 + 2.0)
        # day 19 has only 19 prior sessions (days 0..18) -- high20 must be NaN (min_periods=20)
        row19 = panel[panel.bar_date == dates[19]].iloc[0]
        assert np.isnan(row19.high20)
    finally:
        os.remove(parquet_path)


# --------------------------------------------------------------------------------------------
# Placebo: FULL day, by design (report-only, look-ahead)
# --------------------------------------------------------------------------------------------

def test_placebo_condition_uses_full_day_range():
    """The placebo (1,455) is the ONLY cell allowed to use the full day's high/low/close -- verify
    the condition function reads day_high/day_low/day_close (not the arm-bar running values)."""
    df = pd.DataFrame({
        'prev_close': [10.0], 'close_j': [10.5], 'float_known': [True], 'float_shares': [1e6],
        'mover_j': [2.0],            # arm-bar mover is tiny -- would NOT qualify 1,448
        'level': [11.0], 'prev_high': [10.8], 'high20': [10.9],
        'bar_density_j': [0.5], 'dollar_vol_j': [0], 'spread_frac': [0.01],
        'placebo_range': [15.0],     # but the FULL-DAY range is a big mover
        'placebo_gap_close': [0.0], 'day_close': [10.5], 'day_high': [11.5], 'day_low': [10.0],
    })
    cond = c1445.cell_conditions(df)
    assert bool(cond['1455'].iloc[0])       # placebo qualifies on the full-day range
    assert not bool(cond['1448'].iloc[0])   # the causal mover cell does NOT (arm-bar mover only 2%)


# --------------------------------------------------------------------------------------------
# Scoring primitives
# --------------------------------------------------------------------------------------------

def test_day_clustered_t_matches_iid_when_one_row_per_day():
    """With one observation per day, the day-clustered t reduces to the usual OLS t (sanity check)."""
    y = pd.Series([1.0, 2.0, -1.0, 0.5, 1.5])
    days = pd.Series(['d1', 'd2', 'd3', 'd4', 'd5'])
    t = c1445.day_clustered_t(y, days)
    import statsmodels.api as sm
    ref = sm.OLS(y.to_numpy(), np.ones((5, 1))).fit().tvalues[0]
    assert t == pytest.approx(ref, rel=1e-6)


def test_ex_top5_mean_drops_top_values():
    y = list(range(1, 101))   # 1..100, mean 50.5
    m = c1445.ex_top5_mean(pd.Series(y))
    # top 5% (5 values: 96..100) dropped -> mean of 1..95
    assert m == pytest.approx(np.mean(range(1, 96)))


def test_winner_capped_mean_caps_at_3R():
    y = pd.Series([10.0, -1.0, 0.5])
    assert c1445.winner_capped_mean(y, cap=3.0) == pytest.approx((3.0 - 1.0 + 0.5) / 3)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
