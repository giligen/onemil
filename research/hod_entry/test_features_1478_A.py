"""Unit tests for build_features_1478_A.bar_features_for_fill on synthetic bars.

Every assert targets ONE causality or arithmetic property named in PREREG_1478.md / FEATURES_A.md:
bars at or after fill_min must never affect the output, and each bar-derived quantity must match
a hand-computed value on a small synthetic day.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from research.hod_entry.build_features_1478_A import bar_features_for_fill, RTH_OPEN_M  # noqa: E402


def make_bars(rows):
    """rows: list of (m, o, h, l, c, v) -> DataFrame with the expected column names."""
    return pd.DataFrame(rows, columns=['m', 'o', 'h', 'l', 'c', 'v'])


def test_basic_arithmetic():
    """10 RTH bars from 09:30; fill at minute 580 (arm bar j = bar at m=579, the 10th bar)."""
    rows = [(RTH_OPEN_M + i, 10.0 + i * 0.1, 10.0 + i * 0.1 + 0.05, 10.0 + i * 0.1 - 0.05,
              10.0 + i * 0.1, 1000.0) for i in range(10)]
    bars = make_bars(rows)
    feat = bar_features_for_fill(bars, fill_min=RTH_OPEN_M + 10, level=10.9)
    assert feat['n_bars_j'] == 10
    assert feat['arm_m'] == RTH_OPEN_M + 9
    assert feat['close_j'] == pytest.approx(10.0 + 9 * 0.1)
    assert feat['dist_from_open_pct'] == pytest.approx((10.9 - 10.0) / 10.0 * 100)
    assert feat['cum_volume_j'] == pytest.approx(10000.0)
    assert feat['bar_density_j'] == pytest.approx(1.0)   # 10 consecutive bars, no gaps


def test_no_bar_before_fill_min_returns_empty():
    """A fill_min at or before every bar's minute -> no arm bar -> {} (NaN row, never fabricated)."""
    rows = [(RTH_OPEN_M, 10.0, 10.1, 9.9, 10.0, 1000.0)]
    bars = make_bars(rows)
    feat = bar_features_for_fill(bars, fill_min=RTH_OPEN_M, level=10.0)
    assert feat == {}


def test_future_bars_never_used():
    """Adding a bar AFTER fill_min with an extreme price/volume must not move any feature --
    the causality contract of arm_bar_features (bars strictly before fill_min only)."""
    rows = [(RTH_OPEN_M + i, 10.0, 10.1, 9.9, 10.0, 1000.0) for i in range(5)]
    bars_before = make_bars(rows)
    feat_before = bar_features_for_fill(bars_before, fill_min=RTH_OPEN_M + 5, level=10.0)

    rows_with_future = rows + [(RTH_OPEN_M + 5, 999.0, 999.0, 999.0, 999.0, 10**9)]
    bars_after = make_bars(rows_with_future)
    feat_after = bar_features_for_fill(bars_after, fill_min=RTH_OPEN_M + 5, level=10.0)

    for k in feat_before:
        if isinstance(feat_before[k], float) and np.isnan(feat_before[k]):
            assert np.isnan(feat_after[k])
        else:
            assert feat_before[k] == feat_after[k], f'feature {k} changed when a future bar was added'


def test_arm_index_counts_prior_crosses():
    """3 flat bars, then a bar whose high pierces the running HOD by >= 0.01 (1 prior cross),
    then another cross before the arm bar -- arm_index must equal the count of such i -> i+1 pairs
    with i+1 <= j (never counting a cross that needs bar j+1)."""
    rows = [
        (RTH_OPEN_M + 0, 10.0, 10.00, 9.95, 10.00, 500.0),
        (RTH_OPEN_M + 1, 10.0, 10.00, 9.95, 10.00, 500.0),
        (RTH_OPEN_M + 2, 10.0, 10.02, 9.95, 10.00, 500.0),   # h=10.02 >= running_hod(10.00)+0.01 -> cross #1 (i=1->2)
        (RTH_OPEN_M + 3, 10.0, 10.02, 9.95, 10.00, 500.0),
        (RTH_OPEN_M + 4, 10.0, 10.05, 9.95, 10.00, 500.0),   # h=10.05 >= running_hod(10.02)+0.01 -> cross #2 (i=3->4)
        (RTH_OPEN_M + 5, 10.0, 10.05, 9.95, 10.00, 500.0),   # arm bar j (bar index 5), fill_min = +6
    ]
    bars = make_bars(rows)
    feat = bar_features_for_fill(bars, fill_min=RTH_OPEN_M + 6, level=10.05)
    assert feat['arm_index'] == 2


def test_level_touches_and_pullback_depth():
    """A level of 10.10: bars whose [l,h] range enters the 0.2% band around it are touches;
    pullback_depth_pct measures the drop from the pre-consolidation high into the K-bar window."""
    rows = [
        (RTH_OPEN_M + 0, 10.00, 10.30, 9.90, 10.20, 1000.0),   # pre-consolidation high = 10.30
        (RTH_OPEN_M + 1, 10.10, 10.11, 10.09, 10.10, 800.0),   # touches level band
        (RTH_OPEN_M + 2, 10.05, 10.06, 10.00, 10.02, 800.0),   # consolidation low bar (lowest low)
        (RTH_OPEN_M + 3, 10.05, 10.10, 10.02, 10.08, 800.0),
        (RTH_OPEN_M + 4, 10.05, 10.10, 10.03, 10.08, 800.0),
        (RTH_OPEN_M + 5, 10.05, 10.10, 10.03, 10.08, 800.0),   # arm bar j (K=5 window = bars 1..5)
    ]
    bars = make_bars(rows)
    feat = bar_features_for_fill(bars, fill_min=RTH_OPEN_M + 6, level=10.10)
    assert feat['level_touches'] >= 1
    expected_pullback = (10.30 - 10.00) / 10.30 * 100   # pre-high (bar 0) vs min low of bars 1..5 (10.00)
    assert feat['pullback_depth_pct'] == pytest.approx(expected_pullback)


def test_premarket_dollar_volume_isolated_from_rth():
    """pm_dollar_vol must sum ONLY bars in [240, 570) and ignore RTH bars entirely."""
    pm_rows = [(300, 5.0, 5.1, 4.9, 5.0, 200.0), (569, 5.0, 5.1, 4.9, 5.0, 300.0)]
    rth_rows = [(RTH_OPEN_M + i, 10.0, 10.1, 9.9, 10.0, 1000.0) for i in range(3)]
    bars = make_bars(pm_rows + rth_rows)
    feat = bar_features_for_fill(bars, fill_min=RTH_OPEN_M + 3, level=10.0)
    expected_pm = 200.0 * 5.0 + 300.0 * 5.0
    assert feat['pm_dollar_vol'] == pytest.approx(expected_pm)


def test_halt_proxy_detects_gap():
    """A >=5 minute gap between consecutive RTH bars before j must set halt_proxy=1; a 1-minute
    cadence with no gap must set it 0."""
    rows_gap = [(RTH_OPEN_M, 10.0, 10.1, 9.9, 10.0, 500.0),
                (RTH_OPEN_M + 6, 10.0, 10.1, 9.9, 10.0, 500.0)]
    feat_gap = bar_features_for_fill(make_bars(rows_gap), fill_min=RTH_OPEN_M + 7, level=10.0)
    assert feat_gap['halt_proxy'] == 1

    rows_ok = [(RTH_OPEN_M + i, 10.0, 10.1, 9.9, 10.0, 500.0) for i in range(3)]
    feat_ok = bar_features_for_fill(make_bars(rows_ok), fill_min=RTH_OPEN_M + 3, level=10.0)
    assert feat_ok['halt_proxy'] == 0


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
