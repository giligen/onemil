"""Unit tests for research/thermo/thermo.py (PREREG.md cells 1,420-1,422). Synthetic data only -- no
CSVs, no DB, no network. Proves causality (SPEC.md step 2: T(d) never uses day d or later; the
expanding median excludes the current day) plus min_n / min_hist undefined and the small pure
functions' arithmetic."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thermo import (daily_outcome_series, ex_top_pct, hot_flags, quintile_table, split_stats,
                     thermometer)


def _synthetic_daily(n_days=200, n_per_day=10, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2024-01-01', periods=n_days).strftime('%Y-%m-%d').tolist()
    values = [rng.random(n_per_day) for _ in range(n_days)]
    return pd.Series(values, index=pd.Index(dates, name='date'))


# --------------------------------------------------------------------------- causality (SPEC step 2)
def test_thermometer_causal_no_lookahead():
    """Mutating day d's outcomes must NOT change T(d) (causal window is strictly prior days), but MUST
    change T(d+1) (whose window pools day d), and must not change any T(i) for i < d."""
    daily = _synthetic_daily(n_days=150, n_per_day=10)
    T_before = thermometer(daily, window=20, min_n=40)
    d = 100
    mutated = daily.copy()
    mutated.iloc[d] = mutated.iloc[d] + 1000.0
    T_after = thermometer(mutated, window=20, min_n=40)

    a, b = T_before.iloc[d], T_after.iloc[d]
    assert a == b or (np.isnan(a) and np.isnan(b)), "T(d) must not use day d's own outcome"

    a, b = T_before.iloc[d + 1], T_after.iloc[d + 1]
    assert a != b and not (np.isnan(a) and np.isnan(b)), "T(d+1) must change (its window pools day d)"

    for i in range(0, d):
        a, b = T_before.iloc[i], T_after.iloc[i]
        assert a == b or (np.isnan(a) and np.isnan(b)), f"T({i}) for i<d must be unaffected"


def test_thermometer_oracle_uses_same_day():
    """end_lag=0 (oracle/look-ahead) DOES pool day d's own outcome into T(d) -- the opposite of the
    causal default, and the basis for the PREREG 'oracle bound' adversary lens."""
    daily = _synthetic_daily(n_days=150, n_per_day=10)
    T_oracle_before = thermometer(daily, window=20, min_n=40, end_lag=0)
    d = 100
    mutated = daily.copy()
    mutated.iloc[d] = mutated.iloc[d] + 1000.0
    T_oracle_after = thermometer(mutated, window=20, min_n=40, end_lag=0)
    assert T_oracle_before.iloc[d] != T_oracle_after.iloc[d]


def test_thermometer_stale_ends_60_days_earlier():
    """end_lag=61 (stale) at day i must equal the causal value (end_lag=1) computed at day i-60 --
    i.e. it is literally the causal thermometer shifted 60 trading days into the past."""
    daily = _synthetic_daily(n_days=200, n_per_day=10)
    T_causal = thermometer(daily, window=20, min_n=40, end_lag=1)
    T_stale = thermometer(daily, window=20, min_n=40, end_lag=61)
    for i in range(60, len(daily)):
        a, b = T_stale.iloc[i], T_causal.iloc[i - 60]
        assert a == b or (np.isnan(a) and np.isnan(b))


def test_hot_flags_expanding_median_excludes_current_day():
    """Golden-value test: hot_flags(d) must equal (T(d) > median of T at STRICTLY earlier days), an
    independently written reference computed directly from vals[:i] (never vals[:i+1])."""
    idx = pd.Index([f"day{i:03d}" for i in range(70)])
    rng = np.random.default_rng(1)
    T = pd.Series(rng.random(70), index=idx)
    min_hist = 60
    hot = hot_flags(T, min_hist=min_hist)
    vals = T.values
    for i in range(len(idx)):
        earlier = vals[:i]
        if len(earlier) < min_hist:
            assert np.isnan(hot.iloc[i]), f"day {i}: expected undefined (< {min_hist} earlier values)"
        else:
            expected = float(vals[i] > np.median(earlier))
            assert hot.iloc[i] == expected, f"day {i}: expected {expected}, got {hot.iloc[i]}"


def test_thermometer_drops_nan_outcomes_without_poisoning_the_window():
    """Regression (found live in the b0_trades smoke run: 125 of 12,135 net_R values are NaN --
    undecided-exit rows). A single NaN in a pooled window must not turn every later T(d) into NaN --
    it must be dropped from the pool and from the min_n count, not propagate via np.mean."""
    daily = _synthetic_daily(n_days=60, n_per_day=10)
    with_nan = daily.copy()
    with_nan.iloc[5] = np.append(with_nan.iloc[5][:-1], np.nan)  # one NaN in an early day's outcomes
    T_clean = thermometer(daily, window=20, min_n=40)
    T_nan = thermometer(with_nan, window=20, min_n=40)
    assert T_nan.notna().sum() > 0, "a single NaN outcome must not blank out the whole thermometer"
    # every T(d) whose window doesn't touch day 5 must be identical with/without the NaN
    for i in range(len(daily)):
        if i <= 5:
            continue
        a, b = T_clean.iloc[i], T_nan.iloc[i]
        if i - 20 > 5:  # window no longer reaches day 5
            assert a == b or (np.isnan(a) and np.isnan(b))


# --------------------------------------------------------------------------- undefined thresholds
def test_thermometer_min_n_gives_undefined():
    """3 outcomes/day * 10 days = 30 pooled < min_n=40 even with every day pooled -> always undefined."""
    daily = _synthetic_daily(n_days=10, n_per_day=3)
    T = thermometer(daily, window=20, min_n=40)
    assert T.isna().all()


def test_thermometer_min_n_becomes_defined_once_enough_pooled():
    daily = _synthetic_daily(n_days=30, n_per_day=5)  # 20-day window * 5/day = 100 >= 40 once i>=20
    T = thermometer(daily, window=20, min_n=40)
    assert T.iloc[:8].isna().all()  # too few prior days pooled yet (< 40 obs)
    assert not T.iloc[20:].isna().any()


def test_hot_flags_min_hist_gives_undefined():
    idx = pd.Index([f"day{i:03d}" for i in range(50)])
    T = pd.Series(np.linspace(0, 1, 50), index=idx)
    hot = hot_flags(T, min_hist=60)  # fewer than 60 earlier defined values anywhere in a 50-day series
    assert hot.isna().all()


# --------------------------------------------------------------------------- split_stats / ex_top_pct / quintile / daily_outcome_series
def test_split_stats_recovers_known_difference():
    rng = np.random.default_rng(2)
    n = 400
    day = pd.Series([f"d{i:04d}" for i in range(n)])
    hot = pd.Series(rng.integers(0, 2, n).astype(float))
    R = pd.Series(rng.normal(0, 0.01, n) + hot * 0.5)  # true hot-cold gap = 0.5
    out = split_stats(R, hot, day)
    assert out['hot_minus_cold'] == pytest.approx(0.5, abs=0.15)
    assert out['t_cluster'] > 2
    assert out['n_hot'] + out['n_cold'] == n


def test_split_stats_degenerate_cohort_is_nan_not_crash():
    R = pd.Series([1.0, 2.0, 3.0])
    hot = pd.Series([1.0, 1.0, 1.0])  # no cold cohort at all
    day = pd.Series(['d1', 'd2', 'd3'])
    out = split_stats(R, hot, day)
    assert out['n_cold'] == 0
    assert np.isnan(out['hot_minus_cold'])


def test_ex_top_pct_drops_top_k_like_bf2024_score():
    R = pd.Series([10, 9, 1, 1, 1, 1, 1, 1, 1, 1])  # n=10, k=max(1,round(10*0.05))=1 -> drop the 10
    out = ex_top_pct(R, pct=0.05)
    assert out == pytest.approx(np.mean([9, 1, 1, 1, 1, 1, 1, 1, 1]))


def test_daily_outcome_series_keeps_empty_calendar_days():
    df = pd.DataFrame({'date': ['2024-01-02', '2024-01-02', '2024-01-04'], 'v': [1.0, 2.0, 3.0]})
    calendar = ['2024-01-02', '2024-01-03', '2024-01-04']  # 01-03 has no rows at all
    s = daily_outcome_series(df, 'date', 'v', calendar_dates=calendar)
    assert list(s.index) == calendar
    assert len(s.loc['2024-01-03']) == 0
    assert sorted(s.loc['2024-01-02']) == [1.0, 2.0]


def test_quintile_table_report_only_shape():
    rng = np.random.default_rng(3)
    T = pd.Series(rng.random(100))
    R = pd.Series(rng.normal(0, 1, 100))
    qt = quintile_table(T, R)
    assert set(qt.columns) >= {'q', 'mean', 'count'}
    assert qt['count'].sum() == 100
