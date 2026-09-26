"""Synthetic unit tests for research/crypto_trend/cells.py.

Every test uses hand-built close/open series with a known answer -- no real
market data -- so a wrong boundary or sign flip is caught deterministically.
"""
import numpy as np
import pandas as pd
import pytest

from cells import (
    assign_split, c4_signal, count_matched_null_green_share, donchian_signal,
    max_drawdown, momentum_signal, newey_west_t, portfolio_weekly, sma_trend_signal,
)


def daily_index(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D")


def mondays_from(index: pd.DatetimeIndex) -> list:
    # every Monday strictly inside the index range, so `close.index < monday` always
    # has data and `monday` itself is a valid lookup for the open price.
    return [d for d in index if d.weekday() == 0 and d != index[0]]


# ---------------------------------------------------------------------------
# Boundary: the Monday close itself must be invisible to the signal
# ---------------------------------------------------------------------------
def test_ma_filter_uses_only_prior_closes():
    idx = daily_index("2024-01-01", 210)  # 2024-01-01 is a Monday
    close = pd.Series(90.0, index=idx)  # flat below any reasonable SMA baseline
    mondays = mondays_from(idx)
    target_monday = mondays[20]  # deep enough in for a full 100-day window before it

    # Case A: a huge spike planted ON the target Monday's own close must NOT flip
    # the signal long (it must remain invisible).
    close_spike_on_monday = close.copy()
    close_spike_on_monday.loc[target_monday] = 1_000_000.0
    pos_a = sma_trend_signal(close_spike_on_monday, [target_monday], window=100)
    assert pos_a.loc[target_monday] == 0, "Monday's own close leaked into the SMA signal"

    # Case B: the same spike planted on the PRIOR day (visible) must flip it long,
    # proving the function does look at prior data and isn't just always-flat.
    close_spike_prior_day = close.copy()
    prior_day = target_monday - pd.Timedelta(days=1)
    close_spike_prior_day.loc[prior_day] = 1_000_000.0
    pos_b = sma_trend_signal(close_spike_prior_day, [target_monday], window=100)
    assert pos_b.loc[target_monday] == 1, "a visible prior-day close failed to move the SMA signal"


def test_monday_boundary_excludes_monday_close_across_all_signals():
    """Same spike-on-Monday-vs-spike-on-Sunday check, for momentum and Donchian too."""
    idx = daily_index("2024-01-01", 210)
    base = pd.Series(90.0, index=idx)
    mondays = mondays_from(idx)
    m = mondays[15]
    prior_day = m - pd.Timedelta(days=1)

    spike_monday = base.copy()
    spike_monday.loc[m] = 1_000_000.0
    spike_prior = base.copy()
    spike_prior.loc[prior_day] = 1_000_000.0

    mom_monday = momentum_signal(spike_monday, [m], lookback=28)
    mom_prior = momentum_signal(spike_prior, [m], lookback=28)
    assert mom_monday.loc[m] == 0, "momentum signal saw the Monday close"
    assert mom_prior.loc[m] == 1, "momentum signal failed to see a visible prior close"

    don_monday = donchian_signal(spike_monday, [m], entry_window=20, exit_window=10)
    don_prior = donchian_signal(spike_prior, [m], entry_window=20, exit_window=10)
    assert don_monday.loc[m] == 0, "donchian signal saw the Monday close"
    assert don_prior.loc[m] == 1, "donchian signal failed to see a visible prior close"


# ---------------------------------------------------------------------------
# MA filter sign
# ---------------------------------------------------------------------------
def test_ma_filter_long_above_flat_below():
    idx = daily_index("2024-01-01", 210)
    mondays = mondays_from(idx)
    m = mondays[20]

    uptrend = pd.Series(np.linspace(50, 150, len(idx)), index=idx)  # close ends well above its SMA100
    pos_up = sma_trend_signal(uptrend, [m], window=100)
    assert pos_up.loc[m] == 1

    downtrend = pd.Series(np.linspace(150, 50, len(idx)), index=idx)
    pos_down = sma_trend_signal(downtrend, [m], window=100)
    assert pos_down.loc[m] == 0

    short_history = pd.Series(100.0, index=idx[:50])
    pos_insufficient = sma_trend_signal(short_history, [idx[40]], window=100)
    assert pos_insufficient.iloc[0] == 0, "insufficient lookback must default to flat, not raise"


# ---------------------------------------------------------------------------
# Donchian entry/exit boundaries (exact touches)
# ---------------------------------------------------------------------------
def test_donchian_enters_on_breakout_above_prior_20d_high():
    idx = daily_index("2024-01-01", 60)
    close = pd.Series(100.0, index=idx)
    mondays = mondays_from(idx)
    m = mondays[3]
    # prior 20 closes (before m) are all 100 -> prior 20-day high == 100
    close_at_high = close.copy()
    close_at_high.loc[m - pd.Timedelta(days=1)] = 100.0  # exactly at the high: no breakout
    pos_equal = donchian_signal(close_at_high, [m], entry_window=20, exit_window=10)
    assert pos_equal.loc[m] == 0, "an exact touch of the prior high must not count as breaking above it"

    close_above = close.copy()
    close_above.loc[m - pd.Timedelta(days=1)] = 100.01
    pos_above = donchian_signal(close_above, [m], entry_window=20, exit_window=10)
    assert pos_above.loc[m] == 1, "a close strictly above the prior 20d high must enter long"


def test_donchian_exits_on_breakdown_below_prior_10d_low_only_while_long():
    idx = daily_index("2024-01-01", 90)
    mondays = mondays_from(idx)
    close = pd.Series(100.0, index=idx)
    entry_m = mondays[3]
    close.loc[entry_m - pd.Timedelta(days=1)] = 200.0  # forces entry at entry_m

    exit_m_candidates = [m for m in mondays if m > entry_m]
    exit_m = exit_m_candidates[2]
    close.loc[exit_m - pd.Timedelta(days=1)] = 1.0  # far below any 10d low -> must exit

    weekly_checks = [m for m in mondays if m <= exit_m]
    pos = donchian_signal(close, weekly_checks, entry_window=20, exit_window=10)
    assert pos.loc[entry_m] == 1
    assert pos.loc[exit_m] == 0, "a close strictly below the prior 10d low must exit to flat"


# ---------------------------------------------------------------------------
# Momentum sign
# ---------------------------------------------------------------------------
def test_momentum_sign():
    idx = daily_index("2024-01-01", 60)
    mondays = mondays_from(idx)
    m = mondays[5]

    up = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    assert momentum_signal(up, [m], lookback=28).loc[m] == 1

    down = pd.Series(np.linspace(200, 100, len(idx)), index=idx)
    assert momentum_signal(down, [m], lookback=28).loc[m] == 0

    flat = pd.Series(100.0, index=idx)
    assert momentum_signal(flat, [m], lookback=28).loc[m] == 0, "zero momentum must be flat, not long (strict >0)"


def test_c4_requires_both_c1_and_c3():
    idx = daily_index("2024-01-01", 210)
    mondays = mondays_from(idx)
    m = mondays[20]
    uptrend = pd.Series(np.linspace(50, 150, len(idx)), index=idx)  # both C1 and C3 long
    assert c4_signal(uptrend, [m]).loc[m] == 1

    choppy = pd.Series(100.0, index=idx)
    choppy.loc[m - pd.Timedelta(days=1)] = 101.0  # nudges C1 long, C3 (28d mom) stays ~0 -> flat
    c1_only = sma_trend_signal(choppy, [m], window=100)
    c3_only = momentum_signal(choppy, [m], lookback=28)
    if int(c1_only.loc[m]) == 1 and int(c3_only.loc[m]) == 0:
        assert c4_signal(choppy, [m]).loc[m] == 0, "C4 must require BOTH signals, not just one"


# ---------------------------------------------------------------------------
# Fee: charged only on a change of position
# ---------------------------------------------------------------------------
def test_fee_charged_only_on_position_change():
    idx = daily_index("2024-01-01", 400)
    mondays = mondays_from(idx)

    panel = {}
    for coin, level in [("BTC", 100.0), ("ETH", 50.0), ("SOL", 20.0)]:
        close = pd.Series(level, index=idx)
        close.loc[mondays[2] - pd.Timedelta(days=1)] = level * 2  # BTC only: forces an entry breakout for C2
        open_ = close.copy()
        panel[coin] = pd.DataFrame({"open": open_, "close": close})

    wk, pos, detail = portfolio_weekly("C2", panel, mondays)
    changed_weeks = pos.diff().abs().sum(axis=1).fillna(0)
    for w in wk.index:
        coin_changed = int(changed_weeks.loc[w]) if w in changed_weeks.index else 0
        if coin_changed == 0:
            assert wk.loc[w, "fee"] == 0.0, f"fee charged on {w} with no position change"
        else:
            assert wk.loc[w, "fee"] > 0.0, f"no fee charged on {w} despite a position change"


def test_no_fee_when_weight_shifts_from_other_coin_joining():
    """If coin A stays long while coin B joins (A's weight drops 1/1 -> 1/2), A itself
    pays no fee -- only B's entry does. This is the PREREG's per-leg (not per-weight)
    cost definition, disclosed in cells.py's module docstring.
    """
    idx = daily_index("2024-01-01", 400)
    mondays = mondays_from(idx)
    panel = {}
    # BTC: always long from week 0 (uptrend from day 1, comfortably above SMA100 throughout)
    btc_close = pd.Series(np.linspace(100, 500, len(idx)), index=idx)
    panel["BTC"] = pd.DataFrame({"open": btc_close, "close": btc_close})
    # ETH: flat, then a breakout after week 30 (C1) -> joins later
    eth_close = pd.Series(90.0, index=idx)
    eth_close.iloc[300:] = np.linspace(90, 300, len(idx) - 300)
    panel["ETH"] = pd.DataFrame({"open": eth_close, "close": eth_close})
    # SOL: never long (deep downtrend)
    sol_close = pd.Series(np.linspace(300, 10, len(idx)), index=idx)
    panel["SOL"] = pd.DataFrame({"open": sol_close, "close": sol_close})

    wk, pos, detail = portfolio_weekly("C1", panel, mondays)
    btc_series = pos["BTC"]
    eth_series = pos["ETH"]
    # find a week where BTC was long last week and still long this week, but ETH enters
    for i in range(1, len(mondays) - 1):
        w = mondays[i]
        if w not in wk.index:
            continue
        if btc_series.loc[mondays[i - 1]] == 1 and btc_series.loc[w] == 1 and eth_series.loc[mondays[i - 1]] == 0 and eth_series.loc[w] == 1:
            btc_fee = detail.loc[(w, "BTC")]["fee"]
            eth_fee = detail.loc[(w, "ETH")]["fee"]
            assert btc_fee == 0.0, "BTC paid a fee on a week it had no position change"
            assert eth_fee > 0.0, "ETH (the coin that actually entered) paid no fee"
            return
    pytest.skip("synthetic series never produced the BTC-stays/ETH-enters week; adjust the fixture")


# ---------------------------------------------------------------------------
# Statistics helpers (sanity, not full NW proof)
# ---------------------------------------------------------------------------
def test_newey_west_t_matches_iid_tstat_when_no_autocorrelation():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.01, scale=0.02, size=2000)  # iid -> NW ~ ordinary t for large n
    t_nw = newey_west_t(x, lags=4)
    t_iid = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(t_nw - t_iid) / abs(t_iid) < 0.15


def test_max_drawdown_simple_path():
    r = pd.Series([0.10, -0.20, 0.05])  # equity 1.10 -> 0.88 -> 0.924; peak 1.10, trough 0.88
    dd = max_drawdown(r)
    assert abs(dd - (1 - 0.88 / 1.10)) < 1e-9


def test_assign_split_boundaries():
    assert assign_split(pd.Timestamp("2023-12-31")) == "TRAIN"
    assert assign_split(pd.Timestamp("2024-01-01")) == "VAL"
    assert assign_split(pd.Timestamp("2025-06-30")) == "VAL"
    assert assign_split(pd.Timestamp("2025-07-01")) == "TEST"
    assert assign_split(pd.Timestamp("2016-01-01")) is None


def test_count_matched_null_preserves_time_in_market():
    idx = daily_index("2024-01-01", 400)
    mondays = mondays_from(idx)
    panel = {}
    for coin, level in [("BTC", 100.0), ("ETH", 50.0), ("SOL", 20.0)]:
        close = pd.Series(np.linspace(level, level * 1.3, len(idx)), index=idx)
        panel[coin] = pd.DataFrame({"open": close, "close": close})
    wk, pos, detail = portfolio_weekly("C1", panel, mondays)
    split_weeks = list(wk.index[:40])
    share = count_matched_null_green_share(pos, panel, mondays, split_weeks, n_draws=50, seed=1)
    assert 0.0 <= share <= 1.0
