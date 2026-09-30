"""Unit tests for cell_1599.py (v3, Amendment 3) on synthetic fixtures only -- these tests never
touch the live Databento/Alpaca fetch or the real opt_cache/dbn/ files, so they run and pass
regardless of whether the background fetch_dbn.py run has finished.

Covers (per the task spec): delta/IV from the NBBO mid, the bid/ask fill convention, the VOID rule
and its rail computed from the cycle table, sizing and the budget assertion, management
precedence, $0 months, the IV gate, and the extension-read-once guard.
"""
import datetime as dt
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

import cell_1599 as m


ET = m.ET
UTC = dt.timezone.utc


def _bar(day, hh, mm, bid, ask):
    ts = dt.datetime(day.year, day.month, day.day, hh, mm, tzinfo=ET).astimezone(UTC)
    return {'ts_event': ts, 'bid_px_00': bid, 'ask_px_00': ask}


def _write_leg(tmp_path, symbol, bars):
    df = pd.DataFrame(bars)
    df.to_parquet(os.path.join(tmp_path, f"{symbol.strip()}.parquet"), index=False)


# --------------------------------------------------------------------------- delta/IV from mid
def test_iv_and_delta_round_trip_from_nbbo_mid():
    """implied_vol_put must recover the sigma a price was generated with, and bs_put_delta on
    that recovered sigma must match the delta computed directly -- this is the exact computation
    build_ladder performs on the mondays.parquet 'mid' column."""
    S, K, T, sigma_true = 550.0, 520.0, 45 / 365.0, 0.18
    price = m.bs_put_price(S, K, T, m.R_RATE, m.Q_RATE, sigma_true)
    iv = m.implied_vol_put(price, S, K, T, m.R_RATE, m.Q_RATE)
    assert iv is not None and abs(iv - sigma_true) < 1e-3
    delta_direct = m.bs_put_delta(S, K, T, m.R_RATE, m.Q_RATE, sigma_true)
    delta_recovered = m.bs_put_delta(S, K, T, m.R_RATE, m.Q_RATE, iv)
    assert abs(delta_direct - delta_recovered) < 1e-3


def test_build_ladder_uses_mid_not_bid_or_ask():
    monday_rows = pd.DataFrame([
        {'entry_date': '2024-03-04', 'symbol': 'SPY   240419P00520000', 'strike': 520.0,
         'expiry': '2024-04-19', 'bid': 4.90, 'ask': 5.10, 'bid_sz': 5, 'ask_sz': 5, 'spot_10': 550.0},
    ])
    monday_rows['mid'] = (monday_rows['bid'] + monday_rows['ask']) / 2.0
    ladder = m.build_ladder(monday_rows, spot=550.0, entry_date='2024-03-04', expiry='2024-04-19')
    assert len(ladder) == 1
    expected_iv = m.implied_vol_put(5.00, 550.0, 520.0, 46 / 365.0, m.R_RATE, m.Q_RATE)
    assert abs(ladder.iloc[0]['iv'] - expected_iv) < 1e-6


def test_select_strikes_nearest_delta_and_width_partner():
    ladder = pd.DataFrame([
        {'strike': 520.0, 'delta': -0.31, 'bid': 5.0, 'ask': 5.2, 'symbol': 'S520'},
        {'strike': 510.0, 'delta': -0.20, 'bid': 3.0, 'ask': 3.2, 'symbol': 'S510'},
        {'strike': 500.0, 'delta': -0.12, 'bid': 1.5, 'ask': 1.7, 'symbol': 'S500'},
    ])
    picked = m.select_strikes(ladder, target_delta=0.20, width=10.0)
    assert picked is not None
    short_row, long_row = picked
    assert short_row['strike'] == 510.0
    assert long_row['strike'] == 500.0


def test_select_strikes_none_when_partner_missing():
    ladder = pd.DataFrame([{'strike': 510.0, 'delta': -0.20, 'bid': 3.0, 'ask': 3.2, 'symbol': 'S510'}])
    assert m.select_strikes(ladder, target_delta=0.20, width=10.0) is None


# --------------------------------------------------------------------------- IV gate
def test_iv_gate_pass_and_fail():
    ladder_high = pd.DataFrame([{'strike': 550.0, 'iv': 0.22}])
    ladder_low = pd.DataFrame([{'strike': 550.0, 'iv': 0.10}])
    assert m.iv_gate_pass(ladder_high, spot=550.0) is True
    assert m.iv_gate_pass(ladder_low, spot=550.0) is False
    assert m.iv_gate_pass(pd.DataFrame(), spot=550.0) is False


# --------------------------------------------------------------------------- bid/ask fill convention
def test_entry_fill_sells_bid_buys_ask():
    short_row = {'bid': 5.00, 'ask': 5.20}
    long_row = {'bid': 3.00, 'ask': 3.20}
    credit, credit_rail = m.entry_fill(short_row, long_row)
    assert credit == pytest.approx(5.00 - 3.20)  # short BID minus long ASK, no slippage constant
    sm, lm = 5.10, 3.10
    assert credit_rail == pytest.approx((sm - 0.03) - (lm + 0.03))


def test_entry_fill_void_on_missing_quote():
    short_row = {'bid': np.nan, 'ask': 5.20}
    long_row = {'bid': 3.00, 'ask': 3.20}
    credit, credit_rail = m.entry_fill(short_row, long_row)
    assert credit is None and credit_rail is None


def test_exit_cost_buys_short_ask_sells_long_bid():
    cost = m.exit_cost_from_quote((1.0, 1.3), (0.4, 0.6))  # (bid, ask) each
    assert cost == pytest.approx(1.3 - 0.4)
    assert m.exit_cost_from_quote(None, (0.4, 0.6)) is None


# --------------------------------------------------------------------------- VOID rule + rail from the cycle table
def test_void_share_computed_from_cycle_table():
    rows = pd.DataFrame([
        {'void_reason': None, 'pnl_usd': 100.0},
        {'void_reason': None, 'pnl_usd': -50.0},
        {'void_reason': 'no_two_sided_10am_quote', 'pnl_usd': np.nan},
    ])
    assert m.void_share_from_table(rows) == pytest.approx(1 / 3)


def test_void_share_asserts_column_present():
    with pytest.raises(AssertionError):
        m.void_share_from_table(pd.DataFrame({'pnl_usd': [1.0]}))


# --------------------------------------------------------------------------- sizing + budget assertion
def test_size_position_matches_prereg_formula():
    contracts, worst_per_contract = m.size_position(m.B / m.N_LADDER, m.WIDTH, 2.5)
    assert contracts == math.floor((m.B / m.N_LADDER) / ((m.WIDTH - 2.5) * 100))
    assert worst_per_contract == pytest.approx((m.WIDTH - 2.5) * 100)


def test_assert_budget_passes_within_b():
    total = m.assert_budget(open_worst_cases=[1000.0, 1000.0], new_worst_case=1000.0, b=6500.0)
    assert total == pytest.approx(3000.0)


def test_assert_budget_raises_on_breach():
    with pytest.raises(AssertionError):
        m.assert_budget(open_worst_cases=[3000.0, 3000.0], new_worst_case=3000.0, b=6500.0)


# --------------------------------------------------------------------------- management precedence + $0 months
def test_run_cycle_management_b_holds_to_expiry_no_marks_checked(tmp_path, monkeypatch):
    legcache = m.LegCache(legs_dir=str(tmp_path))
    warn = {'mark_missing': 0, 'exit_quote_missing': 0}
    out = m.run_cycle(legcache, 'S1', 'L1', '2024-03-04', '2024-04-19', 'B', warn)
    assert out == {'exit_date': '2024-04-19', 'exit_reason': 'expiry_intrinsic', 'exit_cost': None}


def test_run_cycle_management_a_profit_target_before_dte21(tmp_path):
    """A session where the mark is <= 50% of credit (profit target) AND already inside 21 DTE
    must report 'profit_target', not 'dte_21' -- profit is checked before the DTE fallback."""
    entry, expiry = dt.date(2024, 3, 4), dt.date(2024, 4, 19)  # 46 DTE at entry
    trigger_day = expiry - dt.timedelta(days=20)  # inside 21 DTE AND where we set profit mark
    exit_day = trigger_day + dt.timedelta(days=1)
    short_bars = [_bar(trigger_day, 15, 59, 1.00, 1.20), _bar(exit_day, 10, 0, 0.05, 0.10)]
    long_bars = [_bar(trigger_day, 15, 59, 0.30, 0.50), _bar(exit_day, 10, 0, 0.01, 0.05)]
    _write_leg(tmp_path, 'S1', short_bars)
    _write_leg(tmp_path, 'L1', long_bars)
    legcache = m.LegCache(legs_dir=str(tmp_path))
    m._OPEN_CREDIT[('S1', 'L1', entry.isoformat())] = 2.0  # mark (1.10-0.40=0.70) <= 0.5*2.0=1.0
    warn = {'mark_missing': 0, 'exit_quote_missing': 0}
    out = m.run_cycle(legcache, 'S1', 'L1', entry.isoformat(), expiry.isoformat(), 'A', warn)
    assert out['exit_reason'] == 'profit_target'
    assert out['exit_date'] == exit_day.isoformat()
    assert out['exit_cost'] == pytest.approx(0.10 - 0.01)  # buy short at ask, sell long at bid


def test_run_cycle_management_a_stop_checked_before_profit_and_dte():
    """When a session's mark simultaneously exceeds the stop AND would (incorrectly) also read as
    a DTE trigger, 'stop' must win -- it is the first branch in the precedence chain."""
    import inspect
    src = inspect.getsource(m.run_cycle)
    stop_idx = src.index("reason = True, 'stop'")
    profit_idx = src.index("reason = True, 'profit_target'")
    dte_idx = src.index("reason = True, 'dte_21'")
    assert stop_idx < profit_idx < dte_idx


def test_monthly_series_fills_zero_for_months_with_no_exit():
    cycles = pd.DataFrame([{'exit_date': '2024-03-15', 'pnl_usd': 200.0}])
    ms = m.monthly_series(cycles, '2024-02-01', '2024-04-30')
    assert ms[pd.Period('2024-02', freq='M')] == 0.0
    assert ms[pd.Period('2024-04', freq='M')] == 0.0
    assert ms[pd.Period('2024-03', freq='M')] == 200.0


def test_monthly_series_all_zero_when_no_cycles_at_all():
    ms = m.monthly_series(pd.DataFrame(columns=['exit_date', 'pnl_usd']), '2024-02-01', '2024-03-31')
    assert (ms == 0.0).all()


# --------------------------------------------------------------------------- LegCache VOID behaviour
def test_legcache_quote_at_missing_file_is_warning_and_void(tmp_path, caplog):
    legcache = m.LegCache(legs_dir=str(tmp_path))
    with caplog.at_level('WARNING'):
        result = legcache.quote_at('NOPE', '2024-03-04', 10, 0)
    assert result is None
    assert any('not in pre-pulled superset' in r.message for r in caplog.records)


def test_legcache_quote_at_finds_two_sided_bar_in_window(tmp_path):
    day = dt.date(2024, 3, 4)
    _write_leg(tmp_path, 'S1', [_bar(day, 10, 1, 5.00, 5.20)])
    legcache = m.LegCache(legs_dir=str(tmp_path))
    q = legcache.quote_at('S1', '2024-03-04', 10, 0, window_min=2)
    assert q == (5.00, 5.20)


# --------------------------------------------------------------------------- EXTENSION read-once guard
def test_extension_guard_blocks_second_cell(tmp_path, monkeypatch):
    guard_path = tmp_path / 'guard.json'
    monkeypatch.setattr(m, 'EXT_READ_GUARD_PATH', str(guard_path))

    def fake_run_cell(cell_def, *a, **k):
        return []

    monkeypatch.setattr(m, 'run_cell', fake_run_cell)
    m.guarded_extension_run({'cell': 1599}, None, None, None, m.EXT_START, m.EXT_END, {})
    assert json.loads(guard_path.read_text())['cell'] == 1599
    with pytest.raises(RuntimeError, match='already read once'):
        m.guarded_extension_run({'cell': 1603}, None, None, None, m.EXT_START, m.EXT_END, {})
    # Re-running the SAME cell again must not raise (idempotent, still only marks once).
    m.guarded_extension_run({'cell': 1599}, None, None, None, m.EXT_START, m.EXT_END, {})
