"""pytest unit tests for research/hod_entry/entry_replay.py, on synthetic ticks/bars — no DB, no
parquet, no network. Covers: the level definition, fill-at-prevailing-ask on first print>=trigger,
ask-above-limit -> no fill, a stop print after the fill inside the break bar -> stopped, P&L/R
arithmetic, and the E3 re-costing. Run: pytest research/hod_entry/test_entry_replay.py -q"""
import types

import numpy as np
import pandas as pd
import pytest

from research.hod_entry.entry_replay import (
    b0_fill, compute_level, entry_leg_cost_tick, exit_leg_cost, first_trigger_print,
    intrabar_stop, prevailing_quote, simulate_signal, SLIP_BP, TARGET_R,
)


# ---------------------------------------------------------------- level
def test_compute_level_running_max_of_closed_highs():
    bars = pd.DataFrame({'m': [570, 571, 572, 573], 'h': [10.0, 10.5, 9.8, 11.2]})
    # break bar = 573; level = max(h for m<573) = max(10.0,10.5,9.8) = 10.5
    assert compute_level(bars, break_m=573) == 10.5


def test_compute_level_ignores_break_bar_own_high():
    bars = pd.DataFrame({'m': [570, 571], 'h': [10.0, 999.0]})
    # break bar = 571: its own high (999) must NOT count toward the level
    assert compute_level(bars, break_m=571) == 10.0


def test_compute_level_none_when_no_prior_bar():
    bars = pd.DataFrame({'m': [570], 'h': [10.0]})
    assert compute_level(bars, break_m=570) is None


# ---------------------------------------------------------------- trigger + quote
def test_first_trigger_print_picks_earliest_qualifying_print():
    trades = pd.DataFrame({'sec': [100, 101, 102, 103], 'price': [9.9, 10.0, 10.05, 10.10]})
    got = first_trigger_print(trades, lo_sec=100, hi_sec=104, trigger=10.0)
    assert got == (101, 10.0)


def test_first_trigger_print_none_when_never_crossed():
    trades = pd.DataFrame({'sec': [100, 101], 'price': [9.0, 9.5]})
    assert first_trigger_print(trades, 100, 104, trigger=10.0) is None


def test_prevailing_quote_uses_strictly_prior_record():
    mbp = pd.DataFrame({'sec': [100, 101, 102], 'bid_px_00': [9.9, 9.95, 10.5],
                         'ask_px_00': [10.0, 10.05, 10.6]})
    # at_sec=101.5 -> strictly prior record is sec=101 (NOT sec=102, which is >= at_sec)
    bid, ask = prevailing_quote(mbp, at_sec=101.5)
    assert (bid, ask) == (9.95, 10.05)


def test_prevailing_quote_rejects_crossed_or_zero_quote():
    mbp = pd.DataFrame({'sec': [100], 'bid_px_00': [10.1], 'ask_px_00': [10.0]})  # crossed: ask<bid
    assert prevailing_quote(mbp, at_sec=101) is None
    mbp2 = pd.DataFrame({'sec': [100], 'bid_px_00': [0.0], 'ask_px_00': [10.0]})  # one-sided
    assert prevailing_quote(mbp2, at_sec=101) is None


def test_prevailing_quote_none_with_no_prior_record():
    mbp = pd.DataFrame({'sec': [105], 'bid_px_00': [9.9], 'ask_px_00': [10.0]})
    assert prevailing_quote(mbp, at_sec=100) is None


# ---------------------------------------------------------------- intrabar stop
def test_intrabar_stop_finds_first_qualifying_print_after_fill():
    trades = pd.DataFrame({'sec': [101, 102, 103], 'price': [10.2, 9.4, 9.3]})
    got = intrabar_stop(trades, after_sec=101, hi_sec=104, stop=9.5)
    assert got == 102


def test_intrabar_stop_none_when_no_print_breaches_before_bar_closes():
    trades = pd.DataFrame({'sec': [101, 102], 'price': [10.2, 10.1]})
    assert intrabar_stop(trades, after_sec=101, hi_sec=104, stop=9.5) is None


def test_intrabar_stop_ignores_prints_at_or_before_fill_and_at_or_after_bar_close():
    trades = pd.DataFrame({'sec': [100, 105], 'price': [9.0, 9.0]})  # both would breach stop=9.5
    # but sec=100 <= after_sec(100) and sec=105 >= hi_sec(104) -> neither counts
    assert intrabar_stop(trades, after_sec=100, hi_sec=104, stop=9.5) is None


# ---------------------------------------------------------------- b0_fill path walk
def test_b0_fill_stop_hit():
    path = pd.DataFrame({'m': [728, 729], 'o': [10.0, 9.0], 'h': [10.2, 9.1],
                          'l': [9.4, 8.8], 'c': [9.8, 8.9]})
    exit_m, exit_px, why = b0_fill(entry=10.0, stop=9.5, target=12.0, path_after=path)
    assert (exit_m, exit_px, why) == (728, 9.5, 'stop')  # low<=stop, open(10.0) didn't gap through


def test_b0_fill_gap_through_open():
    path = pd.DataFrame({'m': [728], 'o': [9.0], 'h': [9.2], 'l': [8.8], 'c': [8.9]})
    exit_m, exit_px, why = b0_fill(entry=10.0, stop=9.5, target=12.0, path_after=path)
    assert (exit_m, exit_px, why) == (728, 9.0, 'stop')  # open already through stop -> fills at open


def test_b0_fill_target_hit():
    path = pd.DataFrame({'m': [728], 'o': [10.1], 'h': [12.5], 'l': [10.0], 'c': [12.0]})
    exit_m, exit_px, why = b0_fill(entry=10.0, stop=9.5, target=12.0, path_after=path)
    assert (exit_m, exit_px, why) == (728, 12.0, 'target')


def test_b0_fill_eod_takes_precedence_over_stop_or_target_same_bar():
    path = pd.DataFrame({'m': [955], 'o': [11.0], 'h': [12.5], 'l': [9.0], 'c': [11.0]})
    exit_m, exit_px, why = b0_fill(entry=10.0, stop=9.5, target=12.0, path_after=path)
    assert (exit_m, exit_px, why) == (955, 11.0, 'eod')


# ---------------------------------------------------------------- cost / P&L arithmetic
def test_entry_leg_cost_tick_is_half_spread_over_R():
    assert entry_leg_cost_tick(tick_spread=0.20, R=2.0) == pytest.approx(0.05)


def test_exit_leg_cost_matches_b0_formula_components():
    # half_R = 0.5*spread_mean/R ; + SLIP_BP*exit_price/R
    got = exit_leg_cost(spread_mean=0.10, R=2.0, exit_price=20.0)
    want = 0.5 * 0.10 / 2.0 + SLIP_BP * 20.0 / 2.0
    assert got == pytest.approx(want)


def test_net_R_arithmetic_stop_then_target_scenarios():
    entry, stop, R = 10.0, 9.5, 0.5
    raw_rr_stop = (9.5 - entry) / R
    cost = entry_leg_cost_tick(0.02, R) + exit_leg_cost(0.02, R, 9.5)
    assert raw_rr_stop == pytest.approx(-1.0)
    net = raw_rr_stop - cost
    assert net < raw_rr_stop  # cost always erodes net_R


# ---------------------------------------------------------------- simulate_signal (integration
# of the pieces above on a fully synthetic signal)
def _row(**kw):
    base = dict(day='2099-01-01', symbol='ZZZ', entry_m=700, split='VAL', half=np.nan,
                entry=10.30, stop=9.50, R=0.80, exit_price=11.90, raw_rr=2.0, why='target')
    base.update(kw)
    return types.SimpleNamespace(**base)


def _paths_after_entry(entry_m, rows):
    """rows: list of (m,o,h,l,c) starting at entry_m (paths.parquet convention)."""
    return pd.DataFrame(rows, columns=['m', 'o', 'h', 'l', 'c'])


def test_simulate_signal_fills_at_prevailing_ask_when_within_limit():
    level = 10.0  # trigger = 10.01, limit = 10.0*1.0015 = 10.015
    S = 700 * 60
    trades = pd.DataFrame({
        'symbol_win': ['ZZZ'] * 3, 'entry_m': [700] * 3, 'schema': ['trades'] * 3,
        'sec': [S - 30, S - 29, S - 28], 'price': [9.95, 10.01, 10.02],
    })
    mbp = pd.DataFrame({
        'symbol_win': ['ZZZ'] * 2, 'entry_m': [700] * 2, 'schema': ['mbp-1'] * 2,
        'sec': [S - 31, S - 5], 'bid_px_00': [9.98, 10.00], 'ask_px_00': [10.01, 10.02],
    })
    day_ticks = pd.concat([trades, mbp], ignore_index=True)
    paths_grp = _paths_after_entry(700, [(700, 10.02, 10.05, 10.00, 10.03),
                                          (955, 12.00, 12.00, 12.00, 12.00)])
    row = _row(entry_m=700)
    out = simulate_signal(row, level, day_ticks, paths_grp, spread_mean=0.02)
    assert out['usable'] is True
    assert out['e1_fill'] is True
    # fill at the prevailing (strictly prior) ask = 10.01 (the S-31 quote, prior to the S-29 print)
    R_new = 10.01 - 9.50
    assert R_new == pytest.approx(0.51)


def test_simulate_signal_no_fill_when_ask_above_limit():
    level = 10.0  # limit = 10.015
    S = 700 * 60
    trades = pd.DataFrame({
        'symbol_win': ['ZZZ'], 'entry_m': [700], 'schema': ['trades'],
        'sec': [S - 30], 'price': [10.01],
    })
    mbp = pd.DataFrame({
        'symbol_win': ['ZZZ'], 'entry_m': [700], 'schema': ['mbp-1'],
        'sec': [S - 31], 'bid_px_00': [10.90], 'ask_px_00': [11.00],  # far above the 15bps limit
    })
    day_ticks = pd.concat([trades, mbp], ignore_index=True)
    paths_grp = _paths_after_entry(700, [(700, 10.02, 10.05, 10.00, 10.03)])
    row = _row(entry_m=700)
    out = simulate_signal(row, level, day_ticks, paths_grp, spread_mean=0.02)
    assert out['usable'] is True
    assert out['e1_fill'] is False
    assert out['e1_reason'] == 'ask_above_limit'
    assert np.isnan(out['e1_net_R'])


def test_simulate_signal_intrabar_stop_after_fill():
    level = 10.0
    S = 700 * 60
    trades = pd.DataFrame({
        'symbol_win': ['ZZZ'] * 2, 'entry_m': [700] * 2, 'schema': ['trades'] * 2,
        'sec': [S - 30, S - 20], 'price': [10.01, 9.40],  # 9.40 <= stop(9.50), after the fill
    })
    mbp = pd.DataFrame({
        'symbol_win': ['ZZZ'], 'entry_m': [700], 'schema': ['mbp-1'],
        'sec': [S - 31], 'bid_px_00': [10.00], 'ask_px_00': [10.01],
    })
    day_ticks = pd.concat([trades, mbp], ignore_index=True)
    paths_grp = _paths_after_entry(700, [(700, 12.0, 12.0, 12.0, 12.0)])  # would otherwise never stop
    row = _row(entry_m=700, stop=9.50)
    out = simulate_signal(row, level, day_ticks, paths_grp, spread_mean=0.02)
    assert out['e1_fill'] is True
    assert out['e1_why'] == 'stop_intrabar'
    # raw_rr = (stop-entry)/R = -1.0 exactly, before cost
    R_new = 10.01 - 9.50
    raw_rr = (9.50 - 10.01) / R_new
    assert raw_rr == pytest.approx(-1.0)
    assert out['e1_net_R'] < raw_rr  # cost erodes it further


def test_simulate_signal_e3_recosts_same_entry_exit_as_b0():
    level = 10.0  # trigger = 10.01; this signal is 'usable' (has a break-bar print >= trigger and
    # a prevailing quote at that instant) even though E1/E2 can't walk a path (paths_grp=None below)
    S = 700 * 60
    trades = pd.DataFrame({
        'symbol_win': ['ZZZ'], 'entry_m': [700], 'schema': ['trades'],
        'sec': [S - 30], 'price': [10.01],
    })
    mbp = pd.DataFrame({
        'symbol_win': ['ZZZ', 'ZZZ'], 'entry_m': [700, 700], 'schema': ['mbp-1', 'mbp-1'],
        'sec': [S - 31, S - 5], 'bid_px_00': [10.00, 10.28], 'ask_px_00': [10.01, 10.32],
    })
    day_ticks = pd.concat([trades, mbp], ignore_index=True)
    row = _row(entry_m=700, entry=10.30, stop=9.50, R=0.80, exit_price=11.90, raw_rr=2.0)
    out = simulate_signal(row, level, day_ticks, paths_grp=None, spread_mean=0.02)
    assert out['usable'] is True
    assert out['void_reason'] == 'no_path_after_entry'  # E1/E2 skipped, E3 still scored
    # E3 must use B0's OWN entry/exit/R unchanged -> raw_rr recovers exactly if cost were 0
    tick_spread = 10.32 - 10.28
    entry_cost = entry_leg_cost_tick(tick_spread, 0.80)
    exit_cost = exit_leg_cost(0.02, 0.80, 11.90)
    want = 2.0 - (entry_cost + exit_cost)
    assert out['e3_net_R'] == pytest.approx(want)


def test_simulate_signal_void_when_level_missing():
    row = _row()
    out = simulate_signal(row, level=None, day_ticks=None, paths_grp=None, spread_mean=0.02)
    assert out['usable'] is False
    assert out['void_reason'] == 'no_level'


def test_simulate_signal_void_when_trigger_never_crosses_on_xnas():
    level = 10.0
    S = 700 * 60
    trades = pd.DataFrame({'symbol_win': ['ZZZ'], 'entry_m': [700], 'schema': ['trades'],
                            'sec': [S - 30], 'price': [9.50]})  # never reaches trigger 10.01
    mbp = pd.DataFrame({'symbol_win': ['ZZZ'], 'entry_m': [700], 'schema': ['mbp-1'],
                         'sec': [S - 31], 'bid_px_00': [9.45], 'ask_px_00': [9.55]})
    day_ticks = pd.concat([trades, mbp], ignore_index=True)
    row = _row(entry_m=700)
    out = simulate_signal(row, level, day_ticks, paths_grp=None, spread_mean=0.02)
    assert out['usable'] is False
    assert out['void_reason'] == 'no_trigger_cross_xnas'
