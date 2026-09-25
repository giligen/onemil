"""Unit tests for cell_1429.py on synthetic fill rows (no live tape/cache reads)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell_1429 as c1429  # noqa: E402


def make_fills(rows):
    """rows: list of dicts with day, symbol, entry_m, level, ask_at, R, exit_price, why, net_R, wk."""
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# add_sizing: d_bps, mult, slip charge
# --------------------------------------------------------------------------------------------

def test_add_sizing_tight_quote_gets_1_5x():
    f = make_fills([dict(day='2026-01-01', symbol='AAA', entry_m=600, level=10.0, ask_at=10.003,
                          R=1.0, exit_price=12.0, why='target', net_R=2.0, wk='w1')])
    out = c1429.add_sizing(f)
    assert out.d_bps.iloc[0] == pytest.approx(3.0, abs=1e-6)   # 10.003/10.0 - 1 = 3 bps
    assert out.mult.iloc[0] == c1429.MULT_TIGHT


def test_add_sizing_loose_quote_gets_1x():
    f = make_fills([dict(day='2026-01-01', symbol='AAA', entry_m=600, level=10.0, ask_at=10.010,
                          R=1.0, exit_price=12.0, why='target', net_R=2.0, wk='w1')])
    out = c1429.add_sizing(f)
    assert out.d_bps.iloc[0] == pytest.approx(10.0, abs=1e-6)  # 10 bps, > 5 -> 1x
    assert out.mult.iloc[0] == c1429.MULT_LOOSE


def test_add_sizing_boundary_5bps_is_tight():
    f = make_fills([dict(day='2026-01-01', symbol='AAA', entry_m=600, level=10.0, ask_at=10.005,
                          R=1.0, exit_price=12.0, why='target', net_R=2.0, wk='w1')])
    out = c1429.add_sizing(f)
    assert out.mult.iloc[0] == c1429.MULT_TIGHT   # d==5 bps -> "<=5" is tight


def test_add_sizing_warns_over_15bps(capsys):
    f = make_fills([dict(day='2026-01-01', symbol='AAA', entry_m=600, level=10.0, ask_at=10.020,
                          R=1.0, exit_price=12.0, why='target', net_R=2.0, wk='w1')])
    c1429.add_sizing(f)
    assert 'WARNING' in capsys.readouterr().err


def test_add_sizing_stop_slip_charged_only_on_stop_why():
    f = make_fills([
        dict(day='2026-01-01', symbol='AAA', entry_m=600, level=10.0, ask_at=10.0003,
             R=1.0, exit_price=9.0, why='stop', net_R=-1.0, wk='w1'),
        dict(day='2026-01-01', symbol='BBB', entry_m=601, level=10.0, ask_at=10.0003,
             R=1.0, exit_price=12.0, why='target', net_R=2.0, wk='w1'),
    ])
    out = c1429.add_sizing(f)
    expected_slip = c1429.c1430.STOP_SLIP_BP * 9.0 / 1.0
    assert out.net_R_slip.iloc[0] == pytest.approx(-1.0 - expected_slip)
    assert out.net_R_slip.iloc[1] == pytest.approx(2.0)          # target row untouched
    assert out.net_R_noslip.iloc[0] == pytest.approx(-1.0)


# --------------------------------------------------------------------------------------------
# book_r_per_risk / worst_week_ratio
# --------------------------------------------------------------------------------------------

def test_book_r_per_risk_matches_hand_calc():
    # two fills: R=2 mult=1.5, R=0 mult=1.0 -> (2*1.5 + 0*1)/(1.5+1) = 3/2.5 = 1.2
    r = c1429.book_r_per_risk([2.0, 0.0], [1.5, 1.0])
    assert r == pytest.approx(1.2)


def test_book_r_per_risk_flat_book_is_plain_mean():
    r = c1429.book_r_per_risk([1.0, 3.0, -1.0], [1.0, 1.0, 1.0])
    assert r == pytest.approx(1.0)


def test_book_r_per_risk_empty_is_nan():
    assert np.isnan(c1429.book_r_per_risk([], []))


def test_worst_week_ratio_picks_minimum_week():
    net_R = [2.0, 2.0, -3.0, -3.0]
    mult = [1.0, 1.0, 1.0, 1.0]
    wk = ['w1', 'w1', 'w2', 'w2']
    assert c1429.worst_week_ratio(net_R, mult, wk) == pytest.approx(-3.0)


# --------------------------------------------------------------------------------------------
# score_split: weighted book beats flat when tight-quote fills are the winners
# --------------------------------------------------------------------------------------------

def test_score_split_positive_dr_when_tight_fills_win_more():
    rows = []
    # 3 tight (mult 1.5) winners at +2R, 3 loose (mult 1.0) losers at -1R
    for i in range(3):
        rows.append(dict(day=f'2026-01-0{i+1}', symbol='AAA', entry_m=600, level=10.0, ask_at=10.002,
                          R=1.0, exit_price=12.0, why='target', net_R=2.0, wk=f'w{i}'))
    for i in range(3):
        rows.append(dict(day=f'2026-01-1{i+1}', symbol='BBB', entry_m=600, level=10.0, ask_at=10.012,
                          R=1.0, exit_price=9.0, why='eod', net_R=-1.0, wk=f'w{i}'))
    f = c1429.add_sizing(make_fills(rows))
    r = c1429.score_split(f, 'noslip')
    # weighted: (3*2*1.5 + 3*-1*1)/(3*1.5+3*1) = (9-3)/7.5 = 0.8; flat: mean = (6-3)/6 = 0.5
    assert r['weighted_R'] == pytest.approx(0.8)
    assert r['flat_R'] == pytest.approx(0.5)
    assert r['dR'] == pytest.approx(0.3)
    assert r['n'] == 6


def test_score_split_slip_reduces_stop_rows_only():
    rows = [
        dict(day='2026-01-01', symbol='AAA', entry_m=600, level=10.0, ask_at=10.0002,
             R=1.0, exit_price=9.0, why='stop', net_R=-1.0, wk='w1'),
    ]
    f = c1429.add_sizing(make_fills(rows))
    r_noslip = c1429.score_split(f, 'noslip')
    r_slip = c1429.score_split(f, 'slip')
    assert r_slip['weighted_R'] < r_noslip['weighted_R']


# --------------------------------------------------------------------------------------------
# trigger_print_size / size_class_report: missing tape -> None/unknown, logged
# --------------------------------------------------------------------------------------------

def test_trigger_print_size_returns_none_when_day_missing(capsys):
    f = make_fills([dict(day='1999-01-01', symbol='ZZZ', entry_m=600, level=10.0, ask_at=10.0002,
                          R=1.0, exit_price=12.0, why='target', net_R=2.0, wk='w1')])
    rep = c1429.size_class_report(f)
    assert len(rep) == 0     # nothing classified -> empty report
    assert 'WARNING' in capsys.readouterr().err


def test_trigger_print_size_finds_first_qualifying_print():
    day = '2026-01-01'
    S_ns = c1429.sr.et_ns(day, 600 * 60)
    trades = pd.DataFrame({
        'ts': [S_ns - 40 * 10**9, S_ns - 30 * 10**9, S_ns - 20 * 10**9],
        'price': [9.98, 10.02, 10.05],   # trigger = level+TICK = 10.01; first hit at ts=-30s
        'size': [5.0, 250.0, 400.0],
    })
    quotes = pd.DataFrame({'ts': [], 'bid': [], 'ask': []})
    tapes_by_day = {day: {c1429.sr.sig_key('AAA', 599): (trades, quotes)}}
    row = pd.Series(dict(day=day, symbol='AAA', entry_m=600, level=10.0))
    size = c1429.trigger_print_size(row, tapes_by_day)
    assert size == pytest.approx(250.0)


def test_trigger_print_size_none_when_no_print_reaches_trigger():
    day = '2026-01-01'
    S_ns = c1429.sr.et_ns(day, 600 * 60)
    trades = pd.DataFrame({'ts': [S_ns - 30 * 10**9], 'price': [9.5], 'size': [10.0]})
    quotes = pd.DataFrame({'ts': [], 'bid': [], 'ask': []})
    tapes_by_day = {day: {c1429.sr.sig_key('AAA', 599): (trades, quotes)}}
    row = pd.Series(dict(day=day, symbol='AAA', entry_m=600, level=10.0))
    assert c1429.trigger_print_size(row, tapes_by_day) is None
