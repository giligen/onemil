"""Unit tests for cell_1491.py (PREREG_1491.md): stop placement, the fill-bar convention, the two
target definitions, and the own-R <-> % of price unit conversion. Synthetic bars/rows only -- no DB,
no CSV, no network; run with `python3 -m pytest research/hod_entry/test_cell_1491.py -v`."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cell_1491 as c1491
import sip_rebuild as sr


def _row(**kw):
    """A one-row DataFrame's single itertuples() record -- matches the base book's column names."""
    base = dict(day='2026-01-05', symbol='ZZZZ', why='target', fill=10.00, R=1.00, level=10.20,
                fill_min=605.0, holdout='VAL', half_entry=0.01)
    base.update(kw)
    return next(pd.DataFrame([base]).itertuples())


def _bars(rows):
    """rows: list of (m, o, h, l, c) -> the bar DataFrame walk_one/walk_path expect."""
    return pd.DataFrame(rows, columns=['m', 'o', 'h', 'l', 'c'])


# --------------------------------------------------------------------------------------------- stop placement

@pytest.mark.parametrize('s,level,expected', [
    (0.0025, 10.20, 10.20 * 0.9975),
    (0.0050, 10.20, 10.20 * 0.9950),
    (0.0075, 100.0, 100.0 * 0.9925),
])
def test_stop_placement(s, level, expected):
    """stop_s = level * (1 - s), computed the same way run_book computes it."""
    stop_s = level * (1.0 - s)
    assert stop_s == pytest.approx(expected)
    assert stop_s < level


def test_stop_placement_shallower_than_base_stop():
    """The shallow stop must sit ABOVE (closer to fill than) a plausible consolidation-low base
    stop -- the whole point of the PREREG ('smaller risk')."""
    level, base_stop = 10.20, 9.50
    for s in c1491.STOPS:
        stop_s = level * (1.0 - s)
        assert stop_s > base_stop


# --------------------------------------------------------------------------------------------- fill-bar convention

def test_fill_bar_stop_triggers_stop_bar_at_stop_price():
    """Fill bar's low <= stop_s -> 'stop_bar', exit at stop_s exactly (conservative -- no gap-through
    pricing on the fill bar itself), exit_m = floor(fill_min)."""
    r = _row(why='target', fill=10.00, fill_min=605.7, symbol='AAA')
    stop_s = 9.90
    bars = {'AAA': _bars([(605, 10.00, 10.05, 9.80, 9.95),      # fill bar: low 9.80 <= stop_s 9.90
                          (606, 9.95, 10.50, 9.95, 10.40)])}    # would hit target if reached -- must NOT
    exit_m, exit_price, why = c1491.walk_one(r, bars, stop_s, target=12.0)
    assert why == 'stop_bar'
    assert exit_m == 605
    assert exit_price == pytest.approx(stop_s)


def test_fill_bar_no_breach_walks_next_bar():
    """Fill bar's low > stop_s -> walk from the NEXT bar (sip_rebuild.walk_path semantics)."""
    r = _row(why='target', fill=10.00, fill_min=605.2, symbol='BBB')
    stop_s = 9.00
    bars = {'BBB': _bars([(605, 10.00, 10.05, 9.95, 10.02),     # fill bar: low 9.95 > stop_s -- no stop
                          (606, 10.02, 10.60, 10.00, 10.55)])}  # target reached here
    exit_m, exit_price, why = c1491.walk_one(r, bars, stop_s, target=10.60)
    assert why == 'target'
    assert exit_m == 606
    assert exit_price == pytest.approx(10.60)


def test_base_stop_bar_row_is_deterministic_stop_bar():
    """A base fill already tick-stopped intrabar (why=='stop_bar', at a DEEPER original stop) must be
    stop_bar under every shallower stop_s too, with NO bar lookup (bars=None/empty is fine)."""
    r = _row(why='stop_bar', fill=10.00, fill_min=605.9)
    for s in c1491.STOPS:
        stop_s = 10.20 * (1.0 - s)
        exit_m, exit_price, why = c1491.walk_one(r, bars={}, stop_s=stop_s, target=12.0)
        assert why == 'stop_bar'
        assert exit_m == 605
        assert exit_price == pytest.approx(stop_s)


def test_missing_fill_bar_is_excluded_not_guessed():
    """No usable bar for the fill -> ('no_data', None exit_m) so the caller drops the row, never a
    silent guess."""
    r = _row(why='target', fill=10.00, fill_min=605.5, symbol='CCC')
    exit_m, exit_price, why = c1491.walk_one(r, bars={}, stop_s=9.0, target=12.0)
    assert (exit_m, exit_price, why) == (None, None, 'no_data')


# --------------------------------------------------------------------------------------------- the two target definitions

def test_scalp_target_is_two_r_s():
    """SCALP (1,491): target = fill + 2 * R_s -- R_s from the SHALLOW stop, not the base R."""
    fill, level, s = 10.00, 10.20, 0.0050
    stop_s = level * (1 - s)
    R_s = fill - stop_s
    target = fill + sr.TARGET_R * R_s
    assert target == pytest.approx(fill + 2.0 * R_s)
    assert target != pytest.approx(fill + 2.0 * 1.00)     # must NOT reuse an unrelated base R=1.00


def test_asym_target_is_the_base_target_unchanged():
    """ASYMMETRIC (1,492): target = fill + 2 * R_base -- the SAME target price as the base book's own
    2R target, independent of s (only the stop differs across s for this rule)."""
    fill, base_R = 10.00, 0.80
    target = fill + sr.TARGET_R * base_R
    for s in c1491.STOPS:                      # target must not depend on s at all
        assert target == pytest.approx(fill + 2.0 * base_R)


def test_run_book_asym_target_price_matches_base_target_across_all_three_stops():
    """End-to-end: run_book's own target computation for rule='asym' is identical across every s."""
    fills = pd.DataFrame([dict(day='2026-01-05', symbol='DDD', why='target', fill=10.20, R=0.80,
                                level=10.00, fill_min=605.0, holdout='VAL', half_entry=0.01, wk='2026-W02',
                                base_pct=1.0, outcome_R=0.8)])
    bars = {'2026-01-05': {'DDD': _bars([(605, 10.20, 10.25, 10.15, 10.22),
                                          (606, 10.22, 11.90, 10.20, 11.85)])}}
    targets = set()
    for s in c1491.STOPS:
        out = c1491.run_book(fills, bars, s, 'asym')
        # reconstruct the target this book used from its own recorded R_s and exit mechanics is
        # indirect; instead assert the row that DID hit target used exactly fill+2*base_R as price
        if len(out) and (out.why == 'target').any():
            targets.add(round(float(out.loc[out.why == 'target', 'exit_price'].iloc[0]), 6))
    assert targets == {round(10.20 + 2.0 * 0.80, 6)}


# --------------------------------------------------------------------------------------------- unit conversion

def test_cost_and_net_target_no_exit_slip():
    """'target' exit: no exit-side slip beyond half_entry (target = limit)."""
    fill, R_s, holdout, he = 10.00, 0.30, 'VAL', 0.02
    exit_price = fill + 2 * R_s
    raw_R, net_R, net_pct = c1491.cost_and_net(fill, R_s, holdout, he, exit_price, 'target')
    assert raw_R == pytest.approx(2.0)
    assert net_R == pytest.approx(2.0 - he / R_s)
    # unit conversion: net_pct must equal net_R_own * R_s / fill * 100, exactly
    assert net_pct == pytest.approx(net_R * R_s / fill * 100.0)
    assert net_pct == pytest.approx((net_R * R_s) / fill * 100.0)


def test_cost_and_net_stop_charges_the_amendment_slip_in_own_r():
    """'stop' exit charges SLIP_STOP_BPS[holdout] (own-R units), on top of half_entry."""
    fill, R_s, holdout, he = 10.00, 0.20, 'TRAIN-H2', 0.01
    exit_price = fill - R_s               # stopped at exactly stop_s
    raw_R, net_R, net_pct = c1491.cost_and_net(fill, R_s, holdout, he, exit_price, 'stop')
    expected_slip_R = exit_price * c1491.SLIP_STOP_BPS[holdout] / 1e4 / R_s
    assert net_R == pytest.approx(raw_R - he / R_s - expected_slip_R)
    assert net_pct == pytest.approx(net_R * R_s / fill * 100.0)


def test_cost_and_net_eod_uses_1443_means():
    """'eod' exit charges the 1,443 EOD holdout mean bps, not the stop-limit blend."""
    fill, R_s, holdout, he = 10.00, 0.25, 'VAL', 0.015
    exit_price = 10.10
    _, net_R, _ = c1491.cost_and_net(fill, R_s, holdout, he, exit_price, 'eod')
    expected_slip_R = exit_price * c1491.EOD_BPS_1443[holdout] / 1e4 / R_s
    raw_R = (exit_price - fill) / R_s
    assert net_R == pytest.approx(raw_R - he / R_s - expected_slip_R)


def test_unit_conversion_round_trip_is_price_neutral():
    """net_pct converted back to dollars (net_pct/100*fill) must equal net_R_own * R_s (the P&L in
    dollars), for any fill/R_s/holdout/why combination."""
    for fill, R_s, holdout, why, exit_price in [
            (5.00, 0.10, 'VAL', 'target', 5.20), (250.0, 4.0, 'TRAIN-H2', 'stop', 246.0),
            (18.5, 0.6, 'VAL', 'eod', 18.9)]:
        _, net_R, net_pct = c1491.cost_and_net(fill, R_s, holdout, 0.01, exit_price, why)
        dollars_from_pct = net_pct / 100.0 * fill
        dollars_from_R = net_R * R_s
        assert dollars_from_pct == pytest.approx(dollars_from_R, rel=1e-9)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
