"""Unit tests for cell_1479.py's pure pyramid logic (bar-walk, R-in-original-units algebra, cost).
No network, no DB -- fabricated minute-bar paths only. Per the bug protocol every prior defect gets a
unit test too; this file is the first build of the cell so it covers the PREREG's own worked cases.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell_1479 as c1479


def bar(m, o, h, l, c):
    return dict(m=m, o=o, h=h, l=l, c=c, v=100)


def path_df(rows):
    return pd.DataFrame(rows)


def test_pyramid_never_adds_stops_at_minus_third():
    """Price drifts straight down to the original stop without ever touching fill+1R: pre-add stop,
    raw_R = (stop-fill)/R0 * 1/3 = -1/3 exactly."""
    fill, stop = 10.0, 9.0                      # R0 = 1
    path = path_df([bar(600, 9.5, 9.5, 9.0, 9.0)])
    mine = c1479.pyramid_walk(fill, stop, fill + 2, path)
    assert mine['why'] == 'stop'
    assert mine['added'] is False
    raw = c1479.raw_R_original_units(fill, stop, mine)
    assert raw == pytest.approx(-1 / 3, abs=1e-9)


def test_pyramid_add_then_stop_at_breakeven_is_minus_two_thirds():
    """Touches fill+1R (adds, stop moves to fill), then reverses straight to the new (breakeven)
    stop: combined_R = (fill-fill)/R0 - 2/3 = -2/3, per the PREREG's own worked case."""
    fill, stop = 10.0, 9.0
    path = path_df([bar(600, 10.5, 11.0, 10.5, 11.0),      # touches add_th=11
                     bar(601, 10.5, 10.5, 9.9, 10.0)])       # then back down through the new stop=fill=10
    mine = c1479.pyramid_walk(fill, stop, fill + 2, path)
    assert mine['added'] is True
    assert mine['why'] == 'stop_be'
    raw = c1479.raw_R_original_units(fill, stop, mine)
    assert raw == pytest.approx(-2 / 3, abs=1e-9)


def test_pyramid_add_then_target_is_plus_one_and_a_third():
    """Adds at fill+1R, then reaches target=fill+2R: combined_R = (fill+2R-fill)/R0 - 2/3 = 2-2/3=4/3."""
    fill, stop = 10.0, 9.0
    path = path_df([bar(600, 10.5, 11.0, 10.5, 10.9),
                     bar(601, 11.0, 12.0, 11.0, 11.9)])      # reaches target=12
    mine = c1479.pyramid_walk(fill, stop, fill + 2, path)
    assert mine['why'] == 'target'
    raw = c1479.raw_R_original_units(fill, stop, mine)
    assert raw == pytest.approx(4 / 3, abs=1e-9)


def test_same_bar_add_and_target_is_still_post_add_weighted():
    """A single bar that touches both add_th and target must be scored as an ADD (weight -2/3
    baseline), not as a plain pre-add target hit -- the walk checks add before target within a bar."""
    fill, stop = 10.0, 9.0
    path = path_df([bar(600, 10.0, 12.5, 10.5, 12.0)])       # one bar clears 11 (add) AND 12 (target)
    mine = c1479.pyramid_walk(fill, stop, fill + 2, path)
    assert mine['added'] is True
    assert mine['why'] == 'target'


def test_eod_exit_uses_bar_open_and_eod_m_constant():
    fill, stop = 10.0, 9.0
    path = path_df([bar(c1479.EOD_M, 10.2, 10.3, 10.1, 10.25)])
    mine = c1479.pyramid_walk(fill, stop, fill + 2, path)
    assert mine['why'] == 'eod'
    assert mine['exit_price'] == pytest.approx(10.2)


def test_cost_1479_post_add_equals_plain_single_trade_cost():
    """Documented invariant: once added, cost_R is IDENTICAL to a plain (non-pyramided) trade's
    cost_R -- HE*(1/3N)+HE*(2/3N) telescopes to HE*N, same as one full-size entry (module docstring).
    """
    row = pd.Series(dict(R=1.0, half_entry=0.02, exit_half=0.03, holdout='VAL', fill=10.0, stop=9.0))
    mine_added = dict(exit_price=12.0, why='target', added=True)
    cost_R, _ = c1479.cost_1479(row, mine_added)
    plain_cost_R = (row.half_entry + row.exit_half + c1479.sr.SLIP_BP * mine_added['exit_price']) / row.R
    assert cost_R == pytest.approx(plain_cost_R, rel=1e-9)


def test_cost_1479_pre_add_is_one_third_of_plain():
    row = pd.Series(dict(R=1.0, half_entry=0.02, exit_half=0.03, holdout='VAL', fill=10.0, stop=9.0))
    mine_plain = dict(exit_price=8.5, why='stop', added=False)
    cost_R, _ = c1479.cost_1479(row, mine_plain)
    plain_full = (row.half_entry + row.exit_half + c1479.sr.SLIP_BP * mine_plain['exit_price']) / row.R
    # the pre-add spread/impact share is exactly 1/3 of the plain full-size charge, PLUS the
    # stop-limit slip term (also weighted 1/3) since why=='stop'.
    slip = c1479.blended_stop_limit_bps('VAL') * 1e-4 * mine_plain['exit_price'] / row.R * (1 / 3)
    assert cost_R == pytest.approx(plain_full / 3 + slip, rel=1e-9)


def test_blended_stop_limit_bps_matches_result_1463_table():
    train = c1479.blended_stop_limit_bps('TRAIN-H2')
    val = c1479.blended_stop_limit_bps('VAL')
    assert train == pytest.approx((272 * 2.9 + 44 * 93.9) / 316, abs=1e-6)
    assert val == pytest.approx((286 * 3.2 + 34 * 75.8) / 320, abs=1e-6)
    assert train > val > 10          # sanity: both well above the flat "filled" mean alone


def test_to_et_minute_handles_dst_and_utc_offset():
    s = pd.Series(['2025-07-01T13:30:00+00:00'])   # 09:30 ET in July (EDT, UTC-4)
    m = c1479._to_et_minute(s)
    assert int(m.iloc[0]) == c1479.OPEN_M


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
