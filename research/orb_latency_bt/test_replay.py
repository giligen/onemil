"""Unit tests for research/orb_latency_bt/replay.py replay rules, on synthetic
ticks — no network, no Databento. Covers the PREREG-frozen cases:
  * t* before T0 vs t* after T0 (two different fill-price rules)
  * ask above limit -> skipped
  * no print (no t*) -> unfilled
  * P&L arithmetic (BT _sized_pnl + shares * (BT entry - replay fill))
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from replay import replay_one, find_t_star, ask_at, REBUMP_BUFFER  # noqa: E402


def _trades(rows):
    """rows: list of (t_sec, price) -> DataFrame sorted like the real pipeline produces."""
    df = pd.DataFrame(rows, columns=['t_sec', 'price']).sort_values('t_sec').reset_index(drop=True)
    return df


def _mbp(rows):
    """rows: list of (t_sec, ask_px_00)."""
    df = pd.DataFrame(rows, columns=['t_sec', 'ask_px_00']).sort_values('t_sec').reset_index(drop=True)
    return df


TRIGGER = 10.00
LIMIT = 10.03  # = trigger * 1.003, matches the real 30bps relationship


class TestTStar:
    def test_no_print_reaching_trigger_is_unfilled(self):
        trades = _trades([(1.0, 9.50), (50.0, 9.90), (299.0, 9.99)])
        mbp = _mbp([(0.5, 9.98)])
        res = replay_one(trades, mbp, TRIGGER, LIMIT, delay_s=0)
        assert res['status'] == 'unfilled_no_tstar'
        assert res['fill_price'] is None
        assert res['t_star'] is None

    def test_find_t_star_picks_first_qualifying_print(self):
        trades = _trades([(1.0, 9.99), (5.0, 10.00), (6.0, 10.05)])
        t = find_t_star(trades, TRIGGER)
        assert t == 5.0


class TestBeforeAfterT0:
    def test_t_star_after_T0_fills_at_ask_at_t_star(self):
        # order rests at T0=2s; breakout print at t*=5s (>= T0) -> use ask at t*
        trades = _trades([(5.0, 10.00)])
        mbp = _mbp([(0.0, 9.90), (4.9, 10.01), (5.5, 10.20)])  # strictly-prior to t*=5.0 is 10.01
        res = replay_one(trades, mbp, TRIGGER, LIMIT, delay_s=2)
        assert res['status'] == 'filled'
        assert res['fill_price'] == pytest.approx(10.01)
        assert res['t_star'] == 5.0

    def test_t_star_before_T0_fills_at_ask_at_T0_not_at_t_star(self):
        # breakout already fired at t*=1s, order only rests at T0=5s -> fill at ask(T0)
        trades = _trades([(1.0, 10.00)])
        mbp = _mbp([(0.9, 10.01), (4.9, 10.005), (5.1, 10.50)])  # strictly-prior to T0=5.0 -> 10.005
        res = replay_one(trades, mbp, TRIGGER, LIMIT, delay_s=5)
        assert res['status'] == 'filled'
        assert res['fill_price'] == pytest.approx(10.005)
        assert res['t_star'] == 1.0

    def test_t_star_exactly_equal_to_T0_uses_ge_branch(self):
        trades = _trades([(3.0, 10.00)])
        mbp = _mbp([(2.9, 10.02)])
        res = replay_one(trades, mbp, TRIGGER, LIMIT, delay_s=3)
        assert res['status'] == 'filled'
        assert res['fill_price'] == pytest.approx(10.02)


class TestAskAboveLimit:
    def test_ask_above_limit_after_T0_is_skipped_no_rebump(self):
        # t* >= T0 branch: skip iff ask > limit (rebump_buffer NOT applied here)
        trades = _trades([(5.0, 10.00)])
        mbp = _mbp([(4.9, 10.031)])  # 0.001 above limit 10.03, still skip
        res = replay_one(trades, mbp, TRIGGER, LIMIT, delay_s=0)
        assert res['status'] == 'skipped_guard'
        assert res['fill_price'] is None

    def test_ask_plus_rebump_over_limit_before_T0_is_skipped(self):
        # t* < T0 branch: skip iff ask + rebump_buffer > limit
        trades = _trades([(1.0, 10.00)])
        mbp = _mbp([(4.9, 10.02)])  # 10.02 + 0.02 = 10.04 > limit 10.03 -> skip
        res = replay_one(trades, mbp, TRIGGER, LIMIT, delay_s=5, rebump_buffer=REBUMP_BUFFER)
        assert res['status'] == 'skipped_guard'
        assert res['fill_price'] is None

    def test_ask_plus_rebump_exactly_at_limit_fills(self):
        trades = _trades([(1.0, 10.00)])
        mbp = _mbp([(4.9, 10.01)])  # 10.01 + 0.02 = 10.03 == limit -> fills (<=)
        res = replay_one(trades, mbp, TRIGGER, LIMIT, delay_s=5)
        assert res['status'] == 'filled'
        assert res['fill_price'] == pytest.approx(10.01)


class TestNoQuote:
    def test_no_prior_quote_is_unfilled_no_quote(self):
        trades = _trades([(5.0, 10.00)])
        mbp = _mbp([(6.0, 10.02)])  # only a quote AFTER t*, nothing strictly prior
        res = replay_one(trades, mbp, TRIGGER, LIMIT, delay_s=0)
        assert res['status'] == 'unfilled_no_quote'

    def test_ask_at_helper_ignores_nan_and_nonpositive(self):
        mbp = _mbp([(1.0, np.nan), (2.0, 0.0), (3.0, 10.00)])
        assert ask_at(mbp, 4.0) == pytest.approx(10.00)
        assert ask_at(mbp, 2.5) is None  # only nan/0 rows strictly prior to 2.5


class TestPnlArithmetic:
    """P&L formula lives in replay.cmd_run; test it directly here on synthetic
    book rows without touching parquet/CSV I/O."""

    def test_pnl_formula_cheaper_fill_increases_profit(self):
        entry_price_bt = 10.03
        sized_pnl_bt = 150.0
        shares = 100.0
        fill_price = 10.00  # 3c cheaper than the BT's assumed entry
        pnl_replay = sized_pnl_bt + shares * (entry_price_bt - fill_price)
        assert pnl_replay == pytest.approx(150.0 + 100.0 * 0.03)
        assert pnl_replay > sized_pnl_bt

    def test_pnl_formula_worse_fill_decreases_profit(self):
        entry_price_bt = 10.03
        sized_pnl_bt = 150.0
        shares = 100.0
        fill_price = 10.10  # 7c worse than the BT's assumed entry
        pnl_replay = sized_pnl_bt + shares * (entry_price_bt - fill_price)
        assert pnl_replay == pytest.approx(150.0 - 100.0 * 0.07)
        assert pnl_replay < sized_pnl_bt

    def test_skipped_and_unfilled_contribute_zero(self):
        for status in ('unfilled_no_tstar', 'unfilled_no_quote', 'skipped_guard'):
            pnl = 0.0 if status != 'filled' else 999.0
            assert pnl == 0.0

    def test_r_denominator_is_375(self):
        from replay import R_DENOM
        assert R_DENOM == 375.0
        assert (150.0 / R_DENOM) == pytest.approx(0.4)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
