"""Shared BF candidate screen (2026-08-29 alignment build).

Pins the causal semantics that replaced the twice-implemented screen:
the BT cache build and the live scanner now route through ONE predicate
(trading/bf_selection.py). The killed drift class: EOD-lookahead terms
(close>open direction gate) that made spike-then-close-red days
BT-invisible while live traded them (IREZ/LUNL/PFSA, ~50% overlap)."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading.bf_selection import (MoverVerdict, intraday_qualifies,
                                  mover_day_qualifies)


def _q(**over):
    base = dict(high=11.5, low=10.0, prev_close=10.0, price_ref=11.0,
                volume=1_000_000, threshold_pct=0.10,
                price_min=1.0, price_max=30.0, min_dollar_volume=0.0)
    base.update(over)
    return mover_day_qualifies(**base)


class TestSharedScreen:
    def test_gap_up_qualifies(self):
        v = _q(high=11.5, prev_close=10.0)           # +15% vs prev close
        assert v.qualifies and v.reason == 'gap_up'

    def test_range_move_qualifies(self):
        v = _q(high=11.5, low=10.0, prev_close=11.4)  # 15% range, no gap
        assert v.qualifies and v.reason == 'range'

    def test_red_close_spike_day_now_qualifies(self):
        """THE alignment case: big range day that closed red. The old BT
        screen's `close > open` term excluded the whole day (live traded
        it — LUNL 8/13 class). The shared screen has no close-color
        term: the detector is the crash filter, not the screen."""
        v = _q(high=12.0, low=10.0, prev_close=11.9, price_ref=10.2)
        assert v.qualifies                            # closed near lows

    def test_small_move_rejected(self):
        v = _q(high=10.5, low=10.0, prev_close=10.4)  # 5% range, no gap
        assert not v.qualifies and v.reason == 'below_threshold'

    def test_price_band(self):
        assert _q(price_ref=0.5).reason == 'price_below_min'
        assert _q(price_ref=31.0).reason == 'price_above_max'

    def test_dollar_volume_gate(self):
        v = _q(min_dollar_volume=50_000_000, volume=1_000_000,
               price_ref=11.0)                        # $11M < $50M
        assert not v.qualifies and v.reason == 'dollar_volume'

    def test_no_prev_close_range_only(self):
        """First bar in history: gap unknown, range test still works."""
        v = _q(prev_close=None)                       # 15% range
        assert v.qualifies and v.reason == 'range'

    def test_bad_bar_rejected(self):
        assert not _q(low=0).qualifies
        assert not _q(low=-1).qualifies


class TestLiveShim:
    def test_threshold_edge(self):
        assert intraday_qualifies(gap_pct=15.0, range_pct=0.0,
                                  threshold_pct_points=15.0)
        assert not intraday_qualifies(gap_pct=14.99, range_pct=14.99,
                                      threshold_pct_points=15.0)

    def test_max_semantics(self):
        assert intraday_qualifies(gap_pct=2.0, range_pct=16.0,
                                  threshold_pct_points=15.0)

    def test_equivalence_with_screen(self):
        """The live shim and the screen agree at the threshold boundary
        for identical geometry (fraction vs percent-points bridged)."""
        for hi, lo, pc in [(11.5, 10.0, 10.0), (11.0, 10.0, 10.9),
                           (10.5, 10.0, 10.45)]:
            gap = (hi - pc) / pc * 100.0
            rng = (hi - lo) / lo * 100.0
            live = intraday_qualifies(gap_pct=gap, range_pct=rng,
                                      threshold_pct_points=10.0)
            bt = mover_day_qualifies(
                high=hi, low=lo, prev_close=pc, price_ref=hi,
                volume=1, threshold_pct=0.10).qualifies
            assert live == bt, (hi, lo, pc)
