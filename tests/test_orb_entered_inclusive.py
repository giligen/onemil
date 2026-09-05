"""Entered-inclusive ORB book (2026-09-05, the PFSA 8/31 lesson).

Non-fill candidates are emitted into the features CSV, rank like every
other candidate, consume a slot at $0 in the pipeline, and are excluded
from the green-check's fill-parity subject set.
"""
import math

import pytest

from scripts.report_common import bt_filled_symbols
from study_orb import OrbTrade
from study_orb_features import NO_FILL_REASON, trade_row
from study_orb_pipeline_static_lock import is_no_fill


def _trade(entered, range_high=3.355, pnl=0.0):
    t = OrbTrade(variant='v', symbol='SHMD', date='2026-08-31',
                 range_high=range_high, range_low=3.19, range_size=range_high - 3.19)
    t.entered = entered
    if entered:
        t.entry_price = range_high
        t.pnl = pnl
        t.pnl_pct = pnl / 100.0
        t.exit_reason = 'lock_stop'
    return t


FEATS = {'gap_pct': 8.15, 'range_size_pct': 5.17}


class TestTradeRow:
    def test_entered_row_flagged_1(self):
        r = trade_row('SHMD', '2026-08-31', _trade(True, pnl=120.0), FEATS)
        assert r['entered'] == 1 and r['pnl'] == 120.0 and r['win'] == 1
        assert r['exit_reason'] == 'lock_stop' and r['gap_pct'] == 8.15

    def test_no_fill_row_books_zero_at_order_level(self):
        r = trade_row('SHMD', '2026-08-31', _trade(False), FEATS)
        assert r['entered'] == 0
        assert r['pnl'] == 0.0 and r['pnl_pct'] == 0.0 and r['win'] == 0
        assert r['exit_reason'] == NO_FILL_REASON
        assert r['entry_price'] == pytest.approx(3.355)   # the stop trigger live places
        assert r['range_size_pct'] == 5.17                # features kept for ranking

    def test_rangeless_candidate_is_dropped(self):
        assert trade_row('X', '2026-08-31', _trade(False, range_high=0.0), FEATS) is None


class TestIsNoFill:
    def test_zero_is_no_fill(self):
        assert is_no_fill({'entered': 0}) is True
        assert is_no_fill({'entered': '0'}) is True

    def test_one_is_filled(self):
        assert is_no_fill({'entered': 1}) is False

    def test_missing_or_nan_is_legacy_filled(self):
        assert is_no_fill({'pnl': 1.0}) is False
        assert is_no_fill({'entered': math.nan}) is False
        assert is_no_fill({'entered': None}) is False


class TestBtFilledSymbols:
    def test_splits_entered_from_no_fill(self):
        rows = [{'symbol': 'PFSA', 'entered': 1}, {'symbol': 'SHMD', 'entered': 0},
                {'symbol': 'LEGACY'}, {'symbol': 'NANROW', 'entered': math.nan}]
        assert bt_filled_symbols(rows) == {'PFSA', 'LEGACY', 'NANROW'}

    def test_empty(self):
        assert bt_filled_symbols([]) == set()
