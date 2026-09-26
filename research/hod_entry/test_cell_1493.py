"""Unit tests for cell_1493.py -- the retest-bounce exit-grid walk, costs and units.

Covers: stop-first on a bar touching both, gap-through at the open, target needs high>target
(strict), the tape-first minute resolving before the bars are ever consulted, the T30/T60 time
exits, the EOD 15:55 cutoff, and the cost-unit conversions (bps -> percentage points -> R).
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell_1493 as C


def bar(m, o, h, l, c):
    return (m, o, h, l, c)


def path_arr(*bars):
    return np.array(bars, dtype=float)


class TestBarWalk(unittest.TestCase):
    """walk_one's bar phase (no tape -- prices_after=None)."""

    def row(self, retest_minute=700):
        return {'day': '2025-07-01', 'symbol': 'TST', 'retest_minute': retest_minute}

    def test_stop_first_on_a_bar_touching_both(self):
        """A bar whose low<=stop AND high>=target must exit STOP, never target (PREREG: 'stop
        first on a bar touching both')."""
        entry, stop, target = 10.0, 9.90, 10.20
        p = path_arr(bar(701, 10.0, 10.30, 9.80, 10.0))  # touches both stop and target
        res = C.walk_one(self.row(), stop, target, None, p, None)
        self.assertEqual(res['why'], 'stop')

    def test_gap_through_at_the_open(self):
        """A bar that opens BELOW the stop fills at the (worse) open, not at the stop price."""
        entry, stop, target = 10.0, 9.90, 10.20
        p = path_arr(bar(701, 9.70, 9.75, 9.60, 9.65))  # opens below stop
        res = C.walk_one(self.row(), stop, target, None, p, None)
        self.assertEqual(res['why'], 'stop')
        self.assertAlmostEqual(res['exit_price'], 9.70)  # the open, not the stop level

    def test_stop_not_gapped_fills_at_stop_level(self):
        """A bar whose open is above the stop but whose low touches it fills AT the stop, not the
        low (a resting stop-limit order, not a market order)."""
        entry, stop, target = 10.0, 9.90, 10.20
        p = path_arr(bar(701, 9.95, 9.97, 9.85, 9.90))
        res = C.walk_one(self.row(), stop, target, None, p, None)
        self.assertEqual(res['why'], 'stop')
        self.assertAlmostEqual(res['exit_price'], 9.90)

    def test_target_needs_high_strictly_above_target(self):
        """high == target must NOT trigger the target (PREREG: 'a target needs high > target')."""
        entry, stop, target = 10.0, 9.90, 10.20
        p = path_arr(bar(701, 10.0, 10.20, 9.95, 10.10))  # high exactly equals target
        res = C.walk_one(self.row(), stop, target, None, p, None)
        self.assertNotEqual(res['why'], 'target')
        p2 = path_arr(bar(701, 10.0, 10.2001, 9.95, 10.10))  # strictly above
        res2 = C.walk_one(self.row(), stop, target, None, p2, None)
        self.assertEqual(res2['why'], 'target')
        self.assertAlmostEqual(res2['exit_price'], target)

    def test_eod_cutoff_exits_at_open(self):
        stop, target = 9.90, 10.20
        p = path_arr(bar(955, 10.05, 10.10, 10.00, 10.08))
        res = C.walk_one(self.row(), stop, target, None, p, None)
        self.assertEqual(res['why'], 'eod')
        self.assertAlmostEqual(res['exit_price'], 10.05)

    def test_time_cap_fires_before_eod(self):
        stop, target = 9.90, 10.50  # target far away, never hit
        p = path_arr(bar(725, 10.00, 10.05, 9.98, 10.02),
                      bar(730, 10.03, 10.06, 10.00, 10.04))  # m_r+30 = 730
        res = C.walk_one(self.row(700), stop, target, 30, p, None)
        self.assertEqual(res['why'], 'time')
        self.assertEqual(res['exit_m'], 730)
        self.assertAlmostEqual(res['exit_price'], 10.03)

    def test_path_runs_out_before_eod_falls_back_to_last_close(self):
        stop, target = 5.0, 50.0  # neither ever hit
        p = path_arr(bar(701, 10.0, 10.05, 9.98, 10.01),
                      bar(702, 10.01, 10.04, 9.99, 10.02))
        res = C.walk_one(self.row(), stop, target, None, p, None)
        self.assertEqual(res['why'], 'eod_fallback')
        self.assertAlmostEqual(res['exit_price'], 10.02)


class TestTapeFirstMinute(unittest.TestCase):
    """The retest minute's own tape decides before the bar walk is ever consulted."""

    def row(self):
        return {'day': '2025-07-01', 'symbol': 'TST', 'retest_minute': 700}

    def test_tape_target_resolves_without_consulting_bars(self):
        stop, target = 9.90, 10.10
        prices = np.array([10.05, 10.12, 10.20])  # crosses target on the 2nd print
        # deliberately give a bar path that would say 'stop' -- must be ignored
        p = path_arr(bar(701, 9.80, 9.85, 9.70, 9.75))
        res = C.walk_one(self.row(), stop, target, None, p, prices)
        self.assertEqual(res['why'], 'target')
        self.assertAlmostEqual(res['exit_price'], target)

    def test_tape_stop_resolves_first_when_earlier_than_target(self):
        stop, target = 9.90, 10.10
        prices = np.array([9.85, 10.20])  # stop print comes first in time order
        res = C.walk_one(self.row(), stop, target, None, None, prices)
        self.assertEqual(res['why'], 'stop')
        self.assertAlmostEqual(res['exit_price'], 9.85)  # worse of print/stop

    def test_empty_tape_falls_through_to_bars(self):
        stop, target = 9.90, 10.10
        prices = np.array([])
        p = path_arr(bar(701, 10.0, 10.05, 9.98, 10.02))
        res = C.walk_one(self.row(), stop, target, None, p, prices)
        self.assertIn(res['why'], ('eod_fallback', 'stop', 'target'))
        self.assertEqual(res['exit_m'], 701)


class TestCostUnits(unittest.TestCase):
    """cost_and_pct: target/entry are free, stop and eod/time carry the programme's bps constants,
    converted to percentage points and then to R (R := entry - stop, in dollars)."""

    def test_target_is_costless(self):
        raw, cost, net, R, netR = C.cost_and_pct(entry=10.0, stop=9.0, exit_price=10.5, why='target', split='TRAIN')
        self.assertEqual(cost, 0.0)
        self.assertAlmostEqual(net, raw)

    def test_stop_cost_matches_slip_stop_bps(self):
        raw, cost, net, R, netR = C.cost_and_pct(entry=10.0, stop=9.9, exit_price=9.9, why='stop', split='VAL')
        self.assertAlmostEqual(cost, C.SLIP_STOP_BPS['VAL'] / 100.0)
        self.assertAlmostEqual(net, raw - cost)

    def test_eod_cost_matches_eod_bid_bps(self):
        raw, cost, net, R, netR = C.cost_and_pct(entry=10.0, stop=9.9, exit_price=10.05, why='eod', split='TRAIN')
        self.assertAlmostEqual(cost, C.EOD_BID_BPS['TRAIN'] / 100.0)

    def test_net_R_conversion(self):
        # 2% gain, 1% stop distance -> exactly 2R before cost
        entry, stop = 100.0, 99.0
        raw, cost, net, R, netR = C.cost_and_pct(entry, stop, exit_price=102.0, why='target', split='TRAIN')
        self.assertAlmostEqual(R, 1.0)
        self.assertAlmostEqual(netR, 2.0)


class TestGridDefinitions(unittest.TestCase):
    def test_grid_has_55_cells(self):
        self.assertEqual(len(C.cell_grid()), 55)

    def test_mirror_stop_uses_the_wider_of_dip_low_and_1pct(self):
        row = {'entry': 100.0, 'dip_low': 99.5}  # dip_low-0.01=99.49 vs entry*0.99=99.0 -> min=99.0
        stop, target, cap = C.stop_target_for_cell(row, 'M', 'M')
        self.assertAlmostEqual(stop, 99.0)
        self.assertAlmostEqual(target, 102.0)

    def test_cl_stop_uses_the_base_fills_own_stop_column(self):
        row = {'entry': 20.0, 'stop': 19.0}
        stop, target, cap = C.stop_target_for_cell(row, 'CL', 'NONE')
        self.assertAlmostEqual(stop, 19.0)
        self.assertIsNone(target)

    def test_ill_defined_stop_excluded(self):
        row = {'entry': 20.0, 'stop': 20.5, 'day': 'd', 'symbol': 's'}  # stop above entry
        stop, target, cap = C.stop_target_for_cell(row, 'CL', 'NONE')
        self.assertIsNone(stop)


if __name__ == '__main__':
    unittest.main(verbosity=2)
