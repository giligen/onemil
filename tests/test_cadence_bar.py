"""Unit tests for scripts/cadence_bar.py on synthetic weekly series.

Per docs/cadence_bar.md: the cadence bar scores WEEKS and the CYCLE between
strong weeks (net >= +5R), not individual trades. These tests build small
synthetic weekly series directly (bypassing CSV/date-range plumbing) and
check the C1-C7 scoring functions make the calls the doc says they should.
"""
import os
import random
import sys
import unittest
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import cadence_bar as cb  # noqa: E402


def weeks_from(values, start=date(2025, 1, 6)):
    """Turn a list of weekly R values into a `weekly` series (list of
    (monday_date, r)) starting at `start` (a Monday), one week apart.
    """
    return [(start + timedelta(weeks=i), r) for i, r in enumerate(values)]


def trades_from(values, start=date(2025, 1, 6)):
    """Turn a list of weekly R values into one synthetic trade per week,
    dated on that week's Monday — enough to drive score_c4/c5's use of
    trades_in_split (date + r) without needing real CSV/date filtering.
    """
    return [{"date": start + timedelta(weeks=i), "r": r, "symbol": None}
            for i, r in enumerate(values)]


class TestPassingBook(unittest.TestCase):
    """A book with frequent, regular strong weeks and shallow bleed between
    them should clear C1 (cadence), C2 (bleed bound) and C3 (shallow reds).
    """

    def setUp(self):
        # 12 cycles of [strong +5, small green +0.6, bleed -1] -> gap=3
        # every time, bleed -0.4/cycle, all cycle_net > 0.
        pattern = [5.0, 0.6, -1.0]
        self.values = pattern * 12
        self.weekly = weeks_from(self.values)
        self.cycles, self.strong_idx = cb.compute_cycles(self.weekly, strong_r=5.0)

    def test_c1_cadence_passes(self):
        c1 = cb.score_c1(self.cycles, gap_median_thresh=3, gap_p90_thresh=6)
        self.assertTrue(c1["pass"])
        self.assertEqual(c1["median"], 3)
        self.assertEqual(c1["p90"], 3)

    def test_c2_bleed_bound_passes(self):
        c2 = cb.score_c2(self.cycles, bleed_p90_thresh=-4, cycle_pos_thresh=0.75)
        self.assertTrue(c2["pass"])
        self.assertAlmostEqual(c2["bleed_p90"], -0.4, places=6)
        self.assertEqual(c2["net_pos_frac"], 1.0)

    def test_c3_shallow_reds_passes(self):
        c3 = cb.score_c3(self.weekly, p10_thresh=-2, min_thresh=-4,
                          mdd_thresh=8, underwater_thresh=6)
        self.assertTrue(c3["pass"])
        self.assertEqual(c3["min"], -1.0)


class TestQuarterlyMonsterFailsC1(unittest.TestCase):
    """A once-a-quarter monster book: strong weeks ~13 weeks apart. Cadence
    (C1) must fail even though each strong week individually clears the bar.
    """

    def test_c1_fails_on_wide_gaps(self):
        values = [0.1] * 52
        for idx in (0, 13, 26, 39):
            values[idx] = 6.0
        weekly = weeks_from(values)
        cycles, _ = cb.compute_cycles(weekly, strong_r=5.0)
        c1 = cb.score_c1(cycles, gap_median_thresh=3, gap_p90_thresh=6)
        self.assertFalse(c1["pass"])
        self.assertEqual(c1["gaps"], [13, 13, 13])
        self.assertEqual(c1["median"], 13)


class TestBleedingBookFailsC2(unittest.TestCase):
    """Strong weeks come often enough (C1 passes) but the single week
    between each one digs a hole deeper than the bleed bound allows.
    """

    def test_c2_fails_on_deep_bleed(self):
        # gap=2 every cycle (passes C1); bleed -6 every cycle (fails C2).
        values = ([5.0, -6.0]) * 10
        weekly = weeks_from(values)
        cycles, _ = cb.compute_cycles(weekly, strong_r=5.0)
        c1 = cb.score_c1(cycles, gap_median_thresh=3, gap_p90_thresh=6)
        c2 = cb.score_c2(cycles, bleed_p90_thresh=-4, cycle_pos_thresh=0.75)
        self.assertTrue(c1["pass"])
        self.assertFalse(c2["pass"])
        self.assertEqual(c2["bleed_p90"], -6.0)
        self.assertEqual(c2["net_pos_frac"], 0.0)  # every cycle_net = 5-6 = -1


class TestShallowRedCheck(unittest.TestCase):
    """C3 distinguishes a book whose worst week is shallow from one that
    has a single deep red week busting the -4R floor.
    """

    def test_shallow_red_passes(self):
        values = [5.0, -1.0, -1.5, 0.2] * 6
        weekly = weeks_from(values)
        c3 = cb.score_c3(weekly, p10_thresh=-2, min_thresh=-4,
                          mdd_thresh=8, underwater_thresh=6)
        self.assertTrue(c3["pass"])
        self.assertEqual(c3["min"], -1.5)

    def test_deep_red_fails(self):
        values = [5.0, -1.0, -1.5, 0.2] * 6
        values[5] = -10.0  # one blown week
        weekly = weeks_from(values)
        c3 = cb.score_c3(weekly, p10_thresh=-2, min_thresh=-4,
                          mdd_thresh=8, underwater_thresh=6)
        self.assertFalse(c3["pass"])
        self.assertEqual(c3["min"], -10.0)


class TestFlatWeekRule(unittest.TestCase):
    """A week with |P&L| < 0.5R is FLAT — neither green nor red, and must
    not count toward the C4 green-share denominator.
    """

    def test_classify_week_boundaries(self):
        self.assertEqual(cb.classify_week(0.49), "flat")
        self.assertEqual(cb.classify_week(-0.49), "flat")
        self.assertEqual(cb.classify_week(0.5), "green")
        self.assertEqual(cb.classify_week(-0.5), "red")
        self.assertEqual(cb.classify_week(5.0), "green")

    def test_flat_weeks_excluded_from_green_share(self):
        weekly = weeks_from([5.0, 0.3, -0.2, -1.0])
        # green: week0 (5.0). red: week3 (-1.0). flat: week1, week2 excluded.
        share = cb._green_share(weekly)
        self.assertEqual(share, 0.5)


class TestNull(unittest.TestCase):
    """C4's count-matched null: a coin-flip book (green share ~50%, no
    real edge) must fail both the absolute floor and the null margin; the
    sign-shuffle null itself should land near 50% on a symmetric series.
    """

    def test_coinflip_book_fails_c4(self):
        # Alternating +1/-1: green share exactly 50%, well under the 55%
        # absolute floor regardless of the null.
        values = [1.0, -1.0] * 30
        weekly = weeks_from(values)
        trades = trades_from(values)
        rng = random.Random(42)
        c4 = cb.score_c4(weekly, trades, green_thresh=0.55, green_margin=0.10,
                          n_null=200, rng=rng)
        self.assertFalse(c4["pass"])
        self.assertAlmostEqual(c4["green"], 0.5, places=6)
        # The null on a sign-symmetric series should itself land near 50%.
        self.assertAlmostEqual(c4["null"], 0.5, delta=0.1)

    def test_real_edge_clears_null_margin(self):
        # Mostly green with a few sharp reds: real green share well above
        # both the 55% floor and the sign-shuffle null + 10pp margin.
        values = ([1.0, 1.0, 1.0, -3.0]) * 15
        weekly = weeks_from(values)
        trades = trades_from(values)
        rng = random.Random(7)
        c4 = cb.score_c4(weekly, trades, green_thresh=0.55, green_margin=0.10,
                          n_null=200, rng=rng)
        self.assertAlmostEqual(c4["green"], 0.75, places=6)
        self.assertTrue(c4["pass"])
        self.assertLess(c4["null"], c4["green"] - 0.10)


if __name__ == "__main__":
    unittest.main()
