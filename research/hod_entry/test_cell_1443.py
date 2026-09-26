"""Unit tests for cell 1,443 (stop slippage on the tape). All tape fetches are synthetic — no network,
no real API key needed — via monkeypatching `causal_arming.fetch_window` (`cell_1443.ca.fetch_window`).
"""
import os
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cell_1443 as c1443                                   # noqa: E402


def _trades(rows):
    """rows: list of (ts_ns, price)."""
    return pd.DataFrame({'ts': [r[0] for r in rows], 'price': [r[1] for r in rows],
                         'size': [100] * len(rows)}).astype({'ts': 'int64'})


def _quotes(rows):
    """rows: list of (ts_ns, bid, ask)."""
    return pd.DataFrame({'ts': [r[0] for r in rows], 'bid': [r[1] for r in rows],
                         'ask': [r[2] for r in rows]}).astype({'ts': 'int64'})


class TestMeasureStop(unittest.TestCase):
    """Stop / stop_bar exit measurement: t0 = first print <= stop, bid_250 = quote at t0+250ms."""

    def test_bid_250_picks_the_right_quote(self):
        """Three quotes straddle t0+250ms; the one at exactly t0+250ms (last valid <= that ts) wins,
        not an earlier or later one."""
        t0 = 1_000_000_000
        trades = _trades([(t0, 10.00)])                     # first print <= stop(10.00) at t0
        quotes = _quotes([(t0 - 10**6, 9.80, 9.90),          # before t0 — stale
                          (t0 + 200_000_000, 9.90, 10.00),   # before t0+250ms — should be picked
                          (t0 + 300_000_000, 9.99, 10.02)])  # after t0+250ms — must NOT be picked
        with mock.patch.object(c1443.ca, 'fetch_window', return_value=(trades, quotes)):
            res = c1443.measure_stop('TEST', '2026-01-05', stop=10.00, exit_m=600, why='stop',
                                      fill_min=590.0)
        self.assertTrue(res['measured'])
        self.assertAlmostEqual(res['bid_250'], 9.90)
        self.assertEqual(res['t0'], t0)

    def test_slip_sign_worse_than_stop_is_positive(self):
        """bid_250 below the stop -> positive slip_bps (worse fill than the stop price)."""
        t0 = 2_000_000_000
        trades = _trades([(t0, 20.00)])
        quotes = _quotes([(t0 + 100_000_000, 19.90, 19.95)])  # bid 19.90 vs stop 20.00 -> worse
        with mock.patch.object(c1443.ca, 'fetch_window', return_value=(trades, quotes)):
            res = c1443.measure_stop('TEST', '2026-01-05', stop=20.00, exit_m=600, why='stop',
                                      fill_min=590.0)
        self.assertTrue(res['measured'])
        expected_bps = (20.00 - 19.90) / 20.00 * 1e4
        self.assertAlmostEqual(res['slip_bps'], expected_bps)
        self.assertGreater(res['slip_bps'], 0)

    def test_slip_sign_better_than_stop_is_negative(self):
        """bid_250 above the stop -> negative slip_bps (better than the stop price)."""
        t0 = 2_000_000_000
        trades = _trades([(t0, 20.00)])
        quotes = _quotes([(t0 + 100_000_000, 20.10, 20.15)])
        with mock.patch.object(c1443.ca, 'fetch_window', return_value=(trades, quotes)):
            res = c1443.measure_stop('TEST', '2026-01-05', stop=20.00, exit_m=600, why='stop',
                                      fill_min=590.0)
        self.assertLess(res['slip_bps'], 0)

    def test_stop_bar_is_flagged_approx_and_uses_fill_minute(self):
        """why == 'stop_bar' -> window is floor(fill_min), and the row is flagged fill_bar_approx
        because the fill instant is not stored in the CSV."""
        t0 = 3_000_000_000
        trades = _trades([(t0, 15.00)])
        quotes = _quotes([(t0 + 100_000_000, 14.95, 15.00)])
        captured = {}

        def fake_fetch_window(symbol, day, m_lo, m_hi):
            captured['m_lo'], captured['m_hi'] = m_lo, m_hi
            return trades, quotes

        with mock.patch.object(c1443.ca, 'fetch_window', side_effect=fake_fetch_window):
            res = c1443.measure_stop('TEST', '2026-01-05', stop=15.00, exit_m=605, why='stop_bar',
                                      fill_min=600.7)
        self.assertEqual(res['flag'], 'fill_bar_approx')
        self.assertEqual(captured['m_lo'], 600)              # floor(600.7), not exit_m (605)
        self.assertEqual(captured['m_hi'], 601)
        self.assertTrue(res['measured'])

    def test_no_print_le_stop_is_unmeasured(self):
        """Every print in the window is above the stop -> no_print_le_stop, not measured."""
        trades = _trades([(1_000_000_000, 25.00)])           # all prints > stop
        quotes = _quotes([(1_000_000_000, 24.90, 24.95)])
        with mock.patch.object(c1443.ca, 'fetch_window', return_value=(trades, quotes)):
            res = c1443.measure_stop('TEST', '2026-01-05', stop=20.00, exit_m=600, why='stop',
                                      fill_min=590.0)
        self.assertFalse(res['measured'])
        self.assertEqual(res['reason'], 'no_print_le_stop')

    def test_empty_tape_is_unmeasured_no_tape(self):
        empty_t = _trades([])
        empty_q = _quotes([])
        with mock.patch.object(c1443.ca, 'fetch_window', return_value=(empty_t, empty_q)):
            res = c1443.measure_stop('TEST', '2026-01-05', stop=20.00, exit_m=600, why='stop',
                                      fill_min=590.0)
        self.assertFalse(res['measured'])
        self.assertEqual(res['reason'], 'no_tape')

    def test_fetch_exception_is_unmeasured_not_raised(self):
        """A fetch failure (rate limit, symbol delisted, etc) must be counted, never raised —
        matches the project's fetch-completeness-gate rule (LOST rows are counted, not silently
        dropped or fatal)."""
        with mock.patch.object(c1443.ca, 'fetch_window', side_effect=RuntimeError('boom')):
            res = c1443.measure_stop('TEST', '2026-01-05', stop=20.00, exit_m=600, why='stop',
                                      fill_min=590.0)
        self.assertFalse(res['measured'])
        self.assertIn('fetch_error', res['reason'])


class TestMeasureEod(unittest.TestCase):
    """EOD exit measurement: t0 = first print of the 15:55 minute, bid AT t0 (no +250ms offset)."""

    def test_eod_uses_bid_at_first_print_not_250ms_later(self):
        t0 = 5_000_000_000
        trades = _trades([(t0, 30.00), (t0 + 10_000_000_000, 30.05)])  # first print is t0
        quotes = _quotes([(t0 - 10**6, 29.90, 29.95),         # prevailing at t0
                          (t0 + 300_000_000, 29.99, 30.02)])  # later — must NOT be picked
        with mock.patch.object(c1443.ca, 'fetch_window', return_value=(trades, quotes)):
            res = c1443.measure_eod('TEST', '2026-01-05', exit_price=30.00)
        self.assertTrue(res['measured'])
        self.assertEqual(res['t0'], t0)
        self.assertAlmostEqual(res['bid_250'], 29.90)
        expected_bps = (30.00 - 29.90) / 30.00 * 1e4
        self.assertAlmostEqual(res['slip_bps'], expected_bps)

    def test_eod_no_tape_is_unmeasured(self):
        empty_t, empty_q = _trades([]), _quotes([])
        with mock.patch.object(c1443.ca, 'fetch_window', return_value=(empty_t, empty_q)):
            res = c1443.measure_eod('TEST', '2026-01-05', exit_price=30.00)
        self.assertFalse(res['measured'])
        self.assertEqual(res['reason'], 'no_tape')


class TestAggregation(unittest.TestCase):
    """slip_R composes with net_R the same way cell_1430's stop_slip_net does: slip_$ / R."""

    def test_slip_R_matches_stop_slip_net_convention(self):
        fills = pd.DataFrame({
            'day': ['2026-01-05'], 'symbol': ['TEST'], 'holdout': ['VAL'], 'why': ['stop'],
            'stop': [20.00], 'exit_price': [20.00], 'R': [0.50], 'net_R': [-1.0],
            'exit_m': [600], 'fill_min': [590.0],
        })
        fills['slip_bps'] = [50.0]                            # 50 bps worse
        fills['measured'] = [True]
        fills['flag'] = ['']
        fills['slip_dollar'] = fills['stop'].where(fills.why != 'eod', fills['exit_price']) \
            * fills['slip_bps'] / 1e4
        fills['slip_R'] = fills['slip_dollar'] / fills['R']
        # slip_$ = 20.00 * 50/1e4 = 0.10; slip_R = 0.10 / 0.50 = 0.20 R
        self.assertAlmostEqual(fills['slip_R'].iloc[0], 0.20)
        after = fills['net_R'] - fills['slip_R']
        self.assertAlmostEqual(after.iloc[0], -1.20)


if __name__ == '__main__':
    unittest.main()
