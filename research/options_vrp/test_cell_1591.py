"""Unit tests for cell_1591.py (PREREG_1567 v2, Amendment 2/2a). Mocked ticks/cache -- no network.

Covers: delta/IV round trip, strike selection incl. the missing-print shift rule, the tick pricing
window and VOID rule, sizing + budget assertion, management precedence, $0 months, the IV gate.
"""
import datetime as dt
import math
import os
import sqlite3
import sys
import unittest
from unittest.mock import MagicMock

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cell_1591 as m  # noqa: E402
from cell_1567 import bs_put_delta, bs_put_price, implied_vol_put  # noqa: E402


class TestDeltaIVRoundTrip(unittest.TestCase):
    def test_bs_price_then_iv_recovers_input_sigma(self):
        S, K, T, r, q, sigma = 500.0, 480.0, 45 / 365.0, 0.045, 0.013, 0.18
        price = bs_put_price(S, K, T, r, q, sigma)
        iv = implied_vol_put(price, S, K, T, r, q)
        self.assertAlmostEqual(iv, sigma, places=3)

    def test_delta_more_negative_as_strike_rises_toward_spot(self):
        S, T, r, q, sigma = 500.0, 45 / 365.0, 0.045, 0.013, 0.18
        d_low = bs_put_delta(S, 470.0, T, r, q, sigma)
        d_high = bs_put_delta(S, 495.0, T, r, q, sigma)
        self.assertLess(abs(d_low), abs(d_high))  # put delta magnitude grows closer to spot

    def test_iv_none_outside_no_arbitrage_bounds(self):
        # a price above the strike's PV (impossible for a put) must not solve
        self.assertIsNone(implied_vol_put(1000.0, 500.0, 480.0, 45 / 365.0, 0.045, 0.013))


class TestTickCache(unittest.TestCase):
    def setUp(self):
        self.con = sqlite3.connect(':memory:')
        m.init_tick_db(self.con)
        self.warn = {}

    def _fake_trade(self, ts_iso, price):
        t = MagicMock()
        t.timestamp = dt.datetime.fromisoformat(ts_iso)
        t.price = price
        t.size = 1.0
        return t

    def _fake_client(self, trades):
        client = MagicMock()
        res = MagicMock()
        res.data = {'SPYXXX': trades}
        client.get_option_trades.return_value = res
        return client

    def test_price_from_30s_window_when_present(self):
        trades = [self._fake_trade('2024-03-04T15:00:10+00:00', 2.50),
                  self._fake_trade('2024-03-04T15:04:00+00:00', 9.99)]
        client = self._fake_client(trades)
        m.fetch_tick(self.con, client, 'SPYXXX', '2024-03-04', self.warn, pause_s=0)
        price, void, fb = m.get_tick_price(self.con, 'SPYXXX', '2024-03-04')
        self.assertFalse(void)
        self.assertFalse(fb)
        self.assertEqual(price, 2.50)

    def test_fallback_to_5min_window_when_30s_empty_and_warns(self):
        trades = [self._fake_trade('2024-03-04T15:04:00+00:00', 2.61)]
        client = self._fake_client(trades)
        m.fetch_tick(self.con, client, 'SPYXXX', '2024-03-04', self.warn, pause_s=0)
        price, void, fb = m.get_tick_price(self.con, 'SPYXXX', '2024-03-04')
        self.assertFalse(void)
        self.assertTrue(fb)
        self.assertEqual(price, 2.61)
        self.assertEqual(self.warn.get('tick_fallback_5m_window'), 1)

    def test_void_when_no_trade_in_5min_window(self):
        client = self._fake_client([])
        m.fetch_tick(self.con, client, 'SPYXXX', '2024-03-04', self.warn, pause_s=0)
        price, void, fb = m.get_tick_price(self.con, 'SPYXXX', '2024-03-04')
        self.assertTrue(void)
        self.assertIsNone(price)
        self.assertEqual(self.warn.get('tick_void_no_trade'), 1)

    def test_resumable_cache_skips_second_fetch(self):
        client = self._fake_client([self._fake_trade('2024-03-04T15:00:05+00:00', 2.0)])
        made1 = m.fetch_tick(self.con, client, 'SPYXXX', '2024-03-04', self.warn, pause_s=0)
        made2 = m.fetch_tick(self.con, client, 'SPYXXX', '2024-03-04', self.warn, pause_s=0)
        self.assertTrue(made1)
        self.assertFalse(made2)
        self.assertEqual(client.get_option_trades.call_count, 1)


class TestSizingAndBudget(unittest.TestCase):
    def test_size_position_floors_and_respects_worst_case(self):
        contracts, worst = m.size_position(m.B / m.N_LADDER, m.WIDTH, net_credit=1.00)
        self.assertEqual(worst, (m.WIDTH - 1.00) * 100.0)
        self.assertEqual(contracts, math.floor((m.B / m.N_LADDER) / worst))

    def test_size_position_zero_when_worst_case_nonpositive(self):
        contracts, worst = m.size_position(1000.0, m.WIDTH, net_credit=m.WIDTH)  # credit == width -> 0 worst
        self.assertEqual(contracts, 0)

    def test_budget_assertion_style_math_never_exceeds_B(self):
        reserved = 0.0
        for _ in range(m.N_LADDER + 2):  # more rungs than the ladder has -> must clamp, never exceed B
            remaining = max(m.B - reserved, 0.0)
            alloc = min(m.B / m.N_LADDER, remaining)
            contracts, worst_per = m.size_position(alloc, m.WIDTH, net_credit=1.00)
            worst_case = worst_per * contracts
            self.assertLessEqual(reserved + worst_case, m.B + 1e-6)
            reserved += worst_case


class TestMonthlySeriesZeroFill(unittest.TestCase):
    def test_months_with_no_exit_are_zero_not_missing(self):
        cycles = pd.DataFrame([
            {'exit_date': '2024-02-10', 'pnl_usd': 100.0},
            {'exit_date': '2024-04-15', 'pnl_usd': -50.0},
        ])
        s = m.monthly_series(cycles, 'pnl_usd', '2024-02-01', '2024-04-30')
        self.assertEqual(len(s), 3)  # Feb, Mar, Apr
        self.assertEqual(s[pd.Period('2024-03', 'M')], 0.0)

    def test_empty_cycles_still_produces_zero_month_series(self):
        s = m.monthly_series(pd.DataFrame(columns=['exit_date', 'pnl_usd']), 'pnl_usd',
                              '2024-02-01', '2024-03-31')
        self.assertTrue((s == 0.0).all())
        self.assertEqual(len(s), 2)


class TestManagementPrecedenceAndGate(unittest.TestCase):
    def test_run_cycle_v2_management_b_uses_intrinsic_settlement_no_ticks(self):
        cache = MagicMock()
        cache.spy_16_close.return_value = 490.0  # short ITM by 5, long ITM by 0 (long strike 485)
        resolved = {'expiry': '2024-04-19', 'short_strike': 495.0, 'long_strike': 485.0,
                    'credits': {0.03: 1.0, 0.05: 0.9, 0.10: 0.7}, 'trigger_day': None,
                    'short_symbol': 'S', 'long_symbol': 'L', 'exit_session': None}
        con = MagicMock()
        warn = {}
        result = m.run_cycle_v2(cache, con, resolved, 'B', warn)
        self.assertEqual(result['exit_reason'], 'expiry')
        self.assertFalse(result['closed_actively'])
        self.assertAlmostEqual(result['pnl'][0.03], 1.0 - 5.0)

    def test_iv_gate_skips_low_iv_weeks(self):
        cache = object()
        cell = {'cell': 9999, 'delta': 0.2, 'mgmt': 'A', 'gate': 1}
        mondays_ctx = [('2024-02-05', {'iv_atm': 0.10})]  # below IV_GATE_MIN=0.15
        con = MagicMock()
        cycles, counts = m.run_cell(cache, con, cell, mondays_ctx, {}, {})
        self.assertEqual(counts['n_skipped_gate'], 1)
        self.assertEqual(cycles, [])


class TestStrikeShiftDiagnostic(unittest.TestCase):
    def test_true_when_closer_neighbour_has_no_print_at_all(self):
        cache = MagicMock()
        cache.opt_minute_entry = pd.DataFrame({'symbol': ['SPY_490P'], 'monday': ['2024-02-05']})
        strikes = pd.DataFrame({'strike': [480.0, 481.0, 495.0],
                                 'symbol': ['SPY_480P', 'SPY_481P', 'SPY_495P']})
        mkt = {'strikes': strikes}
        entry = {'short_strike': 480.0}
        shifted = m.strike_shift_diagnostic(cache, '2024-02-05', mkt, 0.20, entry)
        self.assertTrue(shifted)  # 481 exists in the grid but has no minute print at all

    def test_false_when_neighbour_has_a_print(self):
        cache = MagicMock()
        cache.opt_minute_entry = pd.DataFrame({'symbol': ['SPY_481P'], 'monday': ['2024-02-05']})
        strikes = pd.DataFrame({'strike': [480.0, 481.0], 'symbol': ['SPY_480P', 'SPY_481P']})
        mkt = {'strikes': strikes}
        entry = {'short_strike': 480.0}
        self.assertFalse(m.strike_shift_diagnostic(cache, '2024-02-05', mkt, 0.20, entry))

    def test_false_when_entry_is_none(self):
        self.assertFalse(m.strike_shift_diagnostic(MagicMock(), '2024-02-05', {}, 0.20, None))


if __name__ == '__main__':
    unittest.main()
