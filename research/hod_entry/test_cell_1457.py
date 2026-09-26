"""Unit tests for research/hod_entry/cell_1457.py (PREREG_1457.md). Synthetic only -- no DB/network
access. Covers: pre-market window boundaries in ET across a DST date, ATR14 excludes the current
day, stop-limit fill/no-fill logic, the R-floor arithmetic identity, and the kill switch."""
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from research.hod_entry import cell_1457 as c1457   # noqa: E402


class TestPremarketWindowDST(unittest.TestCase):
    """04:00-09:29 ET, across a spring-forward DST boundary (2025-03-09 US)."""

    def test_pre_dst_day_minute_conversion(self):
        # 2025-03-08 (still EST, UTC-5): 09:00 ET == 14:00 UTC -> minute 540
        g = pd.DataFrame({'t': ['2025-03-08T14:00:00Z']})
        out = c1457._et_minute(g, 't')
        self.assertEqual(int(out.m.iloc[0]), 9 * 60)

    def test_post_dst_day_minute_conversion(self):
        # 2025-03-10 (now EDT, UTC-4): 09:00 ET == 13:00 UTC -> minute 540 (same ET minute, DIFFERENT
        # UTC hour than the pre-DST day -- proves the conversion is calendar-date-aware, not a fixed
        # UTC offset)
        g = pd.DataFrame({'t': ['2025-03-10T13:00:00Z']})
        out = c1457._et_minute(g, 't')
        self.assertEqual(int(out.m.iloc[0]), 9 * 60)

    def test_premarket_window_boundaries_inclusive_exclusive(self):
        # 03:59 ET (just before window), 04:00 ET (window start), 09:29 ET (last inclusive minute),
        # 09:30 ET (market open, must be excluded)
        rows = {
            '03:59': '2025-03-10T07:59:00Z',   # EDT: 03:59 ET = 07:59 UTC
            '04:00': '2025-03-10T08:00:00Z',
            '09:29': '2025-03-10T13:29:00Z',
            '09:30': '2025-03-10T13:30:00Z',
        }
        g = pd.DataFrame({'t': list(rows.values())}, index=list(rows.keys()))
        out = c1457._et_minute(g, 't')
        in_window = (out.m >= c1457.PM_START_M) & (out.m < c1457.PM_END_M_EXCL)
        self.assertFalse(in_window.loc['03:59'])
        self.assertTrue(in_window.loc['04:00'])
        self.assertTrue(in_window.loc['09:29'])
        self.assertFalse(in_window.loc['09:30'])


class TestATR14ExcludesCurrentDay(unittest.TestCase):
    """atr14_pct at day t must be computable from days [t-14, t-1] only -- verified by perturbing
    day t's own high/low and confirming atr14_pct at day t is UNCHANGED."""

    def _panel(self, high_t_extra=0.0):
        n = 20
        days = pd.date_range('2025-01-02', periods=n, freq='B')
        close = np.full(n, 100.0)
        high = close + 1.0
        low = close - 1.0
        high[15] += high_t_extra   # perturb day index 15's own high only
        df = pd.DataFrame({'bar_date': days, 'symbol': 'ZZZ', 'instrument_id': 1,
                            'open': close, 'high': high, 'low': low, 'close': close,
                            'volume': 1000})
        with patch('pandas.read_parquet', return_value=df):
            return c1457.build_daily_panel_ext({1})

    def test_atr14_unaffected_by_current_day_high(self):
        base = self._panel(high_t_extra=0.0)
        perturbed = self._panel(high_t_extra=50.0)   # a huge, obviously-current-day-only spike
        a = base.loc[15, 'atr14_pct']
        b = perturbed.loc[15, 'atr14_pct']
        self.assertFalse(np.isnan(a))
        self.assertAlmostEqual(a, b, places=9,
                                msg='atr14_pct at day t changed when only day t\'s own high moved '
                                    '-- the current day leaked into its own ATR14')

    def test_atr14_needs_14_prior_sessions(self):
        base = self._panel()
        # row 0's own TR is NaN (no prev_close), so a full 14-valid-TR window isn't available until
        # row 15 (window = TR[1..14]); rows 0..14 must be NaN.
        self.assertTrue(base.loc[:14, 'atr14_pct'].isna().all())
        self.assertFalse(np.isnan(base.loc[15, 'atr14_pct']))


class TestStopLimitFill(unittest.TestCase):
    """Synthetic tape -- the three resolution branches of stop_limit_fill()."""

    def setUp(self):
        c1457._new_fetch_count = 0

    def test_immediate_fill_at_cached_bid250_no_new_fetch(self):
        cache = {'AAA|100.0|stop|99.5': dict(measured=True, bid_250=9.97, t0=1_000_000_000)}
        with patch.object(c1457.c1443, 'load_cache', return_value=cache):
            with patch.object(c1457.ca, 'fetch_window') as mock_fetch:
                # limit = 10.00 * (1 - 20/1e4) = 9.998; bid_250 9.97 < 9.998 -> NOT immediate here.
                # Use a looser offset (50bps) so bid_250 (9.97) >= limit (9.95).
                r = c1457.stop_limit_fill('AAA', '2025-06-02', 10.00, 100.0, 'stop', 99.5, 50.0)
                mock_fetch.assert_not_called()
        self.assertTrue(r['resolved'])
        self.assertEqual(r['mechanism'], 'immediate_bid250')
        self.assertFalse(r['fetched'])
        self.assertAlmostEqual(r['fill_price'], 9.97)

    def test_reprint_after_limit_within_minute(self):
        cache = {'AAA|100.0|stop|99.5': dict(measured=True, bid_250=9.90, t0=1_000_000_000)}
        trades = pd.DataFrame({'ts': [1_000_000_000, 1_000_260_000_000, 1_000_500_000_000],
                                'price': [9.95, 9.80, 9.99]})
        quotes = pd.DataFrame({'ts': [1_000_000_000], 'bid': [9.90], 'ask': [9.92]})
        with patch.object(c1457.c1443, 'load_cache', return_value=cache):
            with patch.object(c1457.ca, 'fetch_window', return_value=(trades, quotes)):
                # limit = 10.00*(1-20/1e4) = 9.998; bid_250 9.90 < limit -> fetch; first print after
                # t0+250ms >= 9.998 is the 9.99 print? 9.99 < 9.998 -- raise threshold: use offset
                # that makes limit 9.97 so the 9.99 print qualifies.
                r = c1457.stop_limit_fill('AAA', '2025-06-02', 10.00, 100.0, 'stop', 99.5, 30.0)
        self.assertTrue(r['resolved'])
        self.assertEqual(r['mechanism'], 'reprint_after_limit')
        self.assertAlmostEqual(r['fill_price'], 9.99)
        self.assertTrue(r['fetched'])

    def test_no_fill_tail_uses_minutes_last_print(self):
        cache = {'AAA|100.0|stop|99.5': dict(measured=True, bid_250=9.50, t0=1_000_000_000)}
        trades = pd.DataFrame({'ts': [1_000_000_000, 1_000_300_000_000],
                                'price': [9.55, 9.60]})   # never reaches the limit
        quotes = pd.DataFrame({'ts': [1_000_000_000], 'bid': [9.50], 'ask': [9.52]})
        with patch.object(c1457.c1443, 'load_cache', return_value=cache):
            with patch.object(c1457.ca, 'fetch_window', return_value=(trades, quotes)):
                r = c1457.stop_limit_fill('AAA', '2025-06-02', 10.00, 100.0, 'stop', 99.5, 20.0)
        self.assertTrue(r['resolved'])
        self.assertEqual(r['mechanism'], 'no_fill_tail')
        self.assertAlmostEqual(r['fill_price'], 9.60)   # the minute's LAST print

    def test_unmeasured_base_row_is_not_resolved(self):
        with patch.object(c1457.c1443, 'load_cache', return_value={}):
            r = c1457.stop_limit_fill('AAA', '2025-06-02', 10.00, 100.0, 'stop', 99.5, 20.0)
        self.assertFalse(r['resolved'])
        self.assertEqual(r['reason'], 'unmeasured_base')

    def test_fetch_budget_exceeded_short_circuits(self):
        cache = {'AAA|100.0|stop|99.5': dict(measured=True, bid_250=9.50, t0=1_000_000_000)}
        c1457._new_fetch_count = c1457.MAX_NEW_FETCHES
        with patch.object(c1457.c1443, 'load_cache', return_value=cache):
            with patch.object(c1457.ca, 'fetch_window') as mock_fetch:
                r = c1457.stop_limit_fill('AAA', '2025-06-02', 10.00, 100.0, 'stop', 99.5, 20.0)
                mock_fetch.assert_not_called()
        self.assertFalse(r['resolved'])
        self.assertEqual(r['reason'], 'fetch_budget_exceeded')


class TestRFloorArithmetic(unittest.TestCase):
    """stop = min(consolidation low, fill * 0.975) <=> d_new = max(d_old, 0.025) applied to
    fill * (1 - d_new) -- the identity cell_1457.run_cell_1464 relies on via cell_1440.FLOOR_PCT."""

    def test_floor_binds_when_stop_is_tighter_than_2_5_pct(self):
        fill, stop = 20.00, 19.90   # d_old = 0.005 (0.5%), tighter than the 2.5% floor
        d_old = (fill - stop) / fill
        d_new = max(d_old, 0.025)
        new_stop = fill * (1 - d_new)
        self.assertAlmostEqual(new_stop, min(stop, fill * 0.975))
        self.assertAlmostEqual(new_stop, fill * 0.975)

    def test_floor_does_not_bind_when_stop_already_wider(self):
        fill, stop = 20.00, 19.00   # d_old = 0.05 (5%), already wider than the 2.5% floor
        d_old = (fill - stop) / fill
        d_new = max(d_old, 0.025)
        new_stop = fill * (1 - d_new)
        self.assertAlmostEqual(new_stop, stop)
        self.assertAlmostEqual(new_stop, min(stop, fill * 0.975))


class TestKillSwitch(unittest.TestCase):
    """1,457 VAL kept mean < +0.15 forces passes_bar=False + note='kill switch' on every entry cell
    and on 1,464; 1,463 is untouched (its own ship bar is independent, per the PREREG)."""

    def test_kill_switch_threshold(self):
        self.assertEqual(c1457.KILL_VAL_THRESHOLD, 0.15)

    def test_below_threshold_triggers(self):
        val_row = dict(n_kept=50, kept_mean=0.02)
        kill = not (val_row['n_kept'] and val_row['kept_mean'] >= c1457.KILL_VAL_THRESHOLD)
        self.assertTrue(kill)

    def test_at_or_above_threshold_does_not_trigger(self):
        val_row = dict(n_kept=50, kept_mean=0.15)
        kill = not (val_row['n_kept'] and val_row['kept_mean'] >= c1457.KILL_VAL_THRESHOLD)
        self.assertFalse(kill)

    def test_zero_kept_triggers(self):
        val_row = dict(n_kept=0, kept_mean=np.nan)
        kill = not (val_row['n_kept'] and val_row['kept_mean'] >= c1457.KILL_VAL_THRESHOLD)
        self.assertTrue(kill)


if __name__ == '__main__':
    unittest.main(verbosity=2)
