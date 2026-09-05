"""Tests for scripts/orb_premarket_backfill.py (2026-09-05)."""
from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from data_sources.alpaca_client import AlpacaAPIError, AlpacaClient
from persistence.database import Database
from scripts.orb_premarket_backfill import (
    PremarketBackfill, bars_to_records, batches, load_candidates_csv,
    premarket_window_utc,
)


class TestWindow:
    def test_edt_window(self):
        s, e = premarket_window_utc(date(2025, 9, 10))   # EDT = UTC-4
        assert s == datetime(2025, 9, 10, 8, 0, tzinfo=timezone.utc)
        assert e == datetime(2025, 9, 10, 13, 34, 59, tzinfo=timezone.utc)

    def test_est_window(self):
        s, e = premarket_window_utc(date(2025, 3, 5))    # EST = UTC-5
        assert s == datetime(2025, 3, 5, 9, 0, tzinfo=timezone.utc)
        assert e == datetime(2025, 3, 5, 14, 34, 59, tzinfo=timezone.utc)

    def test_dst_switch_day(self):
        # 2025-03-09 is the spring-forward Sunday; Monday 3/10 is EDT
        s, _ = premarket_window_utc(date(2025, 3, 10))
        assert s == datetime(2025, 3, 10, 8, 0, tzinfo=timezone.utc)


class TestHelpers:
    def test_batches(self):
        assert list(batches(['a', 'b', 'c', 'd', 'e'], 2)) == [['a', 'b'], ['c', 'd'], ['e']]

    def test_bars_to_records_empty(self):
        assert bars_to_records(None) == []
        assert bars_to_records(pd.DataFrame()) == []

    def test_bars_to_records_shape(self):
        df = pd.DataFrame([{'timestamp': pd.Timestamp('2025-09-10T08:00:00Z'), 'open': 1.0,
                            'high': 1.2, 'low': 0.9, 'close': 1.1, 'volume': 10}])
        r = bars_to_records(df)
        assert r == [{'timestamp': pd.Timestamp('2025-09-10T08:00:00Z'), 'open': 1.0,
                      'high': 1.2, 'low': 0.9, 'close': 1.1, 'volume': 10}]

    def test_load_candidates_csv_filters_range(self, tmp_path):
        p = tmp_path / 'c.csv'
        p.write_text('bar_date,symbol\n2025-01-02,AAA\n2025-01-02,BBB\n2025-02-01,CCC\n')
        got = load_candidates_csv(str(p), date(2025, 1, 1), date(2025, 1, 31))
        assert got == {'2025-01-02': ['AAA', 'BBB']}


def _df(n, start='2025-09-10T08:00:00Z'):
    ts = pd.date_range(start, periods=n, freq='1min', tz='UTC')
    return pd.DataFrame({'timestamp': ts, 'open': 1.0, 'high': 1.1, 'low': 0.9,
                         'close': 1.0, 'volume': 100})


class TestBackfill:
    @pytest.fixture
    def db(self):
        db = MagicMock(spec=Database)
        db.get_premarket_backfilled_symbols.return_value = set()
        db.save_intraday_bars.side_effect = lambda sym, d, recs: len(recs)
        return db

    @pytest.fixture
    def alpaca(self):
        return MagicMock(spec=AlpacaClient)

    def test_fetches_saves_and_marks(self, db, alpaca):
        alpaca.get_1min_bars_range_multi.return_value = {'AAA': _df(3), 'BBB': pd.DataFrame()}
        job = PremarketBackfill(alpaca, db, batch=50)
        job.run({'2025-09-10': ['AAA', 'BBB']})
        alpaca.get_1min_bars_range_multi.assert_called_once()
        syms, s, e = alpaca.get_1min_bars_range_multi.call_args[0]
        assert syms == ['AAA', 'BBB']
        assert (s, e) == premarket_window_utc(date(2025, 9, 10))
        db.save_intraday_bars.assert_called_once()
        assert db.save_intraday_bars.call_args[0][:2] == ('AAA', '2025-09-10')
        # BOTH marked — BBB with n_bars=0 (fetched, no prints)
        marks = {c[0][0]: c[0][2] for c in db.mark_premarket_backfilled.call_args_list}
        assert marks == {'AAA': 3, 'BBB': 0}
        assert job.stats['fetched'] == 2 and job.stats['with_prints'] == 1
        assert job.stats['bars_saved'] == 3 and job.stats['skipped_failed'] == 0

    def test_resume_skips_done_pairs(self, db, alpaca):
        db.get_premarket_backfilled_symbols.return_value = {'AAA'}
        alpaca.get_1min_bars_range_multi.return_value = {'BBB': _df(2)}
        job = PremarketBackfill(alpaca, db)
        job.run({'2025-09-10': ['AAA', 'BBB']})
        assert alpaca.get_1min_bars_range_multi.call_args[0][0] == ['BBB']
        assert job.stats['already_done'] == 1

    def test_all_done_makes_no_api_call(self, db, alpaca):
        db.get_premarket_backfilled_symbols.return_value = {'AAA'}
        PremarketBackfill(alpaca, db).run({'2025-09-10': ['AAA']})
        alpaca.get_1min_bars_range_multi.assert_not_called()

    def test_batching_splits_calls(self, db, alpaca):
        alpaca.get_1min_bars_range_multi.return_value = {}
        job = PremarketBackfill(alpaca, db, batch=2)
        job.run({'2025-09-10': ['A', 'B', 'C']})
        assert alpaca.get_1min_bars_range_multi.call_count == 2
        # symbols absent from the response are marked with 0 bars
        assert db.mark_premarket_backfilled.call_count == 3

    def test_failed_batch_retries_then_skips_unmarked(self, db, alpaca):
        alpaca.get_1min_bars_range_multi.side_effect = AlpacaAPIError('boom')
        job = PremarketBackfill(alpaca, db, retries=1, retry_sleep_s=0)
        job.run({'2025-09-10': ['AAA', 'BBB']})
        assert alpaca.get_1min_bars_range_multi.call_count == 2
        db.mark_premarket_backfilled.assert_not_called()
        db.save_intraday_bars.assert_not_called()
        assert job.stats['skipped_failed'] == 2

    def test_dry_run_touches_nothing(self, db, alpaca):
        job = PremarketBackfill(alpaca, db, dry_run=True)
        job.run({'2025-09-10': ['AAA']})
        alpaca.get_1min_bars_range_multi.assert_not_called()
        db.save_intraday_bars.assert_not_called()
        db.mark_premarket_backfilled.assert_not_called()
        db.get_premarket_backfilled_symbols.assert_not_called()

    def test_limit_restricts_dates(self, db, alpaca):
        alpaca.get_1min_bars_range_multi.return_value = {}
        job = PremarketBackfill(alpaca, db)
        job.run({'2025-09-10': ['A'], '2025-09-11': ['B'], '2025-09-12': ['C']}, limit=2)
        assert job.stats['dates'] == 2
