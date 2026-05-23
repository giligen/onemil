"""Unit tests for batch.intraday_bars_backfill.

Mocks AlpacaClient + Database to validate the audit + fetch + save flow
without making real API calls or touching SQLite.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, call

import pandas as pd
import pytest

from batch.intraday_bars_backfill import (
    DEFAULT_COMPLETENESS_THRESHOLD,
    audit_coverage,
    backfill,
    fetch_and_save,
)


# =========================================================================
# audit_coverage
# =========================================================================

class TestAuditCoverage:

    def test_empty_input_returns_empty(self):
        db = MagicMock()
        assert audit_coverage(db, []) == []
        db.get_intraday_bars_bulk.assert_not_called()

    def test_pairs_missing_from_cache_report_zero(self):
        db = MagicMock()
        db.get_intraday_bars_bulk.return_value = {}  # no hits
        out = audit_coverage(db, [('AAA', '2026-05-08'), ('BBB', '2026-05-08')])
        assert out == [('AAA', '2026-05-08', 0), ('BBB', '2026-05-08', 0)]

    def test_partial_and_full_pairs_correctly_counted(self):
        db = MagicMock()
        db.get_intraday_bars_bulk.return_value = {
            ('AAA', '2026-05-08'): [{}] * 138,   # partial
            ('BBB', '2026-05-08'): [{}] * 388,   # near-full
            ('CCC', '2026-05-08'): [],            # empty
        }
        out = audit_coverage(db, [
            ('AAA', '2026-05-08'),
            ('BBB', '2026-05-08'),
            ('CCC', '2026-05-08'),
        ])
        # Sorted ascending by bar count
        assert out == [
            ('CCC', '2026-05-08', 0),
            ('AAA', '2026-05-08', 138),
            ('BBB', '2026-05-08', 388),
        ]


# =========================================================================
# fetch_and_save
# =========================================================================

class TestFetchAndSave:

    def test_happy_path_saves_bars(self):
        client = MagicMock()
        df = pd.DataFrame({
            'timestamp': pd.date_range('2026-05-08 13:30',
                                         periods=3, freq='1min', tz='UTC'),
            'open': [10.0, 10.1, 10.2],
            'high': [10.1, 10.2, 10.3],
            'low': [9.9, 10.0, 10.1],
            'close': [10.05, 10.15, 10.25],
            'volume': [1000, 1100, 1200],
        })
        client.get_historical_1min_bars.return_value = df
        db = MagicMock()
        db.save_intraday_bars.return_value = 3

        fetched, saved = fetch_and_save(client, db, 'AAA', '2026-05-08')

        assert fetched == 3
        assert saved == 3
        client.get_historical_1min_bars.assert_called_once()
        # 3rd positional kwarg is the symbol
        args, _ = client.get_historical_1min_bars.call_args
        assert args[0] == 'AAA'
        db.save_intraday_bars.assert_called_once()
        save_args, _ = db.save_intraday_bars.call_args
        assert save_args[0] == 'AAA'
        assert save_args[1] == '2026-05-08'
        assert len(save_args[2]) == 3

    def test_empty_alpaca_response_returns_zero(self):
        client = MagicMock()
        client.get_historical_1min_bars.return_value = pd.DataFrame()
        db = MagicMock()

        fetched, saved = fetch_and_save(client, db, 'AAA', '2026-05-08')
        assert fetched == 0
        assert saved == 0
        db.save_intraday_bars.assert_not_called()

    def test_alpaca_exception_returns_zero_no_raise(self):
        client = MagicMock()
        client.get_historical_1min_bars.side_effect = RuntimeError("api down")
        db = MagicMock()

        # MUST NOT raise — single-pair failure cannot abort a batch run.
        fetched, saved = fetch_and_save(client, db, 'AAA', '2026-05-08')
        assert fetched == 0
        assert saved == 0

    def test_bad_date_string_returns_zero(self):
        client = MagicMock()
        db = MagicMock()
        fetched, saved = fetch_and_save(client, db, 'AAA', 'not-a-date')
        assert fetched == 0
        assert saved == 0
        client.get_historical_1min_bars.assert_not_called()

    def test_save_exception_returns_zero_no_raise(self):
        client = MagicMock()
        df = pd.DataFrame({'timestamp': [pd.Timestamp.now(tz='UTC')],
                            'open': [10], 'high': [10], 'low': [10],
                            'close': [10], 'volume': [100]})
        client.get_historical_1min_bars.return_value = df
        db = MagicMock()
        db.save_intraday_bars.side_effect = RuntimeError("disk full")

        fetched, saved = fetch_and_save(client, db, 'AAA', '2026-05-08')
        assert fetched == 0
        assert saved == 0


# =========================================================================
# backfill (integration)
# =========================================================================

class TestBackfill:

    def _df(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({
            'timestamp': pd.date_range('2026-05-08 13:30', periods=n,
                                         freq='1min', tz='UTC'),
            'open': [10.0] * n, 'high': [10.1] * n, 'low': [9.9] * n,
            'close': [10.05] * n, 'volume': [1000] * n,
        })

    def test_dry_run_no_api_calls(self):
        db = MagicMock()
        db.get_intraday_bars_bulk.return_value = {
            ('AAA', '2026-05-08'): [{}] * 100,
        }
        client = MagicMock()
        result = backfill(db, client, [('AAA', '2026-05-08')], dry_run=True)
        assert result['audited'] == 1
        assert result['incomplete'] == 1
        assert result['refetched'] == 0
        assert result['bars_added'] == 0
        client.get_historical_1min_bars.assert_not_called()

    def test_skips_pairs_above_threshold(self):
        db = MagicMock()
        db.get_intraday_bars_bulk.return_value = {
            ('AAA', '2026-05-08'): [{}] * 380,  # above default 350
            ('BBB', '2026-05-08'): [{}] * 138,  # below
        }
        client = MagicMock()
        client.get_historical_1min_bars.return_value = self._df(388)
        db.save_intraday_bars.return_value = 388

        result = backfill(db, client,
                          [('AAA', '2026-05-08'), ('BBB', '2026-05-08')])
        # AAA skipped (above threshold), BBB refetched
        assert result['audited'] == 2
        assert result['incomplete'] == 1
        assert result['refetched'] == 1
        assert result['bars_added'] == 388 - 138
        # Only ONE API call (BBB)
        assert client.get_historical_1min_bars.call_count == 1

    def test_no_refetch_when_alpaca_returns_same_or_fewer(self):
        """If Alpaca returns the same number we already cache, don't
        count as refetched. (Stock is thinly traded.)"""
        db = MagicMock()
        db.get_intraday_bars_bulk.return_value = {
            ('THIN', '2026-05-08'): [{}] * 120,
        }
        client = MagicMock()
        client.get_historical_1min_bars.return_value = self._df(120)
        db.save_intraday_bars.return_value = 120

        result = backfill(db, client, [('THIN', '2026-05-08')])
        # Audited + flagged incomplete, but refetched=0 because Alpaca
        # didn't have more data than cache already had.
        assert result['incomplete'] == 1
        assert result['refetched'] == 0
        assert result['bars_added'] == 0

    def test_threshold_override(self):
        db = MagicMock()
        db.get_intraday_bars_bulk.return_value = {
            ('AAA', '2026-05-08'): [{}] * 300,
        }
        client = MagicMock()
        # threshold=350: AAA flagged incomplete → call API
        result = backfill(db, client, [('AAA', '2026-05-08')],
                          threshold=350, dry_run=True)
        assert result['incomplete'] == 1
        # threshold=200: AAA above → no work
        result = backfill(db, client, [('AAA', '2026-05-08')],
                          threshold=200, dry_run=True)
        assert result['incomplete'] == 0

    def test_empty_input(self):
        db = MagicMock()
        client = MagicMock()
        result = backfill(db, client, [])
        assert result['audited'] == 0
        assert result['incomplete'] == 0
        assert result['refetched'] == 0

    def test_single_failure_does_not_abort_batch(self):
        db = MagicMock()
        db.get_intraday_bars_bulk.return_value = {
            ('FAIL', '2026-05-08'): [{}] * 100,
            ('OK', '2026-05-08'): [{}] * 100,
        }
        client = MagicMock()

        def selective_fetch(symbol, start, end):
            if symbol == 'FAIL':
                raise RuntimeError("rate limited")
            return self._df(380)

        client.get_historical_1min_bars.side_effect = selective_fetch
        db.save_intraday_bars.return_value = 380

        result = backfill(db, client, [
            ('FAIL', '2026-05-08'), ('OK', '2026-05-08'),
        ])
        # Batch processes both, but only OK succeeded
        assert result['incomplete'] == 2
        assert result['refetched'] == 1
        assert result['bars_added'] == 280

    def test_default_threshold_value(self):
        """The 350-bar default must not silently drift."""
        assert DEFAULT_COMPLETENESS_THRESHOLD == 350
