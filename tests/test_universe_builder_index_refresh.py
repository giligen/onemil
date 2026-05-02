"""Tests for UniverseBuilder._refresh_index_symbols (SPY etc).

This step was added 2026-05-02 as part of the EAF post-mortem fix — SPY
daily_bars were 14 days stale on prod because nothing was refreshing them
(SPY is not in the tradeable universe). The universe-rebuild cron now owns
this refresh; tests pin that ownership.
"""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest

from batch.universe_builder import (
    INDEX_REFRESH_LOOKBACK_DAYS,
    INDEX_SYMBOLS,
    UniverseBuilder,
)
from data_sources.alpaca_client import AlpacaClient
from data_sources.float_provider import FloatProvider
from persistence.database import Database


@pytest.fixture
def builder():
    """Builder with all I/O dependencies mocked via spec= for safety."""
    return UniverseBuilder(
        alpaca_client=MagicMock(spec=AlpacaClient),
        float_provider=MagicMock(spec=FloatProvider),
        db=MagicMock(spec=Database),
    )


# ---------------------------------------------------------------------------
# Module-level constants — frozen API
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_spy_is_in_default_index_symbols(self):
        # Removing SPY from this list reintroduces the post-mortem bug.
        assert 'SPY' in INDEX_SYMBOLS

    def test_lookback_covers_50_sma(self):
        # Conviction's SPY 3-day range needs ~3 prior bars; the regime
        # filter needs 50 SMA. 100-day lookback covers both with a buffer.
        assert INDEX_REFRESH_LOOKBACK_DAYS >= 70


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_fetches_with_correct_window(self, builder):
        builder.alpaca.get_daily_bars_range.return_value = {
            'SPY': [
                {'date': date(2026, 4, 30), 'open': 1, 'high': 2, 'low': 0.5, 'close': 1.5, 'volume': 100},
            ],
        }
        builder._refresh_index_symbols(['SPY'], lookback_days=100)

        builder.alpaca.get_daily_bars_range.assert_called_once()
        args, _ = builder.alpaca.get_daily_bars_range.call_args
        symbols, start, end = args
        assert symbols == ['SPY']
        assert (end - start).days == 100

    def test_persists_bars_to_db(self, builder):
        builder.alpaca.get_daily_bars_range.return_value = {
            'SPY': [
                {'date': date(2026, 4, 28), 'open': 712.0, 'high': 712.88, 'low': 709.25, 'close': 711.0, 'volume': 1},
                {'date': date(2026, 4, 29), 'open': 711.0, 'high': 712.20, 'low': 708.37, 'close': 711.5, 'volume': 1},
                {'date': date(2026, 4, 30), 'open': 714.6, 'high': 719.79, 'low': 710.445, 'close': 718.66, 'volume': 1},
            ],
        }
        rows_written = builder._refresh_index_symbols(['SPY'], lookback_days=100)

        assert rows_written == 3
        builder.db.save_daily_bars.assert_called_once()
        rows = builder.db.save_daily_bars.call_args[0][0]
        assert len(rows) == 3
        for r in rows:
            assert r['symbol'] == 'SPY'
            assert isinstance(r['date'], str)  # ISO format expected by save_daily_bars
            for k in ('open', 'high', 'low', 'close', 'volume'):
                assert k in r

    def test_handles_multiple_index_symbols(self, builder):
        builder.alpaca.get_daily_bars_range.return_value = {
            'SPY': [{'date': date(2026, 5, 1), 'open': 1, 'high': 2, 'low': 0.5, 'close': 1.5, 'volume': 1}],
            'QQQ': [{'date': date(2026, 5, 1), 'open': 1, 'high': 2, 'low': 0.5, 'close': 1.5, 'volume': 1}],
        }
        rows_written = builder._refresh_index_symbols(['SPY', 'QQQ'], lookback_days=100)

        assert rows_written == 2
        rows = builder.db.save_daily_bars.call_args[0][0]
        assert {r['symbol'] for r in rows} == {'SPY', 'QQQ'}


# ---------------------------------------------------------------------------
# Failure modes — must NOT abort universe build, MUST log loudly
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_empty_symbols_list_skips_silently(self, builder):
        rows = builder._refresh_index_symbols([], lookback_days=100)
        assert rows == 0
        builder.alpaca.get_daily_bars_range.assert_not_called()
        builder.db.save_daily_bars.assert_not_called()

    def test_api_exception_logs_error_returns_zero(self, builder, caplog):
        import logging
        builder.alpaca.get_daily_bars_range.side_effect = RuntimeError("alpaca down")
        with caplog.at_level(logging.ERROR):
            rows = builder._refresh_index_symbols(['SPY'], lookback_days=100)
        assert rows == 0
        builder.db.save_daily_bars.assert_not_called()
        assert any("Step 8" in r.message and "FAILED" in r.message for r in caplog.records)

    def test_empty_response_logs_error(self, builder, caplog):
        import logging
        builder.alpaca.get_daily_bars_range.return_value = {'SPY': []}
        with caplog.at_level(logging.ERROR):
            rows = builder._refresh_index_symbols(['SPY'], lookback_days=100)
        assert rows == 0
        builder.db.save_daily_bars.assert_not_called()
        assert any("NO bars returned" in r.message for r in caplog.records)

    def test_missing_symbol_in_response_logs_per_missing(self, builder, caplog):
        import logging
        # API returned only SPY; QQQ is missing — should log ERROR for QQQ
        # and still persist SPY rows.
        builder.alpaca.get_daily_bars_range.return_value = {
            'SPY': [{'date': date(2026, 5, 1), 'open': 1, 'high': 2, 'low': 0.5, 'close': 1.5, 'volume': 1}],
        }
        with caplog.at_level(logging.ERROR):
            rows = builder._refresh_index_symbols(['SPY', 'QQQ'], lookback_days=100)
        assert rows == 1  # only SPY persisted
        assert any("QQQ" in r.message and "NO bars" in r.message for r in caplog.records)

    def test_isodate_strings_passthrough(self, builder):
        # Bar with date as ISO string instead of date object — must still serialize
        builder.alpaca.get_daily_bars_range.return_value = {
            'SPY': [{'date': '2026-05-01', 'open': 1, 'high': 2, 'low': 0.5, 'close': 1.5, 'volume': 1}],
        }
        rows = builder._refresh_index_symbols(['SPY'], lookback_days=100)
        assert rows == 1
        persisted = builder.db.save_daily_bars.call_args[0][0]
        assert persisted[0]['date'] == '2026-05-01'
