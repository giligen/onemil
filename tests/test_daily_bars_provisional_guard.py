"""Regression for 2026-04-22 → 2026-04-23 cache-pollution bug.

Bug: `orb_backtest.py` ran at 12:03 PM ET on 4/22, called
`fill_daily_bars_for_dates`, which fetched Alpaca daily bars for a
window that included 4/22 (today). Alpaca returned a provisional bar
whose `close` was the last trade price at 12:03 PM ET (~$13.66 for
BMNZ), NOT the end-of-session close ($13.43). The bar was written to
cache. On 4/23 at 9:35 ET, live read that polluted prev_close and
computed gap_pct=4.17% instead of BT's 5.96% — assigning BMNZ to Q4
instead of Q5 and sizing it differently.

Fix: `save_daily_bars` silently drops rows for today when the regular
session hasn't ended yet (< 16:15 ET). Both `orb_backtest.py` and
`batch_backtest.py` benefit without code changes — every daily-bars
writer is funneled through the same DB API.
"""
from __future__ import annotations

from datetime import datetime, date, timezone
from unittest.mock import MagicMock, patch

import pytest

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo('America/New_York')
except Exception:  # pragma: no cover
    ET = None

from persistence.database import (
    Database,
    _drop_today_provisional_bars,
)


@pytest.fixture
def db(tmp_path):
    return Database(db_path=str(tmp_path / 'test_cache.db'))


# Representative timestamps (Friday market day, weekend, after-hours).
MARKET_OPEN_10AM = datetime(2026, 4, 24, 10, 0, tzinfo=ET)
MARKET_OPEN_1203 = datetime(2026, 4, 22, 12, 3, tzinfo=ET)  # the BMNZ-bug time
MARKET_CLOSE_1615 = datetime(2026, 4, 24, 16, 15, tzinfo=ET)
POST_MARKET_1630 = datetime(2026, 4, 24, 16, 30, tzinfo=ET)
PRE_MARKET_8AM = datetime(2026, 4, 24, 8, 0, tzinfo=ET)
SATURDAY = datetime(2026, 4, 25, 12, 0, tzinfo=ET)


class TestDropProvisionalHelper:

    def test_no_bars_returns_empty(self):
        kept, dropped = _drop_today_provisional_bars([], now_et=MARKET_OPEN_10AM)
        assert kept == [] and dropped == 0

    def test_empty_weekend_passthrough(self):
        # Even a "today" row on a weekend passes — no risk of provisional bar.
        bars = [{'symbol': 'X', 'date': '2026-04-25', 'open': 1, 'high': 1,
                 'low': 1, 'close': 1, 'volume': 0}]
        kept, dropped = _drop_today_provisional_bars(bars, now_et=SATURDAY)
        assert len(kept) == 1 and dropped == 0

    def test_market_open_today_row_dropped(self):
        today_row = {'symbol': 'BMNZ', 'date': '2026-04-22',
                     'open': 14.12, 'high': 14.31, 'low': 13.24,
                     'close': 13.66, 'volume': 3_000_000}  # provisional
        prior = {'symbol': 'BMNZ', 'date': '2026-04-21',
                 'open': 14.77, 'high': 15.78, 'low': 14.33,
                 'close': 15.78, 'volume': 1_274_836}
        kept, dropped = _drop_today_provisional_bars(
            [today_row, prior], now_et=MARKET_OPEN_1203,
        )
        assert dropped == 1
        assert kept == [prior]

    def test_post_close_today_row_persists(self):
        today_row = {'symbol': 'BMNZ', 'date': '2026-04-24',
                     'open': 14.23, 'high': 14.64, 'low': 13.71,
                     'close': 14.10, 'volume': 905_603}
        kept, dropped = _drop_today_provisional_bars(
            [today_row], now_et=POST_MARKET_1630,
        )
        assert dropped == 0
        assert kept == [today_row]

    def test_exactly_1615_is_post_close_threshold(self):
        today_row = {'symbol': 'X', 'date': '2026-04-24',
                     'open': 1, 'high': 1, 'low': 1, 'close': 1, 'volume': 1}
        kept, dropped = _drop_today_provisional_bars(
            [today_row], now_et=MARKET_CLOSE_1615,
        )
        assert dropped == 0 and kept == [today_row]

    def test_pre_market_today_row_dropped(self):
        """Pre-9:30 ET: daily bar for today can't be valid (session hasn't
        opened). Drop to be safe — we never want a placeholder."""
        today_row = {'symbol': 'X', 'date': '2026-04-24',
                     'open': 1, 'high': 1, 'low': 1, 'close': 1, 'volume': 1}
        kept, dropped = _drop_today_provisional_bars(
            [today_row], now_et=PRE_MARKET_8AM,
        )
        assert dropped == 1 and kept == []

    def test_historical_rows_always_pass(self):
        """Past-dated rows are never dropped regardless of wall clock."""
        rows = [
            {'symbol': 'X', 'date': '2026-04-10', 'open': 1, 'high': 1,
             'low': 1, 'close': 1, 'volume': 0},
            {'symbol': 'X', 'date': '2026-04-21', 'open': 1, 'high': 1,
             'low': 1, 'close': 1, 'volume': 0},
        ]
        kept, dropped = _drop_today_provisional_bars(rows, now_et=MARKET_OPEN_10AM)
        assert dropped == 0 and len(kept) == 2

    def test_accepts_datetime_date_for_date_field(self):
        today_row = {'symbol': 'X', 'date': date(2026, 4, 22),
                     'open': 1, 'high': 1, 'low': 1, 'close': 1, 'volume': 0}
        kept, dropped = _drop_today_provisional_bars(
            [today_row], now_et=MARKET_OPEN_1203,
        )
        assert dropped == 1 and kept == []

    def test_accepts_datetime_for_date_field(self):
        today_row = {'symbol': 'X',
                     'date': datetime(2026, 4, 22, 20, 0, tzinfo=timezone.utc),
                     'open': 1, 'high': 1, 'low': 1, 'close': 1, 'volume': 0}
        kept, dropped = _drop_today_provisional_bars(
            [today_row], now_et=MARKET_OPEN_1203,
        )
        assert dropped == 1 and kept == []


class TestSaveDailyBarsIntegration:
    """End-to-end through Database.save_daily_bars — the actual choke
    point that both orb_backtest and batch_backtest funnel through."""

    def test_market_open_writes_history_but_skips_today(self, db):
        rows = [
            {'symbol': 'BMNZ', 'date': '2026-04-21', 'open': 14.77, 'high': 15.78,
             'low': 14.33, 'close': 15.78, 'volume': 1_274_836},
            {'symbol': 'BMNZ', 'date': '2026-04-22', 'open': 14.12, 'high': 14.31,
             'low': 13.24, 'close': 13.66, 'volume': 3_000_000},  # PROVISIONAL
        ]
        n_saved = db.save_daily_bars(rows, now_et=MARKET_OPEN_1203)
        assert n_saved == 1  # only the prior row

        # Verify cache contents — get_daily_bars_cached returns `date` as
        # datetime.date (or str, depending on adapter); normalize to ISO.
        bulk = db.get_daily_bars_cached(['BMNZ'], '2026-04-18', '2026-04-22')
        dates = {str(b['date']) for b in bulk.get('BMNZ', [])}
        assert '2026-04-21' in dates
        assert '2026-04-22' not in dates, (
            "Provisional 4/22 row must not reach the cache during market hours"
        )

    def test_post_close_writes_today(self, db):
        rows = [
            {'symbol': 'BMNZ', 'date': '2026-04-24', 'open': 14.23, 'high': 14.64,
             'low': 13.71, 'close': 14.10, 'volume': 905_603},
        ]
        n_saved = db.save_daily_bars(rows, now_et=POST_MARKET_1630)
        assert n_saved == 1

    def test_provisional_write_cant_overwrite_prior_final(self, db):
        """Guard prevents a post-market FINAL row from being replaced by a
        subsequent intra-day PROVISIONAL write on the following day.
        (INSERT OR REPLACE would otherwise clobber good data.)"""
        # Yesterday: FINAL close written post-close.
        final_row = {'symbol': 'BMNZ', 'date': '2026-04-22', 'open': 14.12,
                     'high': 14.31, 'low': 13.24, 'close': 13.43,
                     'volume': 4_150_038}
        post_close_yesterday = datetime(2026, 4, 22, 17, 0, tzinfo=ET)
        db.save_daily_bars([final_row], now_et=post_close_yesterday)

        # Today mid-day: fetch returns a provisional "today" row AND the
        # prior-day row. Guard must drop ONLY today; the prior row is
        # past-dated now and passes through (re-written with same values).
        now_midday_today = datetime(2026, 4, 23, 12, 3, tzinfo=ET)
        prov_today = {'symbol': 'BMNZ', 'date': '2026-04-23', 'open': 14.0,
                      'high': 14.05, 'low': 13.90, 'close': 13.95,  # provisional
                      'volume': 100_000}
        db.save_daily_bars([final_row, prov_today], now_et=now_midday_today)

        bulk = db.get_daily_bars_cached(['BMNZ'], '2026-04-20', '2026-04-23')
        bars = bulk.get('BMNZ', [])
        dates_to_close = {str(b['date']): b['close'] for b in bars}
        assert dates_to_close.get('2026-04-22') == pytest.approx(13.43), (
            "Prior-day FINAL close must survive (guard only drops today)"
        )
        assert '2026-04-23' not in dates_to_close

    def test_empty_bars_is_noop(self, db):
        assert db.save_daily_bars([], now_et=MARKET_OPEN_10AM) == 0
        assert db.save_daily_bars(None or [], now_et=POST_MARKET_1630) == 0

    def test_weekend_writes_today_row_without_filtering(self, db):
        """Weekend — no pollution risk, Alpaca won't return a weekend row
        in practice, but if a caller passes one it should persist."""
        sat_row = {'symbol': 'X', 'date': '2026-04-25', 'open': 1, 'high': 1,
                   'low': 1, 'close': 1, 'volume': 0}
        assert db.save_daily_bars([sat_row], now_et=SATURDAY) == 1
