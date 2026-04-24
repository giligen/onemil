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


# ---------------------------------------------------------------------------
# Sidecar table for mid-day BT runs
# ---------------------------------------------------------------------------


class TestProvisionalSideTable:
    """`daily_bars_provisional` is the opt-in escape hatch for BT flows that
    want to see today's still-open trades without poisoning the main
    `daily_bars` cache that live reads."""

    def test_save_provisional_persists(self, db):
        row = {'symbol': 'BMNZ', 'date': '2026-04-24',
               'open': 14.23, 'high': 14.64, 'low': 13.71,
               'close': 14.10, 'volume': 900_000}
        assert db.save_daily_bars_provisional([row]) == 1

    def test_provisional_hidden_by_default_from_main_reader(self, db):
        """Default reads (what live uses) must NOT see provisional rows."""
        db.save_daily_bars_provisional([
            {'symbol': 'X', 'date': '2026-04-24', 'open': 1, 'high': 1,
             'low': 1, 'close': 1, 'volume': 0}
        ])
        res = db.get_daily_bars_cached(['X'], '2026-04-24', '2026-04-24')
        assert res.get('X', []) == []

    def test_provisional_visible_with_include_flag(self, db):
        db.save_daily_bars_provisional([
            {'symbol': 'X', 'date': '2026-04-24', 'open': 14.23, 'high': 14.64,
             'low': 13.71, 'close': 14.10, 'volume': 900_000}
        ])
        res = db.get_daily_bars_cached(
            ['X'], '2026-04-24', '2026-04-24', include_provisional=True,
        )
        bars = res.get('X', [])
        assert len(bars) == 1
        assert bars[0]['close'] == pytest.approx(14.10)

    def test_final_wins_over_provisional_same_key(self, db):
        """If a FINAL row exists for (sym, date), the provisional row for
        the same key is SHADOWED when reading with include_provisional."""
        # FINAL written for yesterday post-close (different day, clean path)
        db.save_daily_bars(
            [{'symbol': 'BMNZ', 'date': '2026-04-22', 'open': 14.12, 'high': 14.31,
              'low': 13.24, 'close': 13.43, 'volume': 4_150_038}],
            now_et=POST_MARKET_1630,
        )
        # Provisional also written for the same date somehow (stale leftover)
        db.save_daily_bars_provisional(
            [{'symbol': 'BMNZ', 'date': '2026-04-22', 'open': 14.12, 'high': 14.20,
              'low': 13.30, 'close': 13.66, 'volume': 1_000_000}]  # mid-day snap
        )
        res = db.get_daily_bars_cached(
            ['BMNZ'], '2026-04-22', '2026-04-22', include_provisional=True,
        )
        bars = res.get('BMNZ', [])
        assert len(bars) == 1
        assert bars[0]['close'] == pytest.approx(13.43), (
            "FINAL close must win over the provisional row on the same date"
        )

    def test_clear_provisional_removes_all(self, db):
        db.save_daily_bars_provisional([
            {'symbol': 'X', 'date': '2026-04-24', 'open': 1, 'high': 1,
             'low': 1, 'close': 1, 'volume': 0},
            {'symbol': 'Y', 'date': '2026-04-24', 'open': 2, 'high': 2,
             'low': 2, 'close': 2, 'volume': 0},
        ])
        assert db.clear_provisional_daily_bars() == 2
        res = db.get_daily_bars_cached(
            ['X', 'Y'], '2026-04-24', '2026-04-24', include_provisional=True,
        )
        assert res == {}

    def test_empty_provisional_write_is_noop(self, db):
        assert db.save_daily_bars_provisional([]) == 0

    def test_provisional_and_final_for_different_dates_both_visible(self, db):
        """Common mid-day case: final for 4/22 (past), provisional for 4/24
        (today). Both visible under include_provisional=True."""
        db.save_daily_bars(
            [{'symbol': 'X', 'date': '2026-04-22', 'open': 5, 'high': 5,
              'low': 5, 'close': 5, 'volume': 100}],
            now_et=POST_MARKET_1630,
        )
        db.save_daily_bars_provisional([
            {'symbol': 'X', 'date': '2026-04-24', 'open': 6, 'high': 6,
             'low': 6, 'close': 6, 'volume': 200}
        ])
        res = db.get_daily_bars_cached(
            ['X'], '2026-04-20', '2026-04-24', include_provisional=True,
            now_et=POST_MARKET_1630,  # market closed — no read-side filter
        )
        bars = res.get('X', [])
        assert len(bars) == 2
        closes = {str(b['date']): b['close'] for b in bars}
        assert closes['2026-04-22'] == 5
        assert closes['2026-04-24'] == 6


# ---------------------------------------------------------------------------
# Read-side defensive guard (P2): legacy/bypassed polluted rows in
# daily_bars must not leak out during market hours.
# ---------------------------------------------------------------------------


class TestReadSideTodayGuard:
    """Direct INSERT of a polluted today-row (simulating legacy pollution
    or a caller that bypasses save_daily_bars) must be FILTERED by
    get_daily_bars_cached when the market is open."""

    def _force_insert_today(self, db, row):
        """Bypass save_daily_bars to simulate legacy polluted data."""
        from datetime import datetime as _dt
        from datetime import timezone as _tz
        db._cache_conn.execute(
            "INSERT OR REPLACE INTO daily_bars "
            "(symbol, bar_date, open, high, low, close, volume, fetched_at) "
            "VALUES (:symbol, :date, :open, :high, :low, :close, :volume, :fetched_at)",
            {**row, 'fetched_at': _dt.now(_tz.utc)},
        )
        db._cache_conn.commit()

    def test_polluted_today_row_hidden_during_market_hours(self, db):
        self._force_insert_today(db, {
            'symbol': 'X', 'date': '2026-04-24',
            'open': 14.0, 'high': 14.1, 'low': 13.9, 'close': 14.0,
            'volume': 100_000,
        })
        # Mid-day read — guard should hide the today-row
        res = db.get_daily_bars_cached(
            ['X'], '2026-04-24', '2026-04-24', now_et=MARKET_OPEN_10AM,
        )
        assert res.get('X', []) == []

    def test_polluted_today_row_visible_post_close(self, db):
        """After 16:15 ET, guard stops filtering; today-row (now final)
        is returned normally."""
        self._force_insert_today(db, {
            'symbol': 'X', 'date': '2026-04-24',
            'open': 14.0, 'high': 14.1, 'low': 13.9, 'close': 14.05,
            'volume': 100_000,
        })
        res = db.get_daily_bars_cached(
            ['X'], '2026-04-24', '2026-04-24', now_et=POST_MARKET_1630,
        )
        bars = res.get('X', [])
        assert len(bars) == 1
        assert bars[0]['close'] == pytest.approx(14.05)

    def test_polluted_today_row_falls_back_to_provisional_when_included(self, db):
        """THE KEY POLLUTION-RECOVERY CASE: if daily_bars has a legacy
        polluted row AND daily_bars_provisional has a fresh correct row,
        the read-side guard drops the polluted one so the provisional
        becomes visible under include_provisional=True. Without the
        read-side guard, the polluted FINAL row would shadow the clean
        PROVISIONAL row (regression of the original bug)."""
        self._force_insert_today(db, {
            'symbol': 'BMNZ', 'date': '2026-04-24',
            'open': 14.23, 'high': 14.30, 'low': 14.10, 'close': 14.10,
            'volume': 500_000,  # polluted mid-day snapshot
        })
        db.save_daily_bars_provisional([{
            'symbol': 'BMNZ', 'date': '2026-04-24',
            'open': 14.23, 'high': 14.64, 'low': 13.71, 'close': 14.55,
            'volume': 800_000,  # fresh provisional fetch
        }])
        res = db.get_daily_bars_cached(
            ['BMNZ'], '2026-04-24', '2026-04-24',
            include_provisional=True, now_et=MARKET_OPEN_10AM,
        )
        bars = res.get('BMNZ', [])
        assert len(bars) == 1
        assert bars[0]['close'] == pytest.approx(14.55), (
            "Fresh provisional row must be returned, not the legacy "
            "polluted row shadowed by the read-side guard."
        )

    def test_past_rows_always_visible(self, db):
        """Guard is scoped to today — any prior date passes through
        regardless of market state."""
        db.save_daily_bars(
            [{'symbol': 'X', 'date': '2026-04-22', 'open': 5, 'high': 5,
              'low': 5, 'close': 5, 'volume': 100}],
            now_et=POST_MARKET_1630,
        )
        res = db.get_daily_bars_cached(
            ['X'], '2026-04-22', '2026-04-22', now_et=MARKET_OPEN_10AM,
        )
        assert len(res.get('X', [])) == 1
