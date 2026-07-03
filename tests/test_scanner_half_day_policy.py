"""Pin the half-day skip policy (scanner._is_trading_day).

POLICY (BT-validated 2026-07-03 money-machine audit): the scanner skips
short trading days (close < 16:00 ET) entirely. Do NOT "optimize" this
away — the 18-month defended BT shows half-days are NEGATIVE edge:

    ORB on 2025-07-03 / 2025-11-28 / 2025-12-24: 10 trades, net -$2,656
    ($100K model). BF Stage-2: ZERO trades on those days.

ORB's entire edge is single-name EOD runners (top-5 trades = 101% of 18mo
P&L); a 3.5-hour session truncates the runway, supply is thin, and any
position held past an early close rides a weekend unprotected (the
force-close is 15:45 ET). Skipping is both the safe AND the profitable
policy.
"""
from unittest.mock import MagicMock

import pytest

from scanner.realtime_scanner import RealtimeScanner


def _scanner_with(alpaca):
    s = RealtimeScanner.__new__(RealtimeScanner)
    s.alpaca = alpaca
    return s


class TestHalfDayPolicy:
    def test_full_day_trades(self):
        a = MagicMock()
        a.is_trading_day.return_value = True
        a.is_short_trading_day.return_value = False
        assert _scanner_with(a)._is_trading_day() is True

    def test_holiday_skips(self):
        a = MagicMock()
        a.is_trading_day.return_value = False
        assert _scanner_with(a)._is_trading_day() is False

    def test_half_day_skips(self):
        """The policy under protection: early close -> no trading at all."""
        a = MagicMock()
        a.is_trading_day.return_value = True
        a.is_short_trading_day.return_value = True
        assert _scanner_with(a)._is_trading_day() is False

    def test_calendar_error_fails_open(self):
        """Calendar API failure lets the scanner run (it just finds no data
        on non-trading days) — pinned so a refactor doesn't silently flip
        this to fail-closed and skip real trading days on API blips."""
        a = MagicMock()
        a.is_trading_day.side_effect = Exception("calendar API down")
        assert _scanner_with(a)._is_trading_day() is True
