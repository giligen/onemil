"""Tests for the central trading-hours helpers in trading/trading_hours.py.

Single source of truth for "is the regular session closed?" — both the
save-side guard in persistence/database.py and the read-side guard in
get_daily_bars_cached consult this, so their decisions can't drift.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from trading.trading_hours import (
    ET,
    REGULAR_SESSION_CLOSE_MINUTES,
    today_et,
    is_regular_session_closed,
)


# Fixture timestamps — Friday 4/24 = trading day, Saturday 4/25 = weekend
FRIDAY_PRE_MARKET = datetime(2026, 4, 24, 8, 0, tzinfo=ET)
FRIDAY_MID_DAY = datetime(2026, 4, 24, 12, 30, tzinfo=ET)
FRIDAY_JUST_BEFORE_THRESHOLD = datetime(2026, 4, 24, 16, 14, 59, tzinfo=ET)
FRIDAY_AT_THRESHOLD = datetime(2026, 4, 24, 16, 15, tzinfo=ET)
FRIDAY_POST_CLOSE = datetime(2026, 4, 24, 17, 30, tzinfo=ET)
SATURDAY_MID_DAY = datetime(2026, 4, 25, 12, 0, tzinfo=ET)
SUNDAY_EVENING = datetime(2026, 4, 26, 20, 0, tzinfo=ET)


class TestIsRegularSessionClosed:

    def test_friday_pre_market_open(self):
        assert is_regular_session_closed(FRIDAY_PRE_MARKET) is False

    def test_friday_mid_day_open(self):
        assert is_regular_session_closed(FRIDAY_MID_DAY) is False

    def test_friday_just_before_1615_still_open(self):
        assert is_regular_session_closed(FRIDAY_JUST_BEFORE_THRESHOLD) is False

    def test_friday_at_1615_is_closed(self):
        """Boundary: the threshold itself counts as closed."""
        assert is_regular_session_closed(FRIDAY_AT_THRESHOLD) is True

    def test_friday_post_close(self):
        assert is_regular_session_closed(FRIDAY_POST_CLOSE) is True

    def test_weekend_is_closed(self):
        assert is_regular_session_closed(SATURDAY_MID_DAY) is True
        assert is_regular_session_closed(SUNDAY_EVENING) is True


class TestTodayEt:

    def test_today_matches_injected_date(self):
        assert today_et(FRIDAY_MID_DAY).isoformat() == '2026-04-24'
        assert today_et(SATURDAY_MID_DAY).isoformat() == '2026-04-25'


class TestConstants:

    def test_close_threshold_is_1615(self):
        """The magic number pins the settlement-grace window. If this
        changes, the save-side + read-side guards both move together."""
        assert REGULAR_SESSION_CLOSE_MINUTES == 16 * 60 + 15
