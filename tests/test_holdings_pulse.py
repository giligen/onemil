"""Holdings pulse — hourly unrealized-P&L Telegram while holding (2026-07-05).

Pins: market-window gate, message format (R-multiple from DB stop),
flat-silence, and error surfacing on a failed account query.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
import holdings_pulse as hp


class TestMarketWindow:
    def test_midday_weekday_open(self):
        # 2026-07-06 is a Monday; 15:00 UTC = 11:00 ET (EDT)
        assert hp.in_market_window(
            datetime(2026, 7, 6, 15, 0, tzinfo=timezone.utc)) is True

    def test_premarket_closed(self):
        # 13:00 UTC = 9:00 ET — before 9:35
        assert hp.in_market_window(
            datetime(2026, 7, 6, 13, 0, tzinfo=timezone.utc)) is False

    def test_after_close_closed(self):
        # 20:30 UTC = 16:30 ET
        assert hp.in_market_window(
            datetime(2026, 7, 6, 20, 30, tzinfo=timezone.utc)) is False

    def test_weekend_closed(self):
        assert hp.in_market_window(
            datetime(2026, 7, 5, 15, 0, tzinfo=timezone.utc)) is False


class TestMessage:
    NOW = datetime(2026, 7, 6, 15, 0, tzinfo=timezone.utc)

    def test_r_multiple_and_totals(self):
        positions = [dict(account='ORB', symbol='TENX', qty=1000,
                          avg_entry=5.00, current=5.20, upl=200.0)]
        meta = {'TENX': {'strategy': 'orb', 'entry_price': 5.00,
                         'stop_price': 4.80}}
        msg = hp.build_message(positions, realized=-150.0, meta=meta,
                               now_utc=self.NOW)
        assert 'TENX' in msg and '+1.0R' in msg          # 200 upl / 200 risk
        assert 'unrealized $+200' in msg
        assert 'realized today $-150' in msg
        assert 'day $+50' in msg

    def test_missing_stop_omits_r(self):
        positions = [dict(account='ORB', symbol='XYZ', qty=100,
                          avg_entry=10.0, current=10.5, upl=50.0)]
        msg = hp.build_message(positions, realized=0.0, meta={},
                               now_utc=self.NOW)
        assert 'R' not in msg.split('$+50</b>')[1].split('\n')[0]

    def test_query_error_surfaced(self):
        positions = [dict(account='ORB', symbol='(QUERY FAILED)', qty=0,
                          avg_entry=0, current=0, upl=0, error='timeout')]
        msg = hp.build_message(positions, realized=0.0, meta={},
                               now_utc=self.NOW)
        assert 'query failed' in msg and 'timeout' in msg
