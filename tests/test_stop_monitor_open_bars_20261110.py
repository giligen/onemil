"""StopMonitor keeps the session-open bars of EVERY streamed symbol and seeds new windows with them.

docs/orb_ws_range_path_fix_spec_20261110.md: ORB subscribes (per symbol) only after the gap gate admits a
candidate, i.e. after the 09:30 ET bar; the window must still start at 09:30. EST in these tests: 09:30 ET == 14:30Z.
"""
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from data_sources.alpaca_client import AlpacaClient
from trading.stop_monitor import StopMonitor

SYM = 'OPNB'


def _bar(minute, hour=14, day=9, symbol=SYM, month=11):
    """A closed 1-min bar as the Alpaca stream delivers it (tz-aware UTC)."""
    return SimpleNamespace(symbol=symbol, timestamp=datetime(2026, month, day, hour, minute, tzinfo=timezone.utc),
                           open=10.0, high=10.2, low=9.9, close=10.1, volume=1000)


def _push(sm, *bars):
    for b in bars:
        asyncio.run(sm._on_bar(b))


@pytest.fixture
def sm():
    return StopMonitor(api_key='k', api_secret='s', alpaca_client=MagicMock(spec=AlpacaClient))


def _window_minutes(sm, symbol=SYM):
    return [(b['timestamp'].hour, b['timestamp'].minute) for b in sm._bar_windows[symbol]]


def test_bulk_symbol_window_is_seeded_with_open_bars_then_live_appends(sm):
    sm.subscribe_bars_many([SYM])
    _push(sm, *[_bar(m) for m in range(30, 37)])            # 14:30Z..14:36Z, bulk: no window
    assert SYM not in sm._bar_windows
    sm.subscribe_bars(SYM)                                    # ORB admits it after the open bars
    assert _window_minutes(sm) == [(14, m) for m in range(30, 37)]
    _push(sm, _bar(37))                                       # live bars append after the seed
    assert _window_minutes(sm)[-2:] == [(14, 36), (14, 37)]
    assert len(sm._bar_windows[SYM]) == 8


def test_open_bars_capped_at_eight(sm):
    sm.subscribe_bars_many([SYM])
    _push(sm, *[_bar(m) for m in range(30, 50)])
    assert len(sm._open_bars[SYM]) == StopMonitor.OPEN_BARS_KEEP == 8
    sm.subscribe_bars(SYM)
    assert _window_minutes(sm) == [(14, m) for m in range(30, 38)]


def test_premarket_bars_ignored_and_first_kept_bar_is_0930(sm):
    sm.subscribe_bars_many([SYM])
    _push(sm, _bar(0, hour=13), _bar(29), _bar(30), _bar(31))   # 08:00 ET, 09:29 ET are pre-market
    assert [b['timestamp'].minute for b in sm._open_bars[SYM]] == [30, 31]
    sm.subscribe_bars(SYM)
    assert _window_minutes(sm) == [(14, 30), (14, 31)]


def test_window_already_holding_a_timestamp_is_not_duplicated(sm):
    sm.subscribe_bars(SYM)                                    # windowed from the start (pre-open admission)
    _push(sm, _bar(30), _bar(31))
    sm.subscribe_bars(SYM)                                    # a second subscribe re-seeds: no duplicates
    assert _window_minutes(sm) == [(14, 30), (14, 31)]
    assert len(sm._open_bars[SYM]) == 2


def test_new_session_date_clears_previous_open_bars(sm, caplog):
    sm.subscribe_bars_many([SYM])
    _push(sm, _bar(30), _bar(31))
    assert sm._open_bars_date == '2026-11-09'
    with caplog.at_level('INFO'):
        _push(sm, _bar(30, day=10), _bar(31, day=10))
    assert sm._open_bars_date == '2026-11-10'
    assert [b['timestamp'].day for b in sm._open_bars[SYM]] == [10, 10]
    assert sum('new session 2026-11-10' in r.message for r in caplog.records) == 1


def test_stale_older_session_bar_does_not_reset(sm):
    sm.subscribe_bars_many([SYM])
    _push(sm, _bar(30, day=10))
    _push(sm, _bar(30, day=9))                                # late backfill of the previous session
    assert sm._open_bars_date == '2026-11-10'
    assert [b['timestamp'].day for b in sm._open_bars[SYM]] == [10]


def test_edt_open_is_1330z(sm):
    """DST: 09:30 ET is 13:30Z in summer - the ET conversion, not a fixed UTC hour, decides."""
    sm.subscribe_bars_many([SYM])
    _push(sm, _bar(29, hour=13, month=7), _bar(30, hour=13, month=7), _bar(30, hour=14, month=7))
    assert [(b['timestamp'].hour, b['timestamp'].minute) for b in sm._open_bars[SYM]] == [(13, 30), (14, 30)]
