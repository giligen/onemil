"""Reproduction: the websocket bar path can never complete an ORB opening range.

docs/orb_ws_range_path_diagnosis_spec_20261110.md / _RESULT.md.

Production sequence (journal 11/9 and 11/10, EST so 09:30 ET == 14:30Z):
  * 12:30Z  HOD bulk-subscribes the seed universe (`StopMonitor.subscribe_bars_many`) —
            bulk symbols get NO rolling window (stop_monitor.py:1017, 1142-1143).
  * ~14:31:00Z the 09:30 bar (ts 14:30Z) is emitted -> HOD's light handler only;
            ORB's universe is still empty (gap gate needs today's open), so it is dropped.
  * 14:31:54Z (11/9) / 14:33:40Z (11/10) ORB admits candidates -> `_subscribe_bars` ->
            `StopMonitor.subscribe_bars` converts the symbol to windowed with an EMPTY window
            (stop_monitor.py:999-1002).
  * Every later bar hands `_on_bar_close` a window starting at 14:31Z, so
            `_first_session_open_ts_utc` returns None forever and no range ever completes.

A REAL StopMonitor (not started: no websocket) is used so the window semantics are
production's own code; bars are pushed through its `_on_bar` coroutine exactly as the
Alpaca stream does. The bar objects stand in for the Alpaca SDK `Bar`.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine
from trading.stop_monitor import StopMonitor

SYM = 'GAPR'
ADMIT_TS = datetime(2026, 11, 9, 14, 31, 54, tzinfo=timezone.utc)   # 11/9 build_universe stamp


class Clock:
    """Frozen clock behind `trading.orb_engine.datetime.now` (keeps the late-add REST
    backfill inert: it only runs at/after 09:35 ET)."""

    def __init__(self, now):
        self.now = now

    def __enter__(self):
        self._p = patch('trading.orb_engine.datetime')
        mock_dt = self._p.start()
        mock_dt.now = lambda tz=None: self.now
        mock_dt.combine = datetime.combine
        mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
        return self

    def __exit__(self, *exc):
        self._p.stop()


def ws_bar(minute: int) -> SimpleNamespace:
    """One closed 1-min bar as the Alpaca stream delivers it (tz-aware UTC timestamp)."""
    base = 10.0 + 0.01 * minute
    return SimpleNamespace(symbol=SYM, timestamp=datetime(2026, 11, 9, 14, minute, tzinfo=timezone.utc),
                           open=base, high=base + 0.20, low=base - 0.10, close=base + 0.05, volume=20000)


@pytest.fixture
def wired():
    """Real StopMonitor + ORBEngine wired the way main.py does (register_on_stop_monitor)."""
    cfg = yaml.safe_load(open(Path(__file__).parent.parent / 'orb.yaml'))
    cfg['strategy']['enabled'] = True
    alpaca = MagicMock(spec=AlpacaClient)
    alpaca.get_open_positions.return_value = []
    alpaca.get_account_info.return_value = {'buying_power': 100_000.0}
    db = MagicMock(spec=Database)
    db.get_open_trades.return_value = []
    db.get_trades_by_date.return_value = []
    sm = StopMonitor(api_key='k', api_secret='s', alpaca_client=MagicMock(spec=AlpacaClient))
    eng = ORBEngine(alpaca_client=alpaca, db=db, stop_monitor=sm, config=cfg)
    eng.register_on_stop_monitor()
    return sm, eng


def _push(sm: StopMonitor, minutes) -> None:
    """Deliver closed bars through StopMonitor's websocket handler."""
    for m in minutes:
        asyncio.run(sm._on_bar(ws_bar(m)))


def test_ws_range_completes_when_candidate_admitted_after_the_open_bar(wired):
    """FAILS today (the defect): HOD bulk-streamed the symbol from 12:30Z, ORB admits it at
    09:31:54 ET, then bars 09:31..09:35 ET stream in. The 09:30 bar WAS on the wire, but the
    window ORB receives starts at 14:31Z -> no session-open bar -> range never completes."""
    sm, eng = wired
    sm.subscribe_bars_many([SYM])                 # HOD's 12:30Z seed subscription (bulk, no window)
    _push(sm, [30])                               # 09:30 ET bar closes; ORB universe still empty
    with Clock(ADMIT_TS):
        eng.build_universe(source_loader=lambda: [SYM])   # gap gate admits at 09:31:54 ET
    _push(sm, [31, 32, 33, 34, 35])
    eng.drain_bar_events()
    window_ts = [b['timestamp'] for b in sm._bar_windows[SYM]]
    assert eng.candidates[SYM].range_data is not None, (
        f"WS path produced no range: the window handed to _on_bar_close starts at "
        f"{window_ts[0].isoformat()} (no 14:30Z session-open bar)")


def test_control_ws_range_completes_when_window_holds_the_open_bar(wired):
    """CONTROL (passes): same bars, but the symbol is admitted before 09:30 ET, so the window
    contains the 14:30Z bar -> range complete. Proves the frame shape / dtype path (H3) and the
    drain (H4) are fine; only the window's start (subscribe time) differs."""
    sm, eng = wired
    with Clock(datetime(2026, 11, 9, 14, 29, 0, tzinfo=timezone.utc)):
        eng.build_universe(source_loader=lambda: [SYM])
    _push(sm, [30, 31, 32, 33, 34])
    eng.drain_bar_events()
    rd = eng.candidates[SYM].range_data
    assert rd is not None
    assert rd.range_start_ts == datetime(2026, 11, 9, 14, 30, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fix (docs/orb_ws_range_path_fix_spec_20261110.md): integration, real StopMonitor + real ORBEngine.
# ---------------------------------------------------------------------------
import logging

import pandas as pd

SWEEP_TS = datetime(2026, 11, 9, 14, 35, 0, tzinfo=timezone.utc)   # 09:35:00 ET


def _rest_bars() -> pd.DataFrame:
    """REST bars with deliberately DIFFERENT numbers (high 99): proves the stream value is the one kept."""
    rows = [{'timestamp': datetime(2026, 11, 9, 14, 30 + i, tzinfo=timezone.utc), 'open': 10.0, 'high': 99.0,
             'low': 1.0, 'close': 10.0, 'volume': 10000} for i in range(5)]
    return pd.DataFrame(rows)


def test_seed_bulk_subscribed_preopen_ws_range_completes_before_any_rest_sweep(wired, caplog):
    """ORB bulk-subscribes the seed pre-open; the 09:30 bar is recorded; the gap gate admits at 09:31:54;
    the 09:34 bar closes -> 'range complete' from the WEBSOCKET path with exact H/L, no REST sweep involved."""
    sm, eng = wired
    assert eng.bulk_subscribe_seed([SYM, 'ZAZZT']) == 1                       # test ticker excluded
    assert sm._bulk_bar_symbols == {SYM}
    _push(sm, [30])                                                           # 09:30 bar: bulk, light handlers only
    with Clock(ADMIT_TS):
        eng.build_universe(source_loader=lambda: [SYM])                       # admitted after the open bar
    with caplog.at_level(logging.INFO):
        _push(sm, [31, 32, 33, 34])
        eng.drain_bar_events()
    rd = eng.candidates[SYM].range_data
    assert rd is not None
    assert rd.range_start_ts == datetime(2026, 11, 9, 14, 30, tzinfo=timezone.utc)
    assert rd.range_high == pytest.approx(10.0 + 0.01 * 34 + 0.20)            # max high of bars 30..34
    assert rd.range_low == pytest.approx(10.0 + 0.01 * 30 - 0.10)             # min low of bars 30..34
    assert any(f'{SYM} range complete' in r.message for r in caplog.records)
    assert eng._sweep_pass_count_today == 0                                   # no REST pass happened


def test_sweep_counts_symbol_as_superseded_when_ws_finishes_the_range_during_the_fetch(wired, caplog):
    """At 09:35:00 the sweep starts with GAPR rangeless (09:34 bar still in flight); the stream delivers it while the
    REST fetch is out -> the stream value wins and the sweep logs it as superseded (first non-zero in production
    history: before the fix the WS path could never complete a range)."""
    sm, eng = wired
    eng.fast_submit_enabled = False
    eng.sweep_retry_delay_s = 0.0
    eng.bulk_subscribe_seed([SYM])
    _push(sm, [30])
    with Clock(ADMIT_TS):
        eng.build_universe(source_loader=lambda: [SYM])
    _push(sm, [31, 32, 33])
    assert eng.candidates[SYM].range_data is None                             # 09:34 bar not yet closed

    def _fetch(symbols, minutes):
        _push(sm, [34])                                                       # stream completes the range mid-fetch
        eng.drain_bar_events()
        return {SYM: _rest_bars()}

    eng._fetch_sweep_bars = _fetch
    with Clock(SWEEP_TS), caplog.at_level(logging.INFO):
        filled = eng._post_open_sweep(release_lock=True)
    assert SYM in filled
    assert eng.candidates[SYM].range_data.range_high == pytest.approx(10.0 + 0.01 * 34 + 0.20)   # stream, not REST 99
    line = [r.message for r in caplog.records if 'post-open sweep fetched' in r.message]
    assert len(line) == 1 and '(0 applied, 1 superseded by the stream)' in line[0]
