"""Gap-input parity (2026-10-02): the live gap gate uses today's official open vs the prior close.

ASTX/AEHG 9/30: the snapshot's daily bar was the previous session's, so live gated on a prior day's open.
"""
import logging
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import yaml
from pathlib import Path

from data_sources.alpaca_client import AlpacaAPIError, AlpacaClient
from persistence.database import Database
from trading import orb_engine as oe
from trading.orb_engine import ORBEngine
from trading.orb_gap_gate import gap_input_needs_today_open
from trading.stop_monitor import StopMonitor

ET = ZoneInfo('America/New_York')
TODAY = datetime.now(timezone.utc).astimezone(ET).date().isoformat()
YDAY = (datetime.fromisoformat(TODAY) - timedelta(days=1)).date().isoformat()


def _engine(alpaca):
    cfg = yaml.safe_load((Path(__file__).parent.parent / 'orb.yaml').read_text())
    cfg['strategy']['enabled'] = True
    db = MagicMock(spec=Database)
    db.get_intraday_bars_for_date.return_value = {}
    return ORBEngine(alpaca_client=alpaca, db=db, stop_monitor=MagicMock(spec=StopMonitor), config=cfg)


def _fresh(o, pc):
    return {'open': o, 'prev_close': pc, 'prev_volume': 2_000_000, 'volume': 500_000,
            'close': o, 'latest_price': o, 'daily_bar_date': TODAY}


def _stale(prior_open, prior_close, snap_prev_close):
    """Snapshot whose daily bar is the PRIOR session (ASTX 9/30: opened 11.65, closed 9.76)."""
    return {'open': prior_open, 'close': prior_close, 'volume': 2_000_000, 'prev_close': snap_prev_close,
            'prev_volume': 1_000_000, 'latest_price': prior_close, 'daily_bar_date': YDAY}


def _bars(o):
    ts = datetime.fromisoformat(TODAY + 'T09:30:00').replace(tzinfo=ET).astimezone(timezone.utc)
    return pd.DataFrame([{'timestamp': ts, 'open': o, 'high': o, 'low': o,
                          'close': o, 'volume': 1}])


def _at_0935():
    """Fixed 09:35:30 ET today (the tests must not depend on the wall clock)."""
    return datetime.fromisoformat(TODAY + 'T09:35:30').replace(tzinfo=ET)


def _clock(monkeypatch, hh, mm):
    now = datetime.fromisoformat(TODAY + f'T{hh:02d}:{mm:02d}:30').replace(tzinfo=ET)

    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)
    monkeypatch.setattr(oe, 'datetime', _DT)


def _alpaca(snaps, bars=None, exc=None):
    a = MagicMock(spec=AlpacaClient)
    a.get_snapshots.return_value = snaps
    if exc:
        a.get_1min_bars_range_multi.side_effect = exc
    else:
        a.get_1min_bars_range_multi.side_effect = lambda syms, s0, s1, **kw: {
            s: (_bars(bars[s]) if s in (bars or {}) else pd.DataFrame()) for s in syms}
    return a


class TestNeedsTodayOpen:
    def test_fresh_needs_nothing_and_stale_liquid_needs_fetch(self):
        assert not gap_input_needs_today_open(_fresh(10, 9), TODAY, 5e5, 3, 30)
        assert gap_input_needs_today_open(_stale(11.65, 9.76, 10.29), TODAY, 5e5, 3, 30)
        assert not gap_input_needs_today_open(_stale(1, 1.0, 1), TODAY, 5e5, 3, 30)      # price prefilter


class TestAstxAehgCase:
    def test_stale_snapshot_with_today_open_below_floor_is_not_admitted(self, monkeypatch):
        """ASTX: stale snapshot would read +13% (11.65 vs 10.29); today's 09:30 open 9.95 vs 9.76 = +1.95%."""
        _clock(monkeypatch, 9, 35)
        a = _alpaca({'ASTX': _stale(11.65, 9.76, 10.29), 'AEHG': _stale(9.16, 9.10, 8.9)},
                    bars={'ASTX': 9.95, 'AEHG': 9.27})
        assert _engine(a).build_orb_universe_from_snapshots(['ASTX', 'AEHG']) == []

    def test_stale_snapshot_with_real_gap_open_is_admitted_on_bar_open(self, monkeypatch):
        _clock(monkeypatch, 9, 35)
        a = _alpaca({'GAPR': _stale(8.0, 10.0, 9.0)}, bars={'GAPR': 10.8})   # +8% vs the stale bar's close
        eng = _engine(a)
        assert eng.build_orb_universe_from_snapshots(['GAPR']) == ['GAPR']
        assert eng._gap_gate_inputs['GAPR']['source'] == 'bar'
        assert eng._gap_gate_inputs['GAPR']['gap_input_prev_close'] == 10.0

    def test_fresh_snapshot_admitted_exactly_as_bt_with_no_rest_call(self, monkeypatch):
        _clock(monkeypatch, 9, 35)
        a = _alpaca({'UP': _fresh(10.6, 10.0), 'FLAT': _fresh(10.1, 10.0)})
        assert _engine(a).build_orb_universe_from_snapshots(['UP', 'FLAT']) == ['UP']
        a.get_1min_bars_range_multi.assert_not_called()


class TestFailureAndBudget:
    def test_batched_fetch_failure_warns_once_and_admits_nothing_stale(self, monkeypatch, caplog):
        _clock(monkeypatch, 9, 35)
        syms = [f"S{i:02d}" for i in range(15)]
        a = _alpaca({s: _stale(11.0, 10.0, 9.0) for s in syms}, exc=AlpacaAPIError("boom"))
        with caplog.at_level(logging.WARNING):
            keep = _engine(a).build_orb_universe_from_snapshots(syms)
        assert keep == []
        warns = [r.getMessage() for r in caplog.records if 'NOT admitted' in r.getMessage()]
        assert len(warns) == 1 and '15 of 15' in warns[0] and 'S09' in warns[0] and 'S10' not in warns[0]
        assert any('batched 09:30 bar fetch failed' in r.getMessage() for r in caplog.records)

    def test_before_0931_nothing_fetched_stale_not_admitted(self, monkeypatch):
        _clock(monkeypatch, 9, 30)
        a = _alpaca({'X': _stale(11.0, 10.0, 9.0)}, bars={'X': 11.0})
        assert _engine(a).build_orb_universe_from_snapshots(['X']) == []
        a.get_1min_bars_range_multi.assert_not_called()

    def test_budget_respected_and_chunks_of_200(self, monkeypatch, caplog):
        _clock(monkeypatch, 9, 35)
        syms = [f"T{i:03d}" for i in range(450)]
        a = _alpaca({s: _stale(11.0, 10.0, 9.0) for s in syms}, bars={})
        eng = _engine(a)
        res = eng._fetch_today_open_bars(syms, _at_0935(), time.time() + 60)
        assert res == {} and a.get_1min_bars_range_multi.call_count == 3
        a.get_1min_bars_range_multi.reset_mock()
        eng2 = _engine(a)
        with caplog.at_level(logging.WARNING):
            eng2._fetch_today_open_bars(syms, _at_0935(), time.time() - 1)
        a.get_1min_bars_range_multi.assert_not_called()
        assert any('budget exhausted' in r.getMessage() for r in caplog.records)

    def test_found_opens_are_cached_for_the_day(self, monkeypatch):
        _clock(monkeypatch, 9, 35)
        a = _alpaca({}, bars={'X': 11.0})
        eng = _engine(a)
        now = _at_0935()
        assert eng._fetch_today_open_bars(['X'], now, time.time() + 60) == {'X': 11.0}
        assert eng._fetch_today_open_bars(['X'], now, time.time() + 60) == {'X': 11.0}
        assert a.get_1min_bars_range_multi.call_count == 1
