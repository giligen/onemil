"""Open-tick stall fixes (docs/orb_open_tick_stall_20261002.md).

(a) a vendor-wide daily-bar lag that flips EVERY cached snapshot stale must
    not trigger a full re-fetch of the candidate set;
(b) the 09:30 minute-bar lookup is ONE batched query, equal in result to the
    per-symbol lookup it replaced;
(c) a hard deadline stops the scan and the snapshot re-fetch, logging WARNING.
"""
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine
from trading.stop_monitor import StopMonitor


def _engine(alpaca, db=None):
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    cfg.setdefault('execution', {})['prewarm_seed'] = True
    return ORBEngine(alpaca_client=alpaca, db=db or MagicMock(spec=Database),
                     stop_monitor=MagicMock(spec=StopMonitor), config=cfg)


def _today_et():
    from zoneinfo import ZoneInfo
    return datetime.now(timezone.utc).astimezone(ZoneInfo('America/New_York')).date().isoformat()


def _snap(o, pc, d):
    return {'open': o, 'prev_close': pc, 'prev_volume': 2_000_000,
            'latest_price': o, 'daily_bar_date': d}


def _yesterday():
    return (datetime.fromisoformat(_today_et()) - timedelta(days=1)).date().isoformat()


class TestFullFlipNoRefetchStorm:
    def test_yesterday_dated_snapshots_are_refetched_not_served(self):
        """A snapshot whose daily bar is still dated yesterday carries YESTERDAY's
        open; it must be re-fetched every tick until it completes (9/25 review
        finding) — never served to the gap gate, even when every cached entry
        is in that state at the open."""
        a = MagicMock(spec=AlpacaClient)
        a.get_snapshots.side_effect = [
            {s: _snap(10.0, 9.0, _yesterday()) for s in ('AAA', 'BBB', 'CCC')},
            {s: _snap(10.0, 9.0, _today_et()) for s in ('AAA', 'BBB', 'CCC', 'DDD')},
        ]
        eng = _engine(a)
        eng.build_orb_universe_from_snapshots(['AAA', 'BBB', 'CCC'])
        eng.build_orb_universe_from_snapshots(['AAA', 'BBB', 'CCC', 'DDD'])
        assert sorted(a.get_snapshots.call_args_list[1].args[0]) == ['AAA', 'BBB', 'CCC', 'DDD']

    def test_cache_survives_session_reset_semantics(self):
        """A warmed complete cache is hit with zero REST calls."""
        a = MagicMock(spec=AlpacaClient)
        a.get_snapshots.return_value = {'AAA': _snap(10.0, 9.0, _today_et())}
        eng = _engine(a)
        eng.build_orb_universe_from_snapshots(['AAA'])
        eng.build_orb_universe_from_snapshots(['AAA'])
        assert a.get_snapshots.call_count == 1


class TestBatchedMinuteBarLookup:
    def test_one_bulk_query_and_no_per_symbol_queries(self):
        a = MagicMock(spec=AlpacaClient)
        syms = [f"S{i}" for i in range(50)]
        a.get_snapshots.return_value = {s: _snap(10.0, 9.0, _today_et()) for s in syms}
        db = MagicMock(spec=Database)
        db.get_intraday_bars_for_date.return_value = {}
        eng = _engine(a, db)
        eng.build_orb_universe_from_snapshots(syms)
        assert db.get_intraday_bars_for_date.call_count == 1
        db.get_intraday_bars_cached.assert_not_called()

    def test_bulk_open_equals_per_symbol_open(self):
        """Gap input uses the 09:30 bar open: 10.0 snapshot vs 9.0 prev is
        +11%, but a settled 09:30 open of 9.2 is +2% -> rejected."""
        from zoneinfo import ZoneInfo
        a = MagicMock(spec=AlpacaClient)
        a.get_snapshots.return_value = {
            'AAA': _snap(10.0, 9.0, _today_et()), 'BBB': _snap(10.0, 9.0, _today_et())}
        ts = datetime.fromisoformat(_today_et() + 'T09:30:00').replace(
            tzinfo=ZoneInfo('America/New_York'))
        db = MagicMock(spec=Database)
        db.get_intraday_bars_for_date.return_value = {
            'AAA': [{'timestamp': ts, 'open': 9.2}]}
        keep = _engine(a, db).build_orb_universe_from_snapshots(['AAA', 'BBB'])
        assert 'AAA' not in keep and 'BBB' in keep


class TestOpenTickBudget:
    def test_expired_deadline_skips_refetch_and_warns(self, caplog):
        a = MagicMock(spec=AlpacaClient)
        eng = _engine(a)
        with caplog.at_level(logging.WARNING):
            eng.build_orb_universe_from_snapshots(['AAA', 'BBB'], deadline=time.time() - 1)
        a.get_snapshots.assert_not_called()
        assert any('budget exhausted' in r.message for r in caplog.records)

    def test_scan_cut_logs_deferred_count_and_returns(self, caplog):
        a = MagicMock(spec=AlpacaClient)
        syms = [f"S{i}" for i in range(20)]
        a.get_snapshots.return_value = {s: _snap(10.0, 9.0, _today_et()) for s in syms}
        eng = _engine(a)
        eng.build_orb_universe_from_snapshots(syms)  # warm the cache
        with caplog.at_level(logging.WARNING):
            # cache warm, deadline already passed: loop cuts immediately
            keep = eng.build_orb_universe_from_snapshots(syms, deadline=time.time() + 0.0)
        assert keep == []
        assert any('20 of 20 candidates not evaluated' in r.message for r in caplog.records)

    def test_slow_scan_still_returns_within_budget(self):
        a = MagicMock(spec=AlpacaClient)
        syms = [f"S{i}" for i in range(200)]
        a.get_snapshots.return_value = {s: _snap(10.0, 9.0, _today_et()) for s in syms}
        db = MagicMock(spec=Database)
        db.get_intraday_bars_for_date.return_value = {}
        eng = _engine(a, db)
        t0 = time.time()
        eng.build_orb_universe_from_snapshots(syms, deadline=time.time() + 5)
        assert time.time() - t0 < 5


class TestIntradayBarsForDate:
    """persistence.Database.get_intraday_bars_for_date: index seeks, one date, many symbols."""

    def _db(self, tmp_path):
        import sqlite3
        db = Database.__new__(Database)
        conn = sqlite3.connect(str(tmp_path / 'c.db')); conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE intraday_bars_1min (symbol TEXT, bar_date TEXT, timestamp TEXT, open REAL, "
                     "high REAL, low REAL, close REAL, volume REAL, UNIQUE(symbol, timestamp))")
        conn.execute("CREATE INDEX idx_intraday_bars_symbol_date ON intraday_bars_1min(symbol, bar_date)")
        rows = [('AAA', '2026-10-02', '2026-10-02T13:31:00+00:00', 2, 2, 2, 2, 20),
                ('AAA', '2026-10-02', '2026-10-02T13:30:00+00:00', 1, 1, 1, 1, 10),
                ('AAA', '2026-10-01', '2026-10-01T13:30:00+00:00', 9, 9, 9, 9, 90),
                ('BBB', '2026-10-02', '2026-10-02T13:30:00+00:00', 5, 5, 5, 5, 50)]
        conn.executemany("INSERT INTO intraday_bars_1min VALUES (?,?,?,?,?,?,?,?)", rows)
        db._cache_conn = conn
        return db

    def test_batched_equals_per_symbol_and_is_date_scoped(self, tmp_path):
        db = self._db(tmp_path)
        got = db.get_intraday_bars_for_date(['AAA', 'BBB', 'CCC', 'AAA'], '2026-10-02', chunk_size=2)
        assert set(got) == {'AAA', 'BBB'}                       # CCC absent, duplicate ignored
        assert [b['open'] for b in got['AAA']] == [1, 2]        # ordered by timestamp, other date excluded
        for sym in ('AAA', 'BBB'):
            per = db.get_intraday_bars_cached(sym, '2026-10-02')
            assert [(b['timestamp'], b['open']) for b in per] == [(b['timestamp'], b['open']) for b in got[sym]]

    def test_empty_input_returns_empty(self, tmp_path):
        assert self._db(tmp_path).get_intraday_bars_for_date([], '2026-10-02') == {}
