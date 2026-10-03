"""Fix A (docs/review_20261003/FIX_A_orb_spec.md): A1-A5."""
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from scanner.realtime_scanner import RealtimeScanner
from trading.orb_engine import ORBEngine
from trading.stop_monitor import StopMonitor


def _engine(alpaca):
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    return ORBEngine(alpaca_client=alpaca, db=MagicMock(spec=Database),
                     stop_monitor=MagicMock(spec=StopMonitor), config=cfg)


def _now_et():
    from zoneinfo import ZoneInfo
    return datetime(2026, 10, 2, 9, 40, tzinfo=ZoneInfo('America/New_York'))


def _bar_df(ts_utc, o):
    return pd.DataFrame([{'timestamp': ts_utc, 'open': o, 'high': o, 'low': o,
                          'close': o, 'volume': 1}])


class TestA1ChunkBudget:
    def test_per_call_timeout_and_no_retry(self):
        a = MagicMock(spec=AlpacaClient)
        a.get_1min_bars_range_multi.return_value = {}
        eng = _engine(a)
        eng._fetch_today_open_bars(['AAA'], _now_et(), time.time() + 100)
        kw = a.get_1min_bars_range_multi.call_args.kwargs
        assert kw['retries'] == 0 and kw['timeout_s'] <= 20.0

    def test_small_remaining_stops_with_warning(self, caplog):
        a = MagicMock(spec=AlpacaClient)
        eng = _engine(a)
        with caplog.at_level(logging.WARNING):
            eng._fetch_today_open_bars(['AAA'], _now_et(), time.time() + 3)
        assert a.get_1min_bars_range_multi.call_count == 0
        assert 'budget exhausted' in caplog.text

    def test_timeout_marks_miss_and_counts(self, caplog):
        a = MagicMock(spec=AlpacaClient)
        a.get_1min_bars_range_multi.side_effect = RuntimeError('timed out')
        eng = _engine(a)
        with caplog.at_level(logging.WARNING):
            eng._fetch_today_open_bars(['AAA'], _now_et(), time.time() + 100)
        assert 'AAA' in eng._open_bar_miss
        assert 'failed' in caplog.text

    def test_client_accepts_timeout_params(self):
        import inspect
        sig = inspect.signature(AlpacaClient.get_1min_bars_range_multi)
        assert 'timeout_s' in sig.parameters and 'retries' in sig.parameters


class TestA2Exact0930Bar:
    def test_only_0931_row_is_a_miss(self):
        a = MagicMock(spec=AlpacaClient)
        t931 = pd.Timestamp('2026-10-02 13:31:00', tz='UTC')
        a.get_1min_bars_range_multi.return_value = {'AAA': _bar_df(t931, 5.0)}
        eng = _engine(a)
        out = eng._fetch_today_open_bars(['AAA'], _now_et(), time.time() + 100)
        assert 'AAA' not in out and 'AAA' in eng._open_bar_miss

    def test_0930_row_taken_tz_aware_and_naive(self):
        a = MagicMock(spec=AlpacaClient)
        aware = pd.Timestamp('2026-10-02 13:30:00', tz='UTC')
        naive = pd.Timestamp('2026-10-02 13:30:00')
        a.get_1min_bars_range_multi.return_value = {
            'AAA': pd.concat([_bar_df(pd.Timestamp('2026-10-02 13:31:00', tz='UTC'), 9.0),
                              _bar_df(aware, 5.0)], ignore_index=True),
            'BBB': _bar_df(naive, 6.0)}
        eng = _engine(a)
        out = eng._fetch_today_open_bars(['AAA', 'BBB'], _now_et(), time.time() + 100)
        assert out == {'AAA': 5.0, 'BBB': 6.0}


class TestA3ExitsFirst:
    def _scanner(self, order, build_raises):
        s = RealtimeScanner.__new__(RealtimeScanner)
        eng = MagicMock()
        eng.enabled = False
        eng.universe_build_due.return_value = True
        def build(**kw):
            order.append('build')
            if build_raises:
                raise RuntimeError('slow build blew up')
        eng.build_universe.side_effect = build
        eng.check_exits.side_effect = lambda: order.append('exits')
        eng.check_entries.side_effect = lambda: order.append('entries')
        eng.is_force_close_time.return_value = False
        s.orb_engine = eng
        return s

    def test_exits_before_build(self):
        order = []
        self._scanner(order, False)._orb_tick()
        assert order.index('exits') < order.index('build')

    def test_build_failure_does_not_starve_exits(self):
        order = []
        with pytest.raises(RuntimeError):
            self._scanner(order, True)._orb_tick()
        assert 'exits' in order


class TestA4NoSilentFailure:
    def test_admission_exceptions_aggregated(self, caplog):
        a = MagicMock(spec=AlpacaClient)
        bad = {'open': 'x', 'prev_close': 1, 'prev_volume': 1}
        a.get_snapshots.return_value = {'AAA': bad, 'BBB': bad}
        eng = _engine(a)
        with caplog.at_level(logging.WARNING):
            eng.build_orb_universe_from_snapshots(['AAA', 'BBB'])
        msgs = [r for r in caplog.records if 'ORB admission' in r.getMessage()]
        assert any(r.levelno == logging.WARNING for r in msgs) or \
            any(r.levelno == logging.ERROR for r in msgs)
        assert sum(r.levelno == logging.ERROR for r in msgs) == 1
        assert any('2 symbol(s) skipped' in r.getMessage() for r in msgs)

    def test_rvol_tilt_failopen_warns_once_per_day(self, caplog, monkeypatch):
        eng = _engine(MagicMock(spec=AlpacaClient))
        eng.rvol_tilt_enabled = True
        eng.rvol_tilt_applies_to = {'production'}
        import trading.orb_engine as oe
        def boom(*a, **k):
            raise ValueError('no rvol')
        monkeypatch.setattr(oe, 'resolve_rvol_tilt_mult', boom)
        cand = MagicMock()
        cand.rel_volume_0935 = None
        with caplog.at_level(logging.WARNING):
            assert eng._get_rvol_tilt_mult(cand) == (1.0, None)
            assert eng._get_rvol_tilt_mult(cand) == (1.0, None)
        warns = [r for r in caplog.records if 'RVOL tilt' in r.getMessage()
                 and r.levelno == logging.WARNING]
        assert len(warns) == 1


class TestA5Reentrancy:
    def test_second_build_skipped_while_first_runs(self, caplog):
        eng = _engine(MagicMock(spec=AlpacaClient))
        seen = {}
        def loader():
            with caplog.at_level(logging.INFO):
                seen['inner'] = eng.build_universe(source_loader=lambda: ['ZZZ'])
            return ['AAA']
        eng.build_universe(source_loader=loader)
        assert 'ZZZ' not in eng.universe and 'AAA' in eng.universe
        assert 'still in progress' in caplog.text
        assert eng._build_in_progress is False

    def test_flag_cleared_after_loader_error(self):
        eng = _engine(MagicMock(spec=AlpacaClient))
        def loader():
            raise RuntimeError('x')
        eng.build_universe(source_loader=loader)
        assert eng._build_in_progress is False
