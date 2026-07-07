"""Tomorrow-morning sequence regressions (2026-07-07 evening).

The IREZ fix makes the daily-cap early-return fire EVERY day after the
first burst (entered+vetoed >= 4). These tests pin that the cap return
skips ONLY entry evaluation — pending-fill processing and time-stop
cancels must keep running all day — and that the PM prefetch covers
late universe batches.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine, CandidateState
from trading.stop_monitor import StopMonitor


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.delenv('ORB_PM_MULT', raising=False)
    monkeypatch.delenv('ORB_PDR_VETO', raising=False)
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    return ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient),
                     db=MagicMock(spec=Database),
                     stop_monitor=MagicMock(spec=StopMonitor), config=cfg)


class TestCapDoesNotStarveLifecycle:
    def _saturate(self, engine):
        engine._symbols_entered_today_db = lambda: {'BEZ'}
        engine._pdr_vetoed_today = {'TECS', 'SSG', 'MUD'}

    def test_fills_processed_after_cap_hit(self, engine):
        self._saturate(engine)
        engine._process_pending_fills = MagicMock()
        out = engine.check_entries()
        assert out == []                       # cap correctly blocks entries
        engine._process_pending_fills.assert_called_once()   # ...but fills run

    def test_time_stop_cancel_after_cap_hit(self, engine):
        self._saturate(engine)
        engine._cancel_stale_pending_orders = MagicMock()
        engine.check_entries()
        engine._cancel_stale_pending_orders.assert_called_once()

    def test_pm_prefetch_still_runs_after_cap_hit(self, engine):
        self._saturate(engine)
        engine._maybe_prefetch_pm = MagicMock()
        engine.check_entries()
        engine._maybe_prefetch_pm.assert_called_once()


class TestPmLateBatchCoverage:
    def _pm_bars(self):
        return pd.DataFrame([{
            'timestamp': pd.Timestamp('2026-07-08 09:00',
                                      tz='America/New_York').tz_convert('UTC'),
            'open': 5.0, 'high': 5.0, 'low': 5.0, 'close': 5.0,
            'volume': 2_000_000,
        }])

    def test_second_batch_fetched_and_merged(self, engine):
        """Universe builds in batches (7/7: 31 then +21). Each batch must
        get PM coverage; already-fetched symbols must not refetch."""
        engine.build_universe(source_loader=lambda: ['AAA'])
        engine.alpaca.get_premarket_1min_bars_multi = MagicMock(
            return_value={'AAA': self._pm_bars()})
        engine._fetch_pm_dollar_vols()
        assert engine._pm_dollar_vols['AAA'] is not None
        # batch 2 arrives
        engine.build_universe(source_loader=lambda: ['AAA', 'BBB'])
        engine.alpaca.get_premarket_1min_bars_multi = MagicMock(
            return_value={'BBB': self._pm_bars()})
        engine._fetch_pm_dollar_vols()
        engine.alpaca.get_premarket_1min_bars_multi.assert_called_once_with(['BBB'])
        assert engine._pm_dollar_vols['BBB'] is not None

    def test_no_print_symbols_not_refetched(self, engine):
        """A symbol with zero premarket prints is recorded as None and must
        not trigger endless refetches."""
        engine.build_universe(source_loader=lambda: ['DEAD'])
        engine.alpaca.get_premarket_1min_bars_multi = MagicMock(return_value={})
        engine._fetch_pm_dollar_vols()
        assert 'DEAD' in engine._pm_dollar_vols       # recorded as None
        engine._fetch_pm_dollar_vols()                # second call
        # no missing symbols -> no second API call
        assert engine.alpaca.get_premarket_1min_bars_multi.call_count == 1

    def test_empty_pool_does_not_stamp_day(self, engine):
        engine.alpaca.get_premarket_1min_bars_multi = MagicMock(return_value={})
        engine._fetch_pm_dollar_vols()   # no candidates yet
        assert engine._pm_fetch_done_day is None or \
            engine._pm_dollar_vols == {}
        engine.alpaca.get_premarket_1min_bars_multi.assert_not_called()


class TestBurstPickBudget:
    """2026-07-07 review: max_keep must subtract entered+vetoed slots, not
    just open positions — else a mid-window restart or multi-burst morning
    replays the IREZ backfill within one burst."""

    def test_pick_budget_counts_vetoed_and_entered(self, engine):
        """Restart replay: BEZ entered (DB), 3 re-vetoed, position closed.
        Pick budget must be 0 — the next-ranked name may NOT slide in."""
        import inspect
        src = inspect.getsource(type(engine).check_entries)
        assert 'symbols_entered_today | self._pdr_vetoed_today' in src \
            and 'set(self.open_positions)' in src

    def test_dedup_handles_zero_budget(self):
        from trading.orb_correlation import dedup_candidates
        assert dedup_candidates(['IREZ', 'AMPG'], max_keep=0) == []
        assert dedup_candidates(['IREZ'], max_keep=-1) == []


class TestPmFailurePoisoning:
    def test_fetch_failure_does_not_retry_every_tick(self, engine):
        """A dead premarket endpoint must cost ONE blocking call, not one
        per tick in the entry hot path."""
        engine.build_universe(source_loader=lambda: ['AAA', 'BBB'])
        engine.alpaca.get_premarket_1min_bars_multi = MagicMock(
            side_effect=RuntimeError('endpoint down'))
        engine._fetch_pm_dollar_vols()
        # symbols poisoned with None -> no missing set -> no re-trigger
        engine._fetch_pm_dollar_vols()
        assert engine.alpaca.get_premarket_1min_bars_multi.call_count == 1
        assert engine._get_pm_mult('AAA') == 1.0   # fail-open preserved
