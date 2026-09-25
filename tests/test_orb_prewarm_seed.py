"""Unit tests for the ORB WARM-phase seed prewarming
(docs/orb_latency_fix_spec_20260925.md, flag orb.yaml execution.prewarm_seed).

Fixture/mocking style follows tests/test_orb_addon_pools.py: MagicMock(spec=...)
for domain/SDK collaborators, ORBEngine built directly off orb.yaml with the
minimal per-test overrides.

Covers:
  * flag off (default) — byte-identical to pre-2026-09-25 behavior, hot path
    still makes a full get_snapshots call every time (regression guard).
  * flag on — a cache hit on an already-warmed symbol makes ZERO further
    REST calls for it; only uncached (newly-qualified) symbols are fetched,
    and that fetch is logged at WARNING with its cost.
  * day-boundary rollover clears the cache (no stale cross-day leakage).
  * reset_daily() clears the cache.
  * parity — the warm path (several small incremental fetches) admits the
    EXACT SAME symbols as the old path (one full fetch), given the same
    underlying snapshot data — the admission logic itself never changes.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine
from trading.stop_monitor import StopMonitor


def _base_cfg(prewarm: bool) -> dict:
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    cfg.setdefault('execution', {})['prewarm_seed'] = prewarm
    return cfg


def _mock_alpaca() -> MagicMock:
    client = MagicMock(spec=AlpacaClient)
    client.get_snapshots.return_value = {}
    return client


def _snap(open_price: float, prev_close: float, prev_volume: int = 2_000_000,
          daily_bar_date: str = None) -> dict:
    snap = {'open': open_price, 'prev_close': prev_close,
            'prev_volume': prev_volume, 'latest_price': open_price}
    if daily_bar_date is not None:
        snap['daily_bar_date'] = daily_bar_date
    return snap


def _today_et() -> str:
    """Same ET-date computation the production code uses (_get_snapshots_warm)."""
    from zoneinfo import ZoneInfo
    return datetime.now(timezone.utc).astimezone(ZoneInfo('America/New_York')).date().isoformat()


def _engine(prewarm: bool, alpaca=None) -> ORBEngine:
    a = alpaca or _mock_alpaca()
    return ORBEngine(alpaca_client=a, db=MagicMock(spec=Database),
                      stop_monitor=MagicMock(spec=StopMonitor),
                      config=_base_cfg(prewarm))


class TestPrewarmFlagOff:
    """Default (flag false) is byte-identical to the pre-fix hot path."""

    def test_flag_defaults_false(self):
        eng = _engine(prewarm=False)
        assert eng.prewarm_seed_enabled is False

    def test_flag_off_calls_get_snapshots_every_time(self):
        a = _mock_alpaca()
        a.get_snapshots.return_value = {'AAA': _snap(10.0, 9.0)}
        eng = _engine(prewarm=False, alpaca=a)
        eng.build_orb_universe_from_snapshots(['AAA'])
        eng.build_orb_universe_from_snapshots(['AAA'])
        assert a.get_snapshots.call_count == 2
        for call in a.get_snapshots.call_args_list:
            assert sorted(call.args[0]) == ['AAA']


class TestPrewarmFlagOnHotPath:
    """Flag true: cached symbols make zero REST calls on repeat ticks."""

    def test_second_call_same_symbols_makes_no_rest_call(self):
        a = _mock_alpaca()
        a.get_snapshots.return_value = {'AAA': _snap(10.0, 9.0, daily_bar_date=_today_et())}
        eng = _engine(prewarm=True, alpaca=a)
        first = eng.build_orb_universe_from_snapshots(['AAA'])
        assert a.get_snapshots.call_count == 1
        second = eng.build_orb_universe_from_snapshots(['AAA'])
        # Hot path: zero additional network calls for an already-warmed symbol.
        assert a.get_snapshots.call_count == 1
        assert first == second == ['AAA']

    def test_incremental_refresh_fetches_only_new_symbol(self):
        """A symbol that qualifies AFTER the warm (e.g. at 09:34:50) triggers
        a fetch for ONLY that symbol, never a full re-fetch."""
        today_et = _today_et()
        a = _mock_alpaca()
        a.get_snapshots.side_effect = [
            {'AAA': _snap(10.0, 9.0, daily_bar_date=today_et)},
            {'BBB': _snap(12.0, 11.0, daily_bar_date=today_et)},
        ]
        eng = _engine(prewarm=True, alpaca=a)
        eng.build_orb_universe_from_snapshots(['AAA'])
        keep = eng.build_orb_universe_from_snapshots(['AAA', 'BBB'])
        assert a.get_snapshots.call_count == 2
        second_call_symbols = a.get_snapshots.call_args_list[1].args[0]
        assert second_call_symbols == ['BBB']  # only the uncached one
        assert set(keep) == {'AAA', 'BBB'}

    def test_incomplete_snapshot_open_zero_is_refetched_and_fresh_value_used(self):
        """A symbol that had not printed yet on tick 1 (open==0) must not
        stick with that stale dict all day — tick 2 re-fetches it and the
        fresh (now-printed) open is what admission sees."""
        today_et = _today_et()
        a = MagicMock(spec=AlpacaClient)
        a.get_snapshots.side_effect = [
            {'AAA': _snap(0.0, 9.0, daily_bar_date=today_et)},   # tick 1: pre-open, incomplete
            {'AAA': _snap(11.0, 9.0, daily_bar_date=today_et)},  # tick 2: printed, gap 22%
        ]
        eng = _engine(prewarm=True, alpaca=a)
        keep1 = eng.build_orb_universe_from_snapshots(['AAA'])
        assert keep1 == []  # open<=0 -> no admission on tick 1
        assert a.get_snapshots.call_count == 1
        keep2 = eng.build_orb_universe_from_snapshots(['AAA'])
        assert a.get_snapshots.call_count == 2  # incomplete cache entry was re-fetched
        assert a.get_snapshots.call_args_list[1].args[0] == ['AAA']
        assert keep2 == ['AAA']  # fresh value used, not the stale open==0 one

    def test_complete_snapshot_is_not_refetched(self):
        """A snapshot with open>0 and today's daily_bar_date is genuinely
        warm — it must NOT be re-fetched on a later tick."""
        today_et = _today_et()
        a = MagicMock(spec=AlpacaClient)
        a.get_snapshots.return_value = {'AAA': _snap(10.0, 9.0, daily_bar_date=today_et)}
        eng = _engine(prewarm=True, alpaca=a)
        eng.build_orb_universe_from_snapshots(['AAA'])
        assert a.get_snapshots.call_count == 1
        eng.build_orb_universe_from_snapshots(['AAA'])
        assert a.get_snapshots.call_count == 1  # complete entry reused, no re-fetch

    def test_cache_miss_logs_warning(self, caplog):
        import logging
        a = _mock_alpaca()
        a.get_snapshots.return_value = {'AAA': _snap(10.0, 9.0)}
        eng = _engine(prewarm=True, alpaca=a)
        with caplog.at_level(logging.WARNING):
            eng.build_orb_universe_from_snapshots(['AAA'])
        assert any('prewarm cache MISS' in r.message for r in caplog.records)

    def test_day_rollover_clears_cache(self):
        a = _mock_alpaca()
        a.get_snapshots.return_value = {'AAA': _snap(10.0, 9.0)}
        eng = _engine(prewarm=True, alpaca=a)
        eng.build_orb_universe_from_snapshots(['AAA'])
        assert a.get_snapshots.call_count == 1
        # Simulate yesterday's cache still sitting in memory.
        eng._snapshot_cache_date = (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).date().isoformat()
        eng.build_orb_universe_from_snapshots(['AAA'])
        assert a.get_snapshots.call_count == 2  # re-warmed, not reused

    def test_reset_daily_clears_cache(self):
        a = _mock_alpaca()
        a.get_snapshots.return_value = {'AAA': _snap(10.0, 9.0)}
        eng = _engine(prewarm=True, alpaca=a)
        eng.build_orb_universe_from_snapshots(['AAA'])
        assert eng._snapshot_cache
        eng.reset_daily()
        assert eng._snapshot_cache == {}
        assert eng._snapshot_cache_date is None


class TestPrewarmParity:
    """The warm path and the old path must admit the EXACT SAME symbols —
    the fix changes WHEN a snapshot is fetched, never the admission logic."""

    def test_incremental_warm_matches_single_full_fetch(self):
        snaps = {
            'PROD': _snap(10.0, 9.0),      # gap 11.1% -> admitted
            'NOPE': _snap(10.0, 9.99),     # gap 0.1% -> rejected
            'LATE': _snap(12.0, 11.0),     # gap 9.1% -> admitted, arrives late
        }

        # Old path: flag off, one full fetch of everything at once.
        a_old = _mock_alpaca()
        a_old.get_snapshots.return_value = dict(snaps)
        eng_old = _engine(prewarm=False, alpaca=a_old)
        old_result = sorted(eng_old.build_orb_universe_from_snapshots(list(snaps.keys())))

        # Warm path: flag on, symbols arrive across three separate ticks.
        a_new = _mock_alpaca()
        a_new.get_snapshots.side_effect = [
            {'PROD': snaps['PROD']},
            {'NOPE': snaps['NOPE']},
            {'LATE': snaps['LATE']},
        ]
        eng_new = _engine(prewarm=True, alpaca=a_new)
        eng_new.build_orb_universe_from_snapshots(['PROD'])
        eng_new.build_orb_universe_from_snapshots(['PROD', 'NOPE'])
        new_result = sorted(
            eng_new.build_orb_universe_from_snapshots(['PROD', 'NOPE', 'LATE'])
        )

        assert old_result == new_result == ['LATE', 'PROD']
