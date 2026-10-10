"""Regression for the 2026-10-09 CIEG incident: the day's FIRST selection
burst must rank the FULL candidate field, not the caller's drain subset.

10/9: CIEG/LOFF/LI/SPAL got ranges at 13:35:02 while the first-rank grace
deferred the call; a drain event for OTHER names cleared the grace at 13:35:09
and ran the first burst on ITS subset only. Fix: while nothing is placed or
vetoed today, ``cand_pool`` = every candidate. Later subset calls unchanged.
Add-on pools derive their symbols from the same ``eligible`` list, so they
inherit the widening (asserted below).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine, RangeData
from trading.stop_monitor import StopMonitor

NOW = datetime(2026, 10, 9, 13, 35, 9, tzinfo=timezone.utc)


@pytest.fixture
def engine():
    """Engine with three ranged candidates A, B, C and a recording selector."""
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    alpaca = MagicMock(spec=AlpacaClient)
    alpaca.get_open_positions.return_value = []
    alpaca.get_account_info.return_value = {'buying_power': 500_000.0}
    db = MagicMock(spec=Database)
    db.save_trade.return_value = 1
    db.get_open_trades.return_value = []
    db.get_trades_by_date.return_value = []
    db.get_daily_bars_cached.return_value = {}
    sm = MagicMock(spec=StopMonitor)
    sm.polling_mode = False
    sm.drain_exit_events.return_value = []
    eng = ORBEngine(alpaca_client=alpaca, db=db, stop_monitor=sm, config=cfg)
    eng.build_universe(source_loader=lambda: ['AAA', 'BBB', 'CCC'])
    for s in ('AAA', 'BBB', 'CCC'):
        eng.candidates[s].range_data = RangeData(
            symbol=s, range_high=11.5, range_low=11.0, range_volume=50_000,
            range_avg_bar_range_pct=2.0, range_close=11.4,
            range_start_ts=pd.Timestamp('2026-10-09 13:30:00+00:00'))
    eng.scored_calls = []

    def _fake_pool(pool_label, cand_syms, *a, **kw):
        eng.scored_calls.append((pool_label, sorted(cand_syms)))
        # the BT-style top pick is the alphabetically first of the field
        return [sorted(cand_syms)[0]] if cand_syms else []

    eng._run_pool_selection = _fake_pool
    eng._post_open_range_sweep_done = True
    return eng


def _run(engine, symbols, defer=False, entered=()):
    with patch('trading.orb_engine.datetime') as dt, \
            patch.object(engine, '_should_defer_first_rank', return_value=defer), \
            patch.object(engine, '_symbols_entered_today_db', return_value=set(entered)):
        dt.now = lambda tz=None: NOW
        return engine.check_entries(symbols=symbols)


def test_a_grace_deferred_names_are_scored_when_a_later_drain_clears_it(engine):
    """Drain 1 (AAA, BBB ranged) is deferred; drain 2 (CCC only) runs the burst:
    all three are scored and the top pick is one of the FIRST two."""
    assert _run(engine, {'AAA', 'BBB'}, defer=True) == []
    placed = _run(engine, {'CCC'}, defer=False)
    assert engine.scored_calls[0] == ('production', ['AAA', 'BBB', 'CCC'])
    assert placed == ['AAA'] and placed != ['CCC']


def test_b_after_the_first_burst_a_subset_call_scores_only_the_subset(engine):
    _run(engine, {'AAA', 'BBB', 'CCC'})  # first burst ranks a non-empty field
    engine.scored_calls.clear()
    _run(engine, {'CCC'}, entered={'AAA'})
    assert engine.scored_calls[0] == ('production', ['CCC'])


def test_b2_flag_is_cleared_by_the_daily_reset(engine):
    _run(engine, {'AAA', 'BBB', 'CCC'})
    assert engine._first_burst_done is True
    engine._first_burst_done = False  # what _reset_daily_locked does
    engine.scored_calls.clear()
    _run(engine, {'CCC'})
    assert engine.scored_calls[0] == ('production', ['AAA', 'BBB', 'CCC'])


def test_e_preplaced_names_do_not_hide_the_1009_shape(engine):
    """10/9: COHH/GLWG preplaced (in symbols_entered_today, slots_used 2);
    drain 1 (AAA ranged) deferred by grace; drain 2 (CCC only) clears it ->
    the full field incl. AAA (the CIEG-equivalent) is scored."""
    pre = {'PRE1', 'PRE2'}
    assert _run(engine, {'AAA'}, defer=True) == []  # 13:35:02, before the preplace
    _run(engine, {'CCC'}, defer=False, entered=pre)
    assert engine.scored_calls[0] == ('production', ['AAA', 'BBB', 'CCC'])


def test_c_widening_log_fires_once_with_counts(engine, caplog):
    with caplog.at_level(logging.INFO, logger='trading.orb_engine'):
        _run(engine, {'CCC'})
    lines = [r.message for r in caplog.records if 'first burst' in r.message]
    assert len(lines) == 1
    assert '3 candidates' in lines[0] and 'caller subset had 1' in lines[0]


def test_c2_no_log_when_subset_already_is_the_full_field(engine, caplog):
    with caplog.at_level(logging.INFO, logger='trading.orb_engine'):
        _run(engine, {'AAA', 'BBB', 'CCC'})
    assert not [r for r in caplog.records if 'first burst' in r.message]


def test_addon_pool_inherits_the_widening(engine):
    """Add-on pool symbols come from `eligible`, which is built from cand_pool."""
    engine.addon_pools_enabled = True
    engine.addon_pools = [{'name': 'p1'}]
    engine._symbol_pool = {'AAA': 'p1', 'BBB': 'production', 'CCC': 'production'}
    _run(engine, {'CCC'})
    assert ('production', ['BBB', 'CCC']) in engine.scored_calls
    assert ('p1', ['AAA']) in engine.scored_calls
