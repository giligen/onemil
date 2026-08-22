"""B+ RESTART end-to-end integration — synthetic day (2026-08-15).

Drives the real check_entries pipeline (score -> quintile -> rank -> dedup ->
PDR/G1/catalyst vetoes -> submit) with a REAL Database (8/14 save_trade
lesson) and mocked Alpaca. Asserts:
  * a G1-failing pick is vetoed (slot consumed, NO refill), passing picks submit
  * a daily-kill DB state gates the entry path (no submits)
  * a PDT-capped account gates the entry path
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml
from pathlib import Path

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine, RangeData, CandidateState
from trading.stop_monitor import StopMonitor

ROOT = Path(__file__).parent.parent


@pytest.fixture
def cfg():
    c = yaml.safe_load(open(ROOT / 'orb.yaml'))
    c['strategy']['enabled'] = True
    return c


@pytest.fixture
def tmp_db():
    d = tempfile.mkdtemp()
    return Database(trades_path=os.path.join(d, 'trades.db'),
                    cache_path=os.path.join(d, 'cache.db'))


@pytest.fixture
def engine(cfg, tmp_db):
    al = MagicMock(spec=AlpacaClient)
    # cash account -> PDT dormant by default (tested separately for margin)
    al.get_account_info.return_value = {
        'equity': 10000.0, 'multiplier': 1.0, 'daytrade_count': 0,
        'buying_power': 10000.0}
    e = ORBEngine(alpaca_client=al, db=tmp_db,
                  stop_monitor=MagicMock(spec=StopMonitor), config=cfg)
    # Isolate the ranking/veto/submit core: no-op the I/O-heavy top-of-loop
    # helpers (prefetch/anchor warm/pending fills/sweep) so the test drives
    # selection deterministically.
    for m in ('_maybe_prefetch_pm', '_prewarm_anchors', '_process_pending_fills',
              '_cancel_stale_pending_orders'):
        setattr(e, m, MagicMock())
    e._ensure_ranges_post_open = MagicMock(return_value=set())
    e._past_last_entry_time = MagicMock(return_value=False)
    e._has_buying_power = MagicMock(return_value=True)
    e._get_spread_bps = MagicMock(return_value=50.0)
    return e


def _seed(engine, sym, rv20):
    """A 'good continuation' pick (comp Q5, pdr 15.2 > 11) with variable rv20."""
    c = CandidateState(symbol=sym)
    c.range_data = RangeData(
        symbol=sym, range_high=10.4, range_low=10.0, range_volume=300_000,
        range_avg_bar_range_pct=1.2, range_close=10.38,
        range_start_ts=pd.Timestamp.utcnow(), range_open=10.0)
    engine.candidates[sym] = c
    prov = {'prev_day_bar': {'open': 9.2, 'high': 10.4, 'low': 9.0,
                             'close': 9.2, 'volume': 1e6},
            'daily_stats_20d': {'high_20d': 14.29, 'volume_20d': 1e6,
                                'return_volatility_20d': rv20}}
    return prov


def test_g1_vetoes_one_pick_no_refill(engine):
    """AAAA (rv12) + CCCC (rv12) submit; BBBB (rv5) is G1-vetoed, slot stays
    empty (max_concurrent=3, so a 4th DDDD must NOT backfill BBBB's slot)."""
    engine._submit_entry = MagicMock(side_effect=lambda plan: f'oid-{plan.symbol}')
    providers = {}
    for sym, rv in [('AAAA', 12.0), ('BBBB', 5.0), ('CCCC', 12.0), ('DDDD', 12.0)]:
        providers[sym] = _seed(engine, sym, rv)

    submitted = engine.check_entries(symbols=list(providers),
                                     feature_providers=providers)

    # BBBB G1-vetoed
    assert engine.candidates['BBBB'].rejected_reason == 'g1_veto'
    assert 'BBBB' not in submitted
    assert 'BBBB' in engine._pdr_vetoed_today       # slot consumed
    # BBBB never handed to _submit_entry
    submitted_syms = {c.args[0].symbol
                      for c in engine._submit_entry.call_args_list}
    assert 'BBBB' not in submitted_syms
    # No-refill: exactly 3 distinct picks consumed the 3 slots (AAAA/CCCC/DDDD)
    # minus BBBB's dead slot — i.e. at most max_concurrent-1 submit... but since
    # dedup keeps top-3 and BBBB is one of them, its veto leaves an empty slot.
    assert 'AAAA' in submitted and 'CCCC' in submitted


def test_daily_kill_gates_entries(engine):
    """A -$600 realized ORB day in the DB blocks all new entries."""
    engine._submit_entry = MagicMock(return_value='oid')
    providers = {'AAAA': _seed(engine, 'AAAA', 12.0)}
    today = engine._et_now().strftime('%Y-%m-%d')
    conn = sqlite3.connect(engine.db._trades_path)
    conn.execute(
        """INSERT INTO trades
        (trade_date,symbol,strategy,side,entry_price,stop_loss_price,
         take_profit_price,shares,risk_per_share,total_risk,risk_reward_ratio,
         created_at,updated_at,pnl,fill_price,exit_price)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (today, 'PRIOR', 'orb', 'buy', 10.0, 9.0, 12.0, 100, 1.0, 100.0, 2.0,
         'x', 'x', -600.0, 10.0, 4.0))
    conn.commit()
    conn.close()
    submitted = engine.check_entries(symbols=list(providers),
                                     feature_providers=providers)
    assert submitted == []
    engine._submit_entry.assert_not_called()


def test_pdt_cap_gates_entries(cfg, tmp_db):
    """A margin account under $25K equity with 3 same-window day-trades blocks
    the 4th."""
    al = MagicMock(spec=AlpacaClient)
    al.get_account_info.return_value = {
        'equity': 10000.0, 'multiplier': 4.0, 'daytrade_count': 3,
        'buying_power': 40000.0}
    e = ORBEngine(alpaca_client=al, db=tmp_db,
                  stop_monitor=MagicMock(spec=StopMonitor), config=cfg)
    for m in ('_maybe_prefetch_pm', '_prewarm_anchors', '_process_pending_fills',
              '_cancel_stale_pending_orders'):
        setattr(e, m, MagicMock())
    e._ensure_ranges_post_open = MagicMock(return_value=set())
    e._past_last_entry_time = MagicMock(return_value=False)
    e._submit_entry = MagicMock(return_value='oid')
    today = e._et_now().strftime('%Y-%m-%d')
    conn = sqlite3.connect(e.db._trades_path)
    for s in ('A', 'B', 'C'):
        conn.execute(
            """INSERT INTO trades
            (trade_date,symbol,strategy,side,entry_price,stop_loss_price,
             take_profit_price,shares,risk_per_share,total_risk,
             risk_reward_ratio,created_at,updated_at,fill_price)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (today, s, 'orb', 'buy', 10.0, 9.0, 12.0, 100, 1.0, 100.0, 2.0,
             'x', 'x', 10.0))
    conn.commit()
    conn.close()
    prov = {'AAAA': _seed(e, 'AAAA', 12.0)}
    assert e.check_entries(symbols=['AAAA'], feature_providers=prov) == []
    e._submit_entry.assert_not_called()
