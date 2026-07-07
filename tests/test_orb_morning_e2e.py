"""End-to-end morning simulation (2026-07-07 evening).

Every hot-path change this week is unit-pinned at its seam — but IREZ and
the grace event-eating both hid BETWEEN seams. This test runs tomorrow's
whole opening sequence through the REAL check_entries code (real orb.yaml
params, real composite math via feature_providers, real PDR veto, real
PM mult, real planner + slot accounting) with only Alpaca mocked:

  9:32  tick        -> PM prefetch fires (batch fetch, once)
  9:35:05 burst     -> one straggler rangeless -> grace DEFERS, nothing submits
  9:35:20 burst     -> field complete -> rank -> 3 PDR-vetoed -> 1 submitted
  9:36  burst       -> slot-saturated: no new entries, fills STILL processed
"""
from __future__ import annotations

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


def _rng(sym, hi=10.0, lo=9.6, open_=9.65, vol=300_000):
    return RangeData(symbol=sym, range_high=hi, range_low=lo, range_volume=vol,
                     range_avg_bar_range_pct=1.2, range_close=hi - 0.02,
                     range_start_ts=pd.Timestamp.utcnow(), range_open=open_)


def _providers(pdr_pct):
    """Feature context producing a healthy composite; prev-day range set
    per test (veto knob). prev_close chosen for a ~6.7% gap."""
    prev_close = 9.05
    prev_low = 9.0
    prev_high = prev_low + prev_close * pdr_pct / 100.0
    return {'prev_day_bar': {'open': 9.0, 'high': prev_high, 'low': prev_low,
                             'close': prev_close, 'volume': 2_000_000},
            'daily_stats_20d': {'high_20d': 14.0, 'volume_20d': 1_000_000}}


PM_BARS = pd.DataFrame([{
    'timestamp': pd.Timestamp('2026-07-08 08:00',
                              tz='America/New_York').tz_convert('UTC'),
    'open': 9.0, 'high': 9.0, 'low': 9.0, 'close': 9.0,
    'volume': 1_000_000,   # $9M > $5.8M cut -> 1.5x on the survivor
}])


@pytest.fixture
def engine(monkeypatch):
    for v in ('ORB_PM_MULT', 'ORB_PDR_VETO', 'ORB_PDR_VETO_MIN_PCT'):
        monkeypatch.delenv(v, raising=False)
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    alpaca = MagicMock(spec=AlpacaClient)
    alpaca.get_latest_quote.return_value = {'bid_price': 9.80, 'ask_price': 9.82,
                                            'bid_size': 10, 'ask_size': 10}
    alpaca.submit_stop_bracket_order.return_value = {
        'id': 'ord-1', 'status': 'pending_new'}
    alpaca.get_premarket_1min_bars_multi = MagicMock(
        return_value={s: PM_BARS for s in ('LOUD', 'Q1', 'Q2', 'Q3')})
    alpaca.get_open_positions.return_value = []
    alpaca.get_account_info.return_value = {'buying_power': 100_000.0,
                                            'equity': 100_000.0, 'cash': 100_000.0,
                                            'daytrade_count': 0,
                                            'pattern_day_trader': False}
    db = MagicMock(spec=Database)
    db.save_trade.return_value = 77
    db.get_open_trades.return_value = []
    sm = MagicMock(spec=StopMonitor)
    sm.drain_exit_events.return_value = []
    eng = ORBEngine(alpaca_client=alpaca, db=db, stop_monitor=sm, config=cfg)
    eng._symbols_entered_today_db = MagicMock(return_value=set())
    return eng


def _at(eng, hh, mm, ss, fn, *a, **kw):
    with patch('trading.orb_engine.datetime') as mdt:
        mdt.now.return_value = datetime(2026, 7, 8, hh + 4, mm, ss,
                                        tzinfo=timezone.utc)  # EDT
        mdt.combine = datetime.combine
        mdt.strptime = datetime.strptime
        return fn(*a, **kw)


class TestMorningSequence:
    def test_full_opening_sequence(self, engine):
        # Universe arrives in two batches (like 7/7: 31 then +21)
        engine.build_universe(source_loader=lambda: ['LOUD', 'Q1', 'Q2'])
        engine.build_universe(source_loader=lambda: ['LOUD', 'Q1', 'Q2', 'Q3'])

        # ---- 9:32 tick: PM prefetch fires once, covers all candidates ----
        _at(engine, 9, 32, 0, engine._maybe_prefetch_pm)
        assert engine.alpaca.get_premarket_1min_bars_multi.call_count == 1
        assert set(engine._pm_dollar_vols) == {'LOUD', 'Q1', 'Q2', 'Q3'}

        # ---- ranges: Q3 is the straggler ----
        for s in ('LOUD', 'Q1', 'Q2'):
            engine.candidates[s].range_data = _rng(s)
        providers = {'LOUD': _providers(pdr_pct=20.0),   # explosive prev day
                     'Q1': _providers(pdr_pct=4.0),      # quiet -> veto
                     'Q2': _providers(pdr_pct=5.0),      # quiet -> veto
                     'Q3': _providers(pdr_pct=6.0)}      # quiet -> veto

        # ---- 9:35:05 burst: straggler rangeless -> grace defers ----
        out = _at(engine, 9, 35, 5, engine.check_entries,
                  symbols=['LOUD', 'Q1', 'Q2'], feature_providers=providers)
        assert out == []
        engine.alpaca.submit_stop_bracket_order.assert_not_called()
        assert engine._post_open_range_sweep_done is False  # re-armed

        # ---- straggler consolidates; 9:35:20 burst: full field ----
        engine.candidates['Q3'].range_data = _rng('Q3')
        engine._post_open_range_sweep_done = True   # sweep satisfied
        out = _at(engine, 9, 35, 20, engine.check_entries,
                  feature_providers=providers)

        # exactly one submission (LOUD); the three quiet-prev-day picks vetoed
        assert out == ['LOUD']
        assert engine.alpaca.submit_stop_bracket_order.call_count == 1
        assert engine._pdr_vetoed_today == {'Q1', 'Q2', 'Q3'}
        # PM mult applied to the survivor: shares reflect 1.5x sizing
        kw = engine.alpaca.submit_stop_bracket_order.call_args.kwargs
        assert kw['qty'] > 0

        # ---- 9:36 burst: slots saturated (1 entered + 3 vetoed = 4) ----
        engine._symbols_entered_today_db = MagicMock(return_value={'LOUD'})
        engine._process_pending_fills = MagicMock()
        out = _at(engine, 9, 36, 0, engine.check_entries,
                  feature_providers=providers)
        assert out == []                                # no backfill, ever
        engine._process_pending_fills.assert_called_once()  # fills keep running

    def test_pm_mult_actually_sized_up(self, engine):
        """The survivor's position must be 1.5x the unboosted plan."""
        engine.build_universe(source_loader=lambda: ['LOUD'])
        engine.candidates['LOUD'].range_data = _rng('LOUD')
        providers = {'LOUD': _providers(pdr_pct=20.0)}
        _at(engine, 9, 32, 0, engine._maybe_prefetch_pm)
        _at(engine, 9, 36, 0, engine.check_entries, feature_providers=providers)
        qty_boosted = engine.alpaca.submit_stop_bracket_order.call_args.kwargs['qty']
        # rebuild fresh engine with PM disabled -> baseline qty
        import copy
        engine2_alpaca = engine.alpaca
        with open(Path(__file__).parent.parent / 'orb.yaml') as f:
            cfg = yaml.safe_load(f)
        cfg['strategy']['enabled'] = True
        cfg['sizing']['pm_dollar_vol_mult']['enabled'] = False
        db = MagicMock(spec=Database); db.save_trade.return_value = 78
        db.get_open_trades.return_value = []
        sm = MagicMock(spec=StopMonitor); sm.drain_exit_events.return_value = []
        eng2 = ORBEngine(alpaca_client=engine2_alpaca, db=db,
                         stop_monitor=sm, config=cfg)
        eng2._symbols_entered_today_db = MagicMock(return_value=set())
        eng2.build_universe(source_loader=lambda: ['LOUD'])
        eng2.candidates['LOUD'].range_data = _rng('LOUD')
        engine2_alpaca.submit_stop_bracket_order.reset_mock()
        _at(eng2, 9, 36, 0, eng2.check_entries, feature_providers=providers)
        qty_base = engine2_alpaca.submit_stop_bracket_order.call_args.kwargs['qty']
        assert qty_boosted == pytest.approx(qty_base * 1.5, rel=0.02)
