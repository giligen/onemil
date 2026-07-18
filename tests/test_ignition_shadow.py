"""Ignition S1 shadow (2026-07-19): journal-only, zero orders, isolated."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from data_sources.alpaca_client import AlpacaClient
from trading.ignition_shadow import IgnitionShadow

def _shadow(tmp_path, **cfg):
    a=MagicMock(spec=AlpacaClient)
    a.get_latest_quote.return_value={'bid_price':10.0,'ask_price':10.05,
                                     'bid_size':5,'ask_size':7}
    bars=pd.DataFrame({'open':[9.0]+[10.0]*40,'high':[9.1]+[10.4]*40,
                       'low':[8.9]+[9.4]*40,'close':[9.05]+[10.2]*40,
                       'volume':[10000]*41})
    a.get_1min_bars.return_value=bars
    s=IgnitionShadow(a,{'ignition_shadow':{'enabled':True,**cfg}},
                     log_dir=str(tmp_path))
    return s,a

def _fire(s,sym='IGNI',chg=15.0,gap=12.0,news=None,minute=(9,50)):
    # seen_at is captured at ENQUEUE time (inside on_mover, under the
    # patch); the worker evaluates with that timestamp, so draining
    # after the patch exits is deterministic and time-correct
    with patch('trading.ignition_shadow.datetime') as md:
        md.now.return_value=datetime(2026,7,20,minute[0]+4,minute[1],
                                     tzinfo=timezone.utc)
        s.on_mover(sym,intraday_change_pct=chg,gap_pct=gap,price=10.35,
                   has_news=news,bar_ts_utc=None)
    assert s.drain(10.0), 'shadow worker failed to drain queue'

def _recs(tmp_path):
    f=list(tmp_path.glob('ignition_shadow_*.jsonl'))
    if not f: return []
    return [json.loads(l) for l in f[0].read_text().splitlines()]

class TestShadow:
    def test_news_trigger_journals_no_orders(self, tmp_path):
        s,a=_shadow(tmp_path)
        _fire(s,news=True)
        r=_recs(tmp_path)
        assert len(r)==1 and r[0]['verdict']=='SHADOW_TRIGGER'
        assert r[0]['catalyst']=='news'
        assert 'spread_bps' in r[0] and r[0]['hypo_position_usd']>0
        # zero orders: no submit-like method ever touched
        assert not [c for c in a.method_calls if 'submit' in str(c).lower()]

    def test_no_catalyst_parked_then_late_confirm(self, tmp_path):
        s,a=_shadow(tmp_path)
        # two wrappers of one underlying, arriving apart
        s._anchors={'IREX':'IREN','IREZ':'IREN'}
        _fire(s,sym='IREX',news=None,minute=(9,45))
        r=_recs(tmp_path)
        assert r[-1]['verdict']=='skip_no_catalyst'
        _fire(s,sym='IREZ',news=None,minute=(9,52))
        r=_recs(tmp_path)
        assert any(x.get('catalyst','').startswith('complex_late')
                   and x['symbol']=='IREX' for x in r)

    def test_true_open_gap_gate_in_finalize(self, tmp_path):
        """Scanner gap semantics are wrong for this gate — verified from
        bars: derived open gap >=5% -> ORB territory skip."""
        s,a=_shadow(tmp_path)
        bars=pd.DataFrame({'open':[9.8]+[10.0]*40,'high':[9.9]+[10.4]*40,
                           'low':[9.7]+[9.4]*40,'close':[9.85]+[10.2]*40,
                           'volume':[10000]*41})
        a.get_1min_bars.return_value=bars
        # price 10.35, scanner gap 12% -> prev_close ~9.24, open 9.8 -> open gap ~6%
        _fire(s,news=True)
        r=_recs(tmp_path)
        assert r[-1]['verdict']=='skip_gap_orb_territory'

    def test_eval_cap(self, tmp_path):
        s,a=_shadow(tmp_path,max_evals_per_day=1)
        _fire(s,sym='AAA1',news=True)
        _fire(s,sym='AAA2',news=True)
        r=_recs(tmp_path)
        assert any(x['verdict']=='skip_eval_cap' for x in r)

    def test_never_raises_into_scanner(self, tmp_path):
        s,a=_shadow(tmp_path)
        a.get_1min_bars.side_effect=RuntimeError('api dead')
        a.get_latest_quote.side_effect=RuntimeError('api dead')
        _fire(s,news=True)   # must not raise
        r=_recs(tmp_path)
        assert r[-1]['verdict']=='no_bars'

    def test_disabled_by_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv('IGNITION_SHADOW','0')
        s,_=_shadow(tmp_path)
        assert s.enabled is False

    def test_on_mover_enqueues_only_never_blocks(self, tmp_path):
        """Scanner-thread isolation: on_mover must not do API work —
        even with every API call hung, enqueue returns instantly and a
        full queue drops silently instead of raising."""
        import time
        s,a=_shadow(tmp_path)
        def _hang(*args,**kw):
            time.sleep(30)
        a.get_1min_bars.side_effect=_hang
        a.get_latest_quote.side_effect=_hang
        t0=time.monotonic()
        with patch('trading.ignition_shadow.datetime') as md:
            md.now.return_value=datetime(2026,7,20,13,50,
                                         tzinfo=timezone.utc)
            s.on_mover('HUNG',intraday_change_pct=15.0,gap_pct=12.0,
                       price=10.35,has_news=True,bar_ts_utc=None)
        assert time.monotonic()-t0 < 0.5   # enqueue-only, no API wait
        # full-queue drop path never raises
        import queue as q
        s2,_=_shadow(tmp_path)
        s2._queue=q.Queue(maxsize=1)
        s2._queue.put_nowait(('X',0,0,0,None,None,None))
        s2.on_mover('DROP',intraday_change_pct=15.0,gap_pct=12.0,
                    price=10.35,has_news=True,bar_ts_utc=None)

    def test_latency_journaled_from_bar_ts(self, tmp_path):
        s,a=_shadow(tmp_path)
        seen=datetime(2026,7,20,13,50,30,tzinfo=timezone.utc)
        with patch('trading.ignition_shadow.datetime') as md:
            md.now.return_value=seen
            s.on_mover('LATN',intraday_change_pct=15.0,gap_pct=12.0,
                       price=10.35,has_news=True,
                       bar_ts_utc=datetime(2026,7,20,13,49,45,
                                           tzinfo=timezone.utc))
        assert s.drain(10.0)
        r=_recs(tmp_path)
        assert r[-1]['latency_s']==45.0

    def test_module_has_no_order_code(self):
        import inspect
        import trading.ignition_shadow as m
        src=inspect.getsource(m)
        for word in ('submit_order','submit_stop','bracket','sell','buy_stop'):
            assert word not in src
