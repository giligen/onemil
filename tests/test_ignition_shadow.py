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
    ts=pd.date_range('2026-07-20 13:30','2026-07-20 14:10',freq='1min',
                     tz='UTC')[:41]
    bars=pd.DataFrame({'timestamp':ts,'open':[9.0]+[10.0]*40,
                       'high':[9.1]+[10.4]*40,'low':[8.9]+[9.4]*40,
                       'close':[9.05]+[10.2]*40,'volume':[10000]*41})
    a.get_1min_bars.return_value=bars
    a.get_premarket_news_multi.return_value={}   # default: known-newsless
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
                   has_news=news,price_ts_utc=None)
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
        assert any((x.get('catalyst') or '').startswith('complex_late')
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

    def test_sub_2_dollar_price_floor(self, tmp_path):
        """BT parity: research book has zero sub-$2 trades (CPHI 7/21
        fantasy-trigger incident). No journal, no eval burn."""
        s,a=_shadow(tmp_path)
        with patch('trading.ignition_shadow.datetime') as md:
            md.now.return_value=datetime(2026,7,20,13,50,
                                         tzinfo=timezone.utc)
            s.on_mover('PNNY',intraday_change_pct=18.0,gap_pct=15.0,
                       price=0.96,has_news=True,price_ts_utc=None)
        assert s.drain(10.0)
        assert _recs(tmp_path)==[]
        assert s._evals_today==0
        a.get_premarket_news_multi.assert_not_called()

    def test_participation_cap_bounds_position(self, tmp_path):
        """BT parity: position <= 15% of latest-bar $vol (CPHI's $906
        bar => $136 cap, not $19.7K)."""
        s,a=_shadow(tmp_path)
        bars=pd.DataFrame({'open':[9.0]+[10.0]*40,'high':[9.1]+[10.4]*40,
                           'low':[8.9]+[9.4]*40,'close':[9.05]+[10.2]*40,
                           'volume':[10000]*40+[50]})   # last bar: $510
        a.get_1min_bars.return_value=bars
        _fire(s,news=True)
        r=_recs(tmp_path)
        assert r[-1]['verdict']=='skip_illiquid'
        assert r[-1]['hypo_position_usd']<2000
        assert r[-1]['participation_cap_usd']==pytest.approx(
            0.15*50*10.2,abs=1)

    def test_chase_guard_skips_extended_entry(self, tmp_path):
        """BT parity (7/24 audit): ask > open*1.155 -> the BT harness
        refuses the entry (chase guard 5% past the +10% level)."""
        s,a=_shadow(tmp_path)
        # open 9.0 -> max entry 10.395; ask fixture 10.05 passes, so
        # raise the quote to force a chase violation
        a.get_latest_quote.return_value={'bid_price':10.60,
            'ask_price':10.65,'bid_size':5,'ask_size':7}
        _fire(s,news=True)
        r=_recs(tmp_path)
        assert r[-1]['verdict']=='skip_chase_guard'
        assert r[-1]['chase_ratio']==pytest.approx(10.65/9.0,abs=1e-3)

    def test_within_chase_bound_still_triggers(self, tmp_path):
        s,a=_shadow(tmp_path)   # ask 10.05 vs open 9.0 = 1.117x -> OK
        _fire(s,news=True)
        assert _recs(tmp_path)[-1]['verdict']=='SHADOW_TRIGGER'

    def test_disabled_by_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv('IGNITION_SHADOW','0')
        s,_=_shadow(tmp_path)
        assert s.enabled is False

    def test_worker_resolves_news_itself(self, tmp_path):
        """The scanner always passes has_news=None — the WORKER must
        resolve the news catalyst channel or it is dead (2026-07-19
        review finding: only complex-confirmation could ever fire)."""
        s,a=_shadow(tmp_path)
        a.get_premarket_news_multi.return_value={
            'IGNI':{'n_articles':3,'headline':'FDA approval'}}
        _fire(s,news=None)
        r=_recs(tmp_path)
        assert r[-1]['verdict']=='SHADOW_TRIGGER'
        assert r[-1]['catalyst']=='news'
        assert r[-1]['has_news'] is True
        assert 'FDA' in r[-1]['news_headline']
        a.get_premarket_news_multi.assert_called_once_with(['IGNI'])

    def test_news_fetch_failure_degrades_to_complex_only(self, tmp_path):
        s,a=_shadow(tmp_path)
        a.get_premarket_news_multi.side_effect=RuntimeError('gateway dead')
        _fire(s,news=None)
        r=_recs(tmp_path)
        assert r[-1]['verdict']=='skip_no_catalyst'
        assert r[-1]['has_news'] is None
        assert 'news_error' in r[-1]

    def test_known_newsless_parks_with_resolved_flag(self, tmp_path):
        s,a=_shadow(tmp_path)     # fixture: fetch returns {} = no news
        _fire(s,news=None)
        r=_recs(tmp_path)
        assert r[-1]['verdict']=='skip_no_catalyst'
        assert r[-1]['has_news'] is False

    def test_finalize_journals_actionable_minute(self, tmp_path):
        s,a=_shadow(tmp_path)
        _fire(s,news=True)
        r=_recs(tmp_path)
        assert r[-1]['verdict']=='SHADOW_TRIGGER'
        assert 'minute_final_et' in r[-1]

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
                       price=10.35,has_news=True,price_ts_utc=None)
        assert time.monotonic()-t0 < 0.5   # enqueue-only, no API wait
        # full-queue drop path never raises
        import queue as q
        s2,_=_shadow(tmp_path)
        s2._queue=q.Queue(maxsize=1)
        s2._queue.put_nowait(('X',0,0,0,None,None,None))
        s2.on_mover('DROP',intraday_change_pct=15.0,gap_pct=12.0,
                    price=10.35,has_news=True,price_ts_utc=None)

    def test_latency_journaled_from_bar_ts(self, tmp_path):
        s,a=_shadow(tmp_path)
        seen=datetime(2026,7,20,13,50,30,tzinfo=timezone.utc)
        with patch('trading.ignition_shadow.datetime') as md:
            md.now.return_value=seen
            s.on_mover('LATN',intraday_change_pct=15.0,gap_pct=12.0,
                       price=10.35,has_news=True,
                       price_ts_utc=datetime(2026,7,20,13,49,45,
                                           tzinfo=timezone.utc))
        assert s.drain(10.0)
        r=_recs(tmp_path)
        assert r[-1]['latency_s']==45.0

    def test_day_rollover_resets_state(self, tmp_path):
        """New ET day: seen/cohort/news caches reset — same symbol gets
        re-evaluated (fresh news fetch) into a NEW journal file."""
        s,a=_shadow(tmp_path)
        for day in (20,21):
            with patch('trading.ignition_shadow.datetime') as md:
                md.now.return_value=datetime(2026,7,day,13,50,
                                             tzinfo=timezone.utc)
                s.on_mover('ROLL',intraday_change_pct=15.0,gap_pct=12.0,
                           price=10.35,has_news=None,price_ts_utc=None)
            assert s.drain(10.0)
        files=sorted(tmp_path.glob('ignition_shadow_*.jsonl'))
        assert len(files)==2
        assert a.get_premarket_news_multi.call_count==2  # cache reset
        assert s._evals_today==1                          # day-2 count

    def test_worker_survives_malformed_item(self, tmp_path):
        s,a=_shadow(tmp_path)
        s._queue.put_nowait(('BAD',))   # unpack error inside worker
        assert s.drain(10.0)
        _fire(s,news=True)              # worker still alive + working
        r=_recs(tmp_path)
        assert r[-1]['verdict']=='SHADOW_TRIGGER'

    def test_eval_cap_never_burns_news_api(self, tmp_path):
        """The cap bounds ALL API work — a capped sighting must journal
        skip_eval_cap without a news fetch."""
        s,a=_shadow(tmp_path,max_evals_per_day=0)
        _fire(s,news=None)
        r=_recs(tmp_path)
        assert r[-1]['verdict']=='skip_eval_cap'
        a.get_premarket_news_multi.assert_not_called()
        a.get_1min_bars.assert_not_called()

    def test_no_late_confirm_after_cutoff(self, tmp_path):
        """A sibling sighted after 10:30 ET cannot retroactively confirm
        a parked symbol — the researched trigger window is closed."""
        s,a=_shadow(tmp_path)
        s._anchors={'LATX':'LATE','LATZ':'LATE'}
        _fire(s,sym='LATX',news=None,minute=(10,15))
        _fire(s,sym='LATZ',news=None,minute=(10,35))   # past 630 cutoff
        r=_recs(tmp_path)
        assert r[-1]['verdict']=='skip_no_catalyst'    # only LATX's park
        assert not any('complex_late' in str(x.get('catalyst','')) for x in r)

    def test_queue_full_drop_counted(self, tmp_path):
        import queue as q
        s,_=_shadow(tmp_path)
        s._queue=q.Queue(maxsize=1)
        s._queue.put_nowait(('X',0,0,0,None,None,None))
        for _ in range(3):
            s.on_mover('DROPME',intraday_change_pct=15.0,gap_pct=12.0,
                       price=10.35,has_news=True,price_ts_utc=None)
        assert s._dropped==3

    def test_module_has_no_order_code(self):
        import inspect
        import trading.ignition_shadow as m
        src=inspect.getsource(m)
        for word in ('submit_order','submit_stop','bracket','sell','buy_stop'):
            assert word not in src


class TestScannerHook:
    """The scanner->shadow seam: a regression here silently disconnects
    the shadow and a dead week looks like a quiet market."""

    def _scanner(self):
        from data_sources.news_provider import NewsProvider
        from persistence.database import Database
        from scanner.criteria import ScannerCriteria
        from scanner.realtime_scanner import RealtimeScanner
        alpaca=MagicMock(spec=AlpacaClient)
        news=MagicMock(spec=NewsProvider)
        news.has_interesting_news.return_value=(False,'')
        news.classify_news.return_value={'has_news':False,'catalyst':None,
                                         'headline':'','reason':''}
        db=MagicMock(spec=Database)
        sc=RealtimeScanner(alpaca_client=alpaca,news_provider=news,db=db,
                           criteria=ScannerCriteria(),verbose=False)
        return sc,alpaca

    def _wire_mover(self,sc,alpaca,trade_ts):
        import pytz
        from datetime import datetime as rdt
        sc._universe=[{'symbol':'MOMO','price_close':4.0,
                       'company_name':'Momo Co','float_shares':2_000_000}]
        trade={'price':5.0}          # +25% vs prev close -> mover
        if trade_ts is not None:
            trade['timestamp']=trade_ts
        alpaca.get_latest_trades.return_value={'MOMO':trade}
        now_et=rdt.now(pytz.timezone('US/Eastern'))
        bar_ts=now_et.replace(minute=(now_et.minute//15)*15,second=0,
                              microsecond=0)
        alpaca.get_current_bars.return_value={
            'MOMO':{'volume':100_000,'timestamp':bar_ts,
                    'high':5.0,'low':4.8}}
        bucket=f"{now_et.hour:02d}:{(now_et.minute//15)*15:02d}"
        sc._volume_profiles={'MOMO':{bucket:10_000}}

    def test_mover_reaches_shadow_with_trade_ts(self):
        sc,alpaca=self._scanner()
        ts=datetime(2026,7,20,13,49,45,tzinfo=timezone.utc)
        self._wire_mover(sc,alpaca,ts)
        sc.ignition_shadow=MagicMock(spec=IgnitionShadow)
        sc._run_intraday_cycle()
        sc.ignition_shadow.on_mover.assert_called_once()
        args,kw=sc.ignition_shadow.on_mover.call_args
        assert args[0]=='MOMO'
        assert kw['intraday_change_pct']==pytest.approx(25.0)
        assert kw['gap_pct']==pytest.approx(25.0)
        assert kw['price']==5.0
        assert kw['has_news'] is None
        assert kw['price_ts_utc']==ts     # trade ts, NOT 15-min bar ts

    def test_iso_string_trade_ts_parsed(self):
        """get_latest_trades serializes timestamp via .isoformat() — the
        hook must parse the STRING (live 7/20: latency_s was empty all
        morning because the seam test mocked a datetime object)."""
        sc,alpaca=self._scanner()
        self._wire_mover(sc,alpaca,'2026-07-20T13:49:45+00:00')
        sc.ignition_shadow=MagicMock(spec=IgnitionShadow)
        sc._run_intraday_cycle()
        _,kw=sc.ignition_shadow.on_mover.call_args
        assert kw['price_ts_utc']==datetime(2026,7,20,13,49,45,
                                            tzinfo=timezone.utc)

    def test_garbage_trade_ts_passes_none(self):
        sc,alpaca=self._scanner()
        self._wire_mover(sc,alpaca,'not-a-timestamp')
        sc.ignition_shadow=MagicMock(spec=IgnitionShadow)
        sc._run_intraday_cycle()
        _,kw=sc.ignition_shadow.on_mover.call_args
        assert kw['price_ts_utc'] is None

    def test_naive_trade_ts_coerced_to_utc(self):
        sc,alpaca=self._scanner()
        self._wire_mover(sc,alpaca,datetime(2026,7,20,13,49,45))
        sc.ignition_shadow=MagicMock(spec=IgnitionShadow)
        sc._run_intraday_cycle()
        _,kw=sc.ignition_shadow.on_mover.call_args
        assert kw['price_ts_utc'].tzinfo is not None

    def test_missing_trade_ts_passes_none(self):
        sc,alpaca=self._scanner()
        self._wire_mover(sc,alpaca,None)
        sc.ignition_shadow=MagicMock(spec=IgnitionShadow)
        sc._run_intraday_cycle()
        _,kw=sc.ignition_shadow.on_mover.call_args
        assert kw['price_ts_utc'] is None

    def _make_qualifiable(self,sc):
        # full qualification (news catalyst) so save_scan_result is the
        # cycle-completed marker downstream of the shadow hook
        sc.news.has_interesting_news.return_value=(True,'Big news')
        sc.news.classify_news.return_value={'has_news':True,'catalyst':True,
                                            'headline':'Big news',
                                            'reason':'test'}

    def test_raising_shadow_never_breaks_cycle(self):
        sc,alpaca=self._scanner()
        self._wire_mover(sc,alpaca,None)
        self._make_qualifiable(sc)
        sc.ignition_shadow=MagicMock(spec=IgnitionShadow)
        sc.ignition_shadow.on_mover.side_effect=RuntimeError('shadow bug')
        sc._run_intraday_cycle()      # must not raise
        sc.db.save_scan_result.assert_called()   # work AFTER hook ran

    def test_none_shadow_cycle_still_works(self):
        sc,alpaca=self._scanner()
        self._wire_mover(sc,alpaca,None)
        self._make_qualifiable(sc)
        sc.ignition_shadow=None
        sc._run_intraday_cycle()
        sc.db.save_scan_result.assert_called()
