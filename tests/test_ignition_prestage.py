"""Ignition pre-staged entry subsystem (2026-08-22 build).

Coverage map — every pre-mortem item (design §A-F), every P0 resolution
(design §3b), and the automatable subset of the review's 12 rehearsal
drills has a named test here. See docs/ignition_prestage_design_aug2026.md
and docs/ignition_prestage_review_aug2026.md.
"""
from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.exit_reasons import ExitReason
from trading.ignition_prestage import (
    MAX_CLIENT_ORDER_ID_LEN,
    PRESTAGE_ID_PREFIX,
    STATE_CANCEL_CONFIRMED,
    STATE_CANCEL_PENDING,
    STATE_FILLED,
    STATE_REJECTED,
    STATE_STAGED,
    PrestageManager,
    check_set_parity,
    round_price_for_tick,
    stage_client_order_id,
)
from trading import ignition_rules as _rules
from trading.stop_monitor import StopMonitor
from trading.order_stream import OrderStreamWatcher

_ET = ZoneInfo('America/New_York')
DAY = '2026-08-24'                       # Monday, EDT season
DAY_COMPACT = '20260824'
EST_DAY = '2026-01-15'                   # EST season (§F20 DST scar)


def _et(day: str, hh: int, mm: int, ss: int = 0) -> datetime:
    y, mo, d = (int(x) for x in day.split('-'))
    return datetime(y, mo, d, hh, mm, ss, tzinfo=_ET)


def _bars(day_open=10.0, cross_high=11.2, pre_low=10.3, n=41,
          day=DAY, vol=10000):
    """41 bars from the 9:30 ET open (UTC-4 in August). Bars 0-4 are the
    pre-window (m 570-574); bars from m=575 carry `cross_high`."""
    utc_open = {'2026-08-24': '2026-08-24 13:30',
                '2026-01-15': '2026-01-15 14:30'}[day]
    ts = pd.date_range(utc_open, periods=n, freq='1min', tz='UTC')
    opens = [day_open] + [day_open + 0.2] * (n - 1)
    highs = [day_open + 0.4] * 5 + [cross_high] * (n - 5)
    lows = [pre_low] * n
    closes = [day_open + 0.1] * n
    return pd.DataFrame({'timestamp': ts, 'open': opens, 'high': highs,
                         'low': lows, 'close': closes,
                         'volume': [vol] * n})


def _alpaca():
    a = MagicMock(spec=AlpacaClient)
    a.get_account_info.return_value = {
        'equity': 73000.0, 'buying_power': 292000.0, 'cash': 73000.0,
        'daytrade_count': 0, 'pattern_day_trader': True,
        'multiplier': 4.0}
    a.submit_stop_limit_order.return_value = {'id': 'ord-1',
                                              'status': 'accepted'}
    a.submit_stop_sell_order.return_value = {'id': 'sl-1',
                                             'status': 'accepted'}
    a.cancel_order.return_value = True
    a.get_order.return_value = {'status': 'canceled', 'filled_qty': 0}
    a.get_open_orders.return_value = []
    a.get_open_positions.return_value = []
    a.get_1min_bars.return_value = _bars()
    return a


def _cfg(tmp_path, **over):
    c = {'enabled': True, 'shadow': False, 'risk_usd': 50.0,
         'cap_bps': 300.0, 'stop_offset_bps': 30.0, 'heap_k': 400,
         'promote_rank_slack': 2, 'demote_rank_slack': 3,
         'promote_distance_pct': 20.0, 'demote_distance_pct': 25.0,
         'promote_consecutive': 1, 'ops_per_min': 100,
         'bp_frac': 0.25, 'bp_abs_usd': 30000.0,
         'stage_start_min': 575, 'cancel_all_min': 780,
         'gap_through_cancel_min': 60, 'watchdog_stale_s': 60.0,
         'max_staged_fills': 10, 'pdt_equity_min': 25000.0,
         'log_dir': str(tmp_path)}
    c.update(over)
    return c


def _mgr(tmp_path, monkeypatch=None, **over):
    if monkeypatch is not None:
        monkeypatch.delenv('IGNITION_PRESTAGE', raising=False)
    a = _alpaca()
    db = MagicMock(spec=Database)
    db.save_trade.return_value = 7
    sm = MagicMock(spec=StopMonitor)
    sm.add_watch.return_value = True
    sm.force_exit.return_value = True
    osw = MagicMock(spec=OrderStreamWatcher)
    osw.snapshot_by_client_prefix.return_value = {}
    m = PrestageManager(a, db=db, stop_monitor=sm, order_stream=osw,
                        notifier=None, cfg=_cfg(tmp_path, **over))
    return m, a, db, sm, osw


def _cand(sym='PSTG', day=DAY, day_open=10.0, price=10.5, news=True,
          anchor=None, stop=10.3):
    return {'symbol': sym, 'day': day, 'day_open': day_open,
            'price': price, 'has_news': news, 'anchor': anchor,
            '_stop': stop}


def _feed_and_tick(m, recs, when, ticks=1):
    """Intake candidates then run `ticks` scheduler cycles at `when`."""
    for r in recs:
        m.on_candidate(r)
    for _ in range(ticks):
        m.process_tick(now_et=when)


def _fill_status(sym, qty=68, price=11.05, status='filled',
                 day_compact=DAY_COMPACT,
                 filled_at='2026-08-24T13:41:20+00:00'):
    coid = f"{PRESTAGE_ID_PREFIX}{day_compact}-{sym}"
    return {coid: {'id': f'ord-{sym}', 'client_order_id': coid,
                   'symbol': sym, 'status': status,
                   'filled_qty': qty, 'filled_avg_price': price,
                   'filled_at': filled_at}}


# ===========================================================================
# helpers: rounding (§F17), id scheme (§F18), parity checker (§B8)
# ===========================================================================
class TestHelpers:
    def test_round_price_tick_property(self):
        """Property test across the price grid: penny at >=$1, 4dp
        below $1, idempotent (§F17 — WILL fire day one otherwise)."""
        import random
        rng = random.Random(17)
        for _ in range(500):
            p = rng.uniform(0.01, 50.0)
            r = round_price_for_tick(p)
            if p >= 1.0:
                assert abs(r * 100 - round(r * 100)) < 1e-6
            else:
                assert abs(r * 10000 - round(r * 10000)) < 1e-6
            assert round_price_for_tick(r) == r          # idempotent
            assert abs(r - p) <= (0.005 if p >= 1.0 else 0.00005)

    def test_round_price_sub_dollar_keeps_4_decimals(self):
        assert round_price_for_tick(0.4567) == 0.4567
        assert round_price_for_tick(0.45678) == 0.4568
        assert round_price_for_tick(1.4567) == 1.46

    def test_client_order_id_scheme(self):
        coid = stage_client_order_id('20260824', 'CRWU')
        assert coid == 'ign-stage-20260824-CRWU'
        assert len(coid) <= MAX_CLIENT_ORDER_ID_LEN

    def test_client_order_id_length_validated(self):
        with pytest.raises(ValueError):
            stage_client_order_id('20260824', 'X' * 40)

    def test_client_order_id_charset_validated(self):
        with pytest.raises(ValueError):
            stage_client_order_id('20260824', 'BAD SYM')

    def test_parity_checker_green_synthetic_day(self):
        """Drill 11: staged + chased + missed + staged-fill-no-trigger,
        all with explicit reasons => green."""
        res = check_set_parity(
            shadow_triggers={'STGD', 'CHSD', 'MISS'},
            staged_fills={'STGD', 'NOTRIG'},
            chase_entries={'CHSD'},
            explicit_reasons={'MISS': 'stage_skip_bp_budget',
                              'NOTRIG': 'stage_reject_structure'})
        assert res['ok'], res

    def test_parity_checker_red_on_uncovered_trigger(self):
        res = check_set_parity({'A', 'B'}, {'A'}, set(), {})
        assert not res['ok'] and res['missing'] == ['B']

    def test_parity_checker_red_on_unexplained_fill(self):
        res = check_set_parity({'A'}, {'A', 'GHOST'}, set(), {})
        assert not res['ok'] and res['unexplained_fills'] == ['GHOST']


# ===========================================================================
# scheduler: windows (§F20), P0-6, hysteresis (§E15), news/sibling (P0-2)
# ===========================================================================
class TestScheduler:
    def test_never_stage_before_935(self, tmp_path, monkeypatch):
        m, a, *_ = _mgr(tmp_path, monkeypatch)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 34))
        a.submit_stop_limit_order.assert_not_called()
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 35))
        a.submit_stop_limit_order.assert_called_once()

    def test_stage_order_shape(self, tmp_path, monkeypatch):
        """stop=level*1.003, limit=level*(1+cap), DAY TIF implied,
        client_order_id scheme, qty from $risk at staging estimate."""
        m, a, *_ = _mgr(tmp_path, monkeypatch)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        kw = a.submit_stop_limit_order.call_args.kwargs
        assert kw['symbol'] == 'PSTG' and kw['side'] == 'buy'
        assert kw['stop_price'] == round(11.0 * 1.003, 2)      # 11.03
        assert kw['limit_price'] == round(11.0 * 1.03, 2)      # 11.33
        assert kw['client_order_id'] == f'ign-stage-{DAY_COMPACT}-PSTG'
        assert kw['tick_rounding'] is True
        # qty = floor(50 / (11.03 - 10.3))
        assert kw['qty'] == int(50.0 / (11.03 - 10.3))

    def test_sub_dollar_prices_tick_rounded(self, tmp_path, monkeypatch):
        """Drill 10 (unit half): $0.xx name stages with 4-decimal
        stop/limit — no broker reject."""
        m, a, *_ = _mgr(tmp_path, monkeypatch)
        _feed_and_tick(m, [_cand(sym='PNNY', day_open=0.50, price=0.52,
                                 stop=0.50)], _et(DAY, 9, 40))
        kw = a.submit_stop_limit_order.call_args.kwargs
        assert kw['stop_price'] == round(0.55 * 1.003, 4)
        assert kw['limit_price'] == round(0.55 * 1.03, 4)

    def test_cancel_all_at_1300(self, tmp_path, monkeypatch):
        m, a, *_ = _mgr(tmp_path, monkeypatch)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        assert m._stages['PSTG']['state'] == STATE_STAGED
        m.process_tick(now_et=_et(DAY, 13, 0))
        assert m._stages['PSTG']['state'] == STATE_CANCEL_CONFIRMED
        a.cancel_order.assert_called()

    def test_no_stage_after_1300(self, tmp_path, monkeypatch):
        m, a, *_ = _mgr(tmp_path, monkeypatch)
        _feed_and_tick(m, [_cand()], _et(DAY, 13, 5))
        a.submit_stop_limit_order.assert_not_called()

    def test_est_season_window_edges(self, tmp_path, monkeypatch):
        """§F20 synthetic EST-season day: window math is ET wall-clock,
        not UTC offsets — 9:35 EST stages exactly like 9:35 EDT."""
        m, a, *_ = _mgr(tmp_path, monkeypatch)
        rec = _cand(day=EST_DAY)
        _feed_and_tick(m, [rec], _et(EST_DAY, 9, 34, 59))
        a.submit_stop_limit_order.assert_not_called()
        _feed_and_tick(m, [rec], _et(EST_DAY, 9, 35))
        kw = a.submit_stop_limit_order.call_args.kwargs
        assert kw['client_order_id'] == 'ign-stage-20260115-PSTG'

    def test_already_crossed_never_staged(self, tmp_path, monkeypatch):
        """P0-6: last >= level => buy-stop would reject; routes to chase
        with an explicit parity reason."""
        m, a, *_ = _mgr(tmp_path, monkeypatch)
        _feed_and_tick(m, [_cand(price=11.5)], _et(DAY, 9, 40))
        a.submit_stop_limit_order.assert_not_called()
        assert m._parity_explicit['PSTG'] == 'stage_skip_already_crossed'
        assert m.chase_allowed('PSTG')

    def test_news_gate_at_935(self, tmp_path, monkeypatch):
        """P0-2: staged set = news-eligible names only (complex leg is
        reactive)."""
        m, a, *_ = _mgr(tmp_path, monkeypatch)
        _feed_and_tick(m, [_cand(sym='NONEWS', news=False),
                           _cand(sym='NEWSY', news=True)],
                       _et(DAY, 9, 40))
        staged = [c.kwargs['symbol'] for c in
                  a.submit_stop_limit_order.call_args_list]
        assert staged == ['NEWSY']

    def test_reactive_sibling_staging(self, tmp_path, monkeypatch):
        """P0-2: first anchor trigger => same-anchor siblings still
        below level become stage-eligible regardless of news."""
        m, a, *_ = _mgr(tmp_path, monkeypatch)
        _feed_and_tick(m, [_cand(sym='CRWU', news=False, anchor='CRWV'),
                           _cand(sym='CRWG', news=False, anchor='CRWV')],
                       _et(DAY, 9, 40))
        a.submit_stop_limit_order.assert_not_called()
        m.notify_trigger({'symbol': 'CRWU', 'anchor': 'CRWV',
                          'day': DAY})
        m.process_tick(now_et=_et(DAY, 9, 41))
        staged = [c.kwargs['symbol'] for c in
                  a.submit_stop_limit_order.call_args_list]
        assert staged == ['CRWG']

    def test_hysteresis_promote_needs_consecutive(self, tmp_path,
                                                  monkeypatch):
        """§E15 dual-threshold hysteresis: promote only after N
        consecutive qualifying cycles."""
        m, a, *_ = _mgr(tmp_path, monkeypatch, promote_consecutive=2)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        a.submit_stop_limit_order.assert_not_called()   # streak=1
        m.process_tick(now_et=_et(DAY, 9, 41))
        a.submit_stop_limit_order.assert_called_once()  # streak=2

    def test_hysteresis_distance_gate_resets_streak(self, tmp_path,
                                                    monkeypatch):
        m, a, *_ = _mgr(tmp_path, monkeypatch, promote_consecutive=2,
                        promote_distance_pct=3.0)
        # distance (11.0-10.5)/11.0 = 4.5% > D_in 3.0 => never promotes
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40), ticks=3)
        a.submit_stop_limit_order.assert_not_called()

    def test_demote_on_distance_out(self, tmp_path, monkeypatch):
        m, a, *_ = _mgr(tmp_path, monkeypatch,
                        demote_distance_pct=8.0)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        assert m._stages['PSTG']['state'] == STATE_STAGED
        m.on_price('PSTG', 9.0)          # distance 18% > D_out 8%
        m.process_tick(now_et=_et(DAY, 9, 41))
        assert m._stages['PSTG']['state'] == STATE_CANCEL_CONFIRMED

    def test_one_stage_per_symbol_day_ever(self, tmp_path, monkeypatch):
        """§A5 double-stage idempotency drill: a demoted/cancelled name
        is NEVER re-staged the same day."""
        m, a, *_ = _mgr(tmp_path, monkeypatch, demote_distance_pct=8.0)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        m.on_price('PSTG', 9.0)
        m.process_tick(now_et=_et(DAY, 9, 41))       # demote-cancel
        m.on_price('PSTG', 10.5)                     # back in range
        m.process_tick(now_et=_et(DAY, 9, 42), )
        m.process_tick(now_et=_et(DAY, 9, 43))
        assert a.submit_stop_limit_order.call_count == 1

    def test_no_day_open_logged_as_explicit_skip(self, tmp_path,
                                                 monkeypatch):
        m, a, *_ = _mgr(tmp_path, monkeypatch)
        m.on_candidate({'symbol': 'THIN', 'day': DAY, 'day_open': None,
                        'price': 3.0})
        assert m._parity_explicit['THIN'] == 'stage_skip_no_day_open'


# ===========================================================================
# budgets: rate (§E15/16), BP watermark (§C9), PDT (§C11)
# ===========================================================================
class TestBudgets:
    def test_ops_per_min_budget_defers(self, tmp_path, monkeypatch):
        m, a, *_ = _mgr(tmp_path, monkeypatch, ops_per_min=3)
        recs = [_cand(sym=f'SYM{i}') for i in range(8)]
        _feed_and_tick(m, recs, _et(DAY, 9, 40))
        assert a.submit_stop_limit_order.call_count == 3
        assert m.telemetry.churn_limiter_activations >= 1

    def test_bp_watermark_never_exceeds_budget(self, tmp_path,
                                               monkeypatch):
        m, a, *_ = _mgr(tmp_path, monkeypatch, bp_abs_usd=2000.0)
        recs = [_cand(sym=f'SYM{i}') for i in range(10)]
        _feed_and_tick(m, recs, _et(DAY, 9, 40))
        assert m._bp_reserved() <= 2000.0
        assert m.telemetry.bp_high_watermark_usd <= 2000.0
        assert any(v == 'stage_skip_bp_budget'
                   for v in m._parity_explicit.values())

    def test_bp_80pct_alert_fires(self, tmp_path, monkeypatch):
        m, a, *_ = _mgr(tmp_path, monkeypatch, bp_abs_usd=2000.0)
        notifier = MagicMock()
        m.notifier = notifier
        recs = [_cand(sym=f'SYM{i}') for i in range(10)]
        _feed_and_tick(m, recs, _et(DAY, 9, 40))
        assert any('80%' in str(c) for c in
                   notifier.send_message_sync.call_args_list)

    def test_budgeter_cluster_day_replay_2026_08_03(self, tmp_path,
                                                    monkeypatch):
        """Drill (§C9 test): replay the 8/3 cluster day (65 names,
        research/ignition_capcheck/trades_2026.csv) through the
        budgeter — envelope never exceeded, fallback (explicit parity
        reasons) engaged for every pruned name, watermark recorded."""
        rows = []
        csv_path = Path(__file__).resolve().parent.parent / 'research' \
            / 'ignition_capcheck' / 'trades_2026.csv'
        with open(csv_path, newline='') as fh:
            for r in csv.DictReader(fh):
                if r['day'] == '2026-08-03':
                    rows.append(r)
        assert len(rows) == 65
        m, a, *_ = _mgr(tmp_path, monkeypatch, bp_abs_usd=10000.0)
        recs = [_cand(sym=r['symbol'], day='2026-08-03',
                      day_open=float(r['day_open']),
                      price=float(r['level']) * 0.98,
                      stop=float(r['stop']))
                for r in rows]
        _feed_and_tick(m, recs, _et('2026-08-03', 9, 40), ticks=2)
        staged = {c.kwargs['symbol'] for c in
                  a.submit_stop_limit_order.call_args_list}
        assert staged, 'budgeter staged nothing on the cluster day'
        assert m._bp_reserved() <= 10000.0
        assert m.telemetry.bp_high_watermark_usd <= 10000.0
        pruned = {s for s, v in m._parity_explicit.items()
                  if v == 'stage_skip_bp_budget'}
        assert pruned, 'expected BP pruning on the cluster day'
        # every 8/3 name is either staged or explicitly pruned — the
        # take-all coverage invariant (pruned names fall to chase)
        assert staged | pruned == {r['symbol'] for r in rows}

    def test_pdt_guard_halves_staging_depth(self, tmp_path, monkeypatch):
        """§C11: equity < $25K halves the staging depth."""
        m, a, *_ = _mgr(tmp_path, monkeypatch, heap_k=4,
                        promote_rank_slack=0)
        a.get_account_info.return_value = {
            'equity': 20000.0, 'buying_power': 80000.0}
        recs = [_cand(sym=f'SYM{i}', price=10.5 + i * 0.01)
                for i in range(6)]
        _feed_and_tick(m, recs, _et(DAY, 9, 40), ticks=2)
        # effective K = 2 => at most 2 staged
        assert a.submit_stop_limit_order.call_count <= 2

    def test_account_fetch_failure_falls_back_to_abs_cap(self, tmp_path,
                                                         monkeypatch,
                                                         caplog):
        m, a, *_ = _mgr(tmp_path, monkeypatch)
        a.get_account_info.side_effect = RuntimeError('api dead')
        with caplog.at_level('WARNING'):
            _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        assert 'account fetch failed' in caplog.text
        a.submit_stop_limit_order.assert_called_once()   # abs cap holds


# ===========================================================================
# state machine (P0-3) + cancel/fill race (§A3)
# ===========================================================================
class TestStateMachine:
    def _staged(self, tmp_path, monkeypatch, **over):
        m, a, db, sm, osw = _mgr(tmp_path, monkeypatch, **over)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        assert m._stages['PSTG']['state'] == STATE_STAGED
        return m, a, db, sm, osw

    def test_chase_blocked_while_staged(self, tmp_path, monkeypatch):
        m, *_ = self._staged(tmp_path, monkeypatch)
        assert not m.chase_allowed('PSTG')

    def test_chase_allowed_when_never_staged(self, tmp_path,
                                             monkeypatch):
        m, *_ = _mgr(tmp_path, monkeypatch)
        assert m.chase_allowed('NEVR')

    def test_resolve_for_chase_proves_cancel_at_broker(self, tmp_path,
                                                       monkeypatch):
        m, a, *_ = self._staged(tmp_path, monkeypatch)
        a.get_order.return_value = {'status': 'canceled',
                                    'filled_qty': 0}
        assert m.resolve_for_chase('PSTG') == 'chase_ok'
        assert m._stages['PSTG']['state'] == STATE_CANCEL_CONFIRMED
        assert m.chase_allowed('PSTG')

    def test_cancel_reject_means_filled_adopts(self, tmp_path,
                                               monkeypatch):
        """§A3 + drill 3: cancel fails, poll says FILLED => adopt
        (watch + DB), never drop tracking on an assumed cancel."""
        m, a, db, sm, _ = self._staged(tmp_path, monkeypatch)
        a.cancel_order.return_value = False   # 422 not cancelable
        a.get_order.return_value = {'status': 'filled',
                                    'filled_qty': 68,
                                    'filled_avg_price': 11.05}
        assert m.resolve_for_chase('PSTG') == 'adopted'
        assert m._stages['PSTG']['state'] == STATE_FILLED
        sm.add_watch.assert_called_once()
        db.save_trade.assert_called_once()

    def test_resolve_blocked_on_broker_error(self, tmp_path,
                                             monkeypatch):
        """P0-3 fail-closed: disposition unprovable => chase blocked
        (a double position is strictly worse than a missed chase)."""
        m, a, *_ = self._staged(tmp_path, monkeypatch)
        a.cancel_order.side_effect = RuntimeError('timeout')
        a.get_order.side_effect = RuntimeError('timeout')
        assert m.resolve_for_chase('PSTG') == 'blocked'
        assert not m.chase_allowed('PSTG')

    def test_illegal_transition_logged_not_applied(self, tmp_path,
                                                   monkeypatch, caplog):
        m, *_ = self._staged(tmp_path, monkeypatch)
        m._transition('PSTG', STATE_CANCEL_CONFIRMED, 'ok-from-staged')
        with caplog.at_level('ERROR'):
            assert not m._transition('PSTG', STATE_FILLED, 'bad')
        assert 'ILLEGAL transition' in caplog.text
        assert m._stages['PSTG']['state'] == STATE_CANCEL_CONFIRMED

    def test_broker_submit_reject_routes_to_chase(self, tmp_path,
                                                  monkeypatch):
        """P0-6 rollout-crossing class: a broker reject at place time
        leaves the symbol chase-eligible with an explicit reason."""
        m, a, *_ = _mgr(tmp_path, monkeypatch)
        a.submit_stop_limit_order.side_effect = RuntimeError(
            '422 stop price must be above current')
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        assert m._stages['PSTG']['state'] == STATE_REJECTED
        assert m.chase_allowed('PSTG')
        assert m._parity_explicit['PSTG'] == 'stage_submit_rejected'


# ===========================================================================
# fills + adoption (P0-1/P0-4/P0-5, §A4, §B7, P1-4, P1-6, P1-7)
# ===========================================================================
class TestFillAdoption:
    def _filled(self, tmp_path, monkeypatch, fill_kw=None, **over):
        m, a, db, sm, osw = _mgr(tmp_path, monkeypatch, **over)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        osw.snapshot_by_client_prefix.return_value = _fill_status(
            'PSTG', **(fill_kw or {}))
        m.process_tick(now_et=_et(DAY, 9, 42))
        return m, a, db, sm, osw

    def test_db_row_created_at_fill_only(self, tmp_path, monkeypatch):
        """P1-4: no trades row at stage time; row exists after fill with
        path=staged pattern_data."""
        m, a, db, sm, osw = _mgr(tmp_path, monkeypatch)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        db.save_trade.assert_not_called()
        osw.snapshot_by_client_prefix.return_value = _fill_status('PSTG')
        m.process_tick(now_et=_et(DAY, 9, 42))
        db.save_trade.assert_called_once()
        rec = db.save_trade.call_args.args[0]
        pd_json = json.loads(rec['pattern_data'])
        assert pd_json['path'] == 'staged'
        assert pd_json['parity_reason'] == 'staged_book'
        assert pd_json['level'] == pytest.approx(11.0)
        assert pd_json['rank_at_stage'] == 1
        assert pd_json['fill_class'] in ('clean', 'gap_into')

    def test_structure_pass_watch_uses_fill_derived_stop(self, tmp_path,
                                                         monkeypatch):
        """P0-5: stop/R recomputed at fill from actual bars via the
        shared helper — never staging-time estimates."""
        m, a, db, sm, osw = self._filled(tmp_path, monkeypatch)
        kw = sm.add_watch.call_args.kwargs
        assert kw['strategy'] == 'ignition'
        assert kw['entry_price'] == 11.05
        # stop = min(pre-30min low 10.3, fill*0.99) = 10.3
        assert kw['stop_price'] == pytest.approx(10.3)
        assert kw['lock_arm_at_r'] == _rules.ARM_R
        assert kw['lock_stop_r'] == _rules.LOCK_R
        assert kw['lock_r_unit'] == pytest.approx(11.05 - 10.3)

    def test_skip_exits_keyed_to_broker_fill_minute(self, tmp_path,
                                                    monkeypatch):
        """P0-5: skip window ends at the FILL minute's end (broker
        filled_at, not confirmation wall-clock)."""
        m, a, db, sm, osw = self._filled(tmp_path, monkeypatch)
        kw = sm.add_watch.call_args.kwargs
        fill_ts = datetime(2026, 8, 24, 13, 41, 20,
                           tzinfo=timezone.utc).timestamp()
        assert kw['skip_exits_until_ts'] == (int(fill_ts // 60) + 1) * 60

    def test_dead_man_sl_placed_at_adopt(self, tmp_path, monkeypatch):
        """P0-4: the fill->watch gap is covered by a broker-side SL
        placed BEFORE the (slow) structure validation."""
        m, a, db, sm, osw = self._filled(tmp_path, monkeypatch)
        a.submit_stop_sell_order.assert_called_once()
        args = a.submit_stop_sell_order.call_args.args
        assert args[0] == 'PSTG' and args[1] == 68

    def test_structure_fail_disposition_exit(self, tmp_path,
                                             monkeypatch):
        """P0-1: at-fill structure reject => immediate force_exit with
        stage_reject_structure + DB parity reason + scratch counter."""
        m, a, db, sm, osw = _mgr(tmp_path, monkeypatch)
        a.get_1min_bars.return_value = _bars(cross_high=10.4)  # no cross
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        osw.snapshot_by_client_prefix.return_value = _fill_status('PSTG')
        m.process_tick(now_et=_et(DAY, 9, 42))
        sm.force_exit.assert_called_once()
        assert sm.force_exit.call_args.kwargs['reason'] == \
            ExitReason.STAGE_REJECT_STRUCTURE.value
        rec = db.save_trade.call_args.args[0]
        pd_json = json.loads(rec['pattern_data'])
        assert pd_json['parity_reason'] == 'stage_reject_structure'
        assert pd_json['structure_reject'] == 'stage_fill_no_trigger'
        assert m.telemetry.scratch_count == 1
        assert m._parity_explicit['PSTG'] == 'stage_reject_structure'

    def test_bars_fetch_failure_treated_as_structure_reject(
            self, tmp_path, monkeypatch, caplog):
        """No bars => cannot validate => disposition exit, never a
        silently-held unvalidated position."""
        m, a, db, sm, osw = _mgr(tmp_path, monkeypatch)
        a.get_1min_bars.side_effect = RuntimeError('api dead')
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        osw.snapshot_by_client_prefix.return_value = _fill_status('PSTG')
        with caplog.at_level('ERROR'):
            m.process_tick(now_et=_et(DAY, 9, 42))
        sm.force_exit.assert_called_once()
        assert 'at-fill bars/gates failed' in caplog.text

    def test_partial_fill_cancels_remainder(self, tmp_path, monkeypatch):
        """§A4: partial fill => watch/DB on filled_qty, remainder
        cancelled immediately."""
        m, a, db, sm, osw = self._filled(
            tmp_path, monkeypatch,
            fill_kw={'qty': 30, 'status': 'partially_filled'})
        a.cancel_order.assert_called()      # remainder cancel
        assert sm.add_watch.call_args.kwargs['shares'] == 30
        rec = db.save_trade.call_args.args[0]
        assert rec['shares'] == 30

    def test_fill_without_trigger_flagged(self, tmp_path, monkeypatch):
        """§B7: staged fill with no shadow trigger raises the telemetry
        flag (odd-print / river class)."""
        m, *_ = self._filled(tmp_path, monkeypatch)
        assert m.telemetry.fills_without_trigger == 1

    def test_fill_with_trigger_not_flagged(self, tmp_path, monkeypatch):
        m, a, db, sm, osw = _mgr(tmp_path, monkeypatch)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        m.notify_trigger({'symbol': 'PSTG', 'day': DAY})
        osw.snapshot_by_client_prefix.return_value = _fill_status('PSTG')
        m.process_tick(now_et=_et(DAY, 9, 42))
        assert m.telemetry.fills_without_trigger == 0

    def test_fill_quality_line_path_staged(self, tmp_path, monkeypatch,
                                           caplog):
        """P1-7: staged fills log path=staged with fill-vs-level."""
        with caplog.at_level('INFO'):
            self._filled(tmp_path, monkeypatch)
        assert 'FILL QUALITY PSTG: path=staged' in caplog.text
        assert 'fill_vs_level=' in caplog.text

    def test_double_adoption_is_noop(self, tmp_path, monkeypatch):
        m, a, db, sm, osw = self._filled(tmp_path, monkeypatch)
        m.process_tick(now_et=_et(DAY, 9, 43))   # same fill status again
        db.save_trade.assert_called_once()
        sm.add_watch.assert_called_once()

    def test_max_staged_fills_breach_sweeps_rest(self, tmp_path,
                                                 monkeypatch):
        """P1-6: staged-fill count at cap => remaining stages swept."""
        m, a, db, sm, osw = _mgr(tmp_path, monkeypatch,
                                 max_staged_fills=1)
        _feed_and_tick(m, [_cand(sym='FILL1'), _cand(sym='REST1')],
                       _et(DAY, 9, 40))
        assert m._stages['REST1']['state'] == STATE_STAGED
        osw.snapshot_by_client_prefix.return_value = _fill_status('FILL1')
        m.process_tick(now_et=_et(DAY, 9, 42))
        assert m._stages['FILL1']['state'] == STATE_FILLED
        assert m._stages['REST1']['state'] == STATE_CANCEL_CONFIRMED

    def test_adoption_never_checks_kill(self, tmp_path, monkeypatch):
        """P1-6 kill-race semantics: a fill racing the kill sweep is
        adopted, managed, counted — never dropped or retro-vetoed."""
        m, a, db, sm, osw = _mgr(tmp_path, monkeypatch)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        # the fill races the kill sweep itself: cancel-reject + poll
        # says FILLED while the sweep runs
        a.cancel_order.return_value = False
        a.get_order.return_value = {'status': 'filled', 'filled_qty': 68,
                                    'filled_avg_price': 11.05}
        m.notify_kill('daily')
        assert m._stages['PSTG']['state'] == STATE_FILLED
        sm.add_watch.assert_called()
        db.save_trade.assert_called_once()

    def test_watch_collision_alerts_loudly(self, tmp_path, monkeypatch,
                                           caplog):
        """P1-2 consumer side: add_watch returning False (cross-strategy
        collision) is surfaced as ERROR + telegram, not swallowed."""
        m, a, db, sm, osw = _mgr(tmp_path, monkeypatch)
        sm.add_watch.return_value = False
        notifier = MagicMock()
        m.notifier = notifier
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        osw.snapshot_by_client_prefix.return_value = _fill_status('PSTG')
        with caplog.at_level('ERROR'):
            m.process_tick(now_et=_et(DAY, 9, 42))
        assert 'collision' in caplog.text
        assert notifier.send_message_sync.called


# ===========================================================================
# sweeps: kill (§C10), watchdog (§D12), gap-through (§B6/P1-5),
#         shutdown (§A2), boot (§A1)
# ===========================================================================
class TestSweeps:
    def test_kill_sweep_cancels_all_staged(self, tmp_path, monkeypatch):
        """§C10 + drill 7: kill fires with stages armed => zero staged
        after, before any new tick; staging stays off for the day."""
        m, a, *_ = _mgr(tmp_path, monkeypatch)
        _feed_and_tick(m, [_cand(sym=f'SYM{i}') for i in range(5)],
                       _et(DAY, 9, 40))
        assert sum(1 for r in m._stages.values()
                   if r['state'] == STATE_STAGED) == 5
        m.notify_kill('daily')
        assert all(r['state'] == STATE_CANCEL_CONFIRMED
                   for r in m._stages.values())
        m.on_candidate(_cand(sym='LATE'))
        m.process_tick(now_et=_et(DAY, 9, 45))
        assert 'LATE' not in m._stages

    def test_watchdog_stale_sweeps_and_goes_chase_only(self, tmp_path,
                                                       monkeypatch):
        """§D12: feed stale > threshold => cancel-all + chase-only."""
        m, a, *_ = _mgr(tmp_path, monkeypatch, watchdog_stale_s=0.05)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        time.sleep(0.1)
        m.process_tick(now_et=_et(DAY, 9, 41))
        assert m._chase_only_mode
        assert m._stages['PSTG']['state'] == STATE_CANCEL_CONFIRMED
        assert m.telemetry.feed_stale_events == 1

    def test_watchdog_recovery_reenables_staging(self, tmp_path,
                                                 monkeypatch):
        m, a, *_ = _mgr(tmp_path, monkeypatch, watchdog_stale_s=0.05)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        time.sleep(0.1)
        m.process_tick(now_et=_et(DAY, 9, 41))
        assert m._chase_only_mode
        m.on_price('OTHER', 5.0)          # feed resumed
        assert not m._chase_only_mode
        m.on_candidate(_cand(sym='NEW1'))
        m.process_tick(now_et=_et(DAY, 9, 42))
        assert 'NEW1' in m._stages

    def test_watchdog_sweep_race_vs_inflight_fill(self, tmp_path,
                                                  monkeypatch):
        """Drill: the watchdog's cancel races a fill — poll-and-adopt
        wins, position is never dropped."""
        m, a, db, sm, _ = _mgr(tmp_path, monkeypatch,
                               watchdog_stale_s=0.05)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        a.cancel_order.return_value = False
        a.get_order.return_value = {'status': 'filled', 'filled_qty': 68,
                                    'filled_avg_price': 11.05}
        time.sleep(0.1)
        m.process_tick(now_et=_et(DAY, 9, 41))
        assert m._stages['PSTG']['state'] == STATE_FILLED
        sm.add_watch.assert_called_once()

    def test_gap_through_cancel_after_n_minutes(self, tmp_path,
                                                monkeypatch):
        """§B6 frozen at N=60 (P1-5): gap-through observed, unfilled
        after N => cancel with explicit parity reason."""
        m, a, *_ = _mgr(tmp_path, monkeypatch, gap_through_cancel_min=0)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        m.on_price('PSTG', 12.0)          # > cap 11.33 => gap-through
        assert m.telemetry.gap_through_count == 1
        time.sleep(0.01)
        m.process_tick(now_et=_et(DAY, 9, 50))
        assert m._stages['PSTG']['state'] == STATE_CANCEL_CONFIRMED
        assert m._parity_explicit['PSTG'] == 'gap_through_expired'

    def test_gap_through_not_cancelled_before_n(self, tmp_path,
                                                monkeypatch):
        m, a, *_ = _mgr(tmp_path, monkeypatch, gap_through_cancel_min=60)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        m.on_price('PSTG', 12.0)
        m.process_tick(now_et=_et(DAY, 9, 50))
        assert m._stages['PSTG']['state'] == STATE_STAGED

    def test_shutdown_sweep(self, tmp_path, monkeypatch):
        """§A2: shutdown cancels every resting stage."""
        m, a, *_ = _mgr(tmp_path, monkeypatch)
        _feed_and_tick(m, [_cand(sym=f'SYM{i}') for i in range(3)],
                       _et(DAY, 9, 40))
        m.shutdown_sweep()
        assert all(r['state'] == STATE_CANCEL_CONFIRMED
                   for r in m._stages.values())

    def test_boot_reconcile_state_file_replay(self, tmp_path,
                                              monkeypatch):
        """§A1 kill −9 drill (state-file replay): STAGED records from a
        dead process — one FILLED while down (adopt + watch), one still
        open (cancel), one already cancelled (confirm)."""
        m1, a1, *_ = _mgr(tmp_path, monkeypatch)
        _feed_and_tick(m1, [_cand(sym='FWHILE'), _cand(sym='OPEN1'),
                            _cand(sym='GONE1')], _et(DAY, 9, 40))
        # state file persisted; simulate kill −9 by making a NEW manager
        m2, a2, db2, sm2, _ = _mgr(tmp_path, monkeypatch)

        def _get_order(oid):
            # order ids are all 'ord-1' from the mock — route on the
            # stage records instead via client_order_id fallback
            return _get_order.by_oid.get(oid, {'status': 'canceled',
                                               'filled_qty': 0})
        _get_order.by_oid = {}
        a2.get_open_orders.return_value = []
        # patch day so _now_et lands on DAY regardless of real clock
        monkeypatch.setattr(PrestageManager, '_now_et',
                            staticmethod(lambda: _et(DAY, 9, 50)))
        # OPEN1: first poll 'new' (still resting), post-cancel poll
        # 'canceled' — the realistic cancel-then-confirm sequence
        calls = {'OPEN1': 0}

        def _seq(oid):
            sym = oid.split('-', 1)[1]
            if sym == 'FWHILE':
                return {'status': 'filled', 'filled_qty': 68,
                        'filled_avg_price': 11.05}
            if sym == 'GONE1':
                return {'status': 'canceled', 'filled_qty': 0}
            calls['OPEN1'] += 1
            return {'status': 'new' if calls['OPEN1'] == 1
                    else 'canceled', 'filled_qty': 0}
        # per-symbol order ids: rewrite state file with distinct ids
        state_path = tmp_path / f'prestage_state_{DAY}.json'
        data = json.loads(state_path.read_text())
        for sym, rec in data['stages'].items():
            rec['order_id'] = f'ord-{sym}'
        state_path.write_text(json.dumps(data))
        a2.get_order.side_effect = _seq
        m2.boot_reconcile()
        assert m2._stages['FWHILE']['state'] == STATE_FILLED
        sm2.add_watch.assert_called_once()   # adopted orphan fill
        db2.save_trade.assert_called_once()
        assert m2._stages['GONE1']['state'] == STATE_CANCEL_CONFIRMED
        assert m2._stages['OPEN1']['state'] == STATE_CANCEL_CONFIRMED

    def test_boot_reconcile_broker_scan_without_state_file(
            self, tmp_path, monkeypatch):
        """§A1: state file lost — resting ign-stage-* found at the
        broker are cancelled (none left unmanaged)."""
        m, a, *_ = _mgr(tmp_path, monkeypatch)
        monkeypatch.setattr(PrestageManager, '_now_et',
                            staticmethod(lambda: _et(DAY, 9, 50)))
        a.get_open_orders.return_value = [
            {'id': 'ord-Z', 'client_order_id':
             f'ign-stage-{DAY_COMPACT}-ZOMBI', 'symbol': 'ZOMBI',
             'status': 'new', 'qty': 40, 'filled_qty': 0,
             'stop_price': 11.03, 'limit_price': 11.33}]
        a.get_order.return_value = {'status': 'canceled', 'filled_qty': 0}
        m.boot_reconcile()
        a.cancel_order.assert_called()
        assert m._stages['ZOMBI']['state'] == STATE_CANCEL_CONFIRMED

    def test_boot_reconcile_shadow_mode_never_touches_broker(
            self, tmp_path, monkeypatch):
        m, a, *_ = _mgr(tmp_path, monkeypatch, shadow=True)
        m.boot_reconcile()
        a.get_open_orders.assert_not_called()
        a.cancel_order.assert_not_called()


# ===========================================================================
# SHADOW MODE (§3 item 5) — zero orders by construction
# ===========================================================================
class TestShadowMode:
    def _shadow_mgr(self, tmp_path, monkeypatch, **over):
        return _mgr(tmp_path, monkeypatch, shadow=True, **over)

    def test_shadow_full_flow_places_zero_orders(self, tmp_path,
                                                 monkeypatch):
        """THE no-order-code-path test (ignition-shadow pattern): heap,
        scheduler, would-stage, tape-cross would-fill, sweep — and not
        one submit/cancel call on the alpaca mock, zero DB writes."""
        m, a, db, sm, osw = self._shadow_mgr(tmp_path, monkeypatch)
        _feed_and_tick(m, [_cand(sym='SHDW1'), _cand(sym='SHDW2')],
                       _et(DAY, 9, 40))
        assert m.telemetry.would_stage == 2
        m.on_price('SHDW1', 11.04)        # crosses stop 11.03
        assert m.telemetry.would_fill == 1
        m.notify_kill('daily')            # sweep path in shadow
        m.process_tick(now_et=_et(DAY, 13, 1))
        order_calls = [c for c in a.method_calls
                       if any(w in str(c).lower() for w in
                              ('submit', 'cancel', 'close'))]
        assert order_calls == []
        db.save_trade.assert_not_called()
        db.update_trade.assert_not_called()
        sm.add_watch.assert_not_called()
        sm.force_exit.assert_not_called()

    def test_shadow_would_fill_telemetry_and_parity(self, tmp_path,
                                                    monkeypatch):
        m, a, db, sm, osw = self._shadow_mgr(tmp_path, monkeypatch)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        m.on_price('PSTG', 11.10)
        assert m.telemetry.fills_shadow_inferred == 1
        assert 'PSTG' in m.parity_ledger()['staged_fills']

    def test_shadow_never_gates_chase(self, tmp_path, monkeypatch):
        m, *_ = self._shadow_mgr(tmp_path, monkeypatch)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        assert m.chase_allowed('PSTG')
        assert m.resolve_for_chase('PSTG') == 'chase_ok'

    def test_shadow_would_cancel_on_demote(self, tmp_path, monkeypatch):
        m, *_ = self._shadow_mgr(tmp_path, monkeypatch,
                                 demote_distance_pct=8.0)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        m.on_price('PSTG', 9.0)
        m.process_tick(now_et=_et(DAY, 9, 41))
        assert m.telemetry.would_cancel == 1
        assert m._stages['PSTG']['state'] == STATE_CANCEL_CONFIRMED

    def test_shadow_log_lines_tagged(self, tmp_path, monkeypatch,
                                     caplog):
        m, *_ = self._shadow_mgr(tmp_path, monkeypatch)
        with caplog.at_level('INFO'):
            _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        assert '[PRESTAGE SHADOW] PSTG would_stage' in caplog.text

    def test_disabled_manager_is_fully_inert(self, tmp_path,
                                             monkeypatch):
        m, a, db, sm, osw = _mgr(tmp_path, monkeypatch, enabled=False)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        m.notify_trigger({'symbol': 'PSTG', 'day': DAY})
        m.notify_kill('daily')
        m.boot_reconcile()
        m.shutdown_sweep()
        assert m._stages == {} and m._candidates == {}
        assert a.method_calls == []

    def test_env_kill_switch(self, tmp_path, monkeypatch):
        monkeypatch.setenv('IGNITION_PRESTAGE', '0')
        a = _alpaca()
        m = PrestageManager(a, cfg=_cfg(tmp_path, enabled=True))
        assert not m.enabled


# ===========================================================================
# telemetry (§G) + event log + parity surfaces
# ===========================================================================
class TestTelemetry:
    def test_snapshot_fields(self, tmp_path, monkeypatch):
        m, *_ = _mgr(tmp_path, monkeypatch)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        snap = m.telemetry_snapshot()
        for key in ('stage_ops', 'cancel_ops', 'fills_staged',
                    'bp_high_watermark_usd', 'bp_budget_usd',
                    'gap_through_count', 'scratch_count',
                    'churn_limiter_activations', 'fallback_activations',
                    'feed_stale_events', 'fills_without_trigger',
                    'rank_at_trigger', 'would_stage', 'would_fill'):
            assert key in snap, key
        assert snap['stage_ops'] == 1

    def test_lifecycle_event_journal_written(self, tmp_path,
                                             monkeypatch):
        m, *_ = _mgr(tmp_path, monkeypatch)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        path = tmp_path / f'prestage_events_{DAY}.jsonl'
        events = [json.loads(l) for l in
                  path.read_text().splitlines()]
        assert any(e['event'] == 'candidate' for e in events)
        assert any(e['event'] == 'staged' for e in events)

    def test_rank_at_trigger_recorded(self, tmp_path, monkeypatch):
        m, *_ = _mgr(tmp_path, monkeypatch)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        m.notify_trigger({'symbol': 'PSTG', 'day': DAY})
        assert m.telemetry.rank_at_trigger == [1]

    def test_bad_client_order_id_skips_with_reason(self, tmp_path,
                                                   monkeypatch, caplog):
        """§F18: an id-scheme violation fails at place decision time
        with an explicit parity reason (never a broker reject)."""
        m, a, *_ = _mgr(tmp_path, monkeypatch)
        long_sym = 'WAYTOOLONGSYMBOLNAMEFORTHESCHEMEXXXXX'
        with caplog.at_level('ERROR'):
            _feed_and_tick(m, [_cand(sym=long_sym)], _et(DAY, 9, 40))
        a.submit_stop_limit_order.assert_not_called()
        assert m._parity_explicit[long_sym] == 'stage_skip_bad_coid'

    def test_zero_qty_skip_labeled(self, tmp_path, monkeypatch):
        """Degenerate stop estimate => <1 share at $risk => explicit
        skip, never a 0-qty order."""
        m, a, *_ = _mgr(tmp_path, monkeypatch, risk_usd=0.01)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        a.submit_stop_limit_order.assert_not_called()
        assert m._parity_explicit['PSTG'] == 'stage_skip_zero_qty'

    def test_boot_open_orders_scan_failure_logged(self, tmp_path,
                                                  monkeypatch, caplog):
        """§A1 degraded path: broker scan fails => loud ERROR, state
        replay continues."""
        m, a, *_ = _mgr(tmp_path, monkeypatch)
        monkeypatch.setattr(PrestageManager, '_now_et',
                            staticmethod(lambda: _et(DAY, 9, 50)))
        a.get_open_orders.side_effect = RuntimeError('api dead')
        with caplog.at_level('ERROR'):
            m.boot_reconcile()
        assert 'open-orders scan FAILED' in caplog.text

    def test_boot_record_without_order_id_confirmed(self, tmp_path,
                                                    monkeypatch):
        m, a, *_ = _mgr(tmp_path, monkeypatch)
        monkeypatch.setattr(PrestageManager, '_now_et',
                            staticmethod(lambda: _et(DAY, 9, 50)))
        state = {'day': DAY, 'stages': {'NOID1': {
            'state': STATE_STAGED, 'symbol': 'NOID1',
            'client_order_id': '', 'order_id': '', 'stop_px': 11.03,
            'limit_px': 11.33, 'qty': 40, 'level': 11.0,
            'rank_at_stage': 1, 'stop_est': 10.3, 'anchor': None,
            'has_news': True, 'staged_minute': 580,
            'stage_ts': time.time(), 'gap_through_ts': None,
            'filled_qty': 0, 'fill_price': None}}}
        (tmp_path / f'prestage_state_{DAY}.json').write_text(
            json.dumps(state))
        m.boot_reconcile()
        assert m._stages['NOID1']['state'] == STATE_CANCEL_CONFIRMED

    def test_consume_fills_stream_failure_logged(self, tmp_path,
                                                 monkeypatch, caplog):
        """P0-4 degraded path: order-stream snapshot failure is loud,
        never wedges the tick."""
        m, a, db, sm, osw = _mgr(tmp_path, monkeypatch)
        _feed_and_tick(m, [_cand()], _et(DAY, 9, 40))
        osw.snapshot_by_client_prefix.side_effect = RuntimeError('ws dead')
        with caplog.at_level('WARNING'):
            m.process_tick(now_et=_et(DAY, 9, 42))
        assert 'order-stream snapshot failed' in caplog.text
        assert m._stages['PSTG']['state'] == STATE_STAGED

    def test_state_file_replay_corrupt_falls_back(self, tmp_path,
                                                  monkeypatch, caplog):
        m, a, *_ = _mgr(tmp_path, monkeypatch)
        monkeypatch.setattr(PrestageManager, '_now_et',
                            staticmethod(lambda: _et(DAY, 9, 50)))
        (tmp_path / f'prestage_state_{DAY}.json').write_text('{corrupt')
        with caplog.at_level('WARNING'):
            m.boot_reconcile()
        assert 'state replay failed' in caplog.text

    def test_chase_entry_ledger(self, tmp_path, monkeypatch):
        m, *_ = _mgr(tmp_path, monkeypatch)
        m.on_candidate(_cand())
        m.notify_chase_entry('PSTG')
        assert 'PSTG' in m.parity_ledger()['chase_entries']
        assert m.telemetry.fallback_activations == 1


# ===========================================================================
# integration: sighting -> stage -> stream fill -> validation -> watch
#              -> DB -> parity (spec item 7)
# ===========================================================================
class TestIntegration:
    def test_full_staged_lifecycle_and_parity(self, tmp_path,
                                              monkeypatch):
        m, a, db, sm, osw = _mgr(tmp_path, monkeypatch)
        # 1) sighting flow: candidate + trigger for a chased name too
        _feed_and_tick(m, [_cand(sym='STGD'),
                           _cand(sym='CHSD', news=False)],
                       _et(DAY, 9, 40))
        assert m._stages['STGD']['state'] == STATE_STAGED
        assert 'CHSD' not in m._stages        # newsless => chase path
        # 2) stream fill arrives
        osw.snapshot_by_client_prefix.return_value = _fill_status('STGD')
        m.process_tick(now_et=_et(DAY, 9, 42))
        assert m._stages['STGD']['state'] == STATE_FILLED
        # 3) at-fill validation passed -> watch + DB row path=staged
        assert sm.add_watch.call_args.kwargs['strategy'] == 'ignition'
        rec = db.save_trade.call_args.args[0]
        assert rec['strategy'] == 'ignition' and rec['shares'] == 68
        assert json.loads(rec['pattern_data'])['path'] == 'staged'
        # 4) shadow triggers observed for both; chase entry submitted
        m.notify_trigger({'symbol': 'STGD', 'day': DAY})
        m.notify_trigger({'symbol': 'CHSD', 'day': DAY})
        m.notify_chase_entry('CHSD')
        # 5) set-equality parity checker green (§B8 HARD gate)
        res = m.check_parity()
        assert res['ok'], res

    def test_parity_red_when_trigger_uncovered(self, tmp_path,
                                               monkeypatch):
        m, *_ = _mgr(tmp_path, monkeypatch)
        m.notify_trigger({'symbol': 'LOST', 'day': DAY})
        res = m.check_parity()
        assert not res['ok'] and res['missing'] == ['LOST']


# ===========================================================================
# shared at-fill structure helper (ignition_rules — P0-1 single source)
# ===========================================================================
class TestStructureGatesAtFill:
    def _g(self, bars):
        ts = pd.to_datetime(bars['timestamp'], utc=True)
        et = ts.dt.tz_convert('America/New_York')
        g = bars.copy()
        g['m'] = et.dt.hour * 60 + et.dt.minute
        return g.sort_values('m').reset_index(drop=True)

    def test_pass_returns_fill_derived_stop_and_r(self):
        res = _rules.structure_gates_at_fill(self._g(_bars()), 11.05)
        assert res.get('ok')
        assert res['stop'] == pytest.approx(10.3)
        assert res['r_pct'] == pytest.approx(
            (11.05 - 10.3) / 11.05 * 100)
        assert res['trigger_m'] == 575

    def test_no_trigger_bar_rejects(self):
        res = _rules.structure_gates_at_fill(
            self._g(_bars(cross_high=10.4)), 11.05)
        assert res['reject'] == 'stage_fill_no_trigger'

    def test_chase_guard_on_fill_price(self):
        res = _rules.structure_gates_at_fill(self._g(_bars()), 11.60)
        assert res['reject'] == 'skip_chase_guard'   # > 11.0*1.05

    def test_r_too_small_rejects(self):
        res = _rules.structure_gates_at_fill(
            self._g(_bars(pre_low=10.99)), 11.05)
        assert res['reject'] == 'skip_r_too_small'

    def test_pre_bars_minimum(self):
        # keep the 9:30 open bar (day_open source) but thin the
        # pre-window to 2 bars (570 + 574) — below PRE_BARS_MIN=5
        g = self._g(_bars())
        g = g[(g['m'] == 570) | (g['m'] >= 574)].reset_index(drop=True)
        res = _rules.structure_gates_at_fill(g, 11.05)
        assert res['reject'] == 'skip_pre_bars'

    def test_illiquid_participation_floor(self):
        res = _rules.structure_gates_at_fill(
            self._g(_bars(vol=50)), 11.05)
        assert res['reject'] == 'skip_illiquid'

    def test_empty_bars_reject(self):
        assert _rules.structure_gates_at_fill(None, 11.0)['reject'] \
            == 'no_bars'


# ===========================================================================
# StopMonitor: P1-2 collision fix + P1-3 whitelist extension
# ===========================================================================
class TestStopMonitorCollision:
    def _monitor(self):
        client = MagicMock(spec=AlpacaClient)
        return StopMonitor(api_key='k', api_secret='s',
                           alpaca_client=client)

    def test_cross_strategy_add_rejected_loudly(self, caplog):
        mon = self._monitor()
        assert mon.add_watch('DUPX', 10.0, 100, 'tp', 'sl',
                             strategy='bull_flag') is True
        with caplog.at_level('ERROR'):
            assert mon.add_watch('DUPX', 9.0, 50, 'tp2', 'sl2',
                                 strategy='ignition') is False
        assert 'REJECTED' in caplog.text
        assert mon._watches['DUPX'].strategy == 'bull_flag'
        assert mon._watches['DUPX'].shares == 100

    def test_same_strategy_readd_still_overwrites(self):
        mon = self._monitor()
        mon.add_watch('SAME1', 10.0, 100, 'tp', 'sl', strategy='orb')
        assert mon.add_watch('SAME1', 9.5, 80, 'tp', 'sl',
                             strategy='orb') is True
        assert mon._watches['SAME1'].stop_price == 9.5

    def test_upgrade_quote_watch_collision_rejected(self, caplog):
        mon = self._monitor()
        mon.add_watch('DUPY', 10.0, 100, 'tp', 'sl',
                      strategy='bull_flag')
        with caplog.at_level('ERROR'):
            mon.upgrade_quote_to_stop_watch('DUPY', 9.0, 50, 'tp2',
                                            'sl2', strategy='ignition')
        assert 'REJECTED' in caplog.text
        assert mon._watches['DUPY'].strategy == 'bull_flag'

    def test_force_exit_whitelist_includes_stage_reasons(self, caplog):
        """P1-3: staged-disposition exits pass the whitelist gate (the
        'no active watch' warning proves the reason was accepted)."""
        mon = self._monitor()
        for reason in (ExitReason.STAGE_REJECT_STRUCTURE.value,
                       ExitReason.STAGE_FORCE_FLAT.value):
            caplog.clear()
            with caplog.at_level('WARNING'):
                assert mon.force_exit('NOWATCH', reason=reason) is False
            assert 'no active watch' in caplog.text
            assert 'not in whitelist' not in caplog.text

    def test_force_exit_unlisted_reason_still_rejected(self, caplog):
        mon = self._monitor()
        with caplog.at_level('ERROR'):
            assert mon.force_exit('X', reason='made_up') is False
        assert 'not in whitelist' in caplog.text


# ===========================================================================
# OrderStreamWatcher: client_order_id prefix routing (P0-4)
# ===========================================================================
class TestOrderStreamPrefix:
    def test_snapshot_by_client_prefix(self):
        w = OrderStreamWatcher('k', 's')
        w._statuses['b1'] = {'id': 'b1', 'client_order_id':
                             'ign-stage-20260824-AAA',
                             'status': 'filled', 'filled_qty': 10}
        w._statuses['b2'] = {'id': 'b2', 'client_order_id': 'other-id',
                             'status': 'new', 'filled_qty': 0}
        w._statuses['b3'] = {'id': 'b3', 'status': 'new'}   # no coid
        snap = w.snapshot_by_client_prefix(PRESTAGE_ID_PREFIX)
        assert set(snap) == {'ign-stage-20260824-AAA'}
        # copies, not references
        snap['ign-stage-20260824-AAA']['status'] = 'mutated'
        assert w._statuses['b1']['status'] == 'filled'

    def test_order_to_status_captures_client_order_id(self):
        from trading.order_stream import _order_to_status
        order = MagicMock()
        order.id = 'oid-1'
        order.client_order_id = 'ign-stage-20260824-ZZZ'
        order.symbol = 'ZZZ'
        st = _order_to_status(order, event='fill')
        assert st['client_order_id'] == 'ign-stage-20260824-ZZZ'


# ===========================================================================
# engine wiring (surgical): P0-3 gate, kill sweep, path=chase, triggers
# ===========================================================================
class TestEngineWiring:
    def _engine(self, tmp_path, monkeypatch, prestage, enabled=True):
        from trading.ignition_engine import IgnitionEngine
        monkeypatch.delenv('IGNITION_LIVE', raising=False)
        a = MagicMock(spec=AlpacaClient)
        a.submit_bracket_order.return_value = {
            'id': 'ord-1', 'status': 'pending_new',
            'legs': [{'id': 'tp-1', 'type': 'limit'},
                     {'id': 'sl-1', 'type': 'stop'}]}
        a.get_latest_quote.return_value = {'bid_price': 9.95,
                                           'ask_price': 10.0,
                                           'bid_size': 5, 'ask_size': 5}
        a.get_open_positions.return_value = []
        db = MagicMock(spec=Database)
        db.save_trade.return_value = 42
        db.get_open_trades.return_value = []
        db._trades_path = tmp_path / 'trades.db'
        import sqlite3
        conn = sqlite3.connect(db._trades_path)
        conn.execute("CREATE TABLE trades (strategy TEXT, "
                     "trade_date TEXT, pnl REAL)")
        conn.commit()
        conn.close()
        sm = MagicMock(spec=StopMonitor)
        sm.drain_exit_events.return_value = []
        sm.add_watch.return_value = True
        cfg = {'enabled': enabled, 'dry_run': False, 'risk_usd': 50.0,
               'daily_kill_usd': -300.0, 'weekly_kill_usd': -750.0,
               'max_concurrent': 15, 'max_notional_usd': 1500.0,
               'entry_buffer_bps': 20.0, 'entry_timeout_s': 90.0}
        eng = IgnitionEngine(a, db, sm, notifier=None, cfg=cfg,
                             prestage=prestage)
        return eng, a, db, sm

    def _rec(self, sym='IGNI'):
        return {'symbol': sym, 'day': DAY, 'price': 10.0,
                '_entry': 10.0, '_stop': 9.0, 'ask': 10.0,
                'r_pct': 10.0, 'hypo_entry': 10.0, 'hypo_stop': 9.0,
                'catalyst': 'news', 'anchor': sym, 'anchor_cohort': 1}

    def test_chase_blocked_by_stage_disposition(self, tmp_path,
                                                monkeypatch):
        """P0-3 composed rule: engine submits ONLY on 'chase_ok'."""
        pre = MagicMock(spec=PrestageManager)
        pre.resolve_for_chase.return_value = 'blocked'
        eng, a, *_ = self._engine(tmp_path, monkeypatch, pre)
        eng._handle_trigger(self._rec())
        a.submit_bracket_order.assert_not_called()

    def test_chase_skipped_when_stage_adopted(self, tmp_path,
                                              monkeypatch):
        pre = MagicMock(spec=PrestageManager)
        pre.resolve_for_chase.return_value = 'adopted'
        eng, a, *_ = self._engine(tmp_path, monkeypatch, pre)
        eng._handle_trigger(self._rec())
        a.submit_bracket_order.assert_not_called()

    def test_chase_proceeds_on_chase_ok_and_ledgered(self, tmp_path,
                                                     monkeypatch):
        pre = MagicMock(spec=PrestageManager)
        pre.resolve_for_chase.return_value = 'chase_ok'
        eng, a, *_ = self._engine(tmp_path, monkeypatch, pre)
        eng._handle_trigger(self._rec())
        a.submit_bracket_order.assert_called_once()
        pre.notify_chase_entry.assert_called_once_with('IGNI')

    def test_resolver_exception_fails_closed(self, tmp_path,
                                             monkeypatch, caplog):
        pre = MagicMock(spec=PrestageManager)
        pre.resolve_for_chase.side_effect = RuntimeError('boom')
        eng, a, *_ = self._engine(tmp_path, monkeypatch, pre)
        with caplog.at_level('ERROR'):
            eng._handle_trigger(self._rec())
        a.submit_bracket_order.assert_not_called()
        assert 'fail-closed' in caplog.text

    def test_kill_sweeps_stages_in_same_path(self, tmp_path,
                                             monkeypatch):
        """§C10: the kill and the stage sweep share one call path."""
        pre = MagicMock(spec=PrestageManager)
        pre.resolve_for_chase.return_value = 'chase_ok'
        eng, a, db, sm = self._engine(tmp_path, monkeypatch, pre)
        import sqlite3
        conn = sqlite3.connect(db._trades_path)
        today = datetime.now(timezone.utc).astimezone(_ET) \
            .strftime('%Y-%m-%d')
        conn.execute("INSERT INTO trades VALUES ('ignition', ?, -400)",
                     (today,))
        conn.commit()
        conn.close()
        assert eng._kill_blocked() == 'daily_kill'
        pre.notify_kill.assert_called_once_with('daily')

    def test_trigger_notifies_prestage_even_when_engine_disabled(
            self, tmp_path, monkeypatch):
        pre = MagicMock(spec=PrestageManager)
        eng, a, *_ = self._engine(tmp_path, monkeypatch, pre,
                                  enabled=False)
        eng.enqueue_trigger(self._rec())
        pre.notify_trigger.assert_called_once()
        a.submit_bracket_order.assert_not_called()

    def test_sync_positions_runs_boot_reconcile_first(self, tmp_path,
                                                      monkeypatch):
        pre = MagicMock(spec=PrestageManager)
        eng, *_ = self._engine(tmp_path, monkeypatch, pre,
                               enabled=False)
        eng.sync_positions()      # early-returns for disabled AFTER pre
        pre.boot_reconcile.assert_called_once()

    def test_finalize_eod_sweeps_stages_first(self, tmp_path,
                                              monkeypatch):
        pre = MagicMock(spec=PrestageManager)
        eng, *_ = self._engine(tmp_path, monkeypatch, pre)
        eng.finalize_eod(timeout_s=0.01)
        pre.shutdown_sweep.assert_called_once()

    def test_process_tick_runs_prestage_even_in_dry_run(self, tmp_path,
                                                        monkeypatch):
        pre = MagicMock(spec=PrestageManager)
        from trading.ignition_engine import IgnitionEngine
        monkeypatch.delenv('IGNITION_LIVE', raising=False)
        eng, a, db, sm = self._engine(tmp_path, monkeypatch, pre)
        eng.dry_run = True
        eng.process_tick()
        pre.process_tick.assert_called_once()

    def test_fill_quality_line_has_path_chase(self, tmp_path,
                                              monkeypatch, caplog):
        """P1-7: the chase-path FILL QUALITY line is path-tagged."""
        from trading.ignition_engine import _Pending
        eng, a, db, sm = self._engine(tmp_path, monkeypatch, None)
        p = _Pending('IGNI', 'ord-1', 42, 10.0, 9.0, 50, 'tp-1',
                     'sl-1', time.time())
        eng.pending['IGNI'] = p
        with caplog.at_level('INFO'):
            eng._confirm_fill(p, {'status': 'filled',
                                  'filled_avg_price': 10.02,
                                  'filled_qty': 50})
        assert 'FILL QUALITY IGNI: path=chase' in caplog.text

    def test_no_prestage_is_byte_identical_noop(self, tmp_path,
                                                monkeypatch):
        """Guard: prestage=None leaves every engine path working."""
        eng, a, *_ = self._engine(tmp_path, monkeypatch, None)
        eng._handle_trigger(self._rec())
        a.submit_bracket_order.assert_called_once()
        eng.process_tick()
        eng.finalize_eod(timeout_s=0.01)


# ===========================================================================
# shadow-side intake hooks (surgical additions to IgnitionShadow)
# ===========================================================================
class TestShadowHooks:
    def _shadow(self, tmp_path):
        from trading.ignition_shadow import IgnitionShadow
        a = MagicMock(spec=AlpacaClient)
        a.get_latest_quote.return_value = {'bid_price': 10.0,
                                           'ask_price': 10.05,
                                           'bid_size': 5, 'ask_size': 7}
        # the canonical shadow-test bar fixture: open bar 9.0 (small
        # TRUE open gap so the ORB-disjointness gate passes), deep pre
        # lows so R >= 5%
        ts = pd.date_range('2026-08-24 13:30', periods=41, freq='1min',
                           tz='UTC')
        a.get_1min_bars.return_value = pd.DataFrame(
            {'timestamp': ts, 'open': [9.0] + [10.0] * 40,
             'high': [9.1] + [10.4] * 40, 'low': [8.9] + [9.4] * 40,
             'close': [9.05] + [10.2] * 40, 'volume': [10000] * 41})
        a.get_premarket_news_multi.return_value = {}
        s = IgnitionShadow(a, {'ignition_shadow': {'enabled': True}},
                           log_dir=str(tmp_path))
        return s, a

    def test_on_price_hook_fires_every_sighting(self, tmp_path):
        s, a = self._shadow(tmp_path)
        seen = []
        s.on_price = lambda sym, price, minute: seen.append(
            (sym, price, minute))
        from unittest.mock import patch
        with patch('trading.ignition_shadow.datetime') as md:
            md.now.return_value = datetime(2026, 8, 24, 13, 50,
                                           tzinfo=timezone.utc)
            s.on_mover('HOOK1', intraday_change_pct=15.0, gap_pct=12.0,
                       price=10.35, has_news=True, price_ts_utc=None)
            s.on_mover('HOOK1', intraday_change_pct=15.5, gap_pct=12.0,
                       price=10.40, has_news=True, price_ts_utc=None)
        assert s.drain(10.0)
        assert len(seen) == 2 and seen[0][0] == 'HOOK1'

    def test_on_candidate_hook_gets_day_open(self, tmp_path):
        s, a = self._shadow(tmp_path)
        cands = []
        s.on_candidate = lambda rec: cands.append(dict(rec))
        from unittest.mock import patch
        with patch('trading.ignition_shadow.datetime') as md:
            md.now.return_value = datetime(2026, 8, 24, 13, 50,
                                           tzinfo=timezone.utc)
            s.on_mover('CANDI', intraday_change_pct=15.0, gap_pct=12.0,
                       price=10.35, has_news=True, price_ts_utc=None)
        assert s.drain(10.0)
        assert len(cands) == 1
        assert cands[0]['day_open'] == 9.0

    def test_hook_errors_never_block_shadow(self, tmp_path):
        s, a = self._shadow(tmp_path)
        s.on_price = lambda *a_, **k: (_ for _ in ()).throw(
            RuntimeError('hook boom'))
        s.on_candidate = lambda rec: (_ for _ in ()).throw(
            RuntimeError('hook boom'))
        from unittest.mock import patch
        with patch('trading.ignition_shadow.datetime') as md:
            md.now.return_value = datetime(2026, 8, 24, 13, 50,
                                           tzinfo=timezone.utc)
            s.on_mover('BOOMY', intraday_change_pct=15.0, gap_pct=12.0,
                       price=10.35, has_news=True, price_ts_utc=None)
        assert s.drain(10.0)
        recs = [json.loads(l) for f in
                Path(tmp_path).glob('ignition_shadow_*.jsonl')
                for l in f.read_text().splitlines()]
        assert any(r.get('verdict') == 'SHADOW_TRIGGER' for r in recs)


# ===========================================================================
# exit-reason catalog additions
# ===========================================================================
class TestExitReasonCatalog:
    def test_stage_reasons_known_and_attributed(self):
        from trading import exit_reasons as er
        assert er.is_known('stage_reject_structure')
        assert er.is_attributed('stage_reject_structure')
        assert er.is_known('stage_force_flat')
        assert er.is_attributed('stage_force_flat')
