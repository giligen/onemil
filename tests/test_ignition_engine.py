"""Ignition S3 live engine (2026-08-14 plan) — every gate, the full
order lifecycle, DB persistence contract, and restart resume."""
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.ignition_engine import IgnitionEngine, STRATEGY_NAME
from trading.stop_monitor import StopMonitor


def _cfg(**over):
    c = {'enabled': True, 'dry_run': False, 'risk_usd': 50.0,
         'daily_kill_usd': -300.0, 'weekly_kill_usd': -750.0,
         'max_concurrent': 15, 'max_notional_usd': 1500.0,
         'entry_buffer_bps': 20.0, 'entry_timeout_s': 90.0}
    c.update(over)
    return c


def _engine(tmp_path, monkeypatch=None, **cfg_over):
    if monkeypatch:
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
    conn.execute("CREATE TABLE trades (strategy TEXT, trade_date TEXT, "
                 "pnl REAL)")
    conn.commit(); conn.close()
    sm = MagicMock(spec=StopMonitor)
    sm.drain_exit_events.return_value = []
    eng = IgnitionEngine(a, db, sm, notifier=None, cfg=_cfg(**cfg_over))
    return eng, a, db, sm


def _rec(sym='IGNI', entry=10.0, stop=9.0, ask=10.0):
    return {'symbol': sym, 'day': '2026-08-14', 'price': entry,
            '_entry': entry, '_stop': stop, 'ask': ask,
            'r_pct': (entry - stop) / entry * 100,
            'hypo_entry': entry, 'hypo_stop': stop,
            'catalyst': 'news', 'spread_bps': 30.0, 'anchor': sym,
            'anchor_cohort': 1, 'chg_from_open': 12.0, 'trigger_m': 580,
            'minute_et': 580, 'latency_s': 30.0,
            'intraday_change_pct': 12.0, 'hypo_position_usd': 500}


class TestSizing:
    def test_shares_floor(self, tmp_path):
        eng, a, db, sm = _engine(tmp_path)
        eng._handle_trigger(_rec(entry=10.0, stop=9.0))   # $1 R -> 50sh
        a.submit_bracket_order.assert_called_once()
        assert a.submit_bracket_order.call_args.kwargs['qty'] == 50

    def test_notional_cap(self, tmp_path):
        # R tiny -> uncapped shares huge -> notional cap binds
        eng, a, db, sm = _engine(tmp_path)
        eng._handle_trigger(_rec(entry=100.0, stop=99.5, ask=100.0))
        qty = a.submit_bracket_order.call_args.kwargs['qty']
        assert qty * 100.0 <= 1500.0

    def test_sub_share_skip(self, tmp_path):
        eng, a, db, sm = _engine(tmp_path)
        eng._handle_trigger(_rec(entry=1000.0, stop=900.0))  # 0 shares
        a.submit_bracket_order.assert_not_called()

    def test_bad_levels_skip(self, tmp_path):
        eng, a, db, sm = _engine(tmp_path)
        eng._handle_trigger(_rec(entry=9.0, stop=10.0))
        a.submit_bracket_order.assert_not_called()


class TestGates:
    def test_disabled_never_starts_worker_or_trades(self, tmp_path):
        eng, a, db, sm = _engine(tmp_path, enabled=False)
        eng.enqueue_trigger(_rec())
        time.sleep(0.1)
        a.submit_bracket_order.assert_not_called()

    def test_env_kill(self, tmp_path, monkeypatch):
        monkeypatch.setenv('IGNITION_LIVE', '0')
        eng, a, db, sm = _engine(tmp_path)
        assert eng.enabled is False

    def test_dry_run_no_orders_but_notes(self, tmp_path):
        eng, a, db, sm = _engine(tmp_path, dry_run=True)
        eng._handle_trigger(_rec())
        a.submit_bracket_order.assert_not_called()
        db.save_trade.assert_not_called()
        assert 'IGNI' in eng._entered_today   # dedup still counted

    def test_daily_kill_blocks_new(self, tmp_path):
        eng, a, db, sm = _engine(tmp_path)
        import sqlite3
        conn = sqlite3.connect(db._trades_path)
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        conn.execute("INSERT INTO trades VALUES (?,?,?)",
                     (STRATEGY_NAME, today, -301.0))
        conn.commit(); conn.close()
        eng._handle_trigger(_rec())
        a.submit_bracket_order.assert_not_called()

    def test_weekly_kill_blocks_new(self, tmp_path):
        eng, a, db, sm = _engine(tmp_path)
        import sqlite3
        from datetime import datetime, timezone, timedelta
        wk = (datetime.now(timezone.utc)
              - timedelta(days=datetime.now(timezone.utc).weekday()))
        conn = sqlite3.connect(db._trades_path)
        conn.execute("INSERT INTO trades VALUES (?,?,?)",
                     (STRATEGY_NAME, wk.strftime('%Y-%m-%d'), -800.0))
        conn.commit(); conn.close()
        eng._handle_trigger(_rec())
        a.submit_bracket_order.assert_not_called()

    def test_pnl_query_failure_fails_closed(self, tmp_path):
        eng, a, db, sm = _engine(tmp_path)
        db._trades_path = tmp_path / 'missing-dir' / 'nope.db'
        eng._handle_trigger(_rec())
        a.submit_bracket_order.assert_not_called()

    def test_dedup_one_per_symbol_per_day(self, tmp_path):
        eng, a, db, sm = _engine(tmp_path)
        eng._handle_trigger(_rec())
        eng._handle_trigger(_rec())
        assert a.submit_bracket_order.call_count == 1

    def test_max_concurrent(self, tmp_path):
        eng, a, db, sm = _engine(tmp_path, max_concurrent=1)
        eng._handle_trigger(_rec('AAA'))
        eng._handle_trigger(_rec('BBB'))
        assert a.submit_bracket_order.call_count == 1


class TestOrderLifecycle:
    def test_submit_persists_full_contract(self, tmp_path):
        """INSERT carries the fixed required key set (incl. None exit
        fields — save_trade binds them); telemetry goes via the
        follow-up update_trade (extras are silently dropped by INSERT,
        verified empirically 8/14)."""
        eng, a, db, sm = _engine(tmp_path)
        eng._handle_trigger(_rec())
        rec = db.save_trade.call_args.args[0]
        assert rec['strategy'] == STRATEGY_NAME
        assert rec['order_status'] == 'pending_new'
        assert rec['total_risk'] == pytest.approx(50.0)
        for k in ('fill_price', 'exit_price', 'pnl', 'exit_reason'):
            assert k in rec and rec[k] is None   # required bindings
        pd = json.loads(rec['pattern_data'])
        assert pd['catalyst'] == 'news'
        assert pd['lock_arm_at_r'] == 1.75
        assert pd['hypo_entry'] == 10.0
        upd = db.update_trade.call_args.args[1]
        assert upd['real_stop_loss_price'] == 9.0
        assert upd['entry_quote_bid'] == 9.95

    def test_fill_adds_watch_with_lock(self, tmp_path):
        eng, a, db, sm = _engine(tmp_path)
        eng._handle_trigger(_rec())
        a.get_order.return_value = {'status': 'filled',
                                    'filled_avg_price': 10.02,
                                    'filled_qty': 50}
        eng.process_tick()
        sm.add_watch.assert_called_once()
        kw = sm.add_watch.call_args.kwargs
        assert kw['strategy'] == STRATEGY_NAME
        assert kw['lock_arm_at_r'] == 1.75
        assert kw['lock_stop_r'] == 0.5
        assert kw['lock_r_unit'] == pytest.approx(1.0)
        assert kw['stop_price'] == 9.0
        db.update_trade.assert_called()
        upd = db.update_trade.call_args.args[1]
        assert upd['order_status'] == 'filled'
        assert upd['fill_price'] == 10.02

    def test_timeout_cancels(self, tmp_path):
        eng, a, db, sm = _engine(tmp_path, entry_timeout_s=0.0)
        eng._handle_trigger(_rec())
        a.get_order.return_value = {'status': 'new'}
        eng.process_tick()
        a.cancel_order.assert_called_once_with('ord-1')
        upd = db.update_trade.call_args.args[1]
        assert upd['order_status'] == 'cancelled'
        assert upd['exit_reason'] == 'entry_timeout_canceled'

    def test_exit_event_writes_pnl(self, tmp_path):
        eng, a, db, sm = _engine(tmp_path)
        eng._handle_trigger(_rec())
        a.get_order.return_value = {'status': 'filled',
                                    'filled_avg_price': 10.0,
                                    'filled_qty': 50}
        eng.process_tick()
        ev = MagicMock()
        ev.symbol = 'IGNI'; ev.exit_price = 10.5; ev.shares = 50
        ev.exit_reason = 'lock_stop'; ev.confirmed = True
        ev.trade_db_id = 42; ev.filled_qty = 50
        ev.exit_limit_price = 10.49; ev.pricing_method = 'stop_bid'
        sm.drain_exit_events.return_value = [ev]
        eng.check_exits()
        upd = db.update_trade.call_args.args[1]
        assert upd['pnl'] == pytest.approx(25.0)   # (10.5-10.0)*50
        assert upd['order_status'] == 'closed'
        assert 'IGNI' not in eng.open_positions

    def test_force_close_cancels_legs_before_selling(self, tmp_path):
        """8/14 review P0: active bracket legs hold the shares — the EOD
        sell MUST be preceded by leg cancels or the broker rejects it."""
        eng, a, db, sm = _engine(tmp_path)
        eng._handle_trigger(_rec('PEND'))
        eng._handle_trigger(_rec('OPEN'))
        a.get_order.side_effect = [
            {'status': 'new'},
            {'status': 'filled', 'filled_avg_price': 10.0,
             'filled_qty': 50}]
        eng.process_tick()
        n = eng.force_close_all()
        assert n == 2
        cancelled = [c.args[0] for c in a.cancel_order.call_args_list]
        assert 'ord-1' in cancelled              # pending entry
        assert 'tp-1' in cancelled and 'sl-1' in cancelled  # legs FIRST
        a.submit_limit_sell_order.assert_called()
        sm.remove_watch.assert_called_with('OPEN')

    def test_real_database_persistence_roundtrip(self, tmp_path):
        """The 8/14 P0 the mocks hid: save_trade's INSERT requires a
        fixed key set and SILENTLY DROPS extras. Drive the REAL Database
        through submit -> fill -> exit and verify every field lands."""
        import os
        from persistence.database import Database
        real_db = Database(cache_path=str(tmp_path / 'c.db'),
                           trades_path=str(tmp_path / 't.db'))
        a = MagicMock(spec=AlpacaClient)
        a.submit_bracket_order.return_value = {
            'id': 'ord-1', 'status': 'pending_new',
            'legs': [{'id': 'tp-1', 'type': 'limit'},
                     {'id': 'sl-1', 'type': 'stop'}]}
        a.get_latest_quote.return_value = {'bid_price': 9.95,
                                           'ask_price': 10.0,
                                           'bid_size': 5, 'ask_size': 5}
        sm = MagicMock(spec=StopMonitor)
        real_db._trades_path = tmp_path / 't.db'
        eng = IgnitionEngine(a, real_db, sm, cfg=_cfg())
        eng._handle_trigger(_rec())
        import sqlite3
        conn = sqlite3.connect(tmp_path / 't.db')
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM trades").fetchone()
        assert row['strategy'] == STRATEGY_NAME
        assert row['order_status'] == 'pending_new'
        assert row['real_stop_loss_price'] == 9.0      # via update_trade
        assert row['entry_quote_bid'] == 9.95           # via update_trade
        assert json.loads(row['pattern_data'])['catalyst'] == 'news'
        # fill
        a.get_order.return_value = {'status': 'filled',
                                    'filled_avg_price': 10.02,
                                    'filled_qty': 50}
        eng.process_tick()
        row = conn.execute("SELECT * FROM trades").fetchone()
        assert row['order_status'] == 'filled'
        assert row['fill_price'] == 10.02
        # exit
        ev = MagicMock()
        ev.symbol = 'IGNI'; ev.exit_price = 10.52; ev.shares = 50
        ev.exit_reason = 'lock_stop'; ev.confirmed = True
        ev.trade_db_id = row['id']; ev.filled_qty = 50
        ev.exit_limit_price = 10.51; ev.pricing_method = 'stop_bid'
        sm.drain_exit_events.return_value = [ev]
        eng.check_exits()
        row = conn.execute("SELECT * FROM trades").fetchone()
        assert row['order_status'] == 'closed'
        assert row['pnl'] == pytest.approx((10.52 - 10.02) * 50)
        conn.close()


class TestResume:
    def test_sync_readds_watch_from_db(self, tmp_path):
        eng, a, db, sm = _engine(tmp_path)
        a.get_open_positions.return_value = [{'symbol': 'HOLD'}]
        db.get_open_trades.return_value = [{
            'id': 7, 'symbol': 'HOLD', 'order_status': 'filled',
            'shares': 50, 'filled_qty': 50, 'fill_price': 10.0,
            'entry_price': 10.0, 'real_stop_loss_price': 9.0,
            'stop_loss_price': 8.1, 'order_id': 'x',
            'pattern_data': json.dumps({'lock_arm_at_r': 1.75,
                                        'lock_stop_r': 0.5})}]
        eng.sync_positions()
        kw = sm.add_watch.call_args.kwargs
        assert kw['stop_price'] == 9.0
        assert kw['lock_arm_at_r'] == 1.75
        assert 'HOLD' in eng.open_positions
        assert 'HOLD' in eng._entered_today

    def test_sync_resumes_pending(self, tmp_path):
        eng, a, db, sm = _engine(tmp_path)
        db.get_open_trades.return_value = [{
            'id': 8, 'symbol': 'PND', 'order_status': 'pending_new',
            'shares': 30, 'entry_price': 5.0,
            'real_stop_loss_price': 4.5, 'stop_loss_price': 4.05,
            'order_id': 'ord-9', 'pattern_data': '{}'}]
        eng.sync_positions()
        assert 'PND' in eng.pending
        assert eng.pending['PND'].order_id == 'ord-9'

    def test_sync_broker_gone_marks_pending_verification(self, tmp_path):
        eng, a, db, sm = _engine(tmp_path)
        a.get_open_positions.return_value = []   # exited while down
        db.get_open_trades.return_value = [{
            'id': 9, 'symbol': 'GONE', 'order_status': 'filled',
            'shares': 50, 'filled_qty': 50, 'fill_price': 10.0,
            'entry_price': 10.0, 'real_stop_loss_price': 9.0,
            'stop_loss_price': 8.1, 'order_id': 'x',
            'pattern_data': '{}'}]
        eng.sync_positions()
        sm.add_watch.assert_not_called()
        upd = db.update_trade.call_args.args[1]
        assert upd['order_status'] == 'exit_pending_verification'


class TestShadowWiring:
    def test_trigger_callback_reaches_engine(self, tmp_path):
        import pandas as pd
        from trading.ignition_shadow import IgnitionShadow
        a = MagicMock(spec=AlpacaClient)
        a.get_latest_quote.return_value = {'bid_price': 10.0,
                                           'ask_price': 10.05,
                                           'bid_size': 5, 'ask_size': 7}
        ts = pd.date_range('2026-07-20 13:30', periods=41, freq='1min',
                           tz='UTC')
        a.get_1min_bars.return_value = pd.DataFrame({
            'timestamp': ts, 'open': [9.0] + [10.0] * 40,
            'high': [9.1] + [10.4] * 40, 'low': [8.9] + [9.4] * 40,
            'close': [9.05] + [10.2] * 40, 'volume': [10000] * 41})
        a.get_premarket_news_multi.return_value = {}
        s = IgnitionShadow(a, {'ignition_shadow': {'enabled': True}},
                           log_dir=str(tmp_path))
        got = []
        s.on_trigger = got.append
        from datetime import datetime, timezone
        from unittest.mock import patch
        with patch('trading.ignition_shadow.datetime') as md:
            md.now.return_value = datetime(2026, 7, 20, 13, 50,
                                           tzinfo=timezone.utc)
            s.on_mover('IGNI', intraday_change_pct=15.0, gap_pct=12.0,
                       price=10.35, has_news=True, price_ts_utc=None)
        assert s.drain(10.0)
        assert len(got) == 1 and got[0]['verdict'] == 'SHADOW_TRIGGER'

    def test_callback_exception_never_blocks_journal(self, tmp_path):
        import pandas as pd
        from trading.ignition_shadow import IgnitionShadow
        a = MagicMock(spec=AlpacaClient)
        a.get_latest_quote.return_value = {'bid_price': 10.0,
                                           'ask_price': 10.05,
                                           'bid_size': 5, 'ask_size': 7}
        ts = pd.date_range('2026-07-20 13:30', periods=41, freq='1min',
                           tz='UTC')
        a.get_1min_bars.return_value = pd.DataFrame({
            'timestamp': ts, 'open': [9.0] + [10.0] * 40,
            'high': [9.1] + [10.4] * 40, 'low': [8.9] + [9.4] * 40,
            'close': [9.05] + [10.2] * 40, 'volume': [10000] * 41})
        a.get_premarket_news_multi.return_value = {}
        s = IgnitionShadow(a, {'ignition_shadow': {'enabled': True}},
                           log_dir=str(tmp_path))
        s.on_trigger = lambda rec: 1 / 0
        from datetime import datetime, timezone
        from unittest.mock import patch
        with patch('trading.ignition_shadow.datetime') as md:
            md.now.return_value = datetime(2026, 7, 20, 13, 50,
                                           tzinfo=timezone.utc)
            s.on_mover('IGNI', intraday_change_pct=15.0, gap_pct=12.0,
                       price=10.35, has_news=True, price_ts_utc=None)
        assert s.drain(10.0)
        recs = [json.loads(l) for l in
                (tmp_path / 'ignition_shadow_2026-07-20.jsonl')
                .read_text().splitlines()]
        assert recs[-1]['verdict'] == 'SHADOW_TRIGGER'   # journaled

    def test_shadow_module_still_has_no_order_code(self):
        import inspect
        import trading.ignition_shadow as m
        src = inspect.getsource(m)
        for word in ('submit_order', 'submit_stop', 'bracket', 'sell',
                     'buy_stop'):
            assert word not in src
