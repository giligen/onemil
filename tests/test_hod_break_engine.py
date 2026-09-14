"""HOD-break engine — gates, entry, fills, exits, force close, restart (mocked broker/DB)."""
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.stop_monitor import StopMonitor
from trading.hod_break_engine import HodBreakEngine, STRATEGY_NAME
from tests.test_hod_break import drive_then_consolidate


def cfg(**over):
    base = {'enabled': True, 'dry_run': False, 'risk_usd': 100.0, 'daily_kill_usd': -600.0, 'weekly_kill_usd': -1500.0, 'max_notional_usd': 5000.0,
            'min_price': 1.0, 'min_adv20': 100_000.0, 'max_spread_bps': 100.0, 'order_timeout_s': 75.0,
            'params': {'consol_bars': 5, 'consol_pct': 0.04, 'min_dist_open_pct': 5.0, 'rv_lo': 1.0, 'rv_hi': 5.0, 'min_r_pct': 1.0, 'cap': 0.006,
                       'target_r': 2.0, 'max_per_day': 8, 'max_concurrent': 4, 'last_entry_minute': 930, 'flat_minute': 955}}
    base.update(over); return base


@pytest.fixture
def trades_db(tmp_path):
    p = tmp_path / 'trades.db'
    con = sqlite3.connect(p); con.execute("create table trades (id integer primary key, strategy text, trade_date text, pnl real)"); con.commit(); con.close()
    return p


@pytest.fixture
def mock_alpaca():
    a = MagicMock(spec=AlpacaClient)
    a.get_latest_quote.return_value = {'bid_price': 11.00, 'ask_price': 11.05, 'bid_size': 100, 'ask_size': 100}
    a.submit_bracket_order.return_value = {'id': 'o1', 'status': 'accepted', 'legs': [{'id': 'tp1', 'limit_price': 11.76, 'stop_price': None}, {'id': 'sl1', 'limit_price': None, 'stop_price': 10.7}]}
    a.get_order.return_value = {'status': 'accepted', 'filled_qty': 0, 'filled_avg_price': None}
    a.get_1min_bars_multi.return_value = {}
    a.get_open_positions.return_value = []
    a.cancel_order.return_value = True
    a.close_position.return_value = {'id': 'c1', 'status': 'accepted'}
    return a


@pytest.fixture
def mock_db(trades_db):
    d = MagicMock(spec=Database)
    d._trades_path = str(trades_db)
    d.get_active_universe.return_value = [{'symbol': 'ABC', 'avg_volume_daily': 1_000_000}, {'symbol': 'THIN', 'avg_volume_daily': 10_000}]
    d.save_trade.return_value = 7
    d.get_open_trades.return_value = []
    return d


@pytest.fixture
def mock_sm():
    s = MagicMock(spec=StopMonitor); s.polling_mode = False; return s


@pytest.fixture
def engine(mock_alpaca, mock_db, mock_sm):
    e = HodBreakEngine(mock_alpaca, mock_db, mock_sm, notifier=None, cfg=cfg())
    e._roll_session()
    return e


def bars_df(tape, minute0=570):
    """closed 1-min bars as the StopMonitor hands them (UTC timestamps)."""
    base = datetime(2026, 9, 14, 13, 30, tzinfo=timezone.utc) + timedelta(minutes=minute0 - 570)
    return pd.DataFrame([{'timestamp': base + timedelta(minutes=i), 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v} for i, (o, h, l, c, v) in enumerate(tape)])


def admit(engine, sym='ABC'):
    engine.on_mover(sym, price=11.0, day_open=10.0, cum_volume=40000, above_open_pct=10.0)
    engine._admit_movers()


class TestGates:
    def test_disabled_is_inert(self, mock_alpaca, mock_db, mock_sm):
        e = HodBreakEngine(mock_alpaca, mock_db, mock_sm, cfg=cfg(enabled=False))
        e.on_mover('ABC', price=11, day_open=10, cum_volume=1, above_open_pct=10); e.process_tick()
        assert e.candidates == {} and not mock_alpaca.submit_bracket_order.called

    def test_admit_filters_thin_and_subscribes(self, engine, mock_sm):
        admit(engine, 'THIN'); admit(engine, 'ABC')
        assert 'THIN' not in engine.candidates and 'ABC' in engine.candidates
        mock_sm.subscribe_bars.assert_called_once_with('ABC')

    def test_signal_on_last_bar_submits_bracket(self, engine, mock_alpaca, mock_db):
        admit(engine); engine._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1]))     # ends on the break bar
        mock_alpaca.submit_bracket_order.assert_called_once()
        kw = mock_alpaca.submit_bracket_order.call_args.kwargs
        assert kw['symbol'] == 'ABC' and kw['side'] == 'buy' and kw['limit_price'] == pytest.approx(round(11.0 * 1.006, 2))
        assert kw['sl_price'] == pytest.approx(10.7) and kw['tp_price'] == pytest.approx(round(kw['limit_price'] + 2 * (11.05 - 10.7), 2))   # R on the ask
        assert kw['qty'] == int(100.0 / (11.05 - 10.7))          # sized on the ask (the expected fill), not the limit
        rec = mock_db.save_trade.call_args.args[0]
        assert rec['strategy'] == STRATEGY_NAME and rec['order_status'] == 'pending_new' and json.loads(rec['pattern_data'])['tp_leg_id'] == 'tp1'
        assert engine.positions['ABC'].status == 'pending' and 'ABC' in engine.entered_today

    def test_stale_break_in_backfill_is_not_traded(self, engine, mock_alpaca):
        admit(engine); engine._ingest_bars('ABC', bars_df(drive_then_consolidate()))          # break bar is NOT the last bar
        assert not mock_alpaca.submit_bracket_order.called

    def test_no_chase_when_ask_above_cap(self, engine, mock_alpaca):
        mock_alpaca.get_latest_quote.return_value = {'bid_price': 11.10, 'ask_price': 11.20}
        admit(engine); engine._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1]))
        assert not mock_alpaca.submit_bracket_order.called and engine.candidates['ABC'].rejected_reason == 'no_chase'

    def test_wide_spread_skips(self, engine, mock_alpaca):
        mock_alpaca.get_latest_quote.return_value = {'bid_price': 10.80, 'ask_price': 11.05}
        admit(engine); engine._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1]))
        assert engine.candidates['ABC'].rejected_reason == 'spread'

    def test_missing_quote_fails_closed(self, engine, mock_alpaca):
        mock_alpaca.get_latest_quote.side_effect = RuntimeError('down')
        admit(engine); engine._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1]))
        assert engine.candidates['ABC'].rejected_reason == 'no_quote' and not mock_alpaca.submit_bracket_order.called

    def test_dry_run_logs_and_submits_nothing(self, mock_alpaca, mock_db, mock_sm):
        e = HodBreakEngine(mock_alpaca, mock_db, mock_sm, cfg=cfg(dry_run=True)); e._roll_session()
        admit(e); e._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1]))
        assert not mock_alpaca.submit_bracket_order.called and not mock_db.save_trade.called and e.candidates['ABC'].rejected_reason == 'dry_run'

    def test_daily_kill_blocks(self, engine, mock_alpaca, trades_db):
        con = sqlite3.connect(trades_db); con.execute("insert into trades(strategy, trade_date, pnl) values (?,?,?)", (STRATEGY_NAME, engine.session_date, -700.0)); con.commit(); con.close()
        admit(engine); engine._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1]))
        assert not mock_alpaca.submit_bracket_order.called

    def test_kill_rail_fails_closed_without_db_path(self, mock_alpaca, mock_sm, trades_db):
        d = MagicMock(spec=Database); d.get_active_universe.return_value = [{'symbol': 'ABC', 'avg_volume_daily': 1_000_000}]; d.get_open_trades.return_value = []
        e = HodBreakEngine(mock_alpaca, d, mock_sm, cfg=cfg()); e._roll_session()
        assert e._kill_rails_blocked() == 'weekly_kill'

    def test_concurrency_and_day_caps(self, engine, mock_alpaca):
        e = engine
        for i in range(4): e.positions[f'P{i}'] = MagicMock(status='open')
        admit(e); e._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1]))
        assert e.candidates['ABC'].rejected_reason == 'concurrency'
        e.positions.clear(); e.entered_today = {f'S{i}' for i in range(8)}; e.candidates.clear()
        e._adv_map['XYZ'] = 1_000_000; admit(e, 'XYZ'); e._ingest_bars('XYZ', bars_df(drive_then_consolidate()[:-1]))
        assert e.candidates['XYZ'].rejected_reason == 'day_cap'


class TestLifecycle:
    def _pending(self, engine):
        admit(engine); engine._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1])); return engine.positions['ABC']

    def test_fill_confirms_and_updates_db(self, engine, mock_alpaca, mock_db):
        pos = self._pending(engine)
        mock_alpaca.get_order.return_value = {'status': 'filled', 'filled_qty': pos.shares, 'filled_avg_price': 11.04}
        engine._process_pending_fills()
        assert pos.status == 'open' and pos.fill_price == 11.04
        assert mock_db.update_trade.call_args.args[1]['order_status'] == 'filled'

    def test_timeout_cancels_unfilled(self, engine, mock_alpaca, mock_db):
        pos = self._pending(engine); pos.submitted_at -= timedelta(seconds=100)
        engine._process_pending_fills()
        mock_alpaca.cancel_order.assert_called_with('o1'); assert 'ABC' not in engine.positions
        assert mock_db.update_trade.call_args.args[1]['order_status'] == 'time_stop_canceled'

    def test_timeout_with_partial_fill_keeps_the_shares(self, engine, mock_alpaca):
        pos = self._pending(engine); pos.submitted_at -= timedelta(seconds=100)
        mock_alpaca.get_order.return_value = {'status': 'partially_filled', 'filled_qty': 3, 'filled_avg_price': 11.05}
        engine._process_pending_fills()
        assert engine.positions['ABC'].status == 'open' and engine.positions['ABC'].shares == 3

    def test_target_leg_fill_records_exit_and_pnl(self, engine, mock_alpaca, mock_db):
        pos = self._pending(engine); pos.status = 'open'; pos.fill_price = 11.05
        def get_order(oid): return {'status': 'filled', 'filled_qty': pos.shares, 'filled_avg_price': pos.target} if oid == 'tp1' else {'status': 'accepted', 'filled_qty': 0}
        mock_alpaca.get_order.side_effect = get_order
        assert engine.check_exits() == ['ABC'] and 'ABC' not in engine.positions
        upd = mock_db.update_trade.call_args.args[1]
        assert upd['exit_reason'] == 'target' and upd['pnl'] == pytest.approx((pos.target - 11.05) * pos.shares) and engine.daily_pnl > 0

    def test_stop_leg_fill_records_loss(self, engine, mock_alpaca, mock_db):
        pos = self._pending(engine); pos.status = 'open'; pos.fill_price = 11.05
        mock_alpaca.get_order.side_effect = lambda oid: {'status': 'filled', 'filled_qty': pos.shares, 'filled_avg_price': 10.69} if oid == 'sl1' else {'status': 'accepted', 'filled_qty': 0}
        engine.check_exits()
        assert mock_db.update_trade.call_args.args[1]['exit_reason'] == 'stop' and engine.daily_pnl < 0

    def test_force_close_sells_only_our_shares_and_retries_until_flat(self, engine, mock_alpaca):
        pos = self._pending(engine); pos.status = 'open'; pos.fill_price = 11.05
        mock_alpaca.submit_limit_sell_order.return_value = {'id': 'c1', 'status': 'accepted'}
        with patch('trading.hod_break_engine.time.sleep'):
            n = engine.force_close_all()
        assert n == 1 and mock_alpaca.cancel_order.call_count == 2 and not mock_alpaca.close_position.called
        kw = mock_alpaca.submit_limit_sell_order.call_args.args
        assert kw[0] == 'ABC' and kw[1] == pos.shares and kw[2] == pytest.approx(round(11.00 * 0.99, 2))   # OUR qty, marketable limit off the bid
        assert pos.close_order_id == 'c1' and not engine._flattened                                          # not flat until the sell fills
        # the close order dies unfilled → re-submitted on the next force-close pass
        mock_alpaca.get_order.side_effect = lambda oid: {'status': 'canceled', 'filled_qty': 0} if oid == 'c1' else {'status': 'accepted', 'filled_qty': 0}
        mock_alpaca.submit_limit_sell_order.return_value = {'id': 'c2', 'status': 'accepted'}
        with patch('trading.hod_break_engine.time.sleep'):
            engine.force_close_all()
        assert pos.close_order_id == 'c2'
        mock_alpaca.get_order.side_effect = lambda oid: {'status': 'filled', 'filled_qty': pos.shares, 'filled_avg_price': 11.2} if oid == 'c2' else {'status': 'canceled', 'filled_qty': 0}
        engine.check_exits(); assert 'ABC' not in engine.positions

    def test_force_close_cancels_pending_entry(self, engine, mock_alpaca):
        self._pending(engine)
        with patch('trading.hod_break_engine.time.sleep'):
            engine.force_close_all()
        mock_alpaca.cancel_order.assert_called_with('o1'); assert engine.positions == {}

    def test_is_force_close_time(self, engine):
        with patch.object(HodBreakEngine, '_minute_of_day', return_value=954): assert not engine.is_force_close_time()
        with patch.object(HodBreakEngine, '_minute_of_day', return_value=955): assert engine.is_force_close_time()


class TestRestart:
    def test_sync_rehydrates_open_and_pending(self, mock_alpaca, mock_db, mock_sm):
        e = HodBreakEngine(mock_alpaca, mock_db, mock_sm, cfg=cfg())
        today = e._et_now().strftime('%Y-%m-%d')
        mock_db.get_open_trades.return_value = [
            {'id': 1, 'symbol': 'AAA', 'order_id': 'o9', 'order_status': 'filled', 'shares': 10, 'entry_price': 11.07, 'stop_loss_price': 10.7, 'take_profit_price': 11.8, 'fill_price': 11.05,
             'pattern_data': json.dumps({'level': 11.0, 'tp_leg_id': 'tpA', 'sl_leg_id': 'slA'})},
            {'id': 2, 'symbol': 'BBB', 'order_id': 'o8', 'order_status': 'pending_new', 'shares': 5, 'entry_price': 5.03, 'stop_loss_price': 4.9, 'take_profit_price': 5.3, 'pattern_data': '{}'},
        ]
        mock_alpaca.get_open_positions.return_value = [{'symbol': 'AAA'}]
        assert e.sync_positions() == 2
        assert e.positions['AAA'].status == 'open' and e.positions['AAA'].tp_leg_id == 'tpA' and e.positions['BBB'].status == 'pending'

    def test_sync_flags_db_open_but_broker_flat(self, mock_alpaca, mock_db, mock_sm):
        e = HodBreakEngine(mock_alpaca, mock_db, mock_sm, cfg=cfg())
        mock_db.get_open_trades.return_value = [{'id': 1, 'symbol': 'AAA', 'order_id': 'o9', 'order_status': 'filled', 'shares': 10, 'entry_price': 11, 'stop_loss_price': 10, 'take_profit_price': 12, 'pattern_data': '{}'}]
        mock_alpaca.get_open_positions.return_value = [{'symbol': 'ZZZ'}]
        assert e.sync_positions() == 0
        assert mock_db.update_trade.call_args.args[1]['order_status'] == 'exit_pending_verification'


def test_bar_handler_enqueues_only_for_candidates(engine):
    engine._on_bar_close('NOPE', bars_df(drive_then_consolidate())); assert engine._bar_queue.empty()
    admit(engine); engine._on_bar_close('ABC', bars_df(drive_then_consolidate())); assert not engine._bar_queue.empty()
    assert engine.drain_bar_events() == ['ABC']


class TestBarMerge:
    """2026-09-14 09:50 DBI defect: the stream only carries bars since subscription; ingest must MERGE."""

    def test_stream_bars_do_not_replace_the_backfilled_open(self, engine):
        admit(engine)
        tape = drive_then_consolidate()
        engine._ingest_bars('ABC', bars_df(tape[:3]))                       # backfill: 09:30-09:32
        engine._ingest_bars('ABC', bars_df(tape[3:9], minute0=573))         # stream: 09:33-09:38 (longer than the backfill)
        bars = engine.candidates['ABC'].bars
        assert len(bars) == 9 and bars[0]['open'] == 10.0                  # the 09:30 open survived
        assert [b['open'] for b in bars] == [t[0] for t in tape[:9]]

    def test_duplicate_minutes_are_deduplicated(self, engine):
        admit(engine)
        tape = drive_then_consolidate()
        engine._ingest_bars('ABC', bars_df(tape[:5]))
        engine._ingest_bars('ABC', bars_df(tape[:5]))
        assert len(engine.candidates['ABC'].bars) == 5

    def test_merged_bars_give_the_true_open_and_hod(self, engine, mock_alpaca):
        """with the early bars lost, the engine would see a lower open (wrong floor) and a lower HOD (false break)"""
        admit(engine)
        tape = drive_then_consolidate()
        engine._ingest_bars('ABC', bars_df(tape[:3]))
        engine._ingest_bars('ABC', bars_df(tape[3:9], minute0=573))          # ends on the break bar → signal
        kw = mock_alpaca.submit_bracket_order.call_args.kwargs
        assert kw['limit_price'] == pytest.approx(round(11.0 * 1.006, 2))   # HOD 11.0 from the EARLY bars, not from the stream


class TestRMinOnTheAsk:
    """9/14 EOD parity: R must be measured on the expected fill (the ask), as the spec measures it on the next open."""

    def test_tight_stop_vs_ask_is_rejected_even_if_ok_vs_limit(self, engine, mock_alpaca):
        # level 11.0 → limit 11.07; stop 10.96 gives 1.0% vs the limit but only 0.9% vs an ask of 11.06
        tape = drive_then_consolidate()
        tape = tape[:3] + [(10.99, 10.995, 10.96, 10.97, 3000)] * 5 + [tape[-2]]
        mock_alpaca.get_latest_quote.return_value = {'bid_price': 11.05, 'ask_price': 11.06}
        admit(engine); engine._ingest_bars('ABC', bars_df(tape))
        assert not mock_alpaca.submit_bracket_order.called and engine.candidates['ABC'].rejected_reason == 'r_min'

    def test_size_uses_the_ask(self, engine, mock_alpaca):
        mock_alpaca.get_latest_quote.return_value = {'bid_price': 11.00, 'ask_price': 11.02}
        admit(engine); engine._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1]))
        kw = mock_alpaca.submit_bracket_order.call_args.kwargs
        assert kw['qty'] == int(100.0 / (11.02 - 10.7)) and kw['limit_price'] == pytest.approx(round(11.0 * 1.006, 2))


class TestRestartSafeCaps:
    def test_day_cap_counts_closed_rows_from_the_db(self, engine, trades_db):
        con = sqlite3.connect(trades_db); con.execute("alter table trades add column symbol text"); con.execute("alter table trades add column order_status text")
        for i in range(8): con.execute("insert into trades(strategy, trade_date, pnl, symbol, order_status) values (?,?,?,?,?)", (STRATEGY_NAME, engine.session_date, 1.0, f'S{i}', 'closed'))
        con.execute("insert into trades(strategy, trade_date, pnl, symbol, order_status) values (?,?,?,?,?)", (STRATEGY_NAME, engine.session_date, None, 'DEAD', 'time_stop_canceled'))
        con.commit(); con.close()
        engine.entered_today.clear()
        assert engine._entered_today_count() == 8 and 'DEAD' not in engine.entered_today

    def test_symbol_closed_earlier_today_is_not_re_entered_after_restart(self, engine, mock_alpaca, trades_db):
        con = sqlite3.connect(trades_db); con.execute("alter table trades add column symbol text"); con.execute("alter table trades add column order_status text")
        con.execute("insert into trades(strategy, trade_date, pnl, symbol, order_status) values (?,?,?,?,?)", (STRATEGY_NAME, engine.session_date, -100.0, 'ABC', 'closed')); con.commit(); con.close()
        engine.candidates.clear(); engine.entered_today.clear()
        admit(engine)
        assert 'ABC' not in engine.candidates and not mock_alpaca.submit_bracket_order.called
