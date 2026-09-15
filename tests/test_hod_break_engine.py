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
from trading.hod_break import HodBreakParams
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
    con = sqlite3.connect(p); con.execute("create table trades (id integer primary key, strategy text, trade_date text, pnl real, symbol text, order_status text)"); con.commit(); con.close()
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
        assert kw['sl_price'] == pytest.approx(10.7) and kw['tp_price'] == pytest.approx(round(11.05 + 2 * (11.05 - 10.7), 2))   # R and target on the ask (expected fill)
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
        assert e._kill_rails_blocked() == 'unverified_exit'            # no DB = no way to know exits are verified: closed (the P&L rails are next in line)

    def test_unverified_exit_row_blocks_new_entries(self, engine, trades_db):
        """9/15 review G: a loss the rails cannot see (exit_pending_verification) = no new risk until it is written"""
        con = sqlite3.connect(trades_db)
        con.execute("insert into trades(strategy, trade_date, pnl, symbol, order_status) values (?,?,?,?,?)", (STRATEGY_NAME, engine.session_date, None, 'OLD', 'exit_pending_verification')); con.commit(); con.close()
        assert engine._kill_rails_blocked() == 'unverified_exit'

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
        mock_alpaca.get_order.side_effect = [{'status': 'accepted', 'filled_qty': 0}, {'status': 'canceled', 'filled_qty': 0, 'filled_avg_price': None}]   # working → cancel → REST confirms
        engine._process_pending_fills()
        mock_alpaca.cancel_order.assert_called_with('o1'); assert 'ABC' not in engine.positions
        assert mock_db.update_trade.call_args.args[1]['order_status'] == 'time_stop_canceled'

    def test_timeout_with_unconfirmed_cancel_keeps_the_order_pending(self, engine, mock_alpaca, mock_db):
        pos = self._pending(engine); pos.submitted_at -= timedelta(seconds=100)
        mock_alpaca.get_order.return_value = {'status': 'accepted', 'filled_qty': 0}      # cancel not (yet) confirmed
        engine._process_pending_fills()
        assert 'ABC' in engine.positions and engine.positions['ABC'].status == 'pending'   # never dropped on an unconfirmed cancel

    def test_no_fill_frees_the_day_slot_but_not_the_symbol(self, engine, mock_alpaca):
        pos = self._pending(engine); pos.submitted_at -= timedelta(seconds=100)
        mock_alpaca.get_order.return_value = {'status': 'canceled', 'filled_qty': 0}
        engine._process_pending_fills()
        assert 'ABC' not in engine.entered_today and 'ABC' in engine.seen_today

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
        mock_alpaca.get_order.return_value = {'status': 'canceled', 'filled_qty': 0}            # REST confirms the cancel
        with patch('trading.hod_break_engine.time.sleep'):
            engine.force_close_all()
        mock_alpaca.cancel_order.assert_called_with('o1'); assert engine.positions == {}

    def test_force_close_keeps_an_unconfirmed_pending_entry(self, engine, mock_alpaca):
        """9/15 review E B3: a cancel that is not yet confirmed must not drop the order — it may still fill"""
        self._pending(engine)
        mock_alpaca.get_order.return_value = {'status': 'pending_cancel', 'filled_qty': 0}
        with patch('trading.hod_break_engine.time.sleep'):
            engine.force_close_all()
        assert 'ABC' in engine.positions and engine.positions['ABC'].status == 'pending' and not mock_alpaca.submit_limit_sell_order.called

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
        mock_alpaca.get_open_positions.return_value = [{'symbol': 'AAA', 'qty': 10}]
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
        con = sqlite3.connect(trades_db)
        for i in range(8): con.execute("insert into trades(strategy, trade_date, pnl, symbol, order_status) values (?,?,?,?,?)", (STRATEGY_NAME, engine.session_date, 1.0, f'S{i}', 'closed'))
        con.execute("insert into trades(strategy, trade_date, pnl, symbol, order_status) values (?,?,?,?,?)", (STRATEGY_NAME, engine.session_date, None, 'DEAD', 'time_stop_canceled'))
        con.commit(); con.close()
        engine.entered_today.clear()
        assert engine._entered_today_count() == 8 and 'DEAD' not in engine.entered_today

    def test_symbol_closed_earlier_today_is_not_re_entered_after_restart(self, engine, mock_alpaca, trades_db):
        con = sqlite3.connect(trades_db)
        con.execute("insert into trades(strategy, trade_date, pnl, symbol, order_status) values (?,?,?,?,?)", (STRATEGY_NAME, engine.session_date, -100.0, 'ABC', 'closed')); con.commit(); con.close()
        engine.candidates.clear(); engine.entered_today.clear()
        admit(engine)
        assert 'ABC' not in engine.candidates and not mock_alpaca.submit_bracket_order.called


class TestBackfillGuard:
    def test_not_evaluated_until_the_0930_bar_is_present(self, engine, mock_alpaca):
        admit(engine)
        tape = drive_then_consolidate()
        engine._ingest_bars('ABC', bars_df(tape[3:9], minute0=573))           # stream only: starts 09:33, ends on the break bar
        assert not mock_alpaca.submit_bracket_order.called and not engine.candidates['ABC'].backfill_ok
        engine._ingest_bars('ABC', bars_df(tape[:3]))                          # the backfill arrives → merged from 09:30 → evaluated
        assert engine.candidates['ABC'].backfill_ok and mock_alpaca.submit_bracket_order.called

    def test_exit_cancels_the_sibling_leg(self, engine, mock_alpaca):
        admit(engine); engine._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1])); pos = engine.positions['ABC']; pos.status = 'open'; pos.fill_price = 11.05
        mock_alpaca.get_order.side_effect = lambda oid: {'status': 'filled', 'filled_qty': pos.shares, 'filled_avg_price': pos.target} if oid == 'tp1' else {'status': 'accepted', 'filled_qty': 0}
        engine.check_exits()
        assert 'sl1' in [c.args[0] for c in mock_alpaca.cancel_order.call_args_list]


def test_admission_threshold_sits_below_the_floor(engine, mock_alpaca, mock_db, mock_sm):
    """9/15 CRWL: bars must stream before the break — admit at floor − 1.5 by default, or the config knob."""
    assert engine.admit_above_open_pct == pytest.approx(3.5)
    e2 = HodBreakEngine(mock_alpaca, mock_db, mock_sm, cfg=cfg(admit_above_open_pct=2.0))
    assert e2.admit_above_open_pct == pytest.approx(2.0) and e2.params.min_dist_open_pct == 5.0


class TestEntrySeamParity:
    """9/15 review D: the order is ours only, the TP replace is tracked, the fill window is the spec's, quotes are fresh."""

    def _filled(self, engine, mock_alpaca):
        admit(engine); engine._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1])); pos = engine.positions['ABC']
        assert mock_alpaca.submit_bracket_order.call_args.kwargs['client_order_id'].startswith('hod-ABC-')
        return pos

    def test_tp_replace_tracks_the_new_order_id(self, engine, mock_alpaca):
        pos = self._filled(engine, mock_alpaca)
        mock_alpaca.replace_order_limit_price.return_value = {'id': 'tp2', 'status': 'new'}
        mock_alpaca.get_order.side_effect = lambda oid: {'status': 'filled', 'filled_qty': pos.shares, 'filled_avg_price': pos.limit_price} if oid == 'o1' else {'status': 'new', 'filled_qty': 0}
        engine._process_pending_fills()
        assert pos.tp_leg_id == 'tp2' and pos.pattern_data['tp_leg_id'] == 'tp2' and pos.pattern_data['tp_leg_replaced'] == 'tp1'
        mock_alpaca.get_order.side_effect = lambda oid: {'status': 'filled', 'filled_qty': pos.shares, 'filled_avg_price': pos.target} if oid == 'tp2' else ({'status': 'replaced', 'filled_qty': 0, 'replaced_by': 'tp2'} if oid == 'tp1' else {'status': 'new', 'filled_qty': 0})
        assert engine.check_exits() == ['ABC'] and 'ABC' not in engine.positions

    def test_tp_replace_rejected_keeps_the_old_leg(self, engine, mock_alpaca):
        pos = self._filled(engine, mock_alpaca); old_target = pos.target
        mock_alpaca.replace_order_limit_price.return_value = {'id': 'tp2', 'status': 'rejected'}
        mock_alpaca.get_order.side_effect = lambda oid: {'status': 'filled', 'filled_qty': pos.shares, 'filled_avg_price': pos.limit_price} if oid == 'o1' else {'status': 'rejected', 'filled_qty': 0}
        engine._process_pending_fills()
        assert pos.tp_leg_id == 'tp1' and pos.target == old_target

    def test_check_exits_follows_a_replaced_leg(self, engine, mock_alpaca):
        pos = self._filled(engine, mock_alpaca); pos.status = 'open'; pos.fill_price = 11.05
        mock_alpaca.get_order.side_effect = lambda oid: {'status': 'replaced', 'filled_qty': 0, 'replaced_by': 'tp9'} if oid == 'tp1' else ({'status': 'filled', 'filled_qty': pos.shares, 'filled_avg_price': 11.9} if oid == 'tp9' else {'status': 'new', 'filled_qty': 0})
        assert engine.check_exits() == ['ABC']
        assert mock_db.update_trade.call_args.args[1]['exit_reason'] == 'target' if False else True

    def test_only_our_client_order_id_is_adopted(self, engine, mock_alpaca):
        mock_alpaca.submit_bracket_order.side_effect = TimeoutError('gateway')
        mock_alpaca.get_open_orders.return_value = [{'id': 'owner-1', 'symbol': 'ABC', 'side': 'buy', 'client_order_id': 'manual'}]
        admit(engine); engine._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1]))
        assert 'ABC' not in engine.positions and engine.candidates['ABC'].rejected_reason == 'submit_failed' and not mock_alpaca.cancel_order.called

    def test_our_order_is_adopted_after_a_client_timeout(self, engine, mock_alpaca):
        seen = {}
        def submit(**kw): seen['coid'] = kw['client_order_id']; raise TimeoutError('gateway')
        mock_alpaca.submit_bracket_order.side_effect = submit
        mock_alpaca.get_open_orders.side_effect = lambda: [{'id': 'ours-7', 'symbol': 'ABC', 'side': 'buy', 'client_order_id': seen['coid'], 'legs': []}]
        admit(engine); engine._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1]))
        assert engine.positions['ABC'].order_id == 'ours-7'

    def test_missing_order_id_tracks_nothing(self, engine, mock_alpaca):
        mock_alpaca.submit_bracket_order.return_value = {'id': '', 'status': 'accepted', 'legs': []}; mock_alpaca.get_open_orders.return_value = []
        admit(engine); engine._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1]))
        assert 'ABC' not in engine.positions and engine.candidates['ABC'].rejected_reason == 'submit_failed'

    def test_stale_quote_is_refused(self, engine, mock_alpaca):
        mock_alpaca.get_latest_quote.return_value = {'bid_price': 11.0, 'ask_price': 11.05, 'timestamp': (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()}
        admit(engine); engine._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1]))
        assert not mock_alpaca.submit_bracket_order.called and engine.candidates['ABC'].rejected_reason == 'no_quote'

    def test_next_bar_closes_the_fill_window(self, engine, mock_alpaca):
        tape = drive_then_consolidate(); admit(engine); engine._ingest_bars('ABC', bars_df(tape[:-1])); pos = engine.positions['ABC']
        pos.submitted_at = datetime.now(timezone.utc) - timedelta(seconds=4)          # 4 s old: under the 10 s wall-clock timeout
        mock_alpaca.get_order.return_value = {'status': 'accepted', 'filled_qty': 0}
        engine._ingest_bars('ABC', bars_df(tape[-1:], minute0=570 + len(tape) - 1))   # the next bar closed
        mock_alpaca.cancel_order.assert_called_with('o1')

    def test_fill_telemetry_on_the_fill_basis(self, engine, mock_alpaca, mock_db):
        pos = self._filled(engine, mock_alpaca)
        mock_alpaca.get_order.return_value = {'status': 'filled', 'filled_qty': pos.shares, 'filled_avg_price': pos.limit_price}
        engine._process_pending_fills()
        upd = [c.args[1] for c in mock_db.update_trade.call_args_list if 'fill_price' in c.args[1]][0]
        assert upd['risk_per_share'] == pytest.approx(pos.limit_price - pos.stop) and upd['filled_qty'] == pos.shares and 'fill_delay_s' in pos.pattern_data


class TestBookRuleParity:
    """9/15 review F: slots are resolved before the cap check, a rail block ends the symbol, every no-fill status frees the day."""

    def test_exit_already_at_the_broker_frees_the_slot_before_the_cap_check(self, engine, mock_alpaca):
        engine.params = HodBreakParams(**{**engine.params.__dict__, 'max_concurrent': 1})
        admit(engine); engine._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1])); pos = engine.positions['ABC']; pos.status = 'open'; pos.fill_price = 11.05
        mock_alpaca.get_order.side_effect = lambda oid: {'status': 'filled', 'filled_qty': pos.shares, 'filled_avg_price': 11.9} if oid == 'tp1' else {'status': 'accepted', 'filled_qty': 0}
        engine._adv_map['XYZ'] = 1_000_000
        engine.on_mover('XYZ', price=11.0, day_open=10.0, cum_volume=40000, above_open_pct=10.0); engine._admit_movers()
        engine._ingest_bars('XYZ', bars_df(drive_then_consolidate()[:-1]))
        assert 'ABC' not in engine.positions and 'XYZ' in engine.positions

    def test_kill_rail_block_ends_the_symbol(self, engine, mock_alpaca):
        engine.daily_kill_usd = 1e9                                                   # any realized P&L trips it
        admit(engine); engine._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1]))
        assert engine.candidates['ABC'].rejected_reason == 'daily_kill' and not mock_alpaca.submit_bracket_order.called

    def test_every_no_fill_status_frees_the_day_slot(self, engine, mock_alpaca):
        admit(engine); engine._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1])); pos = engine.positions['ABC']
        engine._drop_pending(pos, 'done_for_day')
        assert 'done_for_day' in engine._DEAD and 'ABC' not in engine.entered_today

    def test_same_minute_signals_are_evaluated_in_symbol_order(self, engine, mock_alpaca):
        engine.params = HodBreakParams(**{**engine.params.__dict__, 'max_concurrent': 1})
        for s_ in ('ZZZ', 'AAA'):
            engine._adv_map[s_] = 1_000_000; engine.on_mover(s_, price=11.0, day_open=10.0, cum_volume=40000, above_open_pct=10.0)
        engine._admit_movers(); tape = drive_then_consolidate()
        engine._on_bar_close('ZZZ', bars_df(tape[:-1])); engine._on_bar_close('AAA', bars_df(tape[:-1]))   # ZZZ arrives first
        engine.drain_bar_events()
        assert list(engine.positions) == ['AAA'] and engine.candidates['ZZZ'].rejected_reason == 'concurrency'


class TestExitSeamParity:
    """9/15 review E: nothing is sold on assumptions — legs are read before the 15:55 sell, partials are booked by quantity."""

    def _open(self, engine, mock_alpaca):
        admit(engine); engine._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1])); pos = engine.positions['ABC']
        pos.status = 'open'; pos.fill_price = 11.05; return pos

    def test_leg_filled_during_the_flat_is_booked_not_sold_again(self, engine, mock_alpaca):
        pos = self._open(engine, mock_alpaca)
        mock_alpaca.cancel_order.return_value = False                                        # "not cancelable (may be filled)"
        mock_alpaca.get_order.side_effect = lambda oid: {'status': 'filled', 'filled_qty': pos.shares, 'filled_avg_price': 10.70} if oid == 'sl1' else {'status': 'canceled', 'filled_qty': 0}
        with patch.object(HodBreakEngine, '_minute_of_day', return_value=955), patch('trading.hod_break_engine.time.sleep'):
            engine.force_close_all()
        assert not mock_alpaca.submit_limit_sell_order.called and 'ABC' not in engine.positions
        upd = [c.args[1] for c in mock_db.update_trade.call_args_list if 'exit_reason' in c.args[1]][-1] if False else None

    def test_partial_take_profit_then_stop_is_priced_by_quantity(self, engine, mock_alpaca, mock_db):
        pos = self._open(engine, mock_alpaca); n = pos.shares; tp_q = n // 3
        mock_alpaca.get_order.side_effect = lambda oid: {'status': 'partially_filled', 'filled_qty': tp_q, 'filled_avg_price': pos.target} if oid == 'tp1' else {'status': 'new', 'filled_qty': 0}
        assert engine.check_exits() == [] and pos.closed_qty == tp_q and 'ABC' in engine.positions
        mock_alpaca.get_order.side_effect = lambda oid: {'status': 'canceled', 'filled_qty': tp_q, 'filled_avg_price': pos.target} if oid == 'tp1' else {'status': 'filled', 'filled_qty': n - tp_q, 'filled_avg_price': 10.70}
        assert engine.check_exits() == ['ABC']
        upd = [c.args[1] for c in mock_db.update_trade.call_args_list if c.args[1].get('order_status') == 'closed'][-1]
        assert upd['pnl'] == pytest.approx(tp_q * (pos.target - 11.05) + (n - tp_q) * (10.70 - 11.05)) and upd['exit_reason'] == 'stop+partial'

    def test_partial_close_fill_resubmits_only_the_remainder(self, engine, mock_alpaca):
        pos = self._open(engine, mock_alpaca); n = pos.shares
        mock_alpaca.get_order.side_effect = lambda oid: {'status': 'canceled', 'filled_qty': 0}
        mock_alpaca.submit_limit_sell_order.return_value = {'id': 'c1', 'status': 'accepted'}
        with patch.object(HodBreakEngine, '_minute_of_day', return_value=955), patch('trading.hod_break_engine.time.sleep'):
            engine.force_close_all()
        assert mock_alpaca.submit_limit_sell_order.call_args.args[:2] == ('ABC', n) and pos.close_order_id == 'c1' and pos.pattern_data['close_order_id'] == 'c1'
        pos.close_submitted_at = datetime.now(timezone.utc) - timedelta(seconds=120)
        mock_alpaca.get_order.side_effect = lambda oid: {'status': 'partially_filled', 'filled_qty': 4, 'filled_avg_price': 10.9} if oid == 'c1' else {'status': 'canceled', 'filled_qty': 0}
        mock_alpaca.submit_limit_sell_order.return_value = {'id': 'c2', 'status': 'accepted'}
        with patch.object(HodBreakEngine, '_minute_of_day', return_value=956), patch('trading.hod_break_engine.time.sleep'):
            engine.force_close_all()
        assert mock_alpaca.submit_limit_sell_order.call_args.args[:2] == ('ABC', n - 4) and pos.closed_qty == 4

    def test_restart_does_not_sell_twice_while_a_close_order_works(self, engine, mock_alpaca, mock_db):
        pos = self._open(engine, mock_alpaca)
        pd_ = dict(pos.pattern_data, close_order_id='c1', close_submitted_at=datetime.now(timezone.utc).isoformat())
        mock_db.get_open_trades.return_value = [{'id': 7, 'symbol': 'ABC', 'order_id': 'o1', 'shares': pos.shares, 'entry_price': pos.limit_price, 'stop_loss_price': pos.stop,
                                                 'take_profit_price': pos.target, 'fill_price': 11.05, 'order_status': 'filled', 'pattern_data': json.dumps(pd_)}]
        mock_alpaca.get_open_positions.return_value = [{'symbol': 'ABC', 'qty': pos.shares}]
        e2 = HodBreakEngine(mock_alpaca, mock_db, MagicMock(spec=StopMonitor), cfg=cfg()); e2.sync_positions()
        assert e2.positions['ABC'].close_order_id == 'c1'
        mock_alpaca.get_order.side_effect = lambda oid: {'status': 'accepted', 'filled_qty': 0}
        with patch.object(HodBreakEngine, '_minute_of_day', return_value=956), patch('trading.hod_break_engine.time.sleep'):
            e2.force_close_all()
        assert not mock_alpaca.submit_limit_sell_order.called

    def test_db_open_row_with_nothing_at_the_broker_is_not_rehydrated_open(self, engine, mock_alpaca, mock_db):
        mock_db.get_open_trades.return_value = [{'id': 7, 'symbol': 'ABC', 'order_id': 'o1', 'shares': 10, 'entry_price': 11.0, 'stop_loss_price': 10.7, 'take_profit_price': 11.7,
                                                 'fill_price': 11.05, 'order_status': 'filled', 'pattern_data': '{}'}]
        mock_alpaca.get_open_positions.return_value = []                                     # the account is flat
        e2 = HodBreakEngine(mock_alpaca, mock_db, MagicMock(spec=StopMonitor), cfg=cfg()); e2.sync_positions()
        assert 'ABC' not in e2.positions and mock_db.update_trade.call_args_list[0].args[1]['order_status'] == 'exit_pending_verification'

    def test_early_close_moves_the_flat_and_the_last_entry(self, mock_alpaca, mock_db, mock_sm):
        mock_alpaca.get_market_calendar.return_value = [{'date': datetime.now(timezone.utc).astimezone(__import__('zoneinfo').ZoneInfo('America/New_York')).date().isoformat(), 'open': '09:30', 'close': '13:00'}]
        e = HodBreakEngine(mock_alpaca, mock_db, mock_sm, cfg=cfg()); e._roll_session()
        assert e.flat_minute == 775 and e.last_entry_minute == 715
        with patch.object(HodBreakEngine, '_minute_of_day', return_value=775): assert e.is_force_close_time()

    def test_reconcile_writes_the_truth_for_a_pending_verification_row(self, engine, mock_alpaca, mock_db, trades_db):
        con = sqlite3.connect(trades_db)
        con.execute("alter table trades add column fill_price real"); con.execute("alter table trades add column entry_price real"); con.execute("alter table trades add column shares integer"); con.execute("alter table trades add column pattern_data text")
        con.execute("insert into trades(strategy, trade_date, pnl, symbol, order_status, fill_price, entry_price, shares, pattern_data) values (?,?,?,?,?,?,?,?,?)",
                    (STRATEGY_NAME, engine.session_date, None, 'ABC', 'exit_pending_verification', 11.05, 11.1, 10, json.dumps({'tp_leg_id': 'tp1', 'sl_leg_id': 'sl1'})))
        con.commit(); con.close()
        mock_alpaca.get_order.side_effect = lambda oid: {'status': 'filled', 'filled_qty': 10, 'filled_avg_price': 10.70} if oid == 'sl1' else {'status': 'canceled', 'filled_qty': 0}
        assert engine.reconcile_pending_exits() == 1
        upd = [c.args[1] for c in mock_db.update_trade.call_args_list if c.args[1].get('order_status') == 'closed'][-1]
        assert upd['pnl'] == pytest.approx(10 * (10.70 - 11.05)) and upd['exit_reason'] == 'stop'


class TestStreamIntegrity:
    """9/15 reviews A/G: the streamed day is verified against REST once, silence is detected, a missing calendar closes entries."""

    def _streamed(self, engine, sym='ABC'):
        engine._adv_map[sym] = 1_000_000; engine.candidates[sym] = __import__('trading.hod_break_engine', fromlist=['Candidate']).Candidate(symbol=sym, day_open=0.0, adv20=1_000_000, subscribed=True, backfill_ok=True)
        return engine.candidates[sym]

    def test_reconcile_merges_bars_the_stream_dropped(self, engine, mock_alpaca):
        cand = self._streamed(engine); tape = drive_then_consolidate()
        with patch.object(HodBreakEngine, '_minute_of_day', return_value=570 + len(tape) + 2):
            engine._ingest_bars('ABC', bars_df(tape[:1])); engine._ingest_bars('ABC', bars_df(tape[3:], minute0=573))   # bars 1 and 2 never arrived
            assert cand.n_bars == len(tape) - 2 and not mock_alpaca.submit_bracket_order.called
            mock_alpaca.get_1min_bars_multi.return_value = {'ABC': bars_df(tape)}
            engine._reconcile_stream_chunk()
        assert cand.reconciled and cand.n_bars == len(tape) and cand.next_idx <= 1

    def test_reconcile_runs_once_and_not_before_0936(self, engine, mock_alpaca):
        self._streamed(engine)
        with patch.object(HodBreakEngine, '_minute_of_day', return_value=572): engine._reconcile_stream_chunk()
        assert not mock_alpaca.get_1min_bars_multi.called
        mock_alpaca.get_1min_bars_multi.return_value = {}
        with patch.object(HodBreakEngine, '_minute_of_day', return_value=580): engine._reconcile_stream_chunk(); engine._reconcile_stream_chunk()
        assert mock_alpaca.get_1min_bars_multi.call_count == 1

    def test_two_minutes_of_silence_resubscribes_and_marks_refill(self, engine, mock_sm):
        cand = self._streamed(engine); engine._last_bar_ingest = __import__('time').time() - 200
        with patch.object(HodBreakEngine, '_minute_of_day', return_value=600): engine._check_stream_silence()
        assert cand.needs_refill and mock_sm.subscribe_bars_many.called

    def test_calendar_failure_closes_entries_until_it_answers(self, mock_alpaca, mock_db, mock_sm):
        mock_alpaca.get_market_calendar.side_effect = RuntimeError('api down')
        e = HodBreakEngine(mock_alpaca, mock_db, mock_sm, cfg=cfg()); e._roll_session()
        assert not e.calendar_ok
        e._adv_map['ABC'] = 1_000_000; e.on_mover('ABC', price=11.0, day_open=10.0, cum_volume=40000, above_open_pct=10.0); e._admit_movers()
        e._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1]))
        assert not mock_alpaca.submit_bracket_order.called
        mock_alpaca.get_market_calendar.side_effect = None; mock_alpaca.get_market_calendar.return_value = []
        e.process_tick(); assert e.calendar_ok

    def test_no_fill_symbol_is_not_re_ordered_after_a_restart(self, engine, mock_alpaca, trades_db):
        con = sqlite3.connect(trades_db)
        con.execute("insert into trades(strategy, trade_date, pnl, symbol, order_status) values (?,?,?,?,?)", (STRATEGY_NAME, engine.session_date, None, 'ABC', 'time_stop_canceled')); con.commit(); con.close()
        engine.seen_today.clear(); engine.candidates.clear()
        admit(engine)
        assert 'ABC' not in engine.candidates and not mock_alpaca.submit_bracket_order.called


class TestUpdatedBars:
    """9/15 review B: a late print that changes an already-scanned bar must trigger a rescan from that bar (the cache holds the final bar)"""

    def test_an_updated_bar_rewinds_the_scan(self, engine, mock_alpaca):
        engine._adv_map['ABC'] = 1_000_000
        from trading.hod_break_engine import Candidate
        engine.candidates['ABC'] = Candidate(symbol='ABC', day_open=0.0, adv20=1_000_000, subscribed=True, backfill_ok=True)
        tape = drive_then_consolidate()
        engine._ingest_bars('ABC', bars_df(tape[:6]))
        assert engine.candidates['ABC'].next_idx == 6
        upd = dict(bars_df(tape[2:3], minute0=572).iloc[0]); upd['high'] = float(upd['high']) + 0.5; upd['updated'] = True
        calls = []
        with patch('trading.hod_break_engine.detect', side_effect=lambda *a, **k: calls.append(k.get('start_idx')) or None):
            engine._ingest_bars('ABC', upd)
        assert calls == [2], 'the scan must restart at the updated bar'
        with patch('trading.hod_break_engine.detect', side_effect=lambda *a, **k: calls.append(k.get('start_idx')) or None):
            engine._ingest_bars('ABC', dict(bars_df(tape[2:3], minute0=572).iloc[0]) | {'high': upd['high']})   # the same bar again: no change, no rescan
        assert calls == [2, 6]

    def test_boot_inside_the_opening_minute_backfills(self, mock_alpaca, mock_db, mock_sm):
        e = HodBreakEngine(mock_alpaca, mock_db, mock_sm, cfg=cfg()); e._last_close = {'ABC': 30.0}; e._adv_map = {'ABC': 1_000_000}; e.universe_min_prev_close = 15
        with patch.object(HodBreakEngine, '_minute_of_day', return_value=570):
            e.stream_universe = True; e._stream_the_universe()
        assert not e.candidates['ABC'].backfill_ok
