"""docs/hod_oco_persistence_fix_20261017.md — the 10/15 QCMU / RBRK defects (docs/hod_week42_triage.md):
1. every safety-net OCO replacement is PERSISTED (trade row + registry agree), 2. the cancelled OCO settles BEFORE the
replacement is submitted (retry once, then ERROR + ungated Telegram), 3. reconcile matches broker sells by symbol/side/
qty/time when stored leg ids are stale, 4. a prior-day open row whose symbol the account still holds is adopted at boot,
6. the mover queue dedups and drains whole, 7. a stale parity-ledger header is rotated and a real fill carries arm['idx'].
(Item 5, the StopMonitor breaker reason, is in tests/test_stop_monitor_circuit_breaker.py.)"""
import csv
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from trading.hod_break_engine import Candidate, HodBreakEngine
from tests.test_hod_live_resting import live_engine, BIG_VOL_ARM
from tests.test_hod_arm_features import BARS, ARM, ADV20_SYNTH
from tests.test_hod_break_engine import cfg
from tests.test_hod_break_engine_boot_defects import real_db, _seed_row, TODAY, hod_cfg

FAST = dict(oco_settle_timeout_s=0.3, oco_settle_poll_s=0.01, oco_retry_delay_s=0.0)


def _lo(**kw):
    """A live_order dict as _arm_live_order builds it."""
    lo = dict(order_id='buy-1', coid='hod-rest-ABC-1', level=11.0, trigger=11.01, limit=11.0165, stop=10.6, qty=63, booked_qty=0,
              arm_ts='', tp_leg_id='', sl_leg_id='', trade_id=None); lo.update(kw); return lo


def _fake_oco(alpaca):
    """OCO ids tp-N/sl-N in submit order; returns the list of (tp, sl) placed."""
    placed = []
    def submit(**kw):
        n = len(placed) + 1; placed.append((f'tp-{n}', f'sl-{n}'))
        return {'id': f'tp-{n}', 'status': 'accepted', 'legs': [{'id': f'sl-{n}', 'type': 'stop'}]}
    alpaca.submit_oco_sell_order.side_effect = submit
    return placed


def _engine(alpaca, db, sm, stream, tmp_path, **over):
    e = live_engine(alpaca, db, sm, stream, tmp_path, **{**FAST, **over})
    cand = Candidate(symbol='ABC', day_open=10.0, adv20=1_000_000); e.candidates['ABC'] = cand
    return e, cand


def _fill(e, cand, lo, qty, status):
    e._on_live_fill(cand, lo, {'filled_qty': qty, 'filled_avg_price': 11.02}, status)


class TestLegPersistence:
    def test_partial_sequence_row_leg_ids_equal_the_last_oco_and_cancelled_ids_are_gone(
            self, hod_live_alpaca, real_db, hod_live_sm, hod_live_stream, tmp_path):
        placed = _fake_oco(hod_live_alpaca)
        e, cand = _engine(hod_live_alpaca, real_db, hod_live_sm, hod_live_stream, tmp_path)
        lo = _lo(); cand.live_order = lo
        for qty, status in ((58, 'partially_filled'), (60, 'partially_filled'), (63, 'filled')):
            _fill(e, cand, lo, qty, status)
        assert placed == [('tp-1', 'sl-1'), ('tp-2', 'sl-2'), ('tp-3', 'sl-3')]
        row = real_db.get_open_trades(TODAY, strategy='hod_break')
        assert len(row) == 1 and row[0]['shares'] in (58, 63) and row[0]['filled_qty'] == 63
        pd_ = json.loads(row[0]['pattern_data'])
        assert (pd_['tp_leg_id'], pd_['sl_leg_id']) == ('tp-3', 'sl-3')                  # DB == last OCO
        assert not any(old in row[0]['pattern_data'] for old in ('tp-1', 'sl-1', 'tp-2', 'sl-2'))
        pos = e.positions['ABC']
        assert (pos.tp_leg_id, pos.sl_leg_id) == ('tp-3', 'sl-3') and pos.pattern_data['tp_leg_id'] == 'tp-3'   # registry == DB
        # a restart rehydrates the LIVE ids, not the cancelled ones
        hod_live_alpaca.get_open_positions.return_value = [{'symbol': 'ABC', 'qty': 63}]
        e2 = live_engine(hod_live_alpaca, real_db, hod_live_sm, hod_live_stream, tmp_path, **FAST); e2.sync_positions()
        assert e2.positions['ABC'].tp_leg_id == 'tp-3' and e2.positions['ABC'].sl_leg_id == 'sl-3'


class TestSettleBeforeReplace:
    def test_replacement_is_submitted_only_after_the_cancelled_legs_are_terminal(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        events = []; polls = {'n': 0}
        _fake_oco(hod_live_alpaca)
        base_submit = hod_live_alpaca.submit_oco_sell_order.side_effect
        hod_live_alpaca.submit_oco_sell_order.side_effect = lambda **kw: (events.append('submit'), base_submit(**kw))[1]
        hod_live_alpaca.cancel_order.side_effect = lambda oid: events.append(f'cancel {oid}') or True
        def get_order(oid):
            polls['n'] += 1                                    # the cancel is still settling for the first polls (held_for_orders)
            return {'id': oid, 'status': 'pending_cancel' if polls['n'] <= 3 else 'canceled', 'filled_qty': 0}
        hod_live_alpaca.get_order.side_effect = get_order
        e, cand = _engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        lo = _lo(); cand.live_order = lo
        _fill(e, cand, lo, 58, 'partially_filled')
        events.clear(); polls['n'] = 0
        _fill(e, cand, lo, 60, 'partially_filled')
        assert events == ['cancel tp-1', 'cancel sl-1', 'submit']
        assert polls['n'] >= 4                                  # it kept polling until BOTH legs read terminal

    def test_a_leg_that_filled_meanwhile_is_booked_and_no_replacement_is_placed(
            self, hod_live_alpaca, real_db, hod_live_sm, hod_live_stream, tmp_path, caplog):
        placed = _fake_oco(hod_live_alpaca)
        e, cand = _engine(hod_live_alpaca, real_db, hod_live_sm, hod_live_stream, tmp_path)
        lo = _lo(); cand.live_order = lo
        _fill(e, cand, lo, 58, 'partially_filled')
        hod_live_alpaca.get_order.side_effect = lambda oid: (
            {'id': oid, 'status': 'filled', 'filled_qty': 58, 'filled_avg_price': 11.60} if oid == 'tp-1'
            else {'id': oid, 'status': 'canceled', 'filled_qty': 0})
        with caplog.at_level(logging.WARNING):
            _fill(e, cand, lo, 60, 'partially_filled')
        assert len(placed) == 1                                 # no 2nd OCO
        assert 'ABC' not in e.positions
        assert real_db.get_open_trades(TODAY, strategy='hod_break') == []
        assert any('FILLED' in r.message and 'NOT re-placing' in r.message for r in caplog.records)

    def test_rejected_replacement_is_retried_once_and_the_retry_is_accepted(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, caplog):
        e, cand = _engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        notifier = MagicMock(); e.notifier = notifier
        hod_live_alpaca.submit_oco_sell_order.side_effect = [
            Exception('insufficient qty available for order (requested: 63, available: 2)'),
            {'id': 'tp-9', 'status': 'accepted', 'legs': [{'id': 'sl-9', 'type': 'stop'}]}]
        hod_live_alpaca.submit_stop_sell_order.side_effect = [Exception('potential wash trade detected')]
        lo = _lo(); cand.live_order = lo
        with caplog.at_level(logging.WARNING):
            _fill(e, cand, lo, 63, 'filled')
        assert hod_live_alpaca.submit_oco_sell_order.call_count == 2
        assert e.positions['ABC'].tp_leg_id == 'tp-9' and e.positions['ABC'].sl_leg_id == 'sl-9'
        assert any('retrying once' in r.message and r.levelno == logging.WARNING for r in caplog.records)
        assert not any(r.levelno >= logging.ERROR and 'FAILED twice' in r.message for r in caplog.records)
        assert not any('UNPROTECTED' in str(c) for c in notifier.send_message.call_args_list)

    def test_rejected_twice_logs_error_and_sends_an_ungated_telegram_with_symbol_qty_and_broker_message(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, caplog):
        e, cand = _engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        assert e.telegram_per_trade is False                    # the gated _notify_trade would have swallowed this
        notifier = MagicMock(); e.notifier = notifier
        hod_live_alpaca.submit_oco_sell_order.side_effect = Exception('insufficient qty available for order (requested: 63, available: 2)')
        hod_live_alpaca.submit_stop_sell_order.side_effect = Exception('insufficient qty available for order (requested: 63, available: 2)')
        lo = _lo(); cand.live_order = lo
        with caplog.at_level(logging.WARNING):
            _fill(e, cand, lo, 63, 'filled')
        assert hod_live_alpaca.submit_oco_sell_order.call_count == 2
        assert any(r.levelno == logging.ERROR and 'FAILED twice' in r.message for r in caplog.records)
        msgs = [str(c.args[0]) for c in notifier.send_message.call_args_list]
        assert any('ABC' in m and '63' in m and 'insufficient qty available' in m for m in msgs)


def _stale_exit_row(real_db, shares=17, fill=115.39, sym='RBRK'):
    pd_ = json.dumps({'tp_leg_id': '35a1499f', 'sl_leg_id': 'b0f5068c', 'level': 116.0})
    tid = _seed_row(real_db, sym, 'hod_break', shares=shares, fill_price=fill, order_status='exit_pending_verification', stop=114.0, pattern_data=pd_)
    return tid


def _sell(oid, qty, px, otype, ts, sym='RBRK'):
    return SimpleNamespace(id=oid, symbol=sym, side='sell', filled_qty=str(qty), filled_avg_price=str(px), order_type=otype, filled_at=ts)


class TestReconcileByBrokerTruth:
    def _engine(self, alpaca, real_db, sm, orders):
        alpaca.get_order.side_effect = lambda oid: {'id': oid, 'status': 'canceled', 'filled_qty': 0, 'filled_avg_price': None}   # the stored ids are cancelled
        alpaca.trading_client = MagicMock(); alpaca.trading_client.get_orders.return_value = orders
        alpaca.get_open_positions.return_value = []
        e = HodBreakEngine(alpaca, real_db, sm, cfg=hod_cfg(**FAST)); e._roll_session(); return e

    def test_limit_sell_after_entry_closes_the_row_as_target_with_pnl_and_clears_the_rail(self, hod_live_alpaca, real_db, hod_live_sm):
        tid = _stale_exit_row(real_db)
        later = datetime.now(timezone.utc) + timedelta(minutes=5)
        e = self._engine(hod_live_alpaca, real_db, hod_live_sm, [_sell('903ef94b', 17, 116.69, 'limit', later)])
        assert e._unverified_exits() == 1
        assert e.reconcile_pending_exits() == 1
        row = real_db.get_trade(tid) if hasattr(real_db, 'get_trade') else None
        con = sqlite3.connect(real_db._trades_path); r = con.execute("select order_status, exit_reason, exit_price, pnl from trades where id=?", (tid,)).fetchone(); con.close()
        assert r[0] == 'closed' and r[1] == 'target' and r[2] == pytest.approx(116.69) and r[3] == pytest.approx((116.69 - 115.39) * 17)
        assert e._unverified_exits() == 0 and e._kill_rails_blocked() != 'unverified_exit'

    def test_stop_type_sell_is_booked_as_stop_loss(self, hod_live_alpaca, real_db, hod_live_sm):
        tid = _stale_exit_row(real_db)
        later = datetime.now(timezone.utc) + timedelta(minutes=5)
        e = self._engine(hod_live_alpaca, real_db, hod_live_sm, [_sell('s1', 17, 114.0, 'stop', later)])
        assert e.reconcile_pending_exits() == 1
        con = sqlite3.connect(real_db._trades_path); r = con.execute("select order_status, exit_reason from trades where id=?", (tid,)).fetchone(); con.close()
        assert r == ('closed', 'stop_loss')

    def test_a_sell_older_than_the_entry_fill_or_with_the_wrong_side_is_ignored(self, hod_live_alpaca, real_db, hod_live_sm):
        tid = _stale_exit_row(real_db)
        old = datetime.now(timezone.utc) - timedelta(days=1)
        buy = SimpleNamespace(id='b1', symbol='RBRK', side='buy', filled_qty='17', filled_avg_price='116', order_type='limit', filled_at=datetime.now(timezone.utc) + timedelta(minutes=1))
        e = self._engine(hod_live_alpaca, real_db, hod_live_sm, [_sell('old', 17, 116.69, 'limit', old), buy])
        assert e.reconcile_pending_exits() == 0
        con = sqlite3.connect(real_db._trades_path); r = con.execute("select order_status from trades where id=?", (tid,)).fetchone(); con.close()
        assert r[0] == 'exit_pending_verification'

    def test_lookup_failure_warns_and_leaves_the_row(self, hod_live_alpaca, real_db, hod_live_sm, caplog):
        tid = _stale_exit_row(real_db)
        e = self._engine(hod_live_alpaca, real_db, hod_live_sm, [])
        hod_live_alpaca.trading_client.get_orders.side_effect = RuntimeError('boom')
        with caplog.at_level(logging.WARNING):
            assert e.reconcile_pending_exits() == 0
        assert any('sell-order lookup failed' in r.message for r in caplog.records)


class TestCarryOverAdoption:
    def _prior_day_row(self, real_db, sym='QCMU', shares=63, status='filled', days=2):
        tid = _seed_row(real_db, sym, 'hod_break', shares=shares, fill_price=20.157, order_status=status, stop=19.4,
                        pattern_data=json.dumps({'tp_leg_id': 'old-tp', 'sl_leg_id': 'old-sl', 'level': 20.0}))
        d = (datetime.strptime(TODAY, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')
        real_db.update_trade(tid, {'trade_date': d}); return tid

    def test_yesterdays_open_row_with_a_broker_position_is_adopted_and_the_oco_is_placed(
            self, hod_live_alpaca, real_db, hod_live_sm, caplog):
        tid = self._prior_day_row(real_db)
        placed = _fake_oco(hod_live_alpaca)
        hod_live_alpaca.get_open_positions.return_value = [{'symbol': 'QCMU', 'qty': 63, 'avg_entry_price': 20.157}]
        notifier = MagicMock()
        e = HodBreakEngine(hod_live_alpaca, real_db, hod_live_sm, notifier=notifier, cfg=hod_cfg(**FAST))
        with caplog.at_level(logging.WARNING):
            e.sync_positions()
        assert 'QCMU' in e.positions and e.positions['QCMU'].open_qty == 63
        kw = hod_live_alpaca.submit_oco_sell_order.call_args.kwargs
        assert kw['symbol'] == 'QCMU' and kw['qty'] == 63
        assert placed == [('tp-1', 'sl-1')] and e.positions['QCMU'].sl_leg_id == 'sl-1'
        con = sqlite3.connect(real_db._trades_path); pdj = con.execute("select pattern_data from trades where id=?", (tid,)).fetchone()[0]; con.close()
        assert json.loads(pdj)['sl_leg_id'] == 'sl-1'                       # persisted, the old ids are gone
        assert 'old-sl' not in pdj
        hod_live_sm.add_watch.assert_called_once()
        assert any('ADOPTED' in r.message and r.levelno == logging.WARNING for r in caplog.records)
        assert any('ADOPTED carried-over QCMU' in str(c) for c in notifier.send_message.call_args_list)

    def test_prior_day_row_the_broker_no_longer_holds_goes_to_exit_pending_verification_then_reconcile(
            self, hod_live_alpaca, real_db, hod_live_sm):
        tid = self._prior_day_row(real_db, sym='RBRK', shares=17)
        hod_live_alpaca.get_open_positions.return_value = []
        hod_live_alpaca.trading_client = MagicMock()
        later = datetime.now(timezone.utc) + timedelta(minutes=5)
        hod_live_alpaca.trading_client.get_orders.return_value = [_sell('x1', 17, 116.69, 'limit', later)]
        hod_live_alpaca.get_order.side_effect = lambda oid: {'id': oid, 'status': 'canceled', 'filled_qty': 0}
        e = HodBreakEngine(hod_live_alpaca, real_db, hod_live_sm, cfg=hod_cfg(**FAST)); e.sync_positions()
        con = sqlite3.connect(real_db._trades_path); r = con.execute("select order_status, exit_reason from trades where id=?", (tid,)).fetchone(); con.close()
        assert r == ('closed', 'target')

    def test_rows_older_than_seven_days_are_left_alone(self, hod_live_alpaca, real_db, hod_live_sm):
        self._prior_day_row(real_db, days=9)
        hod_live_alpaca.get_open_positions.return_value = [{'symbol': 'QCMU', 'qty': 63, 'avg_entry_price': 20.157}]
        e = HodBreakEngine(hod_live_alpaca, real_db, hod_live_sm, cfg=hod_cfg(**FAST)); e.sync_positions()
        assert 'QCMU' not in e.positions and not hod_live_alpaca.submit_oco_sell_order.called


class TestMoverQueue:
    def _eng(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        return live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, **FAST)

    def test_10000_duplicate_calls_produce_no_drops_and_at_most_the_unique_names(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, caplog):
        e = self._eng(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        with caplog.at_level(logging.WARNING):
            for i in range(10_000):
                e.on_mover(f'S{i % 40}', price=11.0, day_open=10.0, cum_volume=1e5, above_open_pct=10.0)
        assert e._mover_queue.qsize() == 40
        assert not any('queue full' in r.message for r in caplog.records)

    def test_symbols_already_candidates_seen_or_positions_are_not_enqueued(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = self._eng(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        e.candidates['A'] = Candidate(symbol='A', day_open=1.0, adv20=1.0); e.seen_today.add('B'); e.positions['C'] = MagicMock()
        for s in 'ABC': e.on_mover(s, price=11.0, day_open=10.0, cum_volume=1e5, above_open_pct=10.0)
        assert e._mover_queue.qsize() == 0

    def test_one_admit_call_drains_the_whole_queue_past_500(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = self._eng(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        for i in range(1_200): e.on_mover(f'Z{i}', price=11.0, day_open=10.0, cum_volume=1e5, above_open_pct=10.0)
        assert e._mover_queue.qsize() == 1_200
        e._admit_movers()
        assert e._mover_queue.qsize() == 0 and not e._queued_movers
        e.on_mover('Z1', price=11.0, day_open=10.0, cum_volume=1e5, above_open_pct=10.0)      # a drained name can be re-offered (rejected ones re-qualify later)
        assert e._mover_queue.qsize() == 1


class TestLedgerRotationAndArmIdx:
    OLD19 = ['date', 'symbol', 'arm_ts', 'level', 'trigger', 'limit', 'qty', 'status']

    def test_parity_ledger_with_a_stale_header_is_rotated_and_the_new_file_has_the_full_header(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, caplog):
        e, cand = _engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        path = tmp_path / 'hod_live_parity_ledger.csv'
        with open(path, 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(self.OLD19); w.writerow(['2026-10-15', 'ZZZ'] + ['x'] * 6)
        with caplog.at_level(logging.WARNING):
            e._append_live_parity_row(cand, _lo(), broker_status='canceled')
            e._append_live_parity_row(cand, _lo(), broker_status='canceled')
        rotated = list(tmp_path.glob('hod_live_parity_ledger.*.csv'))
        assert len(rotated) == 1 and rotated[0].read_text().splitlines()[0] == ','.join(self.OLD19)
        with open(path, newline='') as fh: rows = list(csv.reader(fh))
        assert 'scanner_qualified_at_arm' in rows[0] and 'cf_floor_stop_px' in rows[0] and len(rows) == 3
        assert sum('rotated the old file' in r.message for r in caplog.records) == 1       # WARNING once

    def test_live_armed_carries_idx_and_arm_features_so_a_fill_has_no_missing_idx_warning(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, caplog):
        e, cand = _engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        cand.day_open = 10.0; cand.adv20 = ADV20_SYNTH
        for i, (o, h, l, c, v) in enumerate(BARS): cand.set_bar(570 + i, o, h, l, c, v)
        e._arm_live_order(cand, dict(ARM, idx=3, arm_ts='2026-10-16T09:33:00', bar_volume=1e6))
        lo = cand.live_order
        assert lo is not None and lo['idx'] == 3 and float(lo['arm_features']['r_pct']) == pytest.approx(10.0)
        with caplog.at_level(logging.WARNING):
            _fill(e, cand, lo, lo['qty'], 'filled')
        assert not any("has no 'idx'" in r.message for r in caplog.records)
        with open(tmp_path / 'hod_dry_entry_ledger.csv', newline='') as fh: rows = list(csv.DictReader(fh))
        assert float(rows[-1]['r_pct']) == pytest.approx(10.0)                              # the live-fill row carries the arm-time features
