"""2026-10-02 defects (docs/hod_killrail_restart_qty_20261002.md): (1) a tripped kill rail must cancel every
resting ENTRY order and a late fill must be a managed, flattened position (VIRT); (2) the registry quantity must
equal the broker's after a restart / after an entry fill (NEBX 21 vs 44, COHX 50 vs 120, SPCF), and P&L is booked
on the quantity sold. Foreign (never-armed) broker positions are never touched."""
import logging

import pytest

from trading.hod_break_engine import Position
from tests.test_hod_break_engine import admit
from tests.hod_fills_helper import set_our_fills
from tests.test_hod_live_resting import live_engine, BIG_VOL_ARM


def _armed(alpaca, db, sm, stream, tmp_path):
    e = live_engine(alpaca, db, sm, stream, tmp_path)
    admit(e)
    cand = e.candidates['ABC']
    e._arm_live_order(cand, dict(BIG_VOL_ARM))
    return e, cand


class TestKillRailCancelsRestingEntries:
    def test_daily_kill_cancels_resting_order_once_and_logs_ids(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, monkeypatch, caplog):
        e, cand = _armed(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        oid = cand.live_order['order_id']
        monkeypatch.setattr(e, '_kill_rails_blocked', lambda: 'daily_kill')
        with caplog.at_level(logging.WARNING):
            e._enforce_kill_rails_on_resting()
            e._last_rail_check = 0.0
            e._enforce_kill_rails_on_resting()                      # second pass: nothing left, nothing re-sent
        hod_live_alpaca.cancel_order.assert_called_once_with(oid)
        assert cand.live_order is None
        msgs = [r.getMessage() for r in caplog.records if 'KILL RAIL (daily_kill)' in r.getMessage()]
        assert len(msgs) == 1 and oid in msgs[0]

    def test_no_rail_no_cancel(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, monkeypatch):
        e, cand = _armed(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        monkeypatch.setattr(e, '_kill_rails_blocked', lambda: None)
        e._enforce_kill_rails_on_resting()
        hod_live_alpaca.cancel_order.assert_not_called()
        assert cand.live_order is not None

    def test_fill_racing_the_rail_cancel_is_registered_with_a_stop(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, monkeypatch):
        e, cand = _armed(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        oid = cand.live_order['order_id']
        hod_live_alpaca.get_order.side_effect = lambda o: {'id': o, 'status': 'partially_filled', 'filled_qty': 24, 'filled_avg_price': 11.02}
        hod_live_alpaca.get_open_positions.return_value = [{'symbol': 'ABC', 'qty': 24, 'avg_entry_price': 11.02}]
        monkeypatch.setattr(e, '_kill_rails_blocked', lambda: 'daily_kill')
        e._enforce_kill_rails_on_resting()
        assert e.positions['ABC'].shares == 24
        hod_live_sm.add_watch.assert_called()


class TestLateFillAfterCancelIsManaged:
    def test_unregistered_broker_long_of_an_armed_name_is_adopted_with_its_arm_stop_and_flattened(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, monkeypatch):
        e, cand = _armed(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        cand.live_order = None                                      # the cancel was 'confirmed', then 24 sh filled anyway
        hod_live_alpaca.get_open_positions.return_value = [{'symbol': 'ABC', 'qty': 24, 'avg_entry_price': 11.02}]
        set_our_fills(hod_live_alpaca, {'ABC': 24})                 # review B2: adoption takes OUR fills
        e._reconcile_positions_to_broker(force=True)
        pos = e.positions['ABC']
        assert pos.shares == 24 and pos.stop == pytest.approx(BIG_VOL_ARM['stop'])
        hod_live_sm.add_watch.assert_called_once()
        # the 15:55 flatten sees it: it is an open registry position
        assert any(p.status == 'open' for p in e.positions.values())

    def test_foreign_position_is_never_touched(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e, cand = _armed(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        hod_live_alpaca.get_open_positions.return_value = [{'symbol': 'ZZZ', 'qty': 500, 'avg_entry_price': 5.0}]
        e._reconcile_positions_to_broker(force=True)
        assert 'ZZZ' not in e.positions
        hod_live_sm.add_watch.assert_not_called()
        hod_live_alpaca.submit_oco_sell_order.assert_not_called()


class TestRegistryQtyEqualsBroker:
    def test_fill_on_a_position_the_broker_already_holds_more_of_takes_the_broker_qty(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        """COHX: the broker held 120 (incl. the 50 just filled); the fill used to overwrite the registry with 50."""
        e, cand = _armed(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        coid = cand.live_order['coid']
        hod_live_alpaca.get_open_positions.return_value = [{'symbol': 'ABC', 'qty': 120, 'avg_entry_price': 11.0}]
        set_our_fills(hod_live_alpaca, {'ABC': 120})                # 50 on this coid + 70 on earlier coids of ours: all OUR fills
        hod_live_stream.snapshot_by_client_prefix.return_value = {
            coid: {'status': 'filled', 'filled_qty': 50, 'filled_avg_price': 11.02, 'client_order_id': coid}}
        e._poll_live_fills()
        assert e.positions['ABC'].shares == 120
        assert hod_live_sm.add_watch.call_args.kwargs['shares'] == 120
        assert hod_live_alpaca.submit_oco_sell_order.call_args.kwargs['qty'] == 120     # target leg resized too

    def test_periodic_reconcile_raises_a_stale_low_registry(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e, cand = _armed(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        e.positions['ABC'] = Position(symbol='ABC', trade_id=7, order_id='o', shares=21, limit_price=11.0, stop=10.6, target=12.0,
                                      level=11.0, submitted_at=None, tp_leg_id='', sl_leg_id='', fill_price=11.0, status='open')
        hod_live_alpaca.get_open_positions.return_value = [{'symbol': 'ABC', 'qty': 44, 'avg_entry_price': 11.0}]
        set_our_fills(hod_live_alpaca, {'ABC': 44})
        e._reconcile_positions_to_broker(force=True)
        assert e.positions['ABC'].shares == 44
        hod_live_db.update_trade.assert_any_call(7, {'shares': 44, 'filled_qty': 44})

    def test_sync_positions_rehydrates_at_the_broker_qty(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        hod_live_db.get_open_trades.return_value = [dict(
            id=9, symbol='NEBX', shares=21, entry_price=28.0, fill_price=28.0, stop_loss_price=27.5, take_profit_price=29.0,
            order_status='filled', pattern_data='{}', order_id='x')]
        hod_live_alpaca.get_open_positions.return_value = [{'symbol': 'NEBX', 'qty': 44, 'avg_entry_price': 28.0}]
        e.sync_positions()
        assert e.positions['NEBX'].shares == 44

    def test_pnl_is_booked_on_the_quantity_sold(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        pos = Position(symbol='ABC', trade_id=7, order_id='o', shares=50, limit_price=10.0, stop=9.0, target=12.0, level=10.0,
                       submitted_at=None, tp_leg_id='', sl_leg_id='', fill_price=10.0, status='open')
        pos.closed_qty = 120
        e.positions['ABC'] = pos
        e._record_exit(pos, 11.0, 'target')
        assert hod_live_db.update_trade.call_args.args[1]['pnl'] == pytest.approx(120.0)
