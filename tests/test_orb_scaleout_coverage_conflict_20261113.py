"""DAMD 2026-11-13 (docs/orb_scaleout_coverage_conflict_20261113.md): scale-out vs exit-coverage
conflict. D1 submit-failure counting + rate limit, D2 coverage id owned by the watch so a
scale-out resizes it, D3 add-on blended entry_price, D5 external close booked from the broker."""
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from data_sources.alpaca_client import AlpacaAPIError, AlpacaClient
from persistence.database import Database
from trading.exit_reasons import ExitReason
from trading.orb_engine import ORBEngine
from trading.stop_monitor import StopMonitor
from tests.test_orb_add_on import _make_pos, orb_cfg, mock_stop_monitor  # noqa: F401  (fixtures)


@pytest.fixture
def client():
    c = MagicMock(spec=AlpacaClient)
    c.replace_order_qty.side_effect = lambda oid, q: {'id': f'new-{oid}', 'status': 'accepted'}
    c.submit_limit_sell_order.return_value = {'id': 'scale-1', 'status': 'accepted'}
    c.get_order.return_value = {'id': 'x', 'status': 'new', 'qty': 444}
    c.get_open_positions.return_value = [{'symbol': 'TEST', 'qty': 444, 'side': 'long'}]
    c.get_open_orders.return_value = []
    c.submit_stop_sell_order.return_value = {'id': 'cov-stop-1', 'status': 'accepted'}
    return c


@pytest.fixture
def mon(client):
    m = StopMonitor(api_key='k', api_secret='s', alpaca_client=client)
    m.add_watch(symbol='TEST', stop_price=7.30, shares=444, tp_leg_id='', sl_leg_id='sl-dead',
                trade_db_id=783, entry_price=7.64, risk_per_share=0.3, strategy='orb',
                lock_arm_at_r=1.75, lock_stop_r=0.5, lock_r_unit=1.0)
    m.arm_scale_out('TEST', 8.02, 88)
    m._watches['TEST'].added_at = time.time() - 600
    return m


def _tick(m, price):
    t = MagicMock()
    t.symbol, t.price = 'TEST', price
    asyncio.run(m._on_trade(t))


# ---------------------------------------------------------------- D1
class TestD1SubmitFailureCountsAndRateLimits:
    def test_three_submit_failures_disarm_with_one_error(self, mon, client, caplog):
        client.submit_limit_sell_order.side_effect = AlpacaAPIError('held_for_orders 444 / available 0')
        w = mon._watches['TEST']
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                if w.scale_qty <= 0:              # disarmed: the triggers never call the core again
                    break
                w.scale_last_attempt_ts = 0.0     # bypass the rate limit: isolate the counter
                mon._scale_submit_core('TEST', w)
        assert w.scale_qty == 0
        assert client.submit_limit_sell_order.call_count == 3
        assert len([r for r in caplog.records if r.levelno == logging.ERROR and 'DISARMED' in r.message]) == 1

    def test_second_call_inside_30s_makes_no_broker_call(self, mon, client):
        w = mon._watches['TEST']
        assert mon._scale_submit_core('TEST', w) is True
        w.scale_order_id = ''                      # as if it had not rested
        client.reset_mock()
        assert mon._scale_submit_core('TEST', w) is False
        client.submit_limit_sell_order.assert_not_called()
        client.get_open_positions.assert_not_called()

    def test_tick_storm_with_failing_submit_hits_broker_once_per_window(self, mon, client):
        client.submit_limit_sell_order.side_effect = AlpacaAPIError('held_for_orders 444 / available 0')
        for _ in range(200):
            _tick(mon, 8.10)
        assert client.submit_limit_sell_order.call_count == 1


# ---------------------------------------------------------------- D2
class TestD2CoverageIdOwnedByWatch:
    def test_coverage_placed_stop_becomes_sl_leg_when_old_leg_dead(self, mon, client):
        client.get_order.return_value = {'id': 'sl-dead', 'status': 'canceled'}
        mon._exit_coverage_cycle()
        assert mon._watches['TEST'].sl_leg_id == 'cov-stop-1'

    def test_live_sl_leg_kept(self, mon, client):
        client.get_open_orders.return_value = [
            {'id': 'sl-dead', 'symbol': 'TEST', 'side': 'sell', 'status': 'new', 'qty': 200,
             'filled_qty': 0, 'order_class': 'simple'}]
        client.get_order.return_value = {'id': 'sl-dead', 'status': 'new', 'qty': 200}
        mon._exit_coverage_cycle()
        assert mon._watches['TEST'].sl_leg_id == 'sl-dead'

    def test_parity_coverage_then_scale_out_succeeds(self, mon, client):
        """Coverage stop (222 of 444) registered; scale-out must resize/skip it and submit."""
        client.get_open_orders.return_value = [
            {'id': 'tp-1', 'symbol': 'TEST', 'side': 'sell', 'status': 'new', 'qty': 222,
             'filled_qty': 0, 'order_class': 'simple'}]
        client.get_order.return_value = {'id': 'sl-dead', 'status': 'canceled'}
        mon._exit_coverage_cycle()
        w = mon._watches['TEST']
        assert w.sl_leg_id == 'cov-stop-1'
        client.get_order.return_value = {'id': 'cov-stop-1', 'status': 'new', 'qty': 222}
        client.submit_limit_sell_order.reset_mock()
        w.scale_last_attempt_ts = 0.0
        _tick(mon, 8.10)
        client.replace_order_qty.assert_not_called()       # 222 <= runner 356: nothing to shrink
        client.submit_limit_sell_order.assert_called_once()
        assert w.scale_order_id == 'scale-1'

    def test_oversized_leg_is_shrunk_to_runner(self, mon, client):
        w = mon._watches['TEST']
        w.sl_leg_id = 'big'
        client.get_order.return_value = {'id': 'big', 'status': 'new', 'qty': 444}
        assert mon._scale_submit_core('TEST', w) is True
        client.replace_order_qty.assert_called_once_with('big', 356)


# ---------------------------------------------------------------- D3
class TestD3AddOnBlendedEntryPrice:
    def test_persist_writes_combined_avg_as_entry_price(self, orb_cfg, mock_stop_monitor):  # noqa: F811
        db = MagicMock(spec=__import__('persistence.database', fromlist=['Database']).Database)
        db.get_trade_by_id.return_value = {'id': 100, 'pattern_data': '{}'}
        orb_cfg['strategy']['enabled'] = True
        eng = ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient), db=db,
                        stop_monitor=mock_stop_monitor, config=orb_cfg)
        pos = _make_pos(shares=444, combined_avg_price=7.6373)
        eng._persist_add_on_state(pos)
        upd = db.update_trade.call_args.args[1]
        assert upd['entry_price'] == pytest.approx(7.6373)
        assert upd['shares'] == 444

    def test_unadded_position_leaves_entry_price_alone(self, orb_cfg, mock_stop_monitor):  # noqa: F811
        db = MagicMock(spec=Database)
        db.get_trade_by_id.return_value = {'id': 100, 'pattern_data': '{}'}
        eng = ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient), db=db,
                        stop_monitor=mock_stop_monitor, config=orb_cfg)
        eng._persist_add_on_state(_make_pos())
        assert 'entry_price' not in db.update_trade.call_args.args[1]

    def test_real_database_round_trip_keeps_base_fill_price(self, tmp_path, orb_cfg, mock_stop_monitor):  # noqa: F811
        db = Database(db_path=str(tmp_path / 'c.db'), cache_path=str(tmp_path / 'c.db'),
                      trades_path=str(tmp_path / 't.db'))
        tid = db.save_trade({
            'trade_date': '2026-11-13', 'symbol': 'TEST', 'side': 'buy', 'entry_price': 7.50,
            'stop_loss_price': 7.3, 'take_profit_price': 22.5, 'shares': 222, 'risk_per_share': 0.2,
            'total_risk': 44.4, 'risk_reward_ratio': 2.0, 'order_id': '', 'order_status': 'filled',
            'fill_price': 7.50, 'filled_at': datetime.now(timezone.utc), 'exit_price': None,
            'exit_reason': None, 'exited_at': None, 'pnl': None, 'pnl_pct': None,
            'pattern_data': json.dumps({'pool_id': 'production'}), 'strategy': 'orb',
            'account': 'paper'})
        eng = ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient), db=db,
                        stop_monitor=mock_stop_monitor, config=orb_cfg)
        pos = _make_pos(shares=444, trade_id=tid, entry_price=7.50, combined_avg_price=7.6373)
        eng._persist_add_on_state(pos)
        row = db.get_trade_by_id(tid)
        assert row['entry_price'] == pytest.approx(7.6373)
        assert row['fill_price'] == pytest.approx(7.50)   # rehydrate base price intact
        assert row['shares'] == 444


# ---------------------------------------------------------------- D5
class TestD5ExternalCloseBooked:
    def _flat(self, client, fill):
        client.get_open_positions.return_value = []
        client.get_last_filled_sell.return_value = fill

    def test_stale_sell_before_the_watch_is_not_booked(self, mon, client, caplog):
        """A filled sell older than the watch (a prior exit of a re-entered name) must never become today's exit."""
        from datetime import datetime, timezone, timedelta
        old = datetime.now(timezone.utc) - timedelta(hours=6)   # the watch was armed 600 s ago
        self._flat(client, {'id': 'old-1', 'qty': 444.0, 'price': 7.01, 'filled_at': old})
        with caplog.at_level(logging.ERROR):
            mon._exit_coverage_cycle()
        assert 'TEST' in mon._watches
        assert mon.drain_exit_events('orb') == []
        assert any('predates this watch' in r.message for r in caplog.records)

    def test_flat_broker_emits_external_exit_and_removes_watch(self, mon, client, caplog):
        from datetime import datetime, timezone
        self._flat(client, {'id': 'cfae5a8c-1', 'qty': 444.0, 'price': 8.44, 'filled_at': datetime.now(timezone.utc)})
        with caplog.at_level(logging.INFO):
            mon._exit_coverage_cycle()
        assert 'TEST' not in mon._watches
        evs = mon.drain_exit_events('orb')
        assert len(evs) == 1
        assert (evs[0].exit_reason, evs[0].exit_price, evs[0].shares, evs[0].trade_db_id) == \
            (ExitReason.EXTERNAL.value, 8.44, 444, 783)
        assert any('closed externally' in r.message for r in caplog.records)

    def test_no_fill_found_keeps_watch_and_logs_error(self, mon, client, caplog):
        self._flat(client, None)
        with caplog.at_level(logging.ERROR):
            mon._exit_coverage_cycle()
        assert 'TEST' in mon._watches
        assert any('NOT booked' in r.message for r in caplog.records)

    def test_partial_broker_qty_not_booked(self, mon, client):
        client.get_open_positions.return_value = [{'symbol': 'TEST', 'qty': 100, 'side': 'long'}]
        mon._exit_coverage_cycle()
        assert 'TEST' in mon._watches
        client.get_last_filled_sell.assert_not_called()

    def test_orb_engine_books_the_row_from_the_event(self, mon, client, orb_cfg, mock_stop_monitor):  # noqa: F811
        """Boundary: StopMonitor event -> ORBEngine._handle_exit_event writes exit_price/reason."""
        from datetime import datetime, timezone
        self._flat(client, {'id': 'cfae5a8c-1', 'qty': 444.0, 'price': 8.44, 'filled_at': datetime.now(timezone.utc)})
        mon._exit_coverage_cycle()
        ev = mon.drain_exit_events('orb')[0]
        db = MagicMock(spec=Database)
        orb_cfg['strategy']['enabled'] = True
        eng = ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient), db=db,
                        stop_monitor=mock_stop_monitor, config=orb_cfg)
        eng.open_positions['TEST'] = _make_pos(symbol='TEST', shares=444, trade_id=783,
                                               entry_price=7.50, combined_avg_price=7.6373)
        eng._handle_exit_event(ev)
        upd = [c.args[1] for c in db.update_trade.call_args_list if c.args[0] == 783][-1]
        assert upd['exit_price'] == 8.44 and upd['exit_reason'] == 'external'
        assert upd['pnl'] == pytest.approx((8.44 - 7.6373) * 444)


def test_alpaca_client_get_last_filled_sell_picks_latest_filled_sell():
    c = AlpacaClient.__new__(AlpacaClient)
    mk = lambda i, side, q, px, ts: MagicMock(id=i, side=MagicMock(value=side), filled_qty=q,
                                              filled_avg_price=px, filled_at=ts)
    orders = [mk('a', 'buy', 100, 7.5, 3), mk('b', 'sell', 0, 0, 5), mk('c', 'sell', 444, 8.44, 4),
              mk('d', 'sell', 88, 8.0, 2)]
    c.trading_client = MagicMock()
    c.trading_client.get_orders.return_value = orders
    c._call_with_timeout = lambda fn, name: fn()
    got = c.get_last_filled_sell('TEST')
    assert got['id'] == 'c' and got['price'] == 8.44 and got['qty'] == 444.0
