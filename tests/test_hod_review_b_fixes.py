"""docs/review_20261003/FIX_B_hod_spec.md — Track B review fixes (2026-10-03), written before the code.

B1  a transient positions-API error at the 15:55 flatten must NOT drop the position (get_signed_broker_qty -> None,
    the flatten keeps it registered and retries, one WARNING per symbol per minute, ONE ERROR if still open at the close).
B2  registry / stop-resize / flatten quantities are OUR quantity (sum of fills on this book's client order ids today),
    never the symbol's total broker quantity (the owner may hold the same name on a shared account).
B3  kill-rail cancels are retried until every resting entry is confirmed cancelled (WARNING per retry, ERROR after 5).
F4  adoption skips a symbol this book closed within the last 120 s (broker positions list lagging our own exit).
"""
import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from trading import exit_qty_guard as eqg
from trading.hod_break_engine import HodBreakEngine, Position
from tests.hod_fills_helper import set_our_fills
from tests.test_hod_break_engine import admit
from tests.test_hod_live_resting import live_engine, BIG_VOL_ARM


def _open_pos(engine, sym='ABC', shares=40, trade_id=7):
    pos = Position(symbol=sym, trade_id=trade_id, order_id='o', shares=shares, limit_price=11.0, stop=10.6, target=12.0,
                   level=11.0, submitted_at=None, tp_leg_id='', sl_leg_id='', fill_price=11.0, status='open')
    engine.positions[sym] = pos
    return pos


def _flatten(engine, minute=955):
    with patch.object(HodBreakEngine, '_minute_of_day', return_value=minute), patch('trading.hod_break_engine.time.sleep'):
        return engine.force_close_all()


@pytest.fixture
def eng(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
    return live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)


# ------------------------------------------------------------------------------------------------------------- B1
class TestGuardReturnsNoneOnApiError:
    def test_api_error_is_unknown_not_flat(self, caplog):
        eqg._LOOKUP_WARN_TS.clear()
        a = MagicMock(); a.get_open_positions.side_effect = RuntimeError('429 too many requests')
        with caplog.at_level(logging.WARNING):
            assert eqg.get_signed_broker_qty(a, 'ABC') is None
            assert eqg.get_signed_broker_qty(a, 'ABC') is None          # throttled: one WARNING per symbol per minute
        assert len([r for r in caplog.records if 'broker position lookup failed' in r.getMessage()]) == 1

    def test_absent_symbol_is_still_zero(self):
        a = MagicMock(); a.get_open_positions.return_value = [{'symbol': 'XYZ', 'qty': 5}]
        assert eqg.get_signed_broker_qty(a, 'ABC') == 0


class TestFlattenSurvivesATransientPositionsError:
    def test_positions_call_raises_once_then_succeeds_one_sell(self, eng, hod_live_alpaca, caplog):
        pos = _open_pos(eng, shares=40)
        hod_live_alpaca.get_open_positions.side_effect = [RuntimeError('timeout'), [{'symbol': 'ABC', 'qty': 40}]]
        with caplog.at_level(logging.WARNING):
            n1 = _flatten(eng)
        assert n1 == 0 and 'ABC' in eng.positions and eng.positions['ABC'].status == 'open'    # kept registered, nothing sold
        hod_live_alpaca.submit_limit_sell_order.assert_not_called()
        assert not eng._flattened
        eng.db.update_trade.assert_not_called()                                                # NOT marked exit_pending_verification
        assert any('retry' in r.getMessage().lower() and 'ABC' in r.getMessage() and r.levelname == 'WARNING' for r in caplog.records)
        n2 = _flatten(eng)
        assert n2 == 1
        hod_live_alpaca.submit_limit_sell_order.assert_called_once()
        assert hod_live_alpaca.submit_limit_sell_order.call_args.args[:2] == ('ABC', 40)
        assert pos.close_order_id == 'tp-1'

    def test_unknown_warning_is_once_per_symbol_per_minute(self, eng, hod_live_alpaca, caplog):
        _open_pos(eng, shares=40)
        hod_live_alpaca.get_open_positions.side_effect = RuntimeError('timeout')
        with caplog.at_level(logging.WARNING):
            for _ in range(4):
                _flatten(eng)
        warns = [r for r in caplog.records if r.levelname == 'WARNING' and 'FORCE CLOSE ABC' in r.getMessage() and 'lookup' in r.getMessage()]
        assert len(warns) == 1
        assert 'ABC' in eng.positions

    def test_still_open_after_the_close_is_ONE_error_naming_it(self, eng, hod_live_alpaca, caplog):
        _open_pos(eng, shares=40)
        hod_live_alpaca.get_open_positions.side_effect = RuntimeError('timeout')
        with caplog.at_level(logging.WARNING):
            _flatten(eng, minute=955)
            _flatten(eng, minute=961)
            _flatten(eng, minute=962)
        errs = [r for r in caplog.records if r.levelname == 'ERROR']
        assert len(errs) == 1 and 'ABC' in errs[0].getMessage()
        hod_live_alpaca.submit_limit_sell_order.assert_not_called()


# ------------------------------------------------------------------------------------------------------------- B2
class TestOurFillsReader:
    def test_sums_prefixed_buys_only(self):
        a = MagicMock(); set_our_fills(a, {'ABC': 40}, foreign_buy=60)
        assert eqg.get_our_buy_fill_qty(a, 'ABC', 'hod-', datetime(2026, 10, 3, 4, tzinfo=timezone.utc)) == 40

    def test_api_error_is_none(self, caplog):
        a = MagicMock(); a.trading_client.get_orders.side_effect = RuntimeError('boom')
        with caplog.at_level(logging.WARNING):
            assert eqg.get_our_buy_fill_qty(a, 'ABC', 'hod-', datetime(2026, 10, 3, 4, tzinfo=timezone.utc)) is None
        assert caplog.records

    def test_client_without_trading_client_is_none(self):
        assert eqg.get_our_buy_fill_qty(object(), 'ABC', 'hod-', datetime(2026, 10, 3, 4, tzinfo=timezone.utc)) is None


class TestOurQuantityNotTheBrokerTotal:
    def test_sync_resizes_to_our_fills_not_the_broker_total(self, eng, hod_live_alpaca, hod_live_sm, hod_live_db, caplog):
        _open_pos(eng, shares=21)
        set_our_fills(hod_live_alpaca, {'ABC': 40}, foreign_buy=60)
        with caplog.at_level(logging.WARNING):
            eng._sync_registry_qty_to_broker('ABC', 100)
        assert eng.positions['ABC'].shares == 40
        hod_live_db.update_trade.assert_any_call(7, {'shares': 40, 'filled_qty': 40})
        assert hod_live_sm.add_watch.call_args.kwargs['shares'] == 40
        assert any('foreign shares present' in r.getMessage() for r in caplog.records)

    def test_flatten_sells_ours_when_the_broker_holds_more(self, eng, hod_live_alpaca, caplog):
        _open_pos(eng, shares=40)
        hod_live_alpaca.get_open_positions.return_value = [{'symbol': 'ABC', 'qty': 100}]
        set_our_fills(hod_live_alpaca, {'ABC': 40}, foreign_buy=60)
        with caplog.at_level(logging.WARNING):
            assert _flatten(eng) == 1
        assert hod_live_alpaca.submit_limit_sell_order.call_args.args[:2] == ('ABC', 40)
        assert any('foreign shares present' in r.getMessage() for r in caplog.records)

    def test_flatten_never_sells_beyond_the_registry_when_fills_unreadable(self, eng, hod_live_alpaca, caplog):
        _open_pos(eng, shares=40)
        hod_live_alpaca.get_open_positions.return_value = [{'symbol': 'ABC', 'qty': 100}]
        hod_live_alpaca.trading_client = MagicMock(); hod_live_alpaca.trading_client.get_orders.side_effect = RuntimeError('503')
        with caplog.at_level(logging.WARNING):
            assert _flatten(eng) == 1
        assert hod_live_alpaca.submit_limit_sell_order.call_args.args[:2] == ('ABC', 40)       # registry, NOT the broker's 100
        assert any('our fills' in r.getMessage() and 'registry' in r.getMessage() for r in caplog.records)

    def test_flatten_covers_a_stale_low_registry_up_to_our_own_fills(self, eng, hod_live_alpaca):
        _open_pos(eng, shares=21)
        hod_live_alpaca.get_open_positions.return_value = [{'symbol': 'ABC', 'qty': 44}]
        set_our_fills(hod_live_alpaca, {'ABC': 44})
        assert _flatten(eng) == 1
        assert hod_live_alpaca.submit_limit_sell_order.call_args.args[:2] == ('ABC', 44)

    def test_adoption_takes_our_fills_not_the_broker_total(self, eng, hod_live_alpaca, hod_live_db):
        admit(eng); cand = eng.candidates['ABC']
        eng._arm_live_order(cand, dict(BIG_VOL_ARM)); cand.live_order = None       # armed today, then cancelled
        hod_live_alpaca.get_open_positions.return_value = [{'symbol': 'ABC', 'qty': 100, 'avg_entry_price': 11.0}]
        set_our_fills(hod_live_alpaca, {'ABC': 40}, foreign_buy=60)
        eng._reconcile_positions_to_broker(force=True)
        assert eng.positions['ABC'].shares == 40

    def test_adoption_skipped_loudly_when_nothing_bounds_our_quantity(self, eng, hod_live_alpaca, caplog):
        admit(eng); cand = eng.candidates['ABC']
        eng._arm_live_order(cand, dict(BIG_VOL_ARM)); cand.live_order = None
        hod_live_alpaca.get_open_positions.return_value = [{'symbol': 'ABC', 'qty': 100, 'avg_entry_price': 11.0}]
        hod_live_alpaca.trading_client = MagicMock(); hod_live_alpaca.trading_client.get_orders.side_effect = RuntimeError('503')
        with caplog.at_level(logging.ERROR):
            eng._reconcile_positions_to_broker(force=True)
        assert 'ABC' not in eng.positions
        assert any(r.levelname == 'ERROR' and 'ABC' in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------------------------------------------------- F4
class TestAdoptionRaceAfterOurOwnExit:
    def test_recently_closed_symbol_is_not_re_adopted(self, eng, hod_live_alpaca, hod_live_sm):
        admit(eng); cand = eng.candidates['ABC']
        eng._arm_live_order(cand, dict(BIG_VOL_ARM)); cand.live_order = None
        pos = _open_pos(eng, shares=40)
        eng._record_exit(pos, 11.5, 'target')                                      # our own exit pops the registry
        assert 'ABC' not in eng.positions
        hod_live_alpaca.get_open_positions.return_value = [{'symbol': 'ABC', 'qty': 40, 'avg_entry_price': 11.0}]   # lagging list
        set_our_fills(hod_live_alpaca, {'ABC': 40})
        eng._reconcile_positions_to_broker(force=True)
        assert 'ABC' not in eng.positions
        hod_live_sm.add_watch.assert_not_called()


# ------------------------------------------------------------------------------------------------------------- B3
class TestKillRailCancelRetried:
    def _armed(self, eng):
        admit(eng); cand = eng.candidates['ABC']
        eng._arm_live_order(cand, dict(BIG_VOL_ARM))
        return cand

    def test_failed_pre_cancel_get_is_retried_and_only_then_marked_handled(self, eng, hod_live_alpaca, monkeypatch):
        cand = self._armed(eng); oid = cand.live_order['order_id']
        monkeypatch.setattr(eng, '_kill_rails_blocked', lambda: 'daily_kill')
        calls = {'n': 0}

        def get_order(o):
            calls['n'] += 1
            if calls['n'] == 1:
                raise RuntimeError('timeout')
            return {'id': o, 'status': 'accepted', 'filled_qty': 0, 'filled_avg_price': None}
        hod_live_alpaca.get_order.side_effect = get_order
        eng._enforce_kill_rails_on_resting()
        assert cand.live_order is not None and 'daily_kill' not in eng._rail_cancel_done      # nothing cancelled, NOT handled
        hod_live_alpaca.cancel_order.assert_not_called()
        eng._last_rail_check = 0.0
        eng._enforce_kill_rails_on_resting()                                                    # next tick: retried
        hod_live_alpaca.cancel_order.assert_called_once_with(oid)
        assert cand.live_order is None and 'daily_kill' in eng._rail_cancel_done

    def test_retry_logs_warning_each_time_and_one_error_after_five_failures(self, eng, hod_live_alpaca, monkeypatch, caplog):
        cand = self._armed(eng)
        monkeypatch.setattr(eng, '_kill_rails_blocked', lambda: 'daily_kill')
        hod_live_alpaca.get_order.side_effect = RuntimeError('timeout')
        with caplog.at_level(logging.WARNING):
            for _ in range(6):
                eng._last_rail_check = 0.0
                eng._enforce_kill_rails_on_resting()
        assert cand.live_order is not None and 'daily_kill' not in eng._rail_cancel_done
        retry_warn = [r for r in caplog.records if r.levelname == 'WARNING' and 'KILL RAIL' in r.getMessage() and 'retry' in r.getMessage().lower()]
        rail_err = [r for r in caplog.records if r.levelname == 'ERROR' and 'KILL RAIL' in r.getMessage()]
        assert len(retry_warn) >= 4 and len(rail_err) == 1

    def test_throttle_still_applies(self, eng, hod_live_alpaca, monkeypatch):
        cand = self._armed(eng)
        monkeypatch.setattr(eng, '_kill_rails_blocked', lambda: 'daily_kill')
        hod_live_alpaca.get_order.side_effect = RuntimeError('timeout')
        eng._enforce_kill_rails_on_resting()
        n = hod_live_alpaca.get_order.call_count
        eng._enforce_kill_rails_on_resting()                                                    # < 10 s later: no second attempt
        assert hod_live_alpaca.get_order.call_count == n


# ------------------------------------------------------------------------------------- other callers (None = unknown)
class TestOtherCallersHandleAnUnknownBrokerQty:
    """get_signed_broker_qty now returns None on an API error; ORB's add-on resize and StopMonitor's scale leg must
    skip (not crash on `None <= 0`, not assume flat and not assume long), loudly."""

    def test_orb_add_on_resize_skips_when_the_lookup_fails(self, caplog):
        from trading.orb_engine import ORBEngine
        fake = MagicMock(spec=ORBEngine); fake.alpaca = MagicMock(); fake.alpaca.get_open_positions.side_effect = RuntimeError('503')
        pos = MagicMock(); pos.symbol = 'ABC'; pos.shares = 40; pos.sl_leg_id = 'sl'; pos.tp_leg_id = 'tp'
        eqg._LOOKUP_WARN_TS.clear()
        with caplog.at_level(logging.WARNING):
            assert ORBEngine._resize_exit_legs_for_add(fake, pos, 80) == 40
        fake.alpaca.replace_order_qty.assert_not_called()
        assert any('lookup failed' in r.getMessage() for r in caplog.records)

    def test_stop_monitor_scale_leg_aborts_when_the_lookup_fails(self, caplog):
        from trading.stop_monitor import StopMonitor
        fake = MagicMock(spec=StopMonitor); client = MagicMock(); client.get_open_positions.side_effect = RuntimeError('503')
        fake._client_for.return_value = client
        watch = MagicMock(); watch.strategy = 'orb'; watch.scale_qty = 50; watch.shares = 100
        eqg._LOOKUP_WARN_TS.clear()
        with caplog.at_level(logging.WARNING):
            assert StopMonitor._scale_submit_core(fake, 'ABC', watch) is False
        client.replace_order_qty.assert_not_called()
        assert any('lookup failed' in r.getMessage() for r in caplog.records)
