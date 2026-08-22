"""ORBEngine winner-stack tests (2026-08-22): floored stop at fill (P0-4),
scale-fill DB lifecycle (P0-2/P0-3 single-writer), scale arming gate (P1-1),
FC sync composition, kill-rail accounting, and scale-aware rehydration that
stays correct with the flags OFF (P1-7 rollback drill).
"""
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from persistence.database import Database
from trading.exit_reasons import ExitReason
from trading.orb_engine import ORBEngine, OpenPosition
from trading.stop_monitor import StopExitEvent, StopMonitor


# =========================================================================
# Helpers
# =========================================================================

def _skeleton_engine(**attrs):
    """Engine without __init__ (repo test pattern) + winner-stack attrs."""
    e = object.__new__(ORBEngine)
    e.alpaca = MagicMock()
    e.db = MagicMock()
    e.stop_monitor = MagicMock(spec=StopMonitor)
    e.stop_monitor.add_watch.return_value = True
    e.stop_monitor.arm_scale_out.return_value = True
    e.stop_monitor.drain_exit_events.return_value = []
    e.notifier = None
    e.tg_prefix = '[ORB]'
    e.notify_on_entry = False
    e.notify_on_exit = False
    e.daily_pnl = 0.0
    e.daily_n_filled = 0
    e.open_positions = {}
    e.candidates = {}
    e.safety_sl_pct = 0.10
    e._bar_windows = {}
    e.touchgo_cfg = SimpleNamespace(
        breakout_bar_source='market', master_enabled=True,
        max_breakout_age_min=15.0)
    e.atr_floor_enabled = False
    e.atr_floor_k = 0.25
    e.scale_out_enabled = False
    e.scale_frac = 0.40
    e.scale_level_r = 3.0
    for k, v in attrs.items():
        setattr(e, k, v)
    return e


def _pos(**kw):
    d = dict(symbol='TEST', entry_price=10.0, stop_price=9.0, shares=1000,
             trade_id=42, order_id='ord-1',
             entry_time=datetime.now(timezone.utc),
             range_high=10.0, range_low=9.0,
             lock_arm_at_r=1.75, lock_stop_r=0.5,
             composite_score=0.5, quintile='Q4')
    d.update(kw)
    return OpenPosition(**d)


def _fill_status(price=10.0, qty=1000):
    return {'filled_avg_price': price, 'filled_qty': qty,
            'filled_at': datetime.now(timezone.utc)}


def _db_updates(mock_db):
    return [c.args[1] for c in mock_db.update_trade.call_args_list]


# =========================================================================
# Config plumbing
# =========================================================================

class TestConfigFlags:
    def _engine(self, cfg):
        sm = MagicMock(spec=StopMonitor)
        sm.polling_mode = False
        db = MagicMock(spec=Database)
        from data_sources.alpaca_client import AlpacaClient
        return ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient), db=db,
                         stop_monitor=sm, config=cfg)

    def test_real_orb_yaml_flags_default_off(self, monkeypatch):
        """The LIVE orb.yaml has no winner-stack keys — flags must be OFF
        (Monday rollout flips them, not this build)."""
        monkeypatch.delenv('ORB_ATR_FLOOR', raising=False)
        monkeypatch.delenv('ORB_SCALE_OUT', raising=False)
        cfg = yaml.safe_load(open(Path(__file__).parent.parent / 'orb.yaml'))
        eng = self._engine(cfg)
        assert eng.atr_floor_enabled is False
        assert eng.scale_out_enabled is False
        assert eng.atr_floor_k == 0.25
        assert eng.scale_frac == 0.40
        assert eng.scale_level_r == 3.0

    def test_flags_on_via_yaml(self, monkeypatch):
        monkeypatch.delenv('ORB_ATR_FLOOR', raising=False)
        monkeypatch.delenv('ORB_SCALE_OUT', raising=False)
        cfg = yaml.safe_load(open(Path(__file__).parent.parent / 'orb.yaml'))
        cfg.setdefault('exit', {})['atr_stop_floor'] = {
            'enabled': True, 'k': 0.25}
        cfg['exit']['scale_out'] = {'enabled': True, 'frac': 0.40,
                                    'level_r': 3.0}
        eng = self._engine(cfg)
        assert eng.atr_floor_enabled is True
        assert eng.scale_out_enabled is True

    def test_env_kills(self, monkeypatch):
        monkeypatch.setenv('ORB_ATR_FLOOR', '0')
        monkeypatch.setenv('ORB_SCALE_OUT', '0')
        cfg = yaml.safe_load(open(Path(__file__).parent.parent / 'orb.yaml'))
        cfg.setdefault('exit', {})['atr_stop_floor'] = {'enabled': True}
        cfg['exit']['scale_out'] = {'enabled': True}
        eng = self._engine(cfg)
        assert eng.atr_floor_enabled is False
        assert eng.scale_out_enabled is False


# =========================================================================
# Floored stop at fill (P0-4)
# =========================================================================

class TestFlooredStopAtFill:
    def test_floor_bound_writes_both_columns_and_watch(self):
        eng = _skeleton_engine(atr_floor_enabled=True)
        pos = _pos(atr14=1.0)          # floor = 10 - 0.25 = 9.75
        eng.open_positions['TEST'] = pos
        eng._confirm_fill(pos, _fill_status(price=10.0))
        assert pos.stop_price == pytest.approx(9.75)
        fill_update = _db_updates(eng.db)[0]
        assert fill_update['stop_loss_price'] == pytest.approx(9.75)
        assert fill_update['real_stop_loss_price'] == pytest.approx(9.75)
        _, kwargs = eng.stop_monitor.add_watch.call_args
        assert kwargs['stop_price'] == pytest.approx(9.75)

    def test_floor_anchors_on_actual_fill(self):
        """P1-3: the anchor is the FILL price, not the planned entry."""
        eng = _skeleton_engine(atr_floor_enabled=True)
        pos = _pos(atr14=1.0)
        eng.open_positions['TEST'] = pos
        eng._confirm_fill(pos, _fill_status(price=10.20))   # slipped fill
        assert pos.stop_price == pytest.approx(10.20 - 0.25)

    def test_no_atr_fail_open(self):
        eng = _skeleton_engine(atr_floor_enabled=True)
        eng._get_feature_context = MagicMock(return_value={})
        pos = _pos(atr14=None)
        eng.open_positions['TEST'] = pos
        eng._confirm_fill(pos, _fill_status())
        assert pos.stop_price == pytest.approx(9.0)          # range_low
        fill_update = _db_updates(eng.db)[0]
        assert fill_update['stop_loss_price'] == pytest.approx(9.0)
        assert fill_update['real_stop_loss_price'] == pytest.approx(9.0)

    def test_degenerate_atr_fail_open(self):
        eng = _skeleton_engine(atr_floor_enabled=True)
        pos = _pos(atr14=0.0)
        eng.open_positions['TEST'] = pos
        eng._confirm_fill(pos, _fill_status())
        assert pos.stop_price == pytest.approx(9.0)

    def test_flag_off_untouched(self):
        eng = _skeleton_engine(atr_floor_enabled=False)
        pos = _pos(atr14=1.0)
        eng.open_positions['TEST'] = pos
        eng._confirm_fill(pos, _fill_status())
        assert pos.stop_price == pytest.approx(9.0)
        fill_update = _db_updates(eng.db)[0]
        assert 'stop_loss_price' not in fill_update
        assert 'real_stop_loss_price' not in fill_update


# =========================================================================
# Scale arming gate (P1-1)
# =========================================================================

class TestScaleArming:
    def test_not_armed_until_touchgo_resolved(self):
        eng = _skeleton_engine(scale_out_enabled=True)
        pos = _pos(order_id='', breakout_bar_ts=datetime.now(timezone.utc))
        eng._maybe_arm_scale(pos)
        assert pos.scale_armed is False
        eng.stop_monitor.arm_scale_out.assert_not_called()

    def test_armed_after_touchgo_resolved(self):
        eng = _skeleton_engine(scale_out_enabled=True)
        pos = _pos(order_id='', breakout_bar_ts=datetime.now(timezone.utc),
                   rule_m_evaluated=True, rule_d_evaluated=True)
        eng._maybe_arm_scale(pos)
        assert pos.scale_armed is True
        eng.stop_monitor.arm_scale_out.assert_called_once_with(
            'TEST', pytest.approx(13.0), 400)   # 10 + 3*1R, floor(0.4*1000)

    def test_armed_when_touchgo_inert(self):
        """No breakout bar (rehydrated) => touchgo inert => arm."""
        eng = _skeleton_engine(scale_out_enabled=True)
        pos = _pos(order_id='', breakout_bar_ts=None)
        eng._maybe_arm_scale(pos)
        assert pos.scale_armed is True
        eng.stop_monitor.arm_scale_out.assert_called_once()

    def test_time_fallback(self):
        eng = _skeleton_engine(scale_out_enabled=True)
        pos = _pos(order_id='', breakout_bar_ts=datetime.now(timezone.utc),
                   entry_time=datetime.now(timezone.utc) - timedelta(minutes=5))
        eng._maybe_arm_scale(pos)
        assert pos.scale_armed is True

    def test_tiny_qty_all_runner(self):
        eng = _skeleton_engine(scale_out_enabled=True)
        pos = _pos(order_id='', breakout_bar_ts=None, shares=2)
        eng._maybe_arm_scale(pos)
        assert pos.scale_armed is True
        eng.stop_monitor.arm_scale_out.assert_not_called()

    def test_flag_off_never_arms(self):
        eng = _skeleton_engine(scale_out_enabled=False)
        pos = _pos(order_id='', breakout_bar_ts=None)
        eng._maybe_arm_scale(pos)
        assert pos.scale_armed is False
        eng.stop_monitor.arm_scale_out.assert_not_called()

    def test_pending_position_never_arms(self):
        eng = _skeleton_engine(scale_out_enabled=True)
        pos = _pos(order_id='still-pending', breakout_bar_ts=None)
        eng._maybe_arm_scale(pos)
        assert pos.scale_armed is False


# =========================================================================
# Scale fill event lifecycle (P0-2 / P0-3)
# =========================================================================

def _scale_event(qty=400, price=13.0):
    return StopExitEvent(
        symbol='TEST', stop_price=0.0, exit_price=price, shares=qty,
        order_id='', exit_reason=ExitReason.SCALE_OUT.value,
        trade_db_id=42, filled_qty=qty, strategy='orb', confirmed=True)


def _final_event(price=10.5, qty=600, reason=ExitReason.LOCK_STOP.value):
    return StopExitEvent(
        symbol='TEST', stop_price=10.5, exit_price=price, shares=qty,
        order_id='o2', exit_reason=reason, trade_db_id=42,
        filled_qty=qty, strategy='orb', confirmed=True)


class TestScaleFillLifecycle:
    def test_scale_event_keeps_row_open(self):
        eng = _skeleton_engine()
        pos = _pos(order_id='')
        eng.open_positions['TEST'] = pos
        eng._handle_exit_event(_scale_event())
        # position NOT popped, shares reduced to the runner
        assert 'TEST' in eng.open_positions
        assert pos.shares == 600
        assert pos.scale_qty == 400
        assert pos.scale_pnl == pytest.approx((13.0 - 10.0) * 400)
        assert pos.scaled_at is not None
        # daily pnl untouched (realized-at-close)
        assert eng.daily_pnl == 0.0
        upd = _db_updates(eng.db)[0]
        assert set(upd) >= {'scale_qty', 'scale_price', 'scale_pnl',
                            'scaled_at'}
        assert 'pnl' not in upd
        assert 'exit_price' not in upd
        assert 'order_status' not in upd

    def test_final_exit_writes_combined_pnl_once(self):
        eng = _skeleton_engine()
        pos = _pos(order_id='')
        eng.open_positions['TEST'] = pos
        eng._handle_exit_event(_scale_event())
        eng._handle_exit_event(_final_event())
        assert 'TEST' not in eng.open_positions
        final_upd = _db_updates(eng.db)[-1]
        expected = (10.5 - 10.0) * 600 + (13.0 - 10.0) * 400
        assert final_upd['pnl'] == pytest.approx(expected)
        assert final_upd['order_status'] == 'closed'
        # runner keeps its OWN exit_reason
        assert final_upd['exit_reason'] == ExitReason.LOCK_STOP.value
        assert eng.daily_pnl == pytest.approx(expected)
        # pnl_pct over the ENTRY notional
        assert final_upd['pnl_pct'] == pytest.approx(
            expected / (10.0 * 1000) * 100)

    def test_unscaled_final_exit_byte_identical(self):
        eng = _skeleton_engine()
        pos = _pos(order_id='')
        eng.open_positions['TEST'] = pos
        eng._handle_exit_event(_final_event(price=9.0, qty=1000,
                                            reason=ExitReason.STOP_LOSS.value))
        upd = _db_updates(eng.db)[-1]
        assert upd['pnl'] == pytest.approx((9.0 - 10.0) * 1000)
        assert upd['pnl_pct'] == pytest.approx((9.0 - 10.0) / 10.0 * 100)

    def test_orphan_scale_event_writes_columns_by_id(self):
        eng = _skeleton_engine()
        eng._handle_exit_event(_scale_event())
        upd = _db_updates(eng.db)[0]
        assert upd['scale_qty'] == 400
        assert 'pnl' not in upd


# =========================================================================
# Kill-rail accounting (realized-at-close)
# =========================================================================

class TestKillRailAccounting:
    def _tmp_trades_db(self, tmp_path):
        p = tmp_path / 'trades.db'
        con = sqlite3.connect(p)
        con.execute("""CREATE TABLE trades (
            id INTEGER PRIMARY KEY, strategy TEXT, trade_date TEXT,
            pnl REAL, scale_pnl REAL, scaled_at TEXT)""")
        con.commit()
        con.close()
        return p

    def test_open_scaled_position_not_realized(self, tmp_path):
        """A booked scale leg on a still-open row (pnl NULL) must NOT count
        toward kill rails until the final close writes the combined pnl."""
        p = self._tmp_trades_db(tmp_path)
        con = sqlite3.connect(p)
        con.execute("INSERT INTO trades (strategy, trade_date, pnl, "
                    "scale_pnl, scaled_at) VALUES "
                    "('orb', '2026-08-24', NULL, 1200.0, '2026-08-24T14:00')")
        con.execute("INSERT INTO trades (strategy, trade_date, pnl) "
                    "VALUES ('orb', '2026-08-24', 100.0)")
        con.commit()
        con.close()
        eng = _skeleton_engine()
        eng.db = MagicMock()
        eng.db._trades_path = str(p)
        assert eng._realized_orb_pnl('2026-08-24') == pytest.approx(100.0)

    def test_closed_scaled_position_counts_once(self, tmp_path):
        p = self._tmp_trades_db(tmp_path)
        con = sqlite3.connect(p)
        con.execute("INSERT INTO trades (strategy, trade_date, pnl, "
                    "scale_pnl, scaled_at) VALUES "
                    "('orb', '2026-08-24', 1500.0, 1200.0, "
                    "'2026-08-24T14:00')")
        con.commit()
        con.close()
        eng = _skeleton_engine()
        eng.db = MagicMock()
        eng.db._trades_path = str(p)
        # combined pnl only — scale_pnl must NOT double count
        assert eng._realized_orb_pnl('2026-08-24') == pytest.approx(1500.0)


# =========================================================================
# Rehydration (P0-4 kill-9 class) + rollback drill (P1-7)
# =========================================================================

def _sync_engine(db_row, scale_out_enabled=False):
    eng = _skeleton_engine(scale_out_enabled=scale_out_enabled)
    eng.alpaca.get_open_positions.return_value = [
        {'symbol': 'TEST', 'qty': db_row.get('shares', 1000)}]
    eng.alpaca.trading_client.get_orders.return_value = []
    eng.db.get_open_trades.return_value = [db_row]
    return eng


def _scaled_db_row():
    import json
    return {
        'id': 42, 'symbol': 'TEST', 'order_id': '', 'order_status': 'filled',
        'entry_price': 10.0, 'fill_price': 10.0, 'stop_loss_price': 9.75,
        'shares': 1000, 'filled_qty': 1000,
        'filled_at': datetime.now(timezone.utc),
        'scale_qty': 400, 'scale_price': 13.0, 'scale_pnl': 1200.0,
        'scaled_at': '2026-08-24T14:00:00+00:00',
        'pattern_data': json.dumps({
            'range_high': 10.0, 'range_low': 9.0, 'lock_arm_at_r': 1.75,
            'lock_stop_r': 0.5, 'composite_score': 0.5, 'quintile': 'Q4',
            'atr14': 1.0}),
    }


class TestRehydration:
    def test_mid_scale_rehydrate(self):
        """Kill -9 after the scale fill: restart must come back with runner
        shares, the FLOORED stop, and scale_done on the watch (P0-4)."""
        eng = _sync_engine(_scaled_db_row())
        eng.sync_positions()
        pos = eng.open_positions['TEST']
        assert pos.shares == 600                     # runner, not full
        assert pos.stop_price == pytest.approx(9.75)  # floored stop survives
        assert pos.scale_qty == 400
        assert pos.scale_pnl == pytest.approx(1200.0)
        assert pos.scaled_at is not None
        assert pos.scale_armed is True               # never re-arms
        _, kwargs = eng.stop_monitor.add_watch.call_args
        assert kwargs['shares'] == 600
        assert kwargs['stop_price'] == pytest.approx(9.75)
        assert kwargs['scale_done'] is True
        assert kwargs['lock_r_unit'] == pytest.approx(1.0)

    def test_rollback_flag_off_still_scale_aware(self):
        """P1-7: flags off + restart with an open scaled position — the
        rehydration is DATA-driven, so shares/stop/scale state stay right
        and no new scale is armed."""
        eng = _sync_engine(_scaled_db_row(), scale_out_enabled=False)
        eng.sync_positions()
        pos = eng.open_positions['TEST']
        assert pos.shares == 600
        assert pos.scale_pnl == pytest.approx(1200.0)
        eng.stop_monitor.arm_scale_out.assert_not_called()
        # ...and the final close still composes the combined pnl:
        eng._handle_exit_event(_final_event())
        final_upd = _db_updates(eng.db)[-1]
        assert final_upd['pnl'] == pytest.approx(
            (10.5 - 10.0) * 600 + 1200.0)

    def test_unscaled_rehydrate_arms_when_flag_on(self):
        row = _scaled_db_row()
        row.update({'scale_qty': None, 'scale_price': None,
                    'scale_pnl': None, 'scaled_at': None})
        eng = _sync_engine(row, scale_out_enabled=True)
        eng.sync_positions()
        pos = eng.open_positions['TEST']
        assert pos.shares == 1000
        eng.stop_monitor.arm_scale_out.assert_called_once()

    def test_unscaled_rehydrate_flag_off_legacy(self):
        row = _scaled_db_row()
        row.update({'scale_qty': None, 'scale_price': None,
                    'scale_pnl': None, 'scaled_at': None,
                    'stop_loss_price': 9.0})
        eng = _sync_engine(row, scale_out_enabled=False)
        eng.sync_positions()
        pos = eng.open_positions['TEST']
        assert pos.shares == 1000
        assert pos.stop_price == pytest.approx(9.0)
        eng.stop_monitor.arm_scale_out.assert_not_called()


# =========================================================================
# FC DB sync composition (P0-3.2)
# =========================================================================

class TestSyncDbAfterFc:
    def _sell_order(self, price=11.0, qty=600):
        return SimpleNamespace(
            id='fc-1', side=SimpleNamespace(value='sell'),
            status=SimpleNamespace(value='filled'),
            filled_avg_price=price, filled_qty=qty,
            filled_at=datetime.now(timezone.utc))

    def test_scaled_row_composes_scale_pnl(self):
        eng = _skeleton_engine()
        row = _scaled_db_row()
        row['exit_price'] = None
        eng.db.get_open_trades.return_value = [row]
        eng.alpaca.trading_client.get_orders.return_value = [
            self._sell_order(price=11.0, qty=600)]
        eng.check_exits = MagicMock(return_value=[])
        eng._sync_db_after_fc(['TEST'])
        upd = _db_updates(eng.db)[-1]
        assert upd['pnl'] == pytest.approx((11.0 - 10.0) * 600 + 1200.0)
        assert upd['exit_reason'] == ExitReason.FORCE_CLOSE.value

    def test_unscaled_row_unchanged(self):
        eng = _skeleton_engine()
        row = _scaled_db_row()
        row.update({'exit_price': None, 'scale_qty': None,
                    'scale_price': None, 'scale_pnl': None,
                    'scaled_at': None})
        eng.db.get_open_trades.return_value = [row]
        eng.alpaca.trading_client.get_orders.return_value = [
            self._sell_order(price=11.0, qty=1000)]
        eng.check_exits = MagicMock(return_value=[])
        eng._sync_db_after_fc(['TEST'])
        upd = _db_updates(eng.db)[-1]
        assert upd['pnl'] == pytest.approx((11.0 - 10.0) * 1000)
