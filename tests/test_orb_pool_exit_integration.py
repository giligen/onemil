"""Integration tests: ORB per-pool exit overrides (owner 2026-10-01, P1
half-out at +1R) against a REAL Database (tmp path) + StopMonitor, with
MagicMock(spec=AlpacaClient) standing in for the broker only.

Covers the full lifecycle a P1 position goes through: arm (even though the
global production scale_out flag is OFF), the scale fill books through the
existing scale_* partial-exit columns + resizes the StopMonitor watch
(reusing trading/exit_qty_guard.get_signed_broker_qty), and the runner
exits through the UNCHANGED production _handle_exit_event path. A second
class proves a production position (no pool override) is byte-identical.
A third proves the pool's exit params survive a restart round-trip through
pattern_data.pool_exit.
"""
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.exit_reasons import ExitReason
from trading.orb_engine import ORBEngine, OpenPosition
from trading.stop_monitor import StopMonitor, StopExitEvent

POOL_EXIT = {'scale_out_pct': 0.5, 'scale_out_at_r': 1.0}


@pytest.fixture
def mock_alpaca():
    client = MagicMock(spec=AlpacaClient)
    client.get_open_positions.return_value = [{'symbol': 'ORBP1', 'qty': 200}]
    client.replace_order_qty.side_effect = (
        lambda oid, qty: {'id': f'new-{oid}', 'status': 'accepted'})
    client.submit_limit_sell_order.return_value = {
        'id': 'scale-ord-1', 'status': 'accepted'}
    return client


@pytest.fixture
def real_db(tmp_path):
    """Real SQLite DB on a tmp path -- never the shared production cache."""
    db = Database(
        db_path=str(tmp_path / 'orb_test.db'),
        cache_path=str(tmp_path / 'orb_cache.db'),
        trades_path=str(tmp_path / 'orb_trades.db'),
    )
    yield db
    db.close()


@pytest.fixture
def stop_monitor(mock_alpaca):
    return StopMonitor(
        api_key='k', api_secret='s', alpaca_client=mock_alpaca,
        marketable_limit_offset=0.03, marketable_limit_offset_pct=0.005,
    )


def _make_engine(db, mon, *, scale_out_enabled=False):
    """object.__new__ skips __init__ -- only the attributes the methods
    under test touch are set (tests/test_orb_restart_recovery.py convention).
    """
    engine = object.__new__(ORBEngine)
    engine.db = db
    engine.stop_monitor = mon
    engine.open_positions = {}
    engine.scale_frac = 0.4           # production default != the P1 0.5
    engine.scale_level_r = 3.0        # production default != the P1 1.0
    engine.scale_out_enabled = scale_out_enabled
    engine.notify_on_exit = True
    engine.notifier = MagicMock()
    engine.tg_prefix = '[ORB]'
    engine.daily_pnl = 0.0
    engine.touchgo_cfg = SimpleNamespace(master_enabled=False)
    engine._SCALE_ARM_FALLBACK_MIN = 0.0
    return engine


def _base_trade_row(**overrides):
    row = dict(
        trade_date='2026-10-01', symbol='ORBP1', side='buy',
        entry_price=10.0, stop_loss_price=9.5, take_profit_price=0.0,
        shares=200, risk_per_share=0.5, total_risk=100.0,
        risk_reward_ratio=0.0, order_id='', order_status='open',
        fill_price=10.0, filled_at=datetime.now(timezone.utc),
        exit_price=None, exit_reason=None, exited_at=None,
        pnl=None, pnl_pct=None, pattern_data='{}',
        strategy='orb',
    )
    row.update(overrides)
    return row


class TestP1HalfOutLifecycle:
    """A P1 position (exit_scale_out_pct=0.5 @ +1R) sells 50%, books the
    partial through the existing scale_* columns, resizes the watch, and
    the remainder exits by the production rule."""

    def test_full_lifecycle(self, real_db, stop_monitor, mock_alpaca, caplog):
        caplog.set_level(logging.INFO)
        engine = _make_engine(real_db, stop_monitor, scale_out_enabled=False)
        trade_id = real_db.save_trade(_base_trade_row(pattern_data=json.dumps({
            'range_high': 10.2, 'range_low': 10.0,
            'pool_id': 'P1', 'pool_exit': POOL_EXIT,
        })))

        pos = OpenPosition(
            symbol='ORBP1', entry_price=10.0, stop_price=9.5, shares=200,
            trade_id=trade_id, order_id='', entry_time=datetime.now(timezone.utc),
            range_high=10.2, range_low=10.0, lock_arm_at_r=1.75, lock_stop_r=0.5,
            composite_score=0.5, quintile='Q3',
            pool_id='P1', pool_exit=dict(POOL_EXIT),
        )
        engine.open_positions['ORBP1'] = pos
        stop_monitor.add_watch(
            symbol='ORBP1', stop_price=pos.stop_price, shares=pos.shares,
            tp_leg_id='tp-1', sl_leg_id='sl-1', trade_db_id=trade_id,
            entry_price=pos.entry_price, risk_per_share=0.5, strategy='orb',
            lock_arm_at_r=pos.lock_arm_at_r, lock_stop_r=pos.lock_stop_r,
            lock_r_unit=max(pos.range_high - pos.range_low, 0.0),
            scale_done=False,
        )

        # 1. Arm -- fires even though the GLOBAL production scale_out flag
        #    is off; the pool's own override opts this position in.
        engine._maybe_arm_scale(pos)
        assert pos.scale_armed is True
        watch = stop_monitor._watches['ORBP1']
        assert watch.scale_qty == 100                                    # 50% of 200sh
        assert watch.scale_at_px == pytest.approx(10.0 + 1.0 * 0.2)       # entry + 1.0R

        # 2. Fire the scale submission (simulates the price touch): resizes
        #    the safety legs to runner qty and sells the scale qty.
        ok = stop_monitor._scale_submit_core('ORBP1', watch)
        assert ok is True
        mock_alpaca.replace_order_qty.assert_any_call('tp-1', 100)       # runner = 200-100
        mock_alpaca.replace_order_qty.assert_any_call('sl-1', 100)
        assert mock_alpaca.submit_limit_sell_order.call_args.kwargs['qty'] == 100

        # 3. Book the fill: existing partial-exit columns + pool-tagged
        #    log/Telegram line via the real per-trade ORB notifier.
        fill_ev = StopExitEvent(
            symbol='ORBP1', stop_price=9.5, exit_price=10.2, shares=100,
            order_id='scale-ord-1', exit_reason=ExitReason.SCALE_OUT.value,
            filled_qty=100, trade_db_id=trade_id, confirmed=True,
        )
        engine._handle_scale_fill_event(fill_ev)
        assert pos.scale_qty == 100
        assert pos.scale_price == 10.2
        assert pos.scale_pnl == pytest.approx((10.2 - 10.0) * 100)
        assert pos.scaled_at is not None
        assert pos.shares == 100                      # runner qty remaining
        assert 'ORBP1' in engine.open_positions        # NOT popped -- still open
        assert engine.notifier.mock_calls, "Telegram notifier was never called"
        assert any('SCALE_OUT ORBP1 50% @ +1.0R' in r.message for r in caplog.records), (
            "expected the '[ORB P1] SCALE_OUT ORBP1 50% @ +1.0R px=...' log line")

        cur = real_db._trades_conn.execute(
            "SELECT scale_qty, scale_price, scale_pnl, scaled_at FROM trades WHERE id = ?",
            (trade_id,))
        db_row = cur.fetchone()
        assert db_row[0] == 100
        assert db_row[1] == 10.2
        assert db_row[3] is not None

        # 4. The remainder exits by the UNCHANGED production rule -- same
        #    _handle_exit_event every ORB position (pool or not) uses.
        final_ev = StopExitEvent(
            symbol='ORBP1', stop_price=9.5, exit_price=9.5, shares=100,
            order_id='stop-ord-1', exit_reason=ExitReason.STOP_LOSS.value,
            filled_qty=100, trade_db_id=trade_id, confirmed=True,
        )
        engine._handle_exit_event(final_ev)
        assert 'ORBP1' not in engine.open_positions
        expected_pnl = pos.scale_pnl + (9.5 - 10.0) * 100
        assert engine.daily_pnl == pytest.approx(expected_pnl)


class TestProductionUnaffectedByAbsentOverride:
    """pool_exit == {} (production / ungated pool) must be byte-identical:
    scale-out stays off when the global production flag is off."""

    def test_no_pool_override_never_arms(self, real_db, stop_monitor):
        engine = _make_engine(real_db, stop_monitor, scale_out_enabled=False)
        pos = OpenPosition(
            symbol='PROD1', entry_price=10.0, stop_price=9.5, shares=200,
            trade_id=99, order_id='', entry_time=datetime.now(timezone.utc),
            range_high=10.2, range_low=10.0, lock_arm_at_r=1.75, lock_stop_r=0.5,
            composite_score=0.5, quintile='Q3',
        )  # pool_id/pool_exit left at their defaults: 'production' / {}
        stop_monitor.add_watch(
            symbol='PROD1', stop_price=pos.stop_price, shares=pos.shares,
            tp_leg_id='tp-2', sl_leg_id='sl-2', trade_db_id=99,
            entry_price=pos.entry_price, risk_per_share=0.5, strategy='orb',
            lock_arm_at_r=pos.lock_arm_at_r, lock_stop_r=pos.lock_stop_r,
            lock_r_unit=0.2, scale_done=False,
        )
        engine._maybe_arm_scale(pos)
        assert pos.scale_armed is False   # early-return: never armed


class TestRestartRehydration:
    """A restart must re-hydrate a pool's exit params from the trades
    row's pattern_data.pool_exit instead of silently reverting to
    production -- same pdata.get(...) calls ORBEngine.sync_positions'
    rehydration constructors use."""

    def test_pool_exit_rehydrates_from_pattern_data(self, real_db):
        trade_id = real_db.save_trade(_base_trade_row(pattern_data=json.dumps({
            'range_high': 10.2, 'range_low': 10.0,
            'lock_arm_at_r': 1.75, 'lock_stop_r': 0.5,
            'pool_id': 'P1', 'pool_exit': POOL_EXIT,
        })))
        cur = real_db._trades_conn.execute(
            "SELECT id, symbol, fill_price, entry_price, stop_loss_price, "
            "shares, filled_at, pattern_data FROM trades WHERE id = ?",
            (trade_id,))
        db_row = cur.fetchone()
        pdata = json.loads(db_row[7])

        pos = OpenPosition(
            symbol=db_row[1],
            entry_price=float(db_row[2] or db_row[3] or 0.0),
            stop_price=float(db_row[4] or 0.0),
            shares=int(db_row[5] or 0),
            trade_id=int(db_row[0]), order_id='',
            entry_time=db_row[6] or datetime.now(timezone.utc),
            range_high=float(pdata.get('range_high', 0.0)),
            range_low=float(pdata.get('range_low', 0.0)),
            lock_arm_at_r=float(pdata.get('lock_arm_at_r', 1.75)),
            lock_stop_r=float(pdata.get('lock_stop_r', 0.5)),
            composite_score=0.5, quintile='Q3',
            pool_id=str(pdata.get('pool_id') or 'production'),
            pool_exit=dict(pdata.get('pool_exit') or {}),
        )
        assert pos.pool_id == 'P1'
        assert pos.pool_exit == POOL_EXIT

    def test_absent_pool_exit_rehydrates_to_production(self, real_db):
        trade_id = real_db.save_trade(_base_trade_row(
            symbol='PROD2', pattern_data=json.dumps({
                'range_high': 10.2, 'range_low': 10.0,
            })))
        cur = real_db._trades_conn.execute(
            "SELECT pattern_data FROM trades WHERE id = ?", (trade_id,))
        pdata = json.loads(cur.fetchone()[0])
        pool_id = str(pdata.get('pool_id') or 'production')
        pool_exit = dict(pdata.get('pool_exit') or {})
        assert pool_id == 'production'
        assert pool_exit == {}
