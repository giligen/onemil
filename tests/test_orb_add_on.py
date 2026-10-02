"""Tests for the ORB ADD-ON ("one more entry after R", owner 2026-10-01).

Three layers, per CLAUDE.md:
  * TestOrbAddOnPure         -- trading/orb_add_on.py, no I/O, no engine.
  * TestResizeWatchQty       -- trading/stop_monitor.py's new
                                 resize_watch_qty, against a REAL
                                 StopMonitor instance (constructed via
                                 __new__ to skip the WebSocket-spinning
                                 __init__ -- this method touches only
                                 _watch_lock/_watches).
  * TestEngineAddOn          -- trading/orb_engine.py wiring, using the
                                 SAME mock fixture shapes as
                                 tests/test_orb_engine.py's `engine`
                                 fixture (MagicMock(spec=AlpacaClient),
                                 MagicMock(spec=Database),
                                 MagicMock(spec=StopMonitor)).
  * TestPersistenceRoundTrip -- a REAL Database on a tmp path: pattern_data
                                 round-trips through save/update/get.
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading import orb_add_on
from trading.orb_engine import ORBEngine, OpenPosition
from trading.stop_monitor import StopMonitor, WatchEntry


# =========================================================================
# Pure module
# =========================================================================

class TestResolveAddOnParams:
    def test_empty_or_none_is_byte_identical(self):
        assert orb_add_on.resolve_add_on_params({}) == {}
        assert orb_add_on.resolve_add_on_params(None) == {}

    def test_valid_cfg_round_trips(self):
        cfg = {'enabled': True, 'at_r': 1.5, 'units': 0.5,
               'stop_mode': 'add_breakeven', 'max_adds': 2,
               'applies_to': ['production', 'gap45']}
        out = orb_add_on.resolve_add_on_params(cfg)
        assert out == cfg

    def test_invalid_values_dropped_not_raised(self):
        cfg = {'at_r': -1.0, 'units': 'nope', 'stop_mode': 'bogus',
               'max_adds': 0, 'applies_to': 'not-a-list', 'enabled': 'yes'}
        out = orb_add_on.resolve_add_on_params(cfg)
        assert out == {}

    def test_partial_override_only_sets_present_keys(self):
        out = orb_add_on.resolve_add_on_params({'at_r': 2.0})
        assert out == {'at_r': 2.0}

    def test_add_on_param_chain_override_then_production_then_default(self):
        production = {**orb_add_on.DEFAULT_ADD_ON, 'at_r': 1.0, 'enabled': True}
        override = {'at_r': 2.0}
        assert orb_add_on.add_on_param(override, production, 'at_r') == 2.0
        assert orb_add_on.add_on_param(override, production, 'enabled') is True
        assert orb_add_on.add_on_param({}, {}, 'max_adds') == orb_add_on.DEFAULT_ADD_ON['max_adds']


class TestAddCutoff:
    def test_before_1500_et_not_blocked(self):
        assert orb_add_on.is_past_add_cutoff(datetime(2026, 9, 25, 14, 59)) is False

    def test_at_or_after_1500_et_blocked(self):
        assert orb_add_on.is_past_add_cutoff(datetime(2026, 9, 25, 15, 0)) is True
        assert orb_add_on.is_past_add_cutoff(datetime(2026, 9, 25, 15, 1)) is True


class TestPoolEligibility:
    def test_production_default(self):
        assert orb_add_on.is_pool_eligible('production', ['production']) is True
        assert orb_add_on.is_pool_eligible('gap45', ['production']) is False

    def test_pool_listed(self):
        assert orb_add_on.is_pool_eligible('gap45', ['production', 'gap45']) is True


class TestAddsRemaining:
    def test_counts(self):
        assert orb_add_on.adds_remaining(0, 1) is True
        assert orb_add_on.adds_remaining(1, 1) is False
        assert orb_add_on.adds_remaining(1, 2) is True


class TestAddTrigger:
    def test_not_reached_below_level(self):
        # entry 10.0, range_size 1.0, at_r 1.0 -> trigger at 11.0
        assert orb_add_on.has_reached_add_trigger(10.99, 10.0, 1.0, 1.0) is False

    def test_reached_at_or_above_level(self):
        assert orb_add_on.has_reached_add_trigger(11.00, 10.0, 1.0, 1.0) is True
        assert orb_add_on.has_reached_add_trigger(12.00, 10.0, 1.0, 1.0) is True

    def test_degenerate_range_fails_closed(self):
        assert orb_add_on.has_reached_add_trigger(100.0, 10.0, 0.0, 1.0) is False

    def test_non_positive_at_r_fails_closed(self):
        assert orb_add_on.has_reached_add_trigger(100.0, 10.0, 1.0, 0.0) is False


class TestComputeAddQty:
    def test_same_size_as_base(self):
        assert orb_add_on.compute_add_qty(400, 1.0) == 400

    def test_half_size(self):
        assert orb_add_on.compute_add_qty(400, 0.5) == 200

    def test_zero_or_negative_inputs(self):
        assert orb_add_on.compute_add_qty(0, 1.0) == 0
        assert orb_add_on.compute_add_qty(400, 0.0) == 0


class TestCombinedPosition:
    def test_weighted_average(self):
        qty, avg = orb_add_on.compute_combined_position(100, 10.0, 100, 12.0)
        assert qty == 200
        assert avg == pytest.approx(11.0)

    def test_unequal_qty(self):
        qty, avg = orb_add_on.compute_combined_position(300, 10.0, 100, 14.0)
        assert qty == 400
        assert avg == pytest.approx((300 * 10.0 + 100 * 14.0) / 400)

    def test_zero_add_is_noop(self):
        qty, avg = orb_add_on.compute_combined_position(100, 10.0, 0, 99.0)
        assert (qty, avg) == (100, 10.0)


class TestStopModes:
    def test_original_and_live_lock_have_no_separate_leg(self):
        assert orb_add_on.resolve_add_breakeven_leg('original', 12.0, 100) is None
        assert orb_add_on.resolve_add_breakeven_leg('live_lock', 12.0, 100) is None

    def test_add_breakeven_builds_leg_at_add_price(self):
        leg = orb_add_on.resolve_add_breakeven_leg('add_breakeven', 12.0, 100)
        assert leg == {'price': 12.0, 'qty': 100, 'done': False}

    def test_breakeven_triggers_on_drop_to_its_own_price(self):
        leg = {'price': 12.0, 'qty': 100, 'done': False}
        assert orb_add_on.add_breakeven_triggered(12.5, leg) is False
        assert orb_add_on.add_breakeven_triggered(12.0, leg) is True
        assert orb_add_on.add_breakeven_triggered(11.9, leg) is True

    def test_breakeven_inert_once_done_or_absent(self):
        assert orb_add_on.add_breakeven_triggered(5.0, None) is False
        assert orb_add_on.add_breakeven_triggered(5.0, {'price': 12.0, 'qty': 100, 'done': True}) is False


class TestBuildAddRecord:
    def test_shape(self):
        ts = datetime(2026, 9, 25, 14, 0, tzinfo=timezone.utc)
        rec = orb_add_on.build_add_record(1.0, 12.34, 50, ts)
        assert rec == {'at_r': 1.0, 'px': 12.34, 'qty': 50, 'ts': ts.isoformat()}


# =========================================================================
# StopMonitor.resize_watch_qty against a REAL StopMonitor
# =========================================================================

class TestResizeWatchQty:
    def _real_monitor(self):
        sm = StopMonitor.__new__(StopMonitor)  # skip __init__ (no WS/threads needed)
        sm._watch_lock = threading.Lock()
        sm._watches = {}
        return sm

    def test_resizes_existing_watch_in_place(self):
        sm = self._real_monitor()
        watch = WatchEntry(symbol='ABCD', stop_price=9.0, shares=100,
                            tp_leg_id='tp1', sl_leg_id='sl1',
                            trailing_active=True, highest_since_entry=11.5)
        sm._watches['ABCD'] = watch
        assert sm.resize_watch_qty('ABCD', 200) is True
        assert sm._watches['ABCD'].shares == 200
        # Every OTHER in-flight field survives untouched (the whole point
        # of resize_watch_qty vs re-calling add_watch).
        assert sm._watches['ABCD'].trailing_active is True
        assert sm._watches['ABCD'].highest_since_entry == 11.5
        assert sm._watches['ABCD'] is watch

    def test_no_watch_returns_false(self):
        sm = self._real_monitor()
        assert sm.resize_watch_qty('NOPE', 100) is False


# =========================================================================
# Engine wiring (mirrors tests/test_orb_engine.py's fixture shapes)
# =========================================================================

@pytest.fixture
def orb_cfg():
    """The live orb.yaml with the add-on block REMOVED: this module's baseline is "no
    exit.add_on" and every test that needs the add sets the flag itself. The live file
    is gitignored and changes with the paper/live plan (2026-10-02 it shipped
    add_on.enabled: true and the pre-boot suite failed on the stale assumption)."""
    yaml_path = Path(__file__).parent.parent / 'orb.yaml'
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    (cfg.get('exit') or {}).pop('add_on', None)
    return cfg


@pytest.fixture
def mock_alpaca():
    client = MagicMock(spec=AlpacaClient)
    client.get_open_positions.return_value = []
    client.get_account_info.return_value = {'buying_power': 100_000.0}
    client.submit_limit_buy_order.return_value = {
        'id': 'add-order-1', 'status': 'filled', 'filled_avg_price': 11.05,
    }
    client.replace_order_qty.return_value = {'id': 'leg-new-1'}
    return client


@pytest.fixture
def mock_db():
    db = MagicMock(spec=Database)
    db.save_trade.return_value = 100
    db.get_open_trades.return_value = []
    db.update_trade.return_value = True
    db.get_trade_by_id.return_value = {'id': 100, 'pattern_data': json.dumps({'pool_id': 'production'})}
    return db


@pytest.fixture
def mock_stop_monitor():
    sm = MagicMock(spec=StopMonitor)
    sm.polling_mode = False
    sm.drain_exit_events.return_value = []
    sm.resize_watch_qty.return_value = True
    return sm


@pytest.fixture
def engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor):
    orb_cfg['strategy']['enabled'] = True
    return ORBEngine(alpaca_client=mock_alpaca, db=mock_db,
                      stop_monitor=mock_stop_monitor, config=orb_cfg)


def _make_pos(**overrides):
    base = dict(
        symbol='TEST', entry_price=10.0, stop_price=9.0, shares=100,
        trade_id=100, order_id='', entry_time=datetime.now(timezone.utc),
        range_high=10.5, range_low=9.5, lock_arm_at_r=1.75, lock_stop_r=0.5,
        composite_score=1.0, quintile='Q3', tp_leg_id='tp-1', sl_leg_id='sl-1',
    )
    base.update(overrides)
    return OpenPosition(**base)


class TestEngineAddOnDisabledByteIdentical:
    def test_disabled_add_on_never_submits_or_touches_position(self, engine, mock_alpaca):
        assert engine.add_on_cfg['enabled'] is False  # orb.yaml ships without exit.add_on
        pos = _make_pos()
        import pandas as pd
        bars_df = pd.DataFrame({'close': [11.0]})
        engine._maybe_fire_add_on(pos, bars_df)
        mock_alpaca.submit_limit_buy_order.assert_not_called()
        assert pos.shares == 100
        assert pos.add_count == 0
        assert pos.combined_avg_price is None


class TestEngineAddOnTrigger:
    def test_trigger_submits_add_resizes_legs_and_books_combined_price(
            self, engine, mock_alpaca, mock_stop_monitor, mock_db, monkeypatch):
        # Pin the cutoff rail via the pure function (avoids real-wall-clock
        # flakiness depending on when this suite happens to run).
        monkeypatch.setattr(orb_add_on, 'is_past_add_cutoff', lambda now_et: False)
        engine.add_on_cfg = {**orb_add_on.DEFAULT_ADD_ON, 'enabled': True, 'at_r': 1.0}
        pos = _make_pos()  # range_size = 1.0 -> trigger at 10.0 + 1.0*1.0 = 11.0
        # Broker-truth guard (trading/exit_qty_guard.py): the resize must
        # see the COMBINED qty actually long on the account post-add-fill.
        mock_alpaca.get_open_positions.return_value = [{'symbol': 'TEST', 'qty': 200}]
        import pandas as pd
        bars_df = pd.DataFrame({'close': [11.00]})

        engine._maybe_fire_add_on(pos, bars_df)

        mock_alpaca.submit_limit_buy_order.assert_called_once()
        kwargs = mock_alpaca.submit_limit_buy_order.call_args.kwargs
        assert kwargs['symbol'] == 'TEST'
        assert kwargs['qty'] == 100  # units=1.0 default -> same as base
        assert kwargs['client_order_id'].startswith('orb-add-')

        # combined: 100sh @ 10.0 + 100sh @ 11.05 -> 200sh @ 10.525
        assert pos.shares == 200
        assert pos.add_count == 1
        assert pos.combined_avg_price == pytest.approx(10.525)
        assert len(pos.adds) == 1
        assert pos.adds[0]['qty'] == 100

        mock_alpaca.replace_order_qty.assert_any_call('tp-1', 200)
        mock_alpaca.replace_order_qty.assert_any_call('sl-1', 200)
        mock_stop_monitor.resize_watch_qty.assert_called_once_with('TEST', 200)
        mock_db.update_trade.assert_called_once()
        assert mock_db.update_trade.call_args.args[1]['shares'] == 200

    def test_not_triggered_below_level_is_noop(self, engine, mock_alpaca):
        engine.add_on_cfg = {**orb_add_on.DEFAULT_ADD_ON, 'enabled': True, 'at_r': 1.0}
        pos = _make_pos()
        import pandas as pd
        bars_df = pd.DataFrame({'close': [10.99]})
        engine._maybe_fire_add_on(pos, bars_df)
        mock_alpaca.submit_limit_buy_order.assert_not_called()
        assert pos.add_count == 0

    def test_max_adds_enforced(self, engine, mock_alpaca):
        engine.add_on_cfg = {**orb_add_on.DEFAULT_ADD_ON, 'enabled': True, 'at_r': 1.0, 'max_adds': 1}
        pos = _make_pos(add_count=1)
        import pandas as pd
        bars_df = pd.DataFrame({'close': [50.0]})
        engine._maybe_fire_add_on(pos, bars_df)
        mock_alpaca.submit_limit_buy_order.assert_not_called()

    def test_applies_to_gates_non_eligible_pool(self, engine, mock_alpaca):
        engine.add_on_cfg = {**orb_add_on.DEFAULT_ADD_ON, 'enabled': True, 'at_r': 1.0,
                              'applies_to': ['production']}
        pos = _make_pos(pool_id='gap45')
        import pandas as pd
        bars_df = pd.DataFrame({'close': [50.0]})
        engine._maybe_fire_add_on(pos, bars_df)
        mock_alpaca.submit_limit_buy_order.assert_not_called()

    def test_past_cutoff_blocks_add(self, engine, mock_alpaca, monkeypatch):
        monkeypatch.setattr(orb_add_on, 'is_past_add_cutoff', lambda now_et: True)
        engine.add_on_cfg = {**orb_add_on.DEFAULT_ADD_ON, 'enabled': True, 'at_r': 1.0}
        pos = _make_pos()
        import pandas as pd
        bars_df = pd.DataFrame({'close': [11.00]})
        engine._maybe_fire_add_on(pos, bars_df)
        mock_alpaca.submit_limit_buy_order.assert_not_called()

    def test_insufficient_buying_power_blocks_add(self, engine, mock_alpaca, monkeypatch):
        monkeypatch.setattr(orb_add_on, 'is_past_add_cutoff', lambda now_et: False)
        engine.add_on_cfg = {**orb_add_on.DEFAULT_ADD_ON, 'enabled': True, 'at_r': 1.0}
        mock_alpaca.get_account_info.return_value = {'buying_power': 1.0}
        pos = _make_pos()
        import pandas as pd
        bars_df = pd.DataFrame({'close': [11.00]})
        engine._maybe_fire_add_on(pos, bars_df)
        mock_alpaca.submit_limit_buy_order.assert_not_called()


class TestCombinedPnlBooking:
    def test_exit_pnl_uses_combined_average_price(self, engine, mock_db):
        pos = _make_pos(shares=200, combined_avg_price=10.525, add_count=1,
                         adds=[{'at_r': 1.0, 'px': 11.05, 'qty': 100, 'ts': 'x'}])
        engine.open_positions['TEST'] = pos
        ev = MagicMock()
        ev.symbol = 'TEST'
        ev.exit_reason = 'stop'
        ev.confirmed = True
        ev.exit_price = 12.0
        engine._handle_exit_event(ev)
        # (12.0 - 10.525) * 200 == 295.0, NOT (12.0 - 10.0) * 200 == 400.0
        assert mock_db.update_trade.call_args.args[1]['pnl'] == pytest.approx(295.0)


# =========================================================================
# Persistence round trip (real Database, tmp path)
# =========================================================================

class TestPersistenceRoundTrip:
    def test_add_on_blob_round_trips_through_real_database(self, tmp_path):
        db = Database(
            db_path=str(tmp_path / 'cache_test.db'),
            cache_path=str(tmp_path / 'cache_test.db'),
            trades_path=str(tmp_path / 'trades_test.db'),
        )
        trade_id = db.save_trade({
            'trade_date': '2026-09-25', 'symbol': 'TEST', 'side': 'buy',
            'entry_price': 10.0, 'stop_loss_price': 9.5, 'take_profit_price': 11.0,
            'shares': 100, 'risk_per_share': 0.5, 'total_risk': 50.0,
            'risk_reward_ratio': 2.0, 'order_id': '', 'order_status': 'filled',
            'fill_price': 10.0, 'filled_at': datetime.now(timezone.utc),
            'exit_price': None, 'exit_reason': None, 'exited_at': None,
            'pnl': None, 'pnl_pct': None,
            'pattern_data': json.dumps({'pool_id': 'production'}),
            'strategy': 'orb', 'account': 'paper',
        })
        add_on_blob = {'adds': [{'at_r': 1.0, 'px': 11.05, 'qty': 100, 'ts': 'x'}],
                       'add_qty': 100, 'add_avg_px': 10.525, 'add_be_leg': None}
        existing = db.get_trade_by_id(trade_id)
        pdata = json.loads(existing['pattern_data'])
        pdata['add_on'] = add_on_blob
        db.update_trade(trade_id, {'shares': 200, 'pattern_data': json.dumps(pdata)})

        row = db.get_trade_by_id(trade_id)
        assert row['shares'] == 200
        got = json.loads(row['pattern_data'])
        assert got['pool_id'] == 'production'  # earlier key preserved, never clobbered
        assert got['add_on'] == add_on_blob
