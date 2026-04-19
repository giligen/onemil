"""Tests for ORB entry + exit slippage telemetry.

ORB must persist the same execution-quality fields bull flag and MACD wave do so
Phase 1 paper validation can measure real slippage vs BT's 30bps assumption.

Covered fields:
  Entry (written in _confirm_fill):
    order_submitted_at, order_filled_at, submit_to_fill_ms,
    bar_close_price, drift_bar_to_fill_bps, drift_ask_to_fill_bps,
    entry_quote_bid/ask/bid_size/ask_size/spread/ofi,
    entry_fill_quote_bid/ask

  Exit (written in _handle_exit_event):
    exit_trigger_price, exit_quote_bid/ask/bid_size/ask_size,
    exit_limit_price, exit_pricing_method, exit_slippage
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine, OpenPosition
from trading.stop_monitor import StopMonitor


@pytest.fixture
def orb_cfg():
    yaml_path = Path(__file__).parent.parent / 'orb.yaml'
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    return cfg


@pytest.fixture
def mock_alpaca():
    c = MagicMock(spec=AlpacaClient)
    c.get_open_positions.return_value = []
    c.get_account_info.return_value = {'buying_power': 100_000.0}
    c.get_latest_quote.return_value = {
        'bid_price': 9.95, 'ask_price': 10.00,
        'bid_size': 500, 'ask_size': 600,
    }
    c.submit_stop_bracket_order.return_value = {'id': 'order-1', 'status': 'accepted'}
    return c


@pytest.fixture
def mock_db():
    db = MagicMock(spec=Database)
    db.save_trade.return_value = 100
    db.get_open_trades.return_value = []
    db.update_trade.return_value = True
    return db


@pytest.fixture
def mock_stop_monitor():
    sm = MagicMock(spec=StopMonitor)
    sm.polling_mode = False
    sm.drain_exit_events.return_value = []
    sm.get_quote_watch_snapshot.return_value = {
        'submit_bid': 9.95, 'submit_ask': 10.00,
        'submit_bid_size': 500, 'submit_ask_size': 600,
        'latest_bid': 9.98, 'latest_ask': 10.04,
        'latest_bid_size': 300, 'latest_ask_size': 200,
        'ofi_cumulative': 1250.0,
        'submitted_at': 1700000000.0,
    }
    return sm


@pytest.fixture
def engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor):
    return ORBEngine(
        alpaca_client=mock_alpaca, db=mock_db,
        stop_monitor=mock_stop_monitor, config=orb_cfg,
    )


def _merged_updates(mock_db) -> dict:
    """Merge every update_trade(trade_id, dict) call into a single dict."""
    merged: dict = {}
    for call in mock_db.update_trade.call_args_list:
        args, kwargs = call
        updates = args[1] if len(args) >= 2 else kwargs.get('updates', {})
        merged.update(updates)
    return merged


# =========================================================================
# Entry-side telemetry (_confirm_fill)
# =========================================================================

class TestEntryTelemetry:
    def _make_pos(self, submitted_at, range_high=10.00, entry_quote_ask=10.00):
        return OpenPosition(
            symbol='ABCD',
            entry_price=range_high,         # pre-fill placeholder from planner
            stop_price=9.00,
            shares=100,
            trade_id=42,
            order_id='order-1',
            entry_time=submitted_at,
            range_high=range_high,
            range_low=9.00,
            lock_arm_at_r=1.5,
            lock_stop_r=1.0,
            composite_score=0.45,
            quintile='Q4',
            bar_close_price=range_high,
            order_submitted_at=submitted_at,
            entry_quote_ask=entry_quote_ask,
        )

    def test_confirm_fill_writes_submit_and_filled_timestamps(self, engine, mock_db):
        # Submit ~500ms before "now" so fill_at - order_submitted_at is a
        # small positive latency (real order lifecycle).
        submitted = datetime.now(timezone.utc) - timedelta(milliseconds=500)
        pos = self._make_pos(submitted)
        engine.open_positions['ABCD'] = pos
        # 600ms latency between submit and fill
        order_status = {'filled_avg_price': 10.03, 'filled_qty': 100, 'status': 'filled'}
        # Freeze "now" by mocking datetime.now used in _confirm_fill — simplest:
        # advance real clock but just assert ordering + presence.
        engine._confirm_fill(pos, order_status)

        merged = _merged_updates(mock_db)
        assert merged['order_status'] == 'filled'
        assert merged['fill_price'] == pytest.approx(10.03)
        assert 'order_submitted_at' in merged
        assert 'order_filled_at' in merged
        assert 'filled_at' in merged
        assert merged['order_submitted_at'] == submitted
        assert isinstance(merged['submit_to_fill_ms'], int)
        assert merged['submit_to_fill_ms'] >= 0

    def test_confirm_fill_drift_bar_to_fill_bps(self, engine, mock_db):
        """Fill at $10.03 with bar_close_price (range_high) $10.00 → +30bps."""
        pos = self._make_pos(datetime.now(timezone.utc))
        engine.open_positions['ABCD'] = pos
        engine._confirm_fill(pos, {'filled_avg_price': 10.03, 'filled_qty': 100})

        merged = _merged_updates(mock_db)
        assert merged['bar_close_price'] == pytest.approx(10.00)
        assert merged['drift_bar_to_fill_bps'] == pytest.approx(30.0, abs=0.01)

    def test_confirm_fill_drift_ask_to_fill_bps(self, engine, mock_db):
        """Ask at submit was $10.00; fill $10.02 → +20bps drift."""
        pos = self._make_pos(datetime.now(timezone.utc), entry_quote_ask=10.00)
        engine.open_positions['ABCD'] = pos
        engine._confirm_fill(pos, {'filled_avg_price': 10.02, 'filled_qty': 100})

        merged = _merged_updates(mock_db)
        assert merged['drift_ask_to_fill_bps'] == pytest.approx(20.0, abs=0.01)

    def test_confirm_fill_persists_entry_microstructure(self, engine, mock_db):
        """Quote-watch snapshot fields land in DB update."""
        pos = self._make_pos(datetime.now(timezone.utc))
        engine.open_positions['ABCD'] = pos
        engine._confirm_fill(pos, {'filled_avg_price': 10.03, 'filled_qty': 100})

        merged = _merged_updates(mock_db)
        assert merged['entry_quote_bid'] == pytest.approx(9.95)
        assert merged['entry_quote_ask'] == pytest.approx(10.00)
        assert merged['entry_quote_bid_size'] == 500
        assert merged['entry_quote_ask_size'] == 600
        assert merged['entry_quote_spread'] == pytest.approx(0.05, abs=1e-9)
        assert merged['entry_quote_ofi'] == pytest.approx(1250.0)
        assert merged['entry_fill_quote_bid'] == pytest.approx(9.98)
        assert merged['entry_fill_quote_ask'] == pytest.approx(10.04)

    def test_confirm_fill_removes_quote_watch(self, engine, mock_stop_monitor):
        pos = self._make_pos(datetime.now(timezone.utc))
        engine.open_positions['ABCD'] = pos
        engine._confirm_fill(pos, {'filled_avg_price': 10.03, 'filled_qty': 100})
        mock_stop_monitor.remove_quote_watch.assert_called_once_with('ABCD')

    def test_confirm_fill_no_drift_when_bar_close_missing(self, engine, mock_db):
        """Old-style position without bar_close_price → no drift fields (no crash)."""
        pos = self._make_pos(datetime.now(timezone.utc))
        pos.bar_close_price = None
        pos.entry_quote_ask = None
        engine.open_positions['ABCD'] = pos
        engine._confirm_fill(pos, {'filled_avg_price': 10.03, 'filled_qty': 100})

        merged = _merged_updates(mock_db)
        assert 'drift_bar_to_fill_bps' not in merged
        assert 'drift_ask_to_fill_bps' not in merged
        # Core fields still present
        assert merged['order_status'] == 'filled'
        assert merged['fill_price'] == pytest.approx(10.03)


# =========================================================================
# Exit-side telemetry (_handle_exit_event)
# =========================================================================

@dataclass
class _FakeExitEvent:
    symbol: str
    exit_price: float
    exit_reason: str
    exit_trigger_price: float = 0.0
    exit_quote_bid: float = 0.0
    exit_quote_ask: float = 0.0
    exit_quote_bid_size: int = 0
    exit_quote_ask_size: int = 0
    exit_limit_price: float = 0.0
    pricing_method: str = 'fixed_offset'
    strategy: str = 'orb'


class TestExitTelemetry:
    def _seed_filled_position(self, engine):
        pos = OpenPosition(
            symbol='ABCD',
            entry_price=10.00, stop_price=9.00, shares=100,
            trade_id=42, order_id='', entry_time=datetime.now(timezone.utc),
            range_high=10.00, range_low=9.00,
            lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.45, quintile='Q4',
        )
        engine.open_positions['ABCD'] = pos
        return pos

    def test_exit_event_fields_persisted(self, engine, mock_db):
        self._seed_filled_position(engine)
        ev = _FakeExitEvent(
            symbol='ABCD', exit_price=9.00, exit_reason='stop_loss',
            exit_trigger_price=8.99,
            exit_quote_bid=8.98, exit_quote_ask=9.01,
            exit_quote_bid_size=400, exit_quote_ask_size=300,
            exit_limit_price=9.02,
            pricing_method='quote_medium',
        )
        engine._handle_exit_event(ev)

        merged = _merged_updates(mock_db)
        assert merged['exit_price'] == pytest.approx(9.00)
        assert merged['exit_reason'] == 'stop_loss'
        assert merged['exit_trigger_price'] == pytest.approx(8.99)
        assert merged['exit_quote_bid'] == pytest.approx(8.98)
        assert merged['exit_quote_ask'] == pytest.approx(9.01)
        assert merged['exit_quote_bid_size'] == 400
        assert merged['exit_quote_ask_size'] == 300
        assert merged['exit_limit_price'] == pytest.approx(9.02)
        assert merged['exit_pricing_method'] == 'quote_medium'

    def test_exit_slippage_computed(self, engine, mock_db):
        """exit_slippage = exit_limit_price - actual exit_price (bull flag convention)."""
        self._seed_filled_position(engine)
        # Sent limit $9.02, actually filled at $9.00 → we lost 2 cents to slip
        ev = _FakeExitEvent(
            symbol='ABCD', exit_price=9.00, exit_reason='stop_loss',
            exit_limit_price=9.02,
        )
        engine._handle_exit_event(ev)

        merged = _merged_updates(mock_db)
        assert merged['exit_slippage'] == pytest.approx(0.02)

    def test_exit_slippage_none_when_limit_missing(self, engine, mock_db):
        """If StopMonitor event has no exit_limit_price (fixed_offset / mock), exit_slippage=None."""
        self._seed_filled_position(engine)
        ev = _FakeExitEvent(symbol='ABCD', exit_price=9.00, exit_reason='stop_loss')
        # exit_limit_price default is 0.0
        engine._handle_exit_event(ev)

        merged = _merged_updates(mock_db)
        assert merged['exit_slippage'] is None
        assert merged['exit_limit_price'] is None

    def test_exit_handles_loose_mock_event(self, engine, mock_db):
        """Regression: handler must tolerate events where attrs are MagicMocks (unspec'd mocks)."""
        self._seed_filled_position(engine)
        ev = MagicMock()
        ev.symbol = 'ABCD'
        ev.exit_price = 9.00
        ev.exit_reason = 'stop_loss'
        # All other attrs will be MagicMock (non-numeric) — must coerce to None safely
        engine._handle_exit_event(ev)

        merged = _merged_updates(mock_db)
        assert merged['exit_slippage'] is None
        assert merged['exit_limit_price'] is None
        assert merged['exit_quote_bid'] is None
        assert merged['exit_quote_ask'] is None
        assert merged['exit_pricing_method'] is None


# =========================================================================
# Submit path captures reference fields on OpenPosition
# =========================================================================

class TestSubmitCapturesReference:
    def test_submit_stores_bar_close_and_submitted_at_on_position(self, engine, mock_alpaca):
        """After _submit_entry, OpenPosition carries bar_close_price = range_high,
        order_submitted_at set, entry_quote_ask from quote, and add_quote_watch
        was started."""
        from trading.orb_planner import OrbTradePlan

        plan = OrbTradePlan(
            symbol='ABCD',
            range_high=10.00, range_low=9.00, range_size=1.00,
            entry_price=10.03, stop_price=9.00,
            shares=100, position_dollars=1003.0,
            lock_arm_at_r=1.5, lock_stop_r=1.0,
            risk_per_share=1.03, total_risk=103.0,
            composite_score=0.45, quintile='Q4', adaptive_mult=0.95,
            range_open=10.00,
        )
        engine._save_pending_trade = MagicMock(return_value=42)

        order_id = engine._submit_entry(plan)
        assert order_id == 'order-1'

        pos = engine.open_positions['ABCD']
        assert pos.bar_close_price == pytest.approx(10.00)  # = range_high
        assert pos.order_submitted_at is not None
        assert pos.entry_quote_ask == pytest.approx(10.00)  # from mock_alpaca.get_latest_quote

        # add_quote_watch was called on the StopMonitor
        engine.stop_monitor.add_quote_watch.assert_called_once()
        call_kwargs = engine.stop_monitor.add_quote_watch.call_args.kwargs
        assert call_kwargs['submit_bid'] == pytest.approx(9.95)
        assert call_kwargs['submit_ask'] == pytest.approx(10.00)
