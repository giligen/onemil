"""Bull flag fill recovery from Alpaca order history.

GLXG 2026-06-11 case: bull flag's `_sync_closed_positions` only checked
bracket-LEG fills when reconciling a "DB says open / Alpaca says flat"
trade. When the actual exit was StopMonitor's market close (after both
bracket legs got cancelled), the bracket-leg lookup returned nothing
and the row was finalized as `exit_reason='unknown_exit'` with
`pnl=$0` placeholder — losing the real P&L silently.

`_recover_exit_from_order_history` ports MACD wave's working pattern
(macd_wave_engine.py:339-381) to bull flag: query the symbol's recent
closed orders, find the first filled sell, classify by order_class or
price relative to planned stop.

These tests pin the helper's contract + the integration path that uses
it before falling back to `unknown_exit`.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _mk_order(side='sell', status='filled', filled_avg_price=10.0,
              order_class=None, order_id='ord-1'):
    """Minimal fake Alpaca SDK Order object — attribute access only."""
    o = MagicMock()
    o.id = order_id
    side_m = MagicMock(); side_m.value = side
    status_m = MagicMock(); status_m.value = status
    o.side = side_m
    o.status = status_m
    o.filled_avg_price = filled_avg_price
    if order_class is None:
        o.order_class = None
    else:
        oc_m = MagicMock(); oc_m.value = order_class
        o.order_class = oc_m
    return o


def _mk_engine():
    """Construct a TradingEngine instance without running __init__."""
    from trading.trading_engine import TradingEngine
    eng = TradingEngine.__new__(TradingEngine)
    eng.alpaca = MagicMock()
    eng.alpaca.trading_client = MagicMock()
    return eng


# =========================================================================
# Helper unit tests
# =========================================================================

class TestRecoverExitHappyPath:

    def test_finds_filled_sell_returns_actual_price(self):
        eng = _mk_engine()
        eng.alpaca.trading_client.get_orders = MagicMock(return_value=[
            _mk_order(side='sell', status='filled', filled_avg_price=9.75,
                      order_class='simple'),
        ])
        price, reason = eng._recover_exit_from_order_history(
            'GLXG', fill_price=10.0, planned_stop=9.50,
        )
        assert price == 9.75
        assert reason is not None  # discriminated, not None

    def test_picks_first_filled_sell_skipping_cancelled(self):
        eng = _mk_engine()
        eng.alpaca.trading_client.get_orders = MagicMock(return_value=[
            _mk_order(side='sell', status='canceled', filled_avg_price=None),
            _mk_order(side='sell', status='canceled', filled_avg_price=None),
            _mk_order(side='sell', status='filled', filled_avg_price=11.20),
            _mk_order(side='buy',  status='filled', filled_avg_price=10.0),
        ])
        price, _ = eng._recover_exit_from_order_history(
            'X', fill_price=10.0, planned_stop=9.50,
        )
        assert price == 11.20


class TestRecoverExitClassification:

    def test_bracket_order_class_yields_bracket_sl_tp(self):
        eng = _mk_engine()
        eng.alpaca.trading_client.get_orders = MagicMock(return_value=[
            _mk_order(filled_avg_price=9.50, order_class='bracket'),
        ])
        _, reason = eng._recover_exit_from_order_history(
            'X', fill_price=10.0, planned_stop=9.50,
        )
        assert reason == 'bracket_sl_tp'

    def test_oto_oco_order_class_also_bracket_sl_tp(self):
        for oc in ('oto', 'oco'):
            eng = _mk_engine()
            eng.alpaca.trading_client.get_orders = MagicMock(return_value=[
                _mk_order(filled_avg_price=9.50, order_class=oc),
            ])
            _, reason = eng._recover_exit_from_order_history(
                'X', fill_price=10.0, planned_stop=9.50,
            )
            assert reason == 'bracket_sl_tp'

    def test_solo_at_planned_stop_yields_stop_loss(self):
        eng = _mk_engine()
        eng.alpaca.trading_client.get_orders = MagicMock(return_value=[
            _mk_order(filled_avg_price=9.50, order_class='simple'),
        ])
        _, reason = eng._recover_exit_from_order_history(
            'X', fill_price=10.0, planned_stop=9.50,
        )
        assert reason == 'stop_loss'

    def test_solo_above_entry_yields_trail_stop(self):
        """GLXG-shape case: filled at $3.55, entry $3.41, planned stop $3.26."""
        eng = _mk_engine()
        eng.alpaca.trading_client.get_orders = MagicMock(return_value=[
            _mk_order(filled_avg_price=3.55, order_class='simple'),
        ])
        _, reason = eng._recover_exit_from_order_history(
            'GLXG', fill_price=3.41, planned_stop=3.26,
        )
        assert reason == 'trail_stop'

    def test_solo_between_stop_and_entry_yields_market_fallback(self):
        eng = _mk_engine()
        eng.alpaca.trading_client.get_orders = MagicMock(return_value=[
            _mk_order(filled_avg_price=9.80, order_class='simple'),
        ])
        _, reason = eng._recover_exit_from_order_history(
            'X', fill_price=10.0, planned_stop=9.50,
        )
        assert reason == 'stop_loss_market_fallback'

    def test_no_planned_stop_yields_stopmonitor_exit_for_solo(self):
        eng = _mk_engine()
        eng.alpaca.trading_client.get_orders = MagicMock(return_value=[
            _mk_order(filled_avg_price=9.80, order_class='simple'),
        ])
        _, reason = eng._recover_exit_from_order_history(
            'X', fill_price=10.0, planned_stop=None,
        )
        assert reason == 'stopmonitor_exit'

    def test_within_half_pct_tolerance_of_stop_counts_as_stop_loss(self):
        eng = _mk_engine()
        eng.alpaca.trading_client.get_orders = MagicMock(return_value=[
            # Filled at $9.54 — within 0.5% of $9.50 stop. Should classify
            # as stop_loss, not market_fallback.
            _mk_order(filled_avg_price=9.54, order_class='simple'),
        ])
        _, reason = eng._recover_exit_from_order_history(
            'X', fill_price=10.0, planned_stop=9.50,
        )
        assert reason == 'stop_loss'


class TestRecoverExitDegenerate:

    def test_no_orders_returns_none_none(self):
        eng = _mk_engine()
        eng.alpaca.trading_client.get_orders = MagicMock(return_value=[])
        price, reason = eng._recover_exit_from_order_history(
            'X', fill_price=10.0, planned_stop=9.50,
        )
        assert price is None and reason is None

    def test_only_cancelled_sells_returns_none(self):
        eng = _mk_engine()
        eng.alpaca.trading_client.get_orders = MagicMock(return_value=[
            _mk_order(side='sell', status='canceled', filled_avg_price=None),
            _mk_order(side='sell', status='canceled', filled_avg_price=None),
        ])
        price, _ = eng._recover_exit_from_order_history(
            'X', fill_price=10.0, planned_stop=9.50,
        )
        assert price is None

    def test_only_buy_orders_returns_none(self):
        eng = _mk_engine()
        eng.alpaca.trading_client.get_orders = MagicMock(return_value=[
            _mk_order(side='buy', status='filled', filled_avg_price=10.0),
        ])
        price, _ = eng._recover_exit_from_order_history(
            'X', fill_price=10.0, planned_stop=9.50,
        )
        assert price is None

    def test_alpaca_exception_returns_none_no_raise(self):
        eng = _mk_engine()
        eng.alpaca.trading_client.get_orders = MagicMock(
            side_effect=RuntimeError("alpaca 500")
        )
        price, reason = eng._recover_exit_from_order_history(
            'X', fill_price=10.0, planned_stop=9.50,
        )
        assert price is None and reason is None

    def test_filled_avg_price_zero_skipped(self):
        eng = _mk_engine()
        eng.alpaca.trading_client.get_orders = MagicMock(return_value=[
            _mk_order(side='sell', status='filled', filled_avg_price=0),
            _mk_order(side='sell', status='filled', filled_avg_price=9.75),
        ])
        price, _ = eng._recover_exit_from_order_history(
            'X', fill_price=10.0, planned_stop=9.50,
        )
        assert price == 9.75


# =========================================================================
# Integration with _sync_closed_positions (source inspection)
# =========================================================================

class TestSyncIntegration:
    """The else-branch of _sync_closed_positions must call the recovery
    helper before falling back to unknown_exit. Source inspection rather
    than full-flow integration because _sync_closed_positions touches
    many other systems (StopMonitor, PositionManager, etc.) that aren't
    in scope for this fix."""

    def test_sync_calls_recovery_helper_before_unknown_exit(self):
        from pathlib import Path
        src = (Path(__file__).parent.parent
                / "trading/trading_engine.py").read_text()
        # The recovery call must appear BEFORE the unknown_exit write.
        recovery_idx = src.find("_recover_exit_from_order_history(")
        # First match is the def, second+ are the call sites
        assert recovery_idx > 0, "helper definition missing"
        # Find call site (the one with `self.`)
        call_idx = src.find("self._recover_exit_from_order_history(")
        assert call_idx > 0, "no call site for recovery helper"
        # 2026-07-03 merge note: exit_reason literals were centralized into
        # the ExitReason enum (trading/exit_reasons.py); the fallback write
        # is now `'exit_reason': ExitReason.UNKNOWN_EXIT.value`.
        unknown_idx = src.find("'exit_reason': ExitReason.UNKNOWN_EXIT.value")
        assert unknown_idx > 0
        # The call must precede the unknown_exit write
        assert call_idx < unknown_idx, (
            "recovery helper must run BEFORE the unknown_exit fallback "
            "— otherwise we lose real P&L (GLXG 2026-06-11)"
        )

    def test_unknown_exit_message_mentions_recovery_also_failed(self):
        """Pin the operator-facing message — when we DO fall back to
        unknown_exit, the log line must say recovery also failed."""
        from pathlib import Path
        src = (Path(__file__).parent.parent
                / "trading/trading_engine.py").read_text()
        assert "order-history recovery also failed" in src
