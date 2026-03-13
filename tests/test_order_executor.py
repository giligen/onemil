"""
Unit tests for OrderExecutor — bracket order submission via Alpaca.

Uses mocked AlpacaClient and real database (temp file).
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from data_sources.alpaca_client import AlpacaClient, AlpacaAPIError
from persistence.database import Database
from trading.pattern_detector import BullFlagPattern
from trading.trade_planner import TradePlan
from trading.order_executor import OrderExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pattern():
    """Create a valid BullFlagPattern."""
    return BullFlagPattern(
        symbol="TEST",
        pole_start_idx=0, pole_end_idx=2,
        flag_start_idx=3, flag_end_idx=4,
        pole_low=4.00, pole_high=4.50,
        pole_height=0.50, pole_gain_pct=12.5,
        flag_low=4.30, flag_high=4.40,
        retracement_pct=40.0, pullback_candle_count=2,
        avg_pole_volume=180000, avg_flag_volume=40000,
        breakout_level=4.40,
    )


def _make_plan():
    """Create a valid TradePlan."""
    return TradePlan(
        symbol="TEST",
        entry_price=4.40,
        stop_loss_price=4.29,
        take_profit_price=4.90,
        risk_per_share=0.11,
        reward_per_share=0.50,
        risk_reward_ratio=4.5,
        shares=113,
        total_risk=12.43,
        pattern=_make_pattern(),
    )


@pytest.fixture
def db(tmp_path):
    """Real database with temp file."""
    database = Database(db_path=str(tmp_path / "test.db"))
    yield database
    database.close()


@pytest.fixture
def mock_alpaca():
    """Mocked AlpacaClient."""
    return MagicMock(spec=AlpacaClient)


@pytest.fixture
def executor(mock_alpaca, db):
    """OrderExecutor with mocked Alpaca and real DB."""
    return OrderExecutor(alpaca_client=mock_alpaca, db=db)


# ===========================================================================
# TESTS
# ===========================================================================

class TestSubmitBracketOrder:
    """Tests for bracket order submission."""

    def test_submits_order_and_saves_trade(self, executor, mock_alpaca, db):
        """Successful order submission saves trade to DB."""
        mock_alpaca.submit_bracket_order.return_value = {
            'id': 'order-123',
            'status': 'accepted',
            'symbol': 'TEST',
        }

        plan = _make_plan()
        result = executor.submit_bracket_order(plan)

        assert result is not None
        assert result['order_id'] == 'order-123'
        assert result['status'] == 'accepted'
        assert result['symbol'] == 'TEST'

        # Verify trade saved to DB
        from datetime import date
        trades = db.get_trades_by_date(date.today().isoformat())
        assert len(trades) == 1
        assert trades[0]['symbol'] == 'TEST'
        assert trades[0]['order_id'] == 'order-123'
        assert trades[0]['entry_price'] == 4.40
        assert trades[0]['shares'] == 113

    def test_calls_alpaca_with_correct_params(self, executor, mock_alpaca):
        """Verifies correct parameters passed to Alpaca."""
        mock_alpaca.submit_bracket_order.return_value = {
            'id': 'order-456', 'status': 'accepted',
        }

        plan = _make_plan()
        executor.submit_bracket_order(plan)

        mock_alpaca.submit_bracket_order.assert_called_once_with(
            symbol='TEST',
            qty=113,
            side='buy',
            limit_price=4.40,
            tp_price=4.90,
            sl_price=4.29,
        )

    def test_handles_api_failure(self, executor, mock_alpaca):
        """Returns None when Alpaca API call fails."""
        mock_alpaca.submit_bracket_order.side_effect = AlpacaAPIError("API down")

        plan = _make_plan()
        result = executor.submit_bracket_order(plan)
        assert result is None

    def test_handles_none_return(self, executor, mock_alpaca):
        """Returns None when Alpaca returns None."""
        mock_alpaca.submit_bracket_order.return_value = None

        plan = _make_plan()
        result = executor.submit_bracket_order(plan)
        assert result is None

    def test_saves_pattern_data_as_json(self, executor, mock_alpaca, db):
        """Pattern data is saved as JSON blob in the trade record."""
        import json
        mock_alpaca.submit_bracket_order.return_value = {
            'id': 'order-789', 'status': 'accepted',
        }

        plan = _make_plan()
        executor.submit_bracket_order(plan)

        from datetime import date
        trades = db.get_trades_by_date(date.today().isoformat())
        pattern_data = json.loads(trades[0]['pattern_data'])
        assert pattern_data['pole_height'] == 0.50
        assert pattern_data['breakout_level'] == 4.40
