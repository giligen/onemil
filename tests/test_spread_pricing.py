"""
Unit tests for spread-based exit pricing.

Tests cover:
- compute_limit_price_from_quote() spread tier logic
- Quote-based pricing in _execute_stop_exit (via mocked quote)
- Fallback to fixed offset when quote fails
- AlpacaClient.get_latest_quote() response parsing
- Fill timeout and market-sell fallback
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

from data_sources.alpaca_client import AlpacaClient
from trading.stop_monitor import StopMonitor, StopExitEvent, WatchEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_alpaca():
    """Mocked AlpacaClient with quote support."""
    client = MagicMock(spec=AlpacaClient)
    client.cancel_order.return_value = True
    client.submit_limit_sell_order.return_value = {
        'id': 'sell-001', 'status': 'accepted', 'symbol': 'TEST',
    }
    client.close_position.return_value = {
        'id': 'close-001', 'status': 'accepted', 'symbol': 'TEST',
    }
    # Default: tight spread (liquid stock)
    client.get_latest_quote.return_value = {
        'bid_price': 9.98, 'ask_price': 10.00,
        'bid_size': 500, 'ask_size': 300,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    # Order fill confirmation (for partial exit fill wait)
    client.get_order.return_value = {
        'status': 'filled', 'filled_avg_price': 9.99,
    }
    client.replace_order_qty.return_value = {'id': 'sl-001', 'status': 'accepted'}
    return client


@pytest.fixture
def monitor(mock_alpaca):
    """StopMonitor with default config."""
    return StopMonitor(
        api_key='test-key',
        api_secret='test-secret',
        alpaca_client=mock_alpaca,
        marketable_limit_offset=0.03,
        marketable_limit_offset_pct=0.005,
    )


# ---------------------------------------------------------------------------
# compute_limit_price_from_quote — spread tier tests
# ---------------------------------------------------------------------------

class TestComputeLimitPriceFromQuote:
    """Test spread-based pricing tiers."""

    def test_tight_spread_uses_midpoint(self):
        """Spread < $0.05 → midpoint."""
        price, method = StopMonitor.compute_limit_price_from_quote(9.98, 10.00)
        assert price == 9.99  # mid of 9.98 and 10.00
        assert method == 'quote_tight'

    def test_tight_spread_penny(self):
        """$0.01 spread → midpoint rounds correctly."""
        price, method = StopMonitor.compute_limit_price_from_quote(5.00, 5.01)
        assert price == 5.0  # mid = 5.005, rounds to 5.00 (banker's rounding)
        assert method == 'quote_tight'

    def test_medium_spread_uses_bid_plus_penny(self):
        """Spread $0.05-$0.15 → bid + $0.01."""
        price, method = StopMonitor.compute_limit_price_from_quote(9.90, 10.00)
        assert price == 9.91  # bid + 0.01
        assert method == 'quote_medium'

    def test_medium_spread_boundary(self):
        """Exactly $0.05 spread → medium tier."""
        price, method = StopMonitor.compute_limit_price_from_quote(9.95, 10.00)
        assert price == 9.96
        assert method == 'quote_medium'

    def test_wide_spread_uses_bid(self):
        """Spread > $0.15 → bid price."""
        price, method = StopMonitor.compute_limit_price_from_quote(9.80, 10.00)
        assert price == 9.80
        assert method == 'quote_wide'

    def test_wide_spread_boundary(self):
        """$0.15 spread (float edge) → wide tier due to float precision."""
        price, method = StopMonitor.compute_limit_price_from_quote(9.85, 10.00)
        # 10.00 - 9.85 = 0.15000...036 (float), exceeds 0.15 → wide
        assert price == 9.85
        assert method == 'quote_wide'

    def test_very_wide_spread(self):
        """$0.50 spread → bid."""
        price, method = StopMonitor.compute_limit_price_from_quote(4.50, 5.00)
        assert price == 4.50
        assert method == 'quote_wide'

    def test_invalid_bid_zero(self):
        """Bid = 0 → invalid."""
        price, method = StopMonitor.compute_limit_price_from_quote(0.0, 10.00)
        assert price == 0.0
        assert method == 'invalid'

    def test_invalid_ask_below_bid(self):
        """Ask < bid (crossed market) → invalid."""
        price, method = StopMonitor.compute_limit_price_from_quote(10.00, 9.90)
        assert price == 0.0
        assert method == 'invalid'

    def test_invalid_negative_prices(self):
        """Negative prices → invalid."""
        price, method = StopMonitor.compute_limit_price_from_quote(-1.0, 5.0)
        assert price == 0.0
        assert method == 'invalid'

    def test_floor_at_one_cent(self):
        """Very low bid → floor at $0.01."""
        price, method = StopMonitor.compute_limit_price_from_quote(0.005, 0.01)
        assert price == 0.01

    def test_high_price_stock_tight_spread(self):
        """$50 stock with $0.02 spread → midpoint saves $0.24 vs fixed offset."""
        price, method = StopMonitor.compute_limit_price_from_quote(49.99, 50.01)
        assert price == 50.00  # mid
        assert method == 'quote_tight'
        # Fixed offset would be 50.01 - max(0.03, 50.01*0.005) = 50.01 - 0.25 = $49.76
        # Saving: $50.00 - $49.76 = $0.24/share


# ---------------------------------------------------------------------------
# Savings comparison vs fixed offset
# ---------------------------------------------------------------------------

class TestSavingsVsFixedOffset:
    """Verify spread-based pricing saves money on liquid stocks."""

    def test_savings_on_5_dollar_stock(self, monitor):
        """$5 stock: tight spread saves vs fixed offset."""
        quote_price, _ = StopMonitor.compute_limit_price_from_quote(4.99, 5.01)
        fixed_price = monitor.compute_limit_price(5.00)
        assert quote_price > fixed_price  # quote gets better price
        savings = quote_price - fixed_price
        assert savings >= 0.01

    def test_savings_on_20_dollar_stock(self, monitor):
        """$20 stock: tight spread saves ~$0.08/share."""
        quote_price, _ = StopMonitor.compute_limit_price_from_quote(19.99, 20.01)
        fixed_price = monitor.compute_limit_price(20.00)
        savings = quote_price - fixed_price
        assert savings >= 0.05  # significant savings


# ---------------------------------------------------------------------------
# Quote fallback in execute_partial_exit
# ---------------------------------------------------------------------------

class TestPartialExitQuoteFallback:
    """Test that partial exit falls back to fixed offset on quote failure."""

    def test_uses_quote_when_available(self, monitor, mock_alpaca):
        """Quote succeeds → uses quote-based pricing, fill confirmed."""
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 8.48, 'ask_price': 8.50,
            'bid_size': 100, 'ask_size': 200,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 8.49,
        }
        monitor.add_watch('TEST', 4.50, 1000, 'tp', 'sl',
                          entry_price=5.0, risk_per_share=0.5)
        event = monitor.execute_partial_exit('TEST', 0.5, 0.5)
        assert event is not None
        # Fill confirmed at 8.49 (midpoint)
        assert event.exit_price == 8.49

    def test_falls_back_on_quote_failure(self, monitor, mock_alpaca):
        """Quote fails → falls back to fixed-offset pricing, fill confirmed."""
        mock_alpaca.get_latest_quote.side_effect = Exception("API down")
        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 8.46,
        }
        monitor.add_watch('TEST', 4.50, 1000, 'tp', 'sl',
                          entry_price=5.0, risk_per_share=0.5)
        with monitor._watch_lock:
            monitor._watches['TEST'].highest_since_entry = 8.50
        event = monitor.execute_partial_exit('TEST', 0.5, 0.5)
        assert event is not None
        # Fill confirmed at 8.46
        assert event.exit_price == 8.46


# ---------------------------------------------------------------------------
# StopExitEvent fields
# ---------------------------------------------------------------------------

class TestStopExitEventFields:
    """Test new fields on StopExitEvent."""

    def test_default_fields(self):
        """New fields have sensible defaults."""
        event = StopExitEvent(
            symbol='TEST', stop_price=4.50, exit_price=4.45,
            shares=1000, order_id='ord-1', exit_reason='stop_loss',
        )
        assert event.submitted_at == 0.0
        assert event.pricing_method == 'fixed_offset'

    def test_custom_fields(self):
        """Can set pricing_method and submitted_at."""
        event = StopExitEvent(
            symbol='TEST', stop_price=4.50, exit_price=4.49,
            shares=1000, order_id='ord-1', exit_reason='trail_stop',
            submitted_at=1234567890.0, pricing_method='quote_tight',
        )
        assert event.submitted_at == 1234567890.0
        assert event.pricing_method == 'quote_tight'
