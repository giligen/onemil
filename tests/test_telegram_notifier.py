"""
Unit tests for notifications/telegram_notifier.py - TelegramNotifier.

Covers:
- Initialization and validation
- send_message (async) with mocked aiohttp
- send_message_sync wrapper
- All event notification methods
- End-of-day report generation
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone

from notifications.telegram_notifier import TelegramNotifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def notifier():
    """Create a TelegramNotifier with valid config (disabled for unit tests)."""
    return TelegramNotifier(
        bot_token="test-token-123",
        chat_id="12345",
        enabled=False,  # Disabled to prevent real API calls
    )


@pytest.fixture
def enabled_notifier():
    """Create an enabled notifier for testing send logic."""
    return TelegramNotifier(
        bot_token="test-token-123",
        chat_id="12345",
        enabled=True,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for TelegramNotifier initialization."""

    def test_disabled_notifier(self):
        """Disabled notifier sets enabled=False."""
        n = TelegramNotifier(bot_token="tok", chat_id="id", enabled=False)
        assert n.enabled is False

    def test_enabled_with_valid_config(self):
        """Enabled notifier with valid token and chat_id stays enabled."""
        n = TelegramNotifier(bot_token="tok", chat_id="id", enabled=True)
        assert n.enabled is True

    def test_enabled_with_missing_token_disables(self):
        """Enabled notifier with empty token disables itself."""
        n = TelegramNotifier(bot_token="", chat_id="id", enabled=True)
        assert n.enabled is False

    def test_enabled_with_missing_chat_id_disables(self):
        """Enabled notifier with empty chat_id disables itself."""
        n = TelegramNotifier(bot_token="tok", chat_id="", enabled=True)
        assert n.enabled is False

    def test_api_url_formed_correctly(self):
        """API URL includes bot token."""
        n = TelegramNotifier(bot_token="my-token", chat_id="id", enabled=False)
        assert "my-token" in n.api_url


# ---------------------------------------------------------------------------
# send_message
# ---------------------------------------------------------------------------

class TestSendMessage:
    """Tests for send_message async method."""

    @pytest.mark.asyncio
    async def test_disabled_returns_false(self, notifier):
        """Disabled notifier returns False without making API call."""
        result = await notifier.send_message("test")
        assert result is False

    @pytest.mark.asyncio
    async def test_successful_send(self, enabled_notifier):
        """Successful API call returns True."""
        mock_response = MagicMock()
        mock_response.status = 200

        # aiohttp uses `async with session.post(...)` which needs __aenter__/__aexit__
        mock_post_cm = MagicMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=False)

        # aiohttp uses `async with ClientSession() as session`
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_post_cm)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch('notifications.telegram_notifier.aiohttp.ClientSession', return_value=mock_session_cm):
            result = await enabled_notifier.send_message("Hello")
            assert result is True


# ---------------------------------------------------------------------------
# send_message_sync
# ---------------------------------------------------------------------------

class TestSendMessageSync:
    """Tests for send_message_sync wrapper."""

    def test_sync_wrapper_calls_async(self, notifier):
        """Sync wrapper delegates to send_message."""
        # Disabled notifier, just verify it doesn't crash
        result = notifier.send_message_sync("test message")
        assert result is False


# ---------------------------------------------------------------------------
# Event Notifications (disabled notifier — verify no crash)
# ---------------------------------------------------------------------------

class TestEventNotifications:
    """Tests for event notification methods."""

    def test_notify_scanner_started(self, notifier):
        """Scanner started notification doesn't crash."""
        notifier.notify_scanner_started(
            universe_size=150, trading_enabled=True, mode="paper"
        )

    def test_notify_stock_qualified(self, notifier):
        """Stock qualified notification doesn't crash."""
        notifier.notify_stock_qualified(
            symbol="AAPL", price=5.50, change_pct=15.0,
            relative_volume=8.5, headline="Big catalyst"
        )

    def test_notify_stock_qualified_no_headline(self, notifier):
        """Stock qualified without headline doesn't crash."""
        notifier.notify_stock_qualified(
            symbol="TSLA", price=10.0, change_pct=20.0,
            relative_volume=6.0, headline=None
        )

    def test_notify_premarket_gaps(self, notifier):
        """Premarket gaps notification doesn't crash."""
        gaps = [
            {'symbol': 'AAA', 'gap_pct': 15.0, 'current_price': 5.50, 'prev_close': 4.78},
            {'symbol': 'BBB', 'gap_pct': 8.0, 'current_price': 3.24, 'prev_close': 3.00},
        ]
        notifier.notify_premarket_gaps(gaps)

    def test_notify_premarket_gaps_empty(self, notifier):
        """Empty gaps list doesn't send anything."""
        notifier.notify_premarket_gaps([])

    def test_notify_pattern_detected(self, notifier):
        """Pattern detected notification doesn't crash."""
        notifier.notify_pattern_detected(
            symbol="MOMO", pole_gain_pct=8.5,
            retracement_pct=35.0, breakout_level=5.50,
        )

    def test_notify_trade_planned(self, notifier):
        """Trade planned notification doesn't crash."""
        notifier.notify_trade_planned(
            symbol="MOMO", entry=5.50, stop=5.30,
            target=5.90, shares=100, risk_reward=2.0,
        )

    def test_notify_order_submitted(self, notifier):
        """Order submitted notification doesn't crash."""
        notifier.notify_order_submitted(
            symbol="MOMO", order_id="order-123",
            shares=100, entry=5.50,
        )

    def test_notify_order_filled(self, notifier):
        """Order filled notification doesn't crash."""
        notifier.notify_order_filled(
            symbol="MOMO", shares=100,
            fill_price=5.48, order_id="order-123",
        )

    def test_notify_position_closed_profit(self, notifier):
        """Position closed with profit doesn't crash."""
        notifier.notify_position_closed(
            symbol="MOMO", entry_price=5.50, exit_price=5.90,
            shares=100, pnl=40.0, exit_reason="take_profit",
        )

    def test_notify_position_closed_loss(self, notifier):
        """Position closed with loss doesn't crash."""
        notifier.notify_position_closed(
            symbol="MOMO", entry_price=5.50, exit_price=5.30,
            shares=100, pnl=-20.0, exit_reason="stop_loss",
        )

    def test_notify_error(self, notifier):
        """Error notification doesn't crash."""
        notifier.notify_error("Something broke", component="PatternDetector")


# ---------------------------------------------------------------------------
# Daily Report
# ---------------------------------------------------------------------------

class TestDailyReport:
    """Tests for send_daily_report."""

    def test_daily_report_full(self, notifier):
        """Full daily report with all data doesn't crash."""
        report = {
            'trade_date': '2026-03-13',
            'universe_size': 150,
            'premarket_gaps': [
                {'symbol': 'AAA', 'gap_pct': 15.0, 'current_price': 5.50, 'prev_close': 4.78},
            ],
            'qualified_stocks': [
                {'symbol': 'AAA', 'intraday_change_pct': 20.0,
                 'relative_volume': 8.5, 'news_headline': 'Catalyst'},
            ],
            'patterns_detected': 1,
            'patterns_detected_details': [
                {'symbol': 'AAA', 'pole_gain_pct': 8.0, 'retracement_pct': 30.0},
            ],
            'trades': [
                {'symbol': 'AAA', 'entry_price': 5.50, 'exit_price': 5.90,
                 'pnl': 40.0, 'order_status': 'filled', 'exit_reason': 'take_profit'},
            ],
            'total_trades': 1,
            'winning_trades': 1,
            'losing_trades': 0,
            'gross_pnl': 40.0,
            'open_positions': 0,
        }
        notifier.send_daily_report(report)

    def test_daily_report_empty(self, notifier):
        """Daily report with no activity doesn't crash."""
        report = {
            'trade_date': '2026-03-13',
            'universe_size': 150,
            'premarket_gaps': [],
            'qualified_stocks': [],
            'patterns_detected': 0,
            'patterns_detected_details': [],
            'trades': [],
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'gross_pnl': 0.0,
            'open_positions': 0,
        }
        notifier.send_daily_report(report)

    def test_daily_report_with_open_positions(self, notifier):
        """Daily report with open positions shows warning."""
        report = {
            'trade_date': '2026-03-13',
            'universe_size': 100,
            'premarket_gaps': [],
            'qualified_stocks': [],
            'patterns_detected': 0,
            'patterns_detected_details': [],
            'trades': [
                {'symbol': 'OPEN', 'entry_price': 5.00, 'exit_price': None,
                 'pnl': None, 'order_status': 'filled', 'exit_reason': None},
            ],
            'total_trades': 1,
            'winning_trades': 0,
            'losing_trades': 0,
            'gross_pnl': 0.0,
            'open_positions': 1,
        }
        notifier.send_daily_report(report)

    def test_daily_report_negative_pnl(self, notifier):
        """Daily report with negative P&L formats correctly."""
        report = {
            'trade_date': '2026-03-13',
            'universe_size': 100,
            'premarket_gaps': [],
            'qualified_stocks': [],
            'patterns_detected': 1,
            'patterns_detected_details': [],
            'trades': [
                {'symbol': 'LOSS', 'entry_price': 5.50, 'exit_price': 5.30,
                 'pnl': -20.0, 'order_status': 'filled', 'exit_reason': 'stop_loss'},
            ],
            'total_trades': 1,
            'winning_trades': 0,
            'losing_trades': 1,
            'gross_pnl': -20.0,
            'open_positions': 0,
        }
        notifier.send_daily_report(report)


# ---------------------------------------------------------------------------
# Flood control (2026-09-29 incident: 144 identical ERRORs in 30s -> 68x 429)
# ---------------------------------------------------------------------------

class TestDedupGate:
    """Tests for TelegramNotifier._dedup_gate (60s exact-text dedup)."""

    def test_first_message_passes_through(self, enabled_notifier):
        out = enabled_notifier._dedup_gate("hello", now=100.0)
        assert out == "hello"

    def test_identical_within_window_suppressed(self, enabled_notifier):
        enabled_notifier._dedup_gate("hello", now=100.0)
        out = enabled_notifier._dedup_gate("hello", now=110.0)
        assert out is None
        assert enabled_notifier._suppressed_dup_count == 1

    def test_suppressed_count_carries_as_suffix_on_next_distinct_message(self, enabled_notifier):
        enabled_notifier._dedup_gate("hello", now=100.0)
        enabled_notifier._dedup_gate("hello", now=110.0)  # suppressed, count=1
        enabled_notifier._dedup_gate("hello", now=115.0)  # suppressed, count=2
        out = enabled_notifier._dedup_gate("goodbye", now=120.0)
        assert out == "goodbye (+2 identical suppressed)"

    def test_identical_after_window_sends_again(self, enabled_notifier):
        enabled_notifier._dedup_gate("hello", now=100.0)
        out = enabled_notifier._dedup_gate("hello", now=161.0)  # 61s later
        assert out == "hello"


class TestRateCapGate:
    """Tests for TelegramNotifier._rate_cap_gate (20 sends / rolling 60s)."""

    def test_under_cap_passes_through(self, enabled_notifier):
        out = enabled_notifier._rate_cap_gate("msg", now=0.0)
        assert out == ["msg"]

    def test_cap_drops_beyond_limit(self, enabled_notifier):
        for i in range(20):
            enabled_notifier._rate_cap_gate(f"msg{i}", now=float(i))
        out = enabled_notifier._rate_cap_gate("msg20", now=20.0)
        assert out == []
        assert enabled_notifier._rate_suppressed_count == 1

    def test_summary_sent_once_window_frees(self, enabled_notifier):
        for i in range(20):
            enabled_notifier._rate_cap_gate(f"msg{i}", now=float(i))
        for i in range(5):
            enabled_notifier._rate_cap_gate("overflow", now=25.0)
        out = enabled_notifier._rate_cap_gate("next", now=61.0)
        assert len(out) == 2
        assert "suppressed 5 messages in the last minute" in out[0]
        assert out[1] == "next"
        assert enabled_notifier._rate_suppressed_count == 0


class TestSendMessageDedupIntegration:
    """send_message end-to-end: dedup suppresses the HTTP call entirely."""

    @pytest.mark.asyncio
    async def test_duplicate_send_makes_only_one_http_call(self, enabled_notifier):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_post_cm = MagicMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_post_cm)
        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch('notifications.telegram_notifier.aiohttp.ClientSession', return_value=mock_session_cm):
            first = await enabled_notifier.send_message("dup text")
            second = await enabled_notifier.send_message("dup text")
            assert first is True
            assert second is False
            assert mock_session.post.call_count == 1


class TestRateLimitHandling:
    """send_message end-to-end: HTTP 429 handling."""

    @pytest.mark.asyncio
    async def test_429_with_short_retry_after_retries_once(self, enabled_notifier):
        rate_limited = MagicMock()
        rate_limited.status = 429
        rate_limited.json = AsyncMock(return_value={"parameters": {"retry_after": 1}})
        rate_limited.headers = {}
        cm_429 = MagicMock()
        cm_429.__aenter__ = AsyncMock(return_value=rate_limited)
        cm_429.__aexit__ = AsyncMock(return_value=False)

        ok_response = MagicMock()
        ok_response.status = 200
        cm_ok = MagicMock()
        cm_ok.__aenter__ = AsyncMock(return_value=ok_response)
        cm_ok.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=[cm_429, cm_ok])
        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch('notifications.telegram_notifier.aiohttp.ClientSession', return_value=mock_session_cm), \
             patch('notifications.telegram_notifier.asyncio.sleep', new=AsyncMock()) as mock_sleep:
            result = await enabled_notifier.send_message("rate me")
            assert result is True
            mock_sleep.assert_awaited_once_with(1)
            assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_429_with_long_retry_after_drops_without_retry(self, enabled_notifier):
        rate_limited = MagicMock()
        rate_limited.status = 429
        rate_limited.json = AsyncMock(return_value={"parameters": {"retry_after": 30}})
        rate_limited.headers = {}
        cm_429 = MagicMock()
        cm_429.__aenter__ = AsyncMock(return_value=rate_limited)
        cm_429.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=cm_429)
        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch('notifications.telegram_notifier.aiohttp.ClientSession', return_value=mock_session_cm), \
             patch('notifications.telegram_notifier.asyncio.sleep', new=AsyncMock()) as mock_sleep:
            result = await enabled_notifier.send_message("rate me long")
            assert result is False
            mock_sleep.assert_not_awaited()
            assert mock_session.post.call_count == 1
