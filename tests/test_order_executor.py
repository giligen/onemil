"""
Unit tests for OrderExecutor — bracket order submission via Alpaca.

Uses mocked AlpacaClient and real database (temp file).
"""

import logging
import pytest
from datetime import datetime, timezone, timedelta
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


class TestSubmitBuyStopOrder:
    """Tests for simple stop-limit order submission (no bracket)."""

    def test_submits_order_and_saves_trade(self, executor, mock_alpaca, db):
        """Successful simple order submission saves trade to DB."""
        mock_alpaca.submit_stop_limit_order.return_value = {
            'id': 'order-simple-1',
            'status': 'accepted',
            'symbol': 'TEST',
            'qty': 113,
        }

        plan = _make_plan()
        result = executor.submit_buy_stop_order(plan)

        assert result is not None
        assert result['order_id'] == 'order-simple-1'
        assert result['order_type'] == 'stop_simple'
        assert result['symbol'] == 'TEST'
        assert result['shares'] == 113

        # Verify trade saved to DB
        from datetime import date
        trades = db.get_trades_by_date(date.today().isoformat())
        assert len(trades) == 1
        assert trades[0]['symbol'] == 'TEST'
        assert trades[0]['order_id'] == 'order-simple-1'

    def test_calls_alpaca_with_correct_params(self, executor, mock_alpaca):
        """Verifies correct parameters passed to Alpaca — no bracket legs."""
        mock_alpaca.submit_stop_limit_order.return_value = {
            'id': 'order-simple-2', 'status': 'accepted',
            'symbol': 'TEST', 'qty': 113,
        }

        plan = _make_plan()
        executor.submit_buy_stop_order(plan, slippage_pct=0.02)

        mock_alpaca.submit_stop_limit_order.assert_called_once_with(
            symbol='TEST',
            qty=113,
            side='buy',
            stop_price=4.40,
            limit_price=round(4.40 * 1.02, 2),
        )

    def test_handles_api_failure(self, executor, mock_alpaca):
        """Returns None when Alpaca API call fails."""
        mock_alpaca.submit_stop_limit_order.side_effect = AlpacaAPIError("API down")

        plan = _make_plan()
        result = executor.submit_buy_stop_order(plan)
        assert result is None

    def test_handles_none_return(self, executor, mock_alpaca):
        """Returns None when Alpaca returns None."""
        mock_alpaca.submit_stop_limit_order.return_value = None

        plan = _make_plan()
        result = executor.submit_buy_stop_order(plan)
        assert result is None

    def test_stop_price_and_limit_price(self, executor, mock_alpaca):
        """Stop price = entry_price, limit = entry * (1 + slippage)."""
        mock_alpaca.submit_stop_limit_order.return_value = {
            'id': 'order-3', 'status': 'accepted',
            'symbol': 'TEST', 'qty': 113,
        }

        plan = _make_plan()
        result = executor.submit_buy_stop_order(plan, slippage_pct=0.03)

        assert result['stop_price'] == 4.40
        assert result['limit_price'] == round(4.40 * 1.03, 2)


class TestMarketableLimitFallback:
    """Marketable-limit fallback (IREZ+TTGT post-mortem 2026-05-08).

    Pinned regression: when the bid is at/above the configured stop_price,
    OrderExecutor MUST submit a marketable LIMIT BUY instead of a stop-limit.
    Alpaca live rejects stop-limit BUY orders where stop is already triggered;
    paper accepts them. This is the parity gap that lost ~24 prod orders in
    the last 5 weeks (TTGT, IREZ, EAF, RMAX, AGCC, OPTX, MLEC, SMX, TOYO).
    """

    def test_falls_back_to_limit_when_bid_above_stop(self, executor, mock_alpaca):
        """Bid > stop → submit_limit_buy_order called instead of stop-limit."""
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 4.45, 'ask_price': 4.46,
        }
        mock_alpaca.submit_limit_buy_order.return_value = {
            'id': 'mlf-1', 'status': 'accepted',
        }
        plan = _make_plan()
        result = executor.submit_buy_stop_order(plan)
        assert result is not None
        assert result['order_type'] == 'marketable_limit_fallback'
        mock_alpaca.submit_limit_buy_order.assert_called_once()
        mock_alpaca.submit_stop_limit_order.assert_not_called()

    def test_falls_back_when_bid_equal_stop(self, executor, mock_alpaca):
        """Bid == stop (boundary case) → marketable limit fallback fires."""
        mock_alpaca.get_latest_quote.return_value = {'bid_price': 4.40}
        mock_alpaca.submit_limit_buy_order.return_value = {
            'id': 'mlf-eq', 'status': 'accepted',
        }
        plan = _make_plan()
        result = executor.submit_buy_stop_order(plan)
        assert result['order_type'] == 'marketable_limit_fallback'
        mock_alpaca.submit_limit_buy_order.assert_called_once()
        mock_alpaca.submit_stop_limit_order.assert_not_called()

    def test_uses_stop_limit_when_bid_below_stop(self, executor, mock_alpaca):
        """Bid < stop → original stop-limit path (current behavior)."""
        mock_alpaca.get_latest_quote.return_value = {'bid_price': 4.30}
        mock_alpaca.submit_stop_limit_order.return_value = {
            'id': 'sl-1', 'status': 'accepted',
            'symbol': 'TEST', 'qty': 113,
        }
        plan = _make_plan()
        result = executor.submit_buy_stop_order(plan)
        assert result['order_type'] == 'stop_simple'
        mock_alpaca.submit_stop_limit_order.assert_called_once()
        mock_alpaca.submit_limit_buy_order.assert_not_called()

    def test_falls_through_when_quote_unavailable(self, executor, mock_alpaca):
        """get_latest_quote raises → defensive default to stop-limit."""
        mock_alpaca.get_latest_quote.side_effect = AlpacaAPIError("flaky")
        mock_alpaca.submit_stop_limit_order.return_value = {
            'id': 'fallthrough-1', 'status': 'accepted',
            'symbol': 'TEST', 'qty': 113,
        }
        plan = _make_plan()
        result = executor.submit_buy_stop_order(plan)
        assert result['order_type'] == 'stop_simple'
        mock_alpaca.submit_stop_limit_order.assert_called_once()
        mock_alpaca.submit_limit_buy_order.assert_not_called()

    def test_falls_through_when_quote_returns_zero_bid(self, executor, mock_alpaca):
        """bid == 0 (degenerate) → defensive default to stop-limit."""
        mock_alpaca.get_latest_quote.return_value = {'bid_price': 0.0}
        mock_alpaca.submit_stop_limit_order.return_value = {
            'id': 'zero-1', 'status': 'accepted',
            'symbol': 'TEST', 'qty': 113,
        }
        plan = _make_plan()
        result = executor.submit_buy_stop_order(plan)
        assert result['order_type'] == 'stop_simple'
        mock_alpaca.submit_stop_limit_order.assert_called_once()
        mock_alpaca.submit_limit_buy_order.assert_not_called()

    def test_disabled_by_config_always_uses_stop_limit(self, mock_alpaca, db):
        """Config kill-switch: enabled=False bypasses the check entirely,
        even when bid > stop. Restores legacy stop-limit-only behavior."""
        executor = OrderExecutor(alpaca_client=mock_alpaca, db=db)
        executor._marketable_limit_fallback_cfg = {'enabled': False}
        mock_alpaca.get_latest_quote.return_value = {'bid_price': 4.45}
        mock_alpaca.submit_stop_limit_order.return_value = {
            'id': 'disabled-1', 'status': 'accepted',
            'symbol': 'TEST', 'qty': 113,
        }
        plan = _make_plan()
        result = executor.submit_buy_stop_order(plan)
        assert result['order_type'] == 'stop_simple'
        mock_alpaca.submit_stop_limit_order.assert_called_once()
        mock_alpaca.submit_limit_buy_order.assert_not_called()

    def test_db_record_saved_in_marketable_path(self, executor, mock_alpaca, db):
        """DB trade row must persist on the fallback path, same as stop-limit."""
        from datetime import date
        mock_alpaca.get_latest_quote.return_value = {'bid_price': 4.45}
        mock_alpaca.submit_limit_buy_order.return_value = {
            'id': 'mlf-db-1', 'status': 'accepted',
        }
        plan = _make_plan()
        executor.submit_buy_stop_order(plan)
        trades = db.get_trades_by_date(date.today().isoformat())
        assert len(trades) == 1
        assert trades[0]['order_id'] == 'mlf-db-1'
        assert trades[0]['symbol'] == 'TEST'

    def test_marketable_path_logs_diagnostic(self, executor, mock_alpaca, caplog):
        """The fallback branch logs `STOP ALREADY TRIGGERED` so journalctl
        spotting the new path is trivial."""
        mock_alpaca.get_latest_quote.return_value = {'bid_price': 4.45}
        mock_alpaca.submit_limit_buy_order.return_value = {
            'id': 'log-1', 'status': 'accepted',
        }
        plan = _make_plan()
        with caplog.at_level(logging.INFO, logger='trading.order_executor'):
            executor.submit_buy_stop_order(plan)
        assert any(
            'STOP ALREADY TRIGGERED' in rec.message
            for rec in caplog.records
        ), "Expected 'STOP ALREADY TRIGGERED' INFO log on the fallback path"

    def test_marketable_path_passes_correct_limit_to_alpaca(
        self, executor, mock_alpaca,
    ):
        """Fallback uses limit_price = stop * (1 + slippage_pct) — same cap
        the stop-limit path would have applied."""
        mock_alpaca.get_latest_quote.return_value = {'bid_price': 4.45}
        mock_alpaca.submit_limit_buy_order.return_value = {
            'id': 'price-1', 'status': 'accepted',
        }
        plan = _make_plan()
        executor.submit_buy_stop_order(plan, slippage_pct=0.02)
        mock_alpaca.submit_limit_buy_order.assert_called_once_with(
            symbol='TEST', qty=113, limit_price=round(4.40 * 1.02, 2),
        )

    def test_returns_none_when_marketable_limit_submit_raises(
        self, executor, mock_alpaca,
    ):
        """If the marketable-limit submission fails after the fallback fires,
        return None — DO NOT silently fall back to the original stop-limit
        path (which would also reject under the same conditions, just from
        Alpaca's pre-trade validation). Trader will retry on next bar.
        Pinned regression for missing test coverage in the original ship."""
        mock_alpaca.get_latest_quote.return_value = {'bid_price': 4.45}
        mock_alpaca.submit_limit_buy_order.side_effect = AlpacaAPIError(
            "Alpaca transient 503"
        )
        plan = _make_plan()
        result = executor.submit_buy_stop_order(plan)
        assert result is None
        mock_alpaca.submit_limit_buy_order.assert_called_once()
        # IMPORTANT: must NOT have fallen through to stop-limit on this path.
        # If we did, Alpaca would reject for the same reason (bid >= stop)
        # and we'd be making 2 wasted API calls per attempted entry.
        mock_alpaca.submit_stop_limit_order.assert_not_called()

    def test_returns_consistent_dict_shape(self, executor, mock_alpaca):
        """Both submission paths return dicts with identical key sets so
        downstream consumers don't need to special-case order_type."""
        mock_alpaca.get_latest_quote.return_value = {'bid_price': 4.45}
        mock_alpaca.submit_limit_buy_order.return_value = {
            'id': 'mlf-shape', 'status': 'accepted',
        }
        mlf_result = executor.submit_buy_stop_order(_make_plan())

        mock_alpaca.reset_mock()
        mock_alpaca.get_latest_quote.return_value = {'bid_price': 4.30}
        mock_alpaca.submit_stop_limit_order.return_value = {
            'id': 'sl-shape', 'status': 'accepted',
            'symbol': 'TEST', 'qty': 113,
        }
        sl_result = executor.submit_buy_stop_order(_make_plan())

        assert set(mlf_result.keys()) == set(sl_result.keys())
        assert mlf_result['order_type'] == 'marketable_limit_fallback'
        assert sl_result['order_type'] == 'stop_simple'


class TestStopRebumpStraddle:
    """Stop re-bump when the bid/ask spread straddles the breakout level
    (IREZ+TTGT post-mortem extended 2026-05-14 after KPTI/TRT).

    Alpaca live rejects a buy stop-limit whenever stop_price <= current ASK
    (the order is immediately marketable). The earlier marketable-limit fix
    only checked the BID, so it missed the straddle case bid < stop <= ask —
    exactly what rejected KPTI ($9.71 stop / $9.79 ask) and TRT ($13.27 /
    $13.30). When the spread straddles the breakout level the order is
    re-bumped to ask + rebump_buffer so it stays a real stop; if the bumped
    stop would exceed limit_price the breakout is too extended and the trade
    is skipped (engine retries next bar).

    Plan under test: entry/stop = 4.40, limit = round(4.40 * 1.02, 2) = 4.49.
    """

    def test_rebumps_stop_when_spread_straddles(self, executor, mock_alpaca):
        """bid < stop <= ask, bumped stop within limit → stop re-bumped to
        ask + 0.02, limit unchanged, order_type tagged stop_rebump."""
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 4.35, 'ask_price': 4.45,
        }
        mock_alpaca.submit_stop_limit_order.return_value = {
            'id': 'rebump-1', 'status': 'accepted',
            'symbol': 'TEST', 'qty': 113,
        }
        result = executor.submit_buy_stop_order(_make_plan())
        assert result is not None
        assert result['order_type'] == 'stop_rebump'
        assert result['stop_price'] == 4.47        # round(4.45 + 0.02, 2)
        assert result['limit_price'] == 4.49       # unchanged
        mock_alpaca.submit_stop_limit_order.assert_called_once_with(
            symbol='TEST', qty=113, side='buy',
            stop_price=4.47, limit_price=4.49,
        )
        mock_alpaca.submit_limit_buy_order.assert_not_called()

    def test_rebumps_when_ask_equals_stop_boundary(self, executor, mock_alpaca):
        """ask == stop exactly (bid below) → straddle branch fires."""
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 4.30, 'ask_price': 4.40,
        }
        mock_alpaca.submit_stop_limit_order.return_value = {
            'id': 'rebump-eq', 'status': 'accepted',
            'symbol': 'TEST', 'qty': 113,
        }
        result = executor.submit_buy_stop_order(_make_plan())
        assert result['order_type'] == 'stop_rebump'
        assert result['stop_price'] == 4.42        # round(4.40 + 0.02, 2)

    def test_skips_when_breakout_extended_past_limit(
        self, executor, mock_alpaca, db,
    ):
        """bid < stop <= ask but ask + buffer > limit → skip entirely:
        return None, submit nothing, persist no trade row. Engine retries
        the symbol on the next bar (result is None == un-traded)."""
        from datetime import date
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 4.35, 'ask_price': 4.48,   # 4.48 + 0.02 = 4.50 > 4.49
        }
        result = executor.submit_buy_stop_order(_make_plan())
        assert result is None
        mock_alpaca.submit_stop_limit_order.assert_not_called()
        mock_alpaca.submit_limit_buy_order.assert_not_called()
        trades = db.get_trades_by_date(date.today().isoformat())
        assert len(trades) == 0                    # no junk rejected row

    def test_normal_stop_limit_when_ask_below_stop(self, executor, mock_alpaca):
        """ask < stop → unchanged: normal stop-limit at the original stop."""
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 4.30, 'ask_price': 4.38,
        }
        mock_alpaca.submit_stop_limit_order.return_value = {
            'id': 'normal-1', 'status': 'accepted',
            'symbol': 'TEST', 'qty': 113,
        }
        result = executor.submit_buy_stop_order(_make_plan())
        assert result['order_type'] == 'stop_simple'
        assert result['stop_price'] == 4.40
        mock_alpaca.submit_stop_limit_order.assert_called_once_with(
            symbol='TEST', qty=113, side='buy',
            stop_price=4.40, limit_price=4.49,
        )

    def test_disabled_config_skips_rebump(self, mock_alpaca, db):
        """enabled=False → straddle quote is ignored, legacy stop-limit at
        the original stop price (byte-identical rollback path)."""
        executor = OrderExecutor(alpaca_client=mock_alpaca, db=db)
        executor._marketable_limit_fallback_cfg = {'enabled': False}
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 4.35, 'ask_price': 4.45,
        }
        mock_alpaca.submit_stop_limit_order.return_value = {
            'id': 'disabled-rebump', 'status': 'accepted',
            'symbol': 'TEST', 'qty': 113,
        }
        result = executor.submit_buy_stop_order(_make_plan())
        assert result['order_type'] == 'stop_simple'
        assert result['stop_price'] == 4.40        # original, not re-bumped
        mock_alpaca.submit_stop_limit_order.assert_called_once()

    def test_rebump_buffer_read_from_config(self, mock_alpaca, db):
        """rebump_buffer is honored from config (not hardcoded 0.02)."""
        executor = OrderExecutor(alpaca_client=mock_alpaca, db=db)
        executor._marketable_limit_fallback_cfg = {
            'enabled': True, 'rebump_buffer': 0.05,
        }
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 4.35, 'ask_price': 4.43,
        }
        mock_alpaca.submit_stop_limit_order.return_value = {
            'id': 'buf-1', 'status': 'accepted',
            'symbol': 'TEST', 'qty': 113,
        }
        result = executor.submit_buy_stop_order(_make_plan())
        assert result['order_type'] == 'stop_rebump'
        assert result['stop_price'] == 4.48        # round(4.43 + 0.05, 2)

    def test_rebump_submit_raises_returns_none(self, executor, mock_alpaca):
        """If the re-bumped stop-limit submission fails, return None — DO NOT
        fall through to the original-stop path (Alpaca would reject it for
        the same reason). Engine retries on the next bar."""
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 4.35, 'ask_price': 4.45,
        }
        mock_alpaca.submit_stop_limit_order.side_effect = AlpacaAPIError(
            "Alpaca transient 503"
        )
        result = executor.submit_buy_stop_order(_make_plan())
        assert result is None
        assert mock_alpaca.submit_stop_limit_order.call_count == 1

    def test_rebump_saves_trade_record(self, executor, mock_alpaca, db):
        """Integration: the re-bumped order persists a trade row, same as the
        normal path — downstream fill tracking depends on it."""
        from datetime import date
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 4.35, 'ask_price': 4.45,
        }
        mock_alpaca.submit_stop_limit_order.return_value = {
            'id': 'rebump-db-1', 'status': 'accepted',
            'symbol': 'TEST', 'qty': 113,
        }
        executor.submit_buy_stop_order(_make_plan())
        trades = db.get_trades_by_date(date.today().isoformat())
        assert len(trades) == 1
        assert trades[0]['order_id'] == 'rebump-db-1'
        assert trades[0]['symbol'] == 'TEST'

    def test_rebump_logs_diagnostic(self, executor, mock_alpaca, caplog):
        """The re-bump branch logs `STOP STRADDLED BY SPREAD` for journalctl."""
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 4.35, 'ask_price': 4.45,
        }
        mock_alpaca.submit_stop_limit_order.return_value = {
            'id': 'log-rebump', 'status': 'accepted',
            'symbol': 'TEST', 'qty': 113,
        }
        with caplog.at_level(logging.INFO, logger='trading.order_executor'):
            executor.submit_buy_stop_order(_make_plan())
        assert any(
            'STOP STRADDLED BY SPREAD' in rec.message
            for rec in caplog.records
        ), "Expected 'STOP STRADDLED BY SPREAD' INFO log on the re-bump path"

    def test_skip_logs_warning(self, executor, mock_alpaca, caplog):
        """The extended-breakout skip logs a WARNING (silent skips hide bugs)."""
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 4.35, 'ask_price': 4.48,
        }
        with caplog.at_level(logging.WARNING, logger='trading.order_executor'):
            executor.submit_buy_stop_order(_make_plan())
        assert any(
            'BUY-STOP SKIPPED' in rec.message
            for rec in caplog.records
        ), "Expected 'BUY-STOP SKIPPED' WARNING on the extended-breakout path"

    def test_degenerate_zero_bid_with_ask_above_stop_rebumps(
        self, executor, mock_alpaca,
    ):
        """bid <= 0 (no bid quote) but ask >= stop → still re-bump. A native
        stop would still be rejected (ask >= stop), and re-bumping stays safe
        (only fills on a genuine upward print)."""
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 0.0, 'ask_price': 4.45,
        }
        mock_alpaca.submit_stop_limit_order.return_value = {
            'id': 'zerobid-1', 'status': 'accepted',
            'symbol': 'TEST', 'qty': 113,
        }
        result = executor.submit_buy_stop_order(_make_plan())
        assert result['order_type'] == 'stop_rebump'
        assert result['stop_price'] == 4.47        # round(4.45 + 0.02, 2)


class TestConflictCheckFastPath:
    """OrderExecutor wash-trade check: prefer OrderStreamWatcher cache when
    available, fall back to REST when stream is absent/unhealthy."""

    @pytest.fixture
    def alpaca_with_client(self):
        """MagicMock without spec so trading_client attribute can be set."""
        m = MagicMock()
        return m

    def test_stream_hit_blocks_without_rest(self, alpaca_with_client, db):
        stream = MagicMock()
        stream.is_healthy.return_value = True
        stream.get_open_order_symbols.return_value = {'AAPL'}
        ex = OrderExecutor(alpaca_client=alpaca_with_client, db=db, order_stream=stream)
        assert ex._has_conflicting_orders('AAPL') is True
        alpaca_with_client.trading_client.get_orders.assert_not_called()

    def test_stream_clear_no_rest(self, alpaca_with_client, db):
        stream = MagicMock()
        stream.is_healthy.return_value = True
        stream.get_open_order_symbols.return_value = {'MSFT'}  # different sym
        ex = OrderExecutor(alpaca_client=alpaca_with_client, db=db, order_stream=stream)
        assert ex._has_conflicting_orders('AAPL') is False
        alpaca_with_client.trading_client.get_orders.assert_not_called()

    def test_unhealthy_stream_falls_back_to_rest(self, alpaca_with_client, db):
        stream = MagicMock()
        stream.is_healthy.return_value = False
        ex = OrderExecutor(alpaca_client=alpaca_with_client, db=db, order_stream=stream)
        alpaca_with_client.trading_client.get_orders.return_value = []
        assert ex._has_conflicting_orders('AAPL') is False
        alpaca_with_client.trading_client.get_orders.assert_called_once()

    def test_no_stream_uses_rest(self, alpaca_with_client, db):
        ex = OrderExecutor(alpaca_client=alpaca_with_client, db=db)
        alpaca_with_client.trading_client.get_orders.return_value = []
        assert ex._has_conflicting_orders('AAPL') is False
        alpaca_with_client.trading_client.get_orders.assert_called_once()

    def test_stream_exception_falls_through(self, alpaca_with_client, db):
        stream = MagicMock()
        stream.is_healthy.return_value = True
        stream.get_open_order_symbols.side_effect = RuntimeError("boom")
        ex = OrderExecutor(alpaca_client=alpaca_with_client, db=db, order_stream=stream)
        alpaca_with_client.trading_client.get_orders.return_value = []
        # Stream raised → REST fallback fires, returns no conflict
        assert ex._has_conflicting_orders('AAPL') is False
        alpaca_with_client.trading_client.get_orders.assert_called_once()


# ===========================================================================
# Pipeline timing telemetry (added 2026-04-15 — see learnings from MACD wave
# 3.3s quote_to_submit on Anthropic+Alpaca cloud incident day. Bull-flag was
# missing this telemetry entirely, so the analogous incident on bull-flag
# would have been invisible.)
# ===========================================================================

class TestSubmitTimingTelemetry:
    """OrderExecutor must capture loop→submit latency so we can detect
    Alpaca/cloud-provider degradation in real time."""

    def _setup_simple_alpaca(self, mock_alpaca):
        mock_alpaca.submit_stop_limit_order.return_value = {
            'id': 'order-timing-1', 'status': 'accepted',
            'symbol': 'TEST', 'qty': 113,
        }

    def test_simple_order_persists_full_pipeline_timing(self, executor, mock_alpaca, db):
        self._setup_simple_alpaca(mock_alpaca)
        bar_close_at = datetime.now(timezone.utc) - timedelta(seconds=2)
        loop_processed_at = bar_close_at + timedelta(milliseconds=600)
        plan = _make_plan()
        result = executor.submit_buy_stop_order(
            plan,
            pipeline_timing={
                'bar_close_at': bar_close_at,
                'loop_processed_at': loop_processed_at,
            },
        )
        assert result is not None
        from datetime import date as _date
        trades = db.get_trades_by_date(_date.today().isoformat())
        assert len(trades) == 1
        t = trades[0]
        # All four timing fields populated; q2s is computed; b2l is 600 ms.
        assert t['order_submitted_at'] is not None
        assert t['loop_processed_at'] is not None
        assert t['bar_close_at'] is not None
        assert t['bar_close_to_loop_ms'] == 600
        assert t['quote_to_submit_ms'] >= 0  # actual elapsed since loop_processed_at

    def test_bracket_order_persists_full_pipeline_timing(self, executor, mock_alpaca, db):
        mock_alpaca.submit_stop_bracket_order.return_value = {
            'id': 'order-timing-2', 'status': 'accepted',
            'symbol': 'TEST', 'qty': 113,
        }
        bar_close_at = datetime.now(timezone.utc) - timedelta(seconds=3)
        loop_processed_at = bar_close_at + timedelta(milliseconds=1500)
        plan = _make_plan()
        result = executor.submit_buy_stop_bracket_order(
            plan,
            pipeline_timing={
                'bar_close_at': bar_close_at,
                'loop_processed_at': loop_processed_at,
            },
        )
        assert result is not None
        from datetime import date as _date
        trades = db.get_trades_by_date(_date.today().isoformat())
        assert len(trades) == 1
        t = trades[0]
        assert t['bar_close_to_loop_ms'] == 1500
        assert t['quote_to_submit_ms'] >= 0

    def test_no_pipeline_timing_leaves_columns_null_safely(
        self, executor, mock_alpaca, db,
    ):
        """Calling without pipeline_timing must NOT crash and must NOT pollute
        the DB with synthetic timestamps."""
        self._setup_simple_alpaca(mock_alpaca)
        plan = _make_plan()
        result = executor.submit_buy_stop_order(plan)  # no pipeline_timing
        assert result is not None
        from datetime import date as _date
        trades = db.get_trades_by_date(_date.today().isoformat())
        assert len(trades) == 1
        t = trades[0]
        # order_submitted_at IS captured (we know when WE submitted)
        assert t['order_submitted_at'] is not None
        # but bar_close_at / loop_processed_at / derived metrics stay NULL
        assert t['bar_close_at'] is None
        assert t['loop_processed_at'] is None
        assert t['bar_close_to_loop_ms'] is None
        assert t['quote_to_submit_ms'] is None

    def test_slow_submit_fires_warn_log(self, executor, mock_alpaca, db, caplog):
        """When loop→submit > 1000ms, a WARN must fire — early signal of
        cloud-provider degradation (the 2026-04-15 Anthropic+Alpaca incident)."""
        self._setup_simple_alpaca(mock_alpaca)
        # Fake an old loop_processed_at so order_submitted_at - loop > 1000ms.
        loop_processed_at = datetime.now(timezone.utc) - timedelta(milliseconds=2500)
        plan = _make_plan()
        with caplog.at_level(logging.WARNING):
            executor.submit_buy_stop_order(
                plan, pipeline_timing={'loop_processed_at': loop_processed_at},
            )
        slow_warns = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and 'SLOW SUBMIT' in r.message
        ]
        assert len(slow_warns) == 1
        assert 'TEST' in slow_warns[0].message
        assert 'strategy=bull_flag' in slow_warns[0].message

    def test_fast_submit_does_not_warn(self, executor, mock_alpaca, db, caplog):
        """Normal-latency submits (<1000ms) must NOT spam WARN logs."""
        self._setup_simple_alpaca(mock_alpaca)
        loop_processed_at = datetime.now(timezone.utc)
        plan = _make_plan()
        with caplog.at_level(logging.WARNING):
            executor.submit_buy_stop_order(
                plan, pipeline_timing={'loop_processed_at': loop_processed_at},
            )
        slow_warns = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and 'SLOW SUBMIT' in r.message
        ]
        assert slow_warns == []


