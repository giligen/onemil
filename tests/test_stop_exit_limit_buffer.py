"""Regression for 2026-04-23 BMNZ stranded-limit bug.

Bug: `_execute_stop_exit` priced the sell limit AT the bid
(`round(bid, 2)`). When a falling stock hits the stop, the bid drops
below our limit within seconds, so the LIMIT SELL sits NEW forever and
the position stays open. Worse, the engine optimistically wrote
`exit_price = limit_price` to the DB on submit, so the trade looked
closed when Alpaca was still long.

Example (BMNZ 2026-04-23):
  trigger at $14.04, bid=$14.03 → limit=$14.03
  bid then fell to $13.86 → limit never filled
  position stranded ~4h, extra −$288 loss vs. what DB recorded

Fix: price BELOW bid using the existing `compute_limit_price(bid)`
helper, which subtracts max(3¢, 50bps) from the price. Keeps the
limit marketable even if bid slips during submit latency.

This test file covers ONLY the price-buffer fix. The DB-accounting
follow-up (wait for fill before writing exit) is tracked separately.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from data_sources.alpaca_client import AlpacaClient
from trading.stop_monitor import StopMonitor


@pytest.fixture
def mock_alpaca():
    client = MagicMock(spec=AlpacaClient)
    client.cancel_order.return_value = True
    client.submit_limit_sell_order.return_value = {
        'id': 'sell-order-123', 'status': 'accepted', 'symbol': 'TEST',
    }
    client.close_position.return_value = {
        'id': 'close-order-456', 'status': 'accepted', 'symbol': 'TEST',
    }
    # REST quote fallback
    client.get_latest_quote.return_value = {
        'bid_price': 14.03, 'ask_price': 14.05,
        'bid_size': 2800, 'ask_size': 100,
    }
    # Default happy-path poll response — override in tests that
    # specifically exercise the timeout/escalation flow.
    client.get_order.return_value = {
        'id': 'sell-order-123',
        'status': 'filled',
        'filled_avg_price': 13.96,
        'filled_qty': 3202,
    }
    # Underlying trading client used for open-order lookup
    client.trading_client = MagicMock()
    client.trading_client.get_orders.return_value = []
    return client


@pytest.fixture
def monitor(mock_alpaca):
    mon = StopMonitor(
        api_key='k', api_secret='s', alpaca_client=mock_alpaca,
        marketable_limit_offset=0.03,
        marketable_limit_offset_pct=0.005,
    )
    # Collapse poll timings so tests finish quickly even on escalation paths.
    mon._STOP_EXIT_FILL_TIMEOUT_S = 0.2
    mon._MARKET_CLOSE_FILL_TIMEOUT_S = 0.2
    mon._STOP_EXIT_POLL_INTERVAL_S = 0.05
    return mon


class TestStopExitLimitBuffer:
    """The submitted limit must sit BELOW the current bid by the configured
    buffer, so normal bid drift doesn't strand the order."""

    @pytest.mark.asyncio
    async def test_ws_cache_path_applies_buffer_below_bid(self, monitor, mock_alpaca):
        """Fresh WS quote path: BMNZ-style scenario — bid=$14.03, limit must be
        <= bid - max(3¢, 50bps) = $14.03 - max($0.03, $0.07) = $13.96."""
        monitor.add_watch('BMNZ', 14.05, 3202, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['BMNZ']
            w.latest_bid = 14.03
            w.latest_ask = 14.05
            w.latest_bid_size = 2800
            w.latest_ask_size = 100
            w.latest_quote_ts = time.time()  # fresh

        await monitor._execute_stop_exit('BMNZ', 14.04, w, exit_reason='stop_loss')

        assert mock_alpaca.submit_limit_sell_order.call_count == 1
        kwargs = mock_alpaca.submit_limit_sell_order.call_args.kwargs
        limit = kwargs['limit_price']
        assert limit < 14.03, f"limit {limit} must be strictly below bid 14.03"
        # $14.03 × 0.005 = $0.07015 → offset rounds to $0.07
        assert limit == pytest.approx(13.96, abs=0.01)

    @pytest.mark.asyncio
    async def test_rest_fallback_path_applies_buffer_below_bid(self, monitor, mock_alpaca):
        """Same buffer semantics on the REST quote fallback (stale WS)."""
        monitor.add_watch('BMNZ', 14.05, 3202, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['BMNZ']
            # Stale quote — forces REST fallback
            w.latest_bid = 0.0
            w.latest_ask = 0.0
            w.latest_quote_ts = 0.0

        await monitor._execute_stop_exit('BMNZ', 14.04, w, exit_reason='stop_loss')
        assert mock_alpaca.submit_limit_sell_order.call_count == 1
        kwargs = mock_alpaca.submit_limit_sell_order.call_args.kwargs
        limit = kwargs['limit_price']
        assert limit < 14.03
        assert limit == pytest.approx(13.96, abs=0.01)

    @pytest.mark.asyncio
    async def test_low_price_uses_fixed_three_cent_floor(self, monitor, mock_alpaca):
        """Low-price stocks (bid < $6): buffer falls back to the 3¢ fixed
        offset since 0.5% is smaller."""
        monitor.add_watch('CHEAP', 5.00, 1000, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['CHEAP']
            w.latest_bid = 5.10
            w.latest_ask = 5.11
            w.latest_bid_size = 500
            w.latest_ask_size = 500
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit('CHEAP', 5.09, w, exit_reason='stop_loss')
        kwargs = mock_alpaca.submit_limit_sell_order.call_args.kwargs
        limit = kwargs['limit_price']
        # 5.10 × 0.005 = $0.0255 < 3¢ → use 3¢ → limit=$5.07
        assert limit == pytest.approx(5.07, abs=0.01)
        assert limit < 5.10

    @pytest.mark.asyncio
    async def test_high_price_uses_pct_buffer(self, monitor, mock_alpaca):
        """High-price stocks: 50bps buffer dominates the 3¢ floor."""
        monitor.add_watch('HIGH', 30.00, 100, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['HIGH']
            w.latest_bid = 28.00
            w.latest_ask = 28.02
            w.latest_bid_size = 300
            w.latest_ask_size = 300
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit('HIGH', 28.01, w, exit_reason='stop_loss')
        kwargs = mock_alpaca.submit_limit_sell_order.call_args.kwargs
        limit = kwargs['limit_price']
        # $28 × 0.005 = $0.14 (dominates 3¢) → limit=$27.86
        assert limit == pytest.approx(27.86, abs=0.01)

    @pytest.mark.asyncio
    async def test_limit_never_zero_or_negative(self, monitor, mock_alpaca):
        """Pathological sub-penny stock: limit floors at $0.01."""
        monitor.add_watch('PENNY', 0.02, 10000, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['PENNY']
            w.latest_bid = 0.02
            w.latest_ask = 0.03
            w.latest_bid_size = 100
            w.latest_ask_size = 100
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit('PENNY', 0.02, w, exit_reason='stop_loss')
        kwargs = mock_alpaca.submit_limit_sell_order.call_args.kwargs
        limit = kwargs['limit_price']
        assert limit >= 0.01


# ---------------------------------------------------------------------------
# Fill-confirmation — the OTHER half of the BMNZ fix (Bug B)
# ---------------------------------------------------------------------------

class TestStopExitFillConfirmation:
    """Before the fix, `_execute_stop_exit` emitted a close event with
    `exit_price = limit_price` the moment it SUBMITTED the limit sell. If
    the limit never filled (BMNZ: limit $14.03 stranded while bid went to
    $13.86), the DB was marked closed at a price Alpaca never executed,
    producing a silent desync that cost real P&L.

    Fix: `_execute_stop_exit` now polls the order for an actual fill and
    escalates to market close on timeout. Exit event is emitted with the
    REAL fill price, not the submitted limit price."""

    @pytest.mark.asyncio
    async def test_happy_path_uses_real_fill_price(self, monitor, mock_alpaca):
        """Real fill price must flow into the event, not the submitted limit."""
        mock_alpaca.get_order.return_value = {
            'id': 'sell-order-123', 'status': 'filled',
            'filled_avg_price': 4.20,   # real fill (below buffered limit ~$4.22)
            'filled_qty': 500,
        }
        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['PLYX']
            w.latest_bid = 4.26
            w.latest_ask = 4.27
            w.latest_bid_size = 300
            w.latest_ask_size = 300
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit('PLYX', 4.25, w, exit_reason='stop_loss')

        events = monitor.drain_exit_events()
        assert len(events) == 1
        # exit_price must reflect the ACTUAL fill ($4.20), not the buffered limit
        assert events[0].exit_price == pytest.approx(4.20, abs=0.01)

    @pytest.mark.asyncio
    async def test_unfilled_limit_escalates_to_market_close(self, monitor, mock_alpaca):
        """Limit that never fills must trigger cancel + market close, and the
        exit event carries the market-close fill price (not the unfilled limit)."""
        # First call (limit poll): NEW forever. Last call (market poll): filled.
        # Use a counter to alternate behavior.
        calls = {'n': 0}

        def _fake_get_order(order_id):
            calls['n'] += 1
            # limit sell order never fills
            if order_id == 'sell-order-123':
                return {
                    'id': 'sell-order-123',
                    'status': 'new',
                    'filled_avg_price': None,
                    'filled_qty': 0,
                }
            # market close fills at $3.95 (below the stranded limit)
            return {
                'id': order_id, 'status': 'filled',
                'filled_avg_price': 3.95, 'filled_qty': 500,
            }
        mock_alpaca.get_order.side_effect = _fake_get_order

        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['PLYX']
            w.latest_bid = 4.26
            w.latest_ask = 4.27
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit('PLYX', 4.25, w, exit_reason='stop_loss')

        # Market close must have been called during escalation
        assert mock_alpaca.close_position.call_count == 1
        events = monitor.drain_exit_events()
        assert len(events) == 1
        # exit_price = market close fill ($3.95), NOT the unfilled limit
        assert events[0].exit_price == pytest.approx(3.95, abs=0.01)
        assert events[0].exit_reason == 'stop_loss_market_fallback'

    @pytest.mark.asyncio
    async def test_explicit_sl_leg_cancel_deferred_until_after_fill(self, monitor, mock_alpaca):
        """The explicit `watch.sl_leg_id` cancel at the end of the exit path is
        a best-effort backstop in case the bulk-cancel-all-open-orders block
        upstream failed. This test asserts that that trailing cancel fires
        AFTER the TP cancel (and AFTER fill confirmation) — i.e., we don't
        bail out early and leave the explicit SL cancel unexecuted.

        Note: in production, the bulk-cancel (which depends on
        `trading_client.get_orders`) usually cancels the SL leg BEFORE the
        limit submit because Alpaca holds shares inside active bracket legs.
        So the naked window during the fill poll is real; the trailing cancel
        is just a safety net for the bulk-cancel-errored case.
        """
        cancel_log = []

        def _track_cancel(order_id):
            cancel_log.append(order_id)
            return True
        mock_alpaca.cancel_order.side_effect = _track_cancel

        monitor.add_watch('PLYX', 4.29, 500, 'tp-leg', 'sl-leg')
        with monitor._watch_lock:
            w = monitor._watches['PLYX']
            w.latest_bid = 4.26
            w.latest_ask = 4.27
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit('PLYX', 4.25, w, exit_reason='stop_loss')

        # Both explicit cancels fire, and SL comes after TP (after fill is
        # confirmed). Bulk-cancel is a no-op here because the fixture's
        # trading_client.get_orders returns [] — production would also kill
        # the SL leg in that block.
        assert 'tp-leg' in cancel_log
        assert 'sl-leg' in cancel_log
        tp_idx = cancel_log.index('tp-leg')
        sl_idx = cancel_log.index('sl-leg')
        assert sl_idx > tp_idx, (
            f"SL leg cancelled at position {sl_idx}, TP at {tp_idx} — "
            f"trailing sl_leg_id cancel must fire AFTER fill confirmation "
            f"(full log: {cancel_log})"
        )

    @pytest.mark.asyncio
    async def test_partial_fill_uses_filled_qty(self, monitor, mock_alpaca):
        """Order marked filled with partial qty must carry that through the
        event — caller uses it for DB accounting."""
        mock_alpaca.get_order.return_value = {
            'id': 'sell-order-123', 'status': 'filled',
            'filled_avg_price': 4.19,
            'filled_qty': 300,  # partial of 500
        }

        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['PLYX']
            w.latest_bid = 4.26
            w.latest_ask = 4.27
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit('PLYX', 4.25, w, exit_reason='stop_loss')
        events = monitor.drain_exit_events()
        assert len(events) == 1
        assert events[0].exit_price == pytest.approx(4.19, abs=0.01)

    @pytest.mark.asyncio
    async def test_market_close_unconfirmed_uses_trigger_price_last_resort(
        self, monitor, mock_alpaca
    ):
        """If BOTH the limit poll and the market close poll time out (very
        unusual — market orders almost always fill in paper), we fall back to
        trigger_price with a loud ERROR log. DB gets SOMETHING (not None) so
        the trade isn't leaked, but we've signalled manual verification."""
        mock_alpaca.get_order.return_value = {
            'id': 'x', 'status': 'new',  # never fills
            'filled_avg_price': None, 'filled_qty': 0,
        }

        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['PLYX']
            w.latest_bid = 4.26
            w.latest_ask = 4.27
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit('PLYX', 4.25, w, exit_reason='stop_loss')
        events = monitor.drain_exit_events()
        assert len(events) == 1
        # trigger_price = 4.25; DB records this so we don't lose the trade row.
        assert events[0].exit_price == pytest.approx(4.25, abs=0.01)


# ---------------------------------------------------------------------------
# Race-condition SL-leg recovery (item 5 from the code review)
# ---------------------------------------------------------------------------
#
# When the primary limit fails to fill AND close_position raises
# "position not found" ("40410000"), the broker-side bracket SL leg most
# likely won the exit race. Before the 2026-04-23 v2 fix the event carried
# trigger_price (often $1+ above the real SL fill — e.g. BMNZ would have
# misreported by ~$3.5K). Now `_escalate_to_market_close` queries the SL
# leg via `_sl_leg_fill_price` and uses the real filled_avg_price when
# recoverable.
# ---------------------------------------------------------------------------


class TestSlLegFillRecoveryOnRace:

    @pytest.mark.asyncio
    async def test_race_on_close_position_recovers_sl_leg_fill(
        self, monitor, mock_alpaca
    ):
        """close_position races → query SL leg → use its real fill price."""
        from alpaca.common.exceptions import APIError

        # Limit sell returns 'new' forever → forces escalation
        def _get_order(order_id):
            if order_id == 'sell-order-123':
                return {'id': 'sell-order-123', 'status': 'new',
                        'filled_avg_price': None, 'filled_qty': 0}
            if order_id == 'sl-1':
                # SL leg won the race and actually filled at $2.70
                return {'id': 'sl-1', 'status': 'filled',
                        'filled_avg_price': 2.70, 'filled_qty': 500}
            return {'id': order_id, 'status': 'new',
                    'filled_avg_price': None, 'filled_qty': 0}
        mock_alpaca.get_order.side_effect = _get_order
        # close_position raises "position not found" (race condition)
        mock_alpaca.close_position.side_effect = Exception(
            "40410000: position not found for PLYX"
        )

        monitor.add_watch('PLYX', 3.00, 500, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['PLYX']
            w.latest_bid = 2.99
            w.latest_ask = 3.00
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit('PLYX', 2.98, w, exit_reason='stop_loss')
        events = monitor.drain_exit_events()
        assert len(events) == 1
        # Event must carry the SL leg's real fill ($2.70), NOT trigger_price ($2.98)
        assert events[0].exit_price == pytest.approx(2.70, abs=0.01)

    @pytest.mark.asyncio
    async def test_race_on_close_position_no_sl_leg_filled_uses_trigger(
        self, monitor, mock_alpaca
    ):
        """close_position races, SL leg status is NOT 'filled' (e.g. someone
        cancelled it, or it's still live but we still got a race) — fall back
        to trigger_price with ERROR (logged for manual reconcile)."""
        def _get_order(order_id):
            if order_id == 'sl-1':
                # SL leg was cancelled — nothing to recover from it
                return {'id': 'sl-1', 'status': 'canceled',
                        'filled_avg_price': None, 'filled_qty': 0}
            return {'id': order_id, 'status': 'new',
                    'filled_avg_price': None, 'filled_qty': 0}
        mock_alpaca.get_order.side_effect = _get_order
        mock_alpaca.close_position.side_effect = Exception(
            "40410000: position not found"
        )

        monitor.add_watch('PLYX', 3.00, 500, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['PLYX']
            w.latest_bid = 2.99
            w.latest_ask = 3.00
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit('PLYX', 2.98, w, exit_reason='stop_loss')
        events = monitor.drain_exit_events()
        assert len(events) == 1
        assert events[0].exit_price == pytest.approx(2.98, abs=0.01)

    @pytest.mark.asyncio
    async def test_race_on_close_position_no_sl_leg_id_uses_trigger(
        self, monitor, mock_alpaca
    ):
        """If the watch was registered without an SL leg id (unusual), the
        race-condition recovery has nothing to query and falls back to
        trigger_price with ERROR."""
        def _get_order(order_id):
            return {'id': order_id, 'status': 'new',
                    'filled_avg_price': None, 'filled_qty': 0}
        mock_alpaca.get_order.side_effect = _get_order
        mock_alpaca.close_position.side_effect = Exception(
            "40410000: position not found"
        )

        # Watch registered with sl_leg_id=None
        monitor.add_watch('PLYX', 3.00, 500, 'tp-1', None)
        with monitor._watch_lock:
            w = monitor._watches['PLYX']
            w.latest_bid = 2.99
            w.latest_ask = 3.00
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit('PLYX', 2.98, w, exit_reason='stop_loss')
        events = monitor.drain_exit_events()
        assert len(events) == 1
        assert events[0].exit_price == pytest.approx(2.98, abs=0.01)

    @pytest.mark.asyncio
    async def test_sl_leg_fill_price_helper_returns_none_on_missing_id(
        self, monitor, mock_alpaca
    ):
        """_sl_leg_fill_price is a pure helper; None id → None, no API call."""
        result = await monitor._sl_leg_fill_price(mock_alpaca, None)
        assert result is None
        mock_alpaca.get_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_sl_leg_fill_price_helper_returns_none_when_not_filled(
        self, monitor, mock_alpaca
    ):
        mock_alpaca.get_order.return_value = {
            'id': 'sl-1', 'status': 'accepted',
            'filled_avg_price': None, 'filled_qty': 0,
        }
        result = await monitor._sl_leg_fill_price(mock_alpaca, 'sl-1')
        assert result is None

    @pytest.mark.asyncio
    async def test_sl_leg_fill_price_helper_returns_price_on_fill(
        self, monitor, mock_alpaca
    ):
        mock_alpaca.get_order.return_value = {
            'id': 'sl-1', 'status': 'filled',
            'filled_avg_price': 12.94, 'filled_qty': 500,
        }
        result = await monitor._sl_leg_fill_price(mock_alpaca, 'sl-1')
        assert result == pytest.approx(12.94, abs=0.001)

    @pytest.mark.asyncio
    async def test_sl_leg_fill_price_helper_swallows_api_errors(
        self, monitor, mock_alpaca
    ):
        """Network / API errors in the SL lookup must not crash the exit
        flow. Return None so caller falls back cleanly."""
        mock_alpaca.get_order.side_effect = Exception("boom")
        result = await monitor._sl_leg_fill_price(mock_alpaca, 'sl-1')
        assert result is None


# ---------------------------------------------------------------------------
# C6 — branch-specific exit_reason for analytics
# ---------------------------------------------------------------------------


class TestExitReasonPerBranch:
    """Each recovery branch in `_escalate_to_market_close` now signals back
    a distinct tag so `_execute_stop_exit` can set a distinct exit_reason.
    Analytics can't separate 'clean market fallback' from 'catastrophic
    SL race recovery' from 'last-resort trigger_price estimate' without
    this."""

    @pytest.mark.asyncio
    async def test_market_close_branch_sets_market_fallback_reason(
        self, monitor, mock_alpaca
    ):
        """Happy-ish fallback: limit didn't fill, market close did."""
        def _get_order(order_id):
            if order_id == 'sell-order-123':
                return {'id': order_id, 'status': 'new',
                        'filled_avg_price': None, 'filled_qty': 0}
            # market close order fills at $4.00
            return {'id': order_id, 'status': 'filled',
                    'filled_avg_price': 4.00, 'filled_qty': 500}
        mock_alpaca.get_order.side_effect = _get_order
        mock_alpaca.close_position.return_value = {
            'id': 'mkt-999', 'status': 'accepted', 'symbol': 'PLYX',
        }

        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['PLYX']
            w.latest_bid = 4.26; w.latest_ask = 4.27
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit('PLYX', 4.25, w, exit_reason='stop_loss')
        events = monitor.drain_exit_events()
        assert len(events) == 1
        assert events[0].exit_reason == 'stop_loss_market_fallback'

    @pytest.mark.asyncio
    async def test_sl_leg_race_branch_sets_sl_race_reason(
        self, monitor, mock_alpaca
    ):
        """Race path: close_position raises 'position not found', SL leg
        reports filled, event picks up sl_race reason + SL order_id."""
        def _get_order(order_id):
            if order_id == 'sl-1':
                return {'id': 'sl-1', 'status': 'filled',
                        'filled_avg_price': 2.70, 'filled_qty': 500}
            return {'id': order_id, 'status': 'new',
                    'filled_avg_price': None, 'filled_qty': 0}
        mock_alpaca.get_order.side_effect = _get_order
        mock_alpaca.close_position.side_effect = Exception(
            "40410000: position not found for PLYX"
        )

        monitor.add_watch('PLYX', 3.00, 500, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['PLYX']
            w.latest_bid = 2.99; w.latest_ask = 3.00
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit('PLYX', 2.98, w, exit_reason='stop_loss')
        events = monitor.drain_exit_events()
        assert len(events) == 1
        assert events[0].exit_reason == 'stop_loss_bracket_sl_race'
        # order_id should point at the SL leg that actually filled
        assert events[0].order_id == 'sl-1'

    @pytest.mark.asyncio
    async def test_last_resort_branch_sets_unconfirmed_reason(
        self, monitor, mock_alpaca
    ):
        """Every recovery path failed → trigger_price + ERROR log +
        distinct exit_reason so analytics can flag these rows for
        manual review."""
        mock_alpaca.get_order.return_value = {
            'id': 'x', 'status': 'new',
            'filled_avg_price': None, 'filled_qty': 0,
        }

        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['PLYX']
            w.latest_bid = 4.26; w.latest_ask = 4.27
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit('PLYX', 4.25, w, exit_reason='stop_loss')
        events = monitor.drain_exit_events()
        assert len(events) == 1
        assert events[0].exit_reason == 'stop_loss_unconfirmed'

    @pytest.mark.asyncio
    async def test_limit_race_branch_keeps_plain_stop_loss_reason(
        self, monitor, mock_alpaca
    ):
        """If the primary limit fills during our cancel attempt, the
        limit did its job — keep 'stop_loss' as the reason rather than
        labeling this a fallback."""
        # After cancel attempt, limit is reported filled.
        def _get_order(order_id):
            if order_id == 'sell-order-123':
                # First poll iteration sees 'new' (triggering escalation),
                # then during escalation's cancel-race check it's 'filled'.
                if getattr(_get_order, 'polled', False):
                    return {'id': 'sell-order-123', 'status': 'filled',
                            'filled_avg_price': 4.22, 'filled_qty': 500}
                _get_order.polled = True
                return {'id': 'sell-order-123', 'status': 'new',
                        'filled_avg_price': None, 'filled_qty': 0}
            return {'id': order_id, 'status': 'new'}
        mock_alpaca.get_order.side_effect = _get_order

        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['PLYX']
            w.latest_bid = 4.26; w.latest_ask = 4.27
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit('PLYX', 4.25, w, exit_reason='stop_loss')
        events = monitor.drain_exit_events()
        assert len(events) == 1
        assert events[0].exit_reason == 'stop_loss'
        assert events[0].exit_price == pytest.approx(4.22, abs=0.01)


class TestMarketCloseExtendedTimeout:
    """GSIT 2026-05-11 regression: market close on thin stocks can take
    30-60s to fill (walking the book). Old 10s timeout fired ERROR + wrote
    trigger_price as last-resort. Verifies extended 60s budget +
    final-retry catches the actual fill, and that the giveup-path uses
    WARNING (not ERROR)."""

    @pytest.mark.asyncio
    async def test_market_close_uses_extended_timeout_constant(
        self, monitor, mock_alpaca
    ):
        """The market-close poll must use _MARKET_CLOSE_FILL_TIMEOUT_S,
        not the shorter _STOP_EXIT_FILL_TIMEOUT_S used for the limit poll.
        Verified by patching _poll_order_fill and asserting timeout_s
        kwarg matches."""
        # Limit poll: always 'new' → escalation.
        # Market-close poll: capture kwargs.
        seen_timeouts = []

        original_poll = monitor._poll_order_fill

        async def _capture_poll(client, order_id, fallback_price,
                                timeout_s=None):
            seen_timeouts.append((order_id, timeout_s))
            return await original_poll(
                client, order_id, fallback_price=fallback_price,
                timeout_s=timeout_s,
            )
        monitor._poll_order_fill = _capture_poll

        # Limit never fills → escalate. Market close also stays 'new' →
        # giveup. Test setup differentiates by order id below.
        def _get_order(order_id):
            return {'id': order_id, 'status': 'new',
                    'filled_avg_price': None, 'filled_qty': 0}
        mock_alpaca.get_order.side_effect = _get_order

        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['PLYX']
            w.latest_bid = 4.26; w.latest_ask = 4.27
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit(
            'PLYX', 4.25, w, exit_reason='stop_loss',
        )

        # Two polls expected: limit (sell-order-123) and market close
        # (close-order-456). Limit uses default ≤ _STOP_EXIT_FILL_TIMEOUT_S,
        # market uses _MARKET_CLOSE_FILL_TIMEOUT_S.
        timeouts_by_order = dict(seen_timeouts)
        assert 'sell-order-123' in timeouts_by_order
        assert 'close-order-456' in timeouts_by_order
        # Limit poll: default (None → falls back to _STOP_EXIT_FILL_TIMEOUT_S)
        assert timeouts_by_order['sell-order-123'] in (
            None, monitor._STOP_EXIT_FILL_TIMEOUT_S
        )
        # Market close: must be the longer constant.
        assert timeouts_by_order['close-order-456'] == (
            monitor._MARKET_CLOSE_FILL_TIMEOUT_S
        )

    @pytest.mark.asyncio
    async def test_market_close_final_retry_catches_late_fill(
        self, monitor, mock_alpaca, caplog
    ):
        """Order returns 'new' throughout the poll loop, then 'filled' on
        the post-timeout final retry. Should use the real fill price (not
        trigger) and emit BRANCH_MARKET_CLOSE (not LAST_RESORT)."""
        # Track calls. Poll iterations all return 'new'. The final
        # one-shot retry (after _poll_order_fill returns None) returns
        # 'filled'. Counter-driven side_effect: every call to mkt order
        # returns 'new' until we've returned at least N times — then
        # 'filled'.
        poll_calls = {'sell-order-123': 0, 'close-order-456': 0}

        def _get_order(order_id):
            poll_calls[order_id] = poll_calls.get(order_id, 0) + 1
            # Sell-order-123 always 'new' → triggers escalation.
            if order_id == 'sell-order-123':
                return {'id': order_id, 'status': 'new',
                        'filled_avg_price': None, 'filled_qty': 0}
            # close-order-456: 'new' during poll loop, 'filled' on final retry.
            # With _MARKET_CLOSE_FILL_TIMEOUT_S=0.2 + poll interval 0.05,
            # we see ~4 poll-loop calls. Switch to 'filled' on call #5+.
            if poll_calls[order_id] >= 5:
                return {'id': order_id, 'status': 'filled',
                        'filled_avg_price': 10.31, 'filled_qty': 500}
            return {'id': order_id, 'status': 'new',
                    'filled_avg_price': None, 'filled_qty': 0}
        mock_alpaca.get_order.side_effect = _get_order

        monitor.add_watch('GSIT', 10.85, 500, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['GSIT']
            w.latest_bid = 10.60; w.latest_ask = 10.75
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit(
            'GSIT', 10.63, w, exit_reason='stop_loss',
        )

        events = monitor.drain_exit_events()
        assert len(events) == 1
        ev = events[0]
        # Final retry recovered the real fill.
        assert ev.exit_price == pytest.approx(10.31, abs=0.01)
        # Branch is MARKET_CLOSE (not LAST_RESORT)
        assert ev.exit_reason == 'stop_loss_market_fallback'

    @pytest.mark.asyncio
    async def test_market_close_giveup_logs_warning_not_error(
        self, monitor, mock_alpaca, caplog
    ):
        """When even the final retry can't confirm, the log level must
        be WARNING — not ERROR. The trigger_price is a placeholder that
        sync_positions() reconciles; not an operator-action alert."""
        import logging as _logging
        caplog.set_level(_logging.WARNING, logger='trading.stop_monitor')

        # All polls + final retry stay 'new' → giveup path.
        mock_alpaca.get_order.return_value = {
            'id': 'x', 'status': 'new',
            'filled_avg_price': None, 'filled_qty': 0,
        }

        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['PLYX']
            w.latest_bid = 4.26; w.latest_ask = 4.27
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit(
            'PLYX', 4.25, w, exit_reason='stop_loss',
        )

        # Find the "did not confirm fill" record — it must be WARNING.
        confirm_records = [
            r for r in caplog.records
            if 'did not confirm fill' in r.getMessage()
        ]
        assert len(confirm_records) >= 1, (
            f"expected 'did not confirm fill' log; got: "
            f"{[r.getMessage() for r in caplog.records]}"
        )
        for r in confirm_records:
            assert r.levelno == _logging.WARNING, (
                f"expected WARNING for unconfirmed market close, "
                f"got {_logging.getLevelName(r.levelno)}: {r.getMessage()}"
            )
        # Message should mention the deferral plan (sync_positions).
        assert any(
            'sync_positions' in r.getMessage() for r in confirm_records
        ), "expected deferral note pointing at sync_positions()"

        # Event still emitted with trigger_price placeholder + unconfirmed
        # reason — preserves event-consumer contract.
        events = monitor.drain_exit_events()
        assert len(events) == 1
        assert events[0].exit_reason == 'stop_loss_unconfirmed'
        assert events[0].exit_price == pytest.approx(4.25, abs=0.01)


class TestExitQtyReconciliation:
    """APT/MLTX 2026-05-11 defense-in-depth: stop_monitor re-queries broker
    position qty right before the sell. If broker has MORE shares than
    watch.shares (the orphan-residual signature), use the broker qty so we
    don't leave shares behind."""

    @pytest.mark.asyncio
    async def test_uses_broker_qty_when_higher_than_watch(
        self, monitor, mock_alpaca, caplog
    ):
        """Broker has 5153 sh; watch.shares=2257 (the APT signature).
        The sell must submit with qty=5153, not 2257, and the event's
        shares field must reflect 5153."""
        import logging as _logging
        caplog.set_level(_logging.WARNING, logger='trading.stop_monitor')

        # Broker says position is 5153 (parent kept filling after partial accept)
        mock_alpaca.get_open_positions.return_value = [
            {'symbol': 'APT', 'qty': 5153, 'avg_entry_price': 7.26,
             'side': 'long', 'market_value': 36000.0,
             'unrealized_pl': -1900.0, 'unrealized_plpc': -0.05},
        ]
        # Happy-path order fill so we don't escalate.
        mock_alpaca.get_order.return_value = {
            'id': 'sell-order-123', 'status': 'filled',
            'filled_avg_price': 7.01, 'filled_qty': 5153,
        }

        # watch.shares set to the (too-low) partial.
        monitor.add_watch('APT', 7.05, 2257, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['APT']
            w.latest_bid = 7.00; w.latest_ask = 7.09
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit(
            'APT', 7.04, w, exit_reason='stop_loss',
        )

        # Sell-order submission used 5153 (the broker view), not 2257.
        mock_alpaca.submit_limit_sell_order.assert_called_once()
        sell_kwargs = mock_alpaca.submit_limit_sell_order.call_args.kwargs
        assert sell_kwargs['qty'] == 5153, (
            f"expected qty=5153 (broker view), got {sell_kwargs.get('qty')}"
        )
        # A mismatch WARNING was logged
        mismatch_logs = [
            r for r in caplog.records
            if 'qty mismatch' in r.getMessage()
        ]
        assert len(mismatch_logs) >= 1, (
            "expected 'qty mismatch' warning when broker_qty != watch.shares"
        )
        # The emitted event also carries the reconciled qty (for P&L math)
        events = monitor.drain_exit_events()
        assert len(events) == 1
        assert events[0].shares == 5153

    @pytest.mark.asyncio
    async def test_uses_watch_shares_when_broker_matches(
        self, monitor, mock_alpaca, caplog
    ):
        """Broker view matches watch.shares — no mismatch warning, sell uses
        the matched qty. The reconcile check should be silent on the happy
        path."""
        import logging as _logging
        caplog.set_level(_logging.WARNING, logger='trading.stop_monitor')

        mock_alpaca.get_open_positions.return_value = [
            {'symbol': 'PLYX', 'qty': 500, 'avg_entry_price': 4.29,
             'side': 'long', 'market_value': 2150.0,
             'unrealized_pl': -25.0, 'unrealized_plpc': -0.01},
        ]
        mock_alpaca.get_order.return_value = {
            'id': 'sell-order-123', 'status': 'filled',
            'filled_avg_price': 4.22, 'filled_qty': 500,
        }
        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['PLYX']
            w.latest_bid = 4.26; w.latest_ask = 4.27
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit(
            'PLYX', 4.25, w, exit_reason='stop_loss',
        )

        # No mismatch warning — broker matched
        mismatch_logs = [
            r for r in caplog.records
            if 'qty mismatch' in r.getMessage()
        ]
        assert len(mismatch_logs) == 0, (
            f"unexpected mismatch warning on match path: "
            f"{[r.getMessage() for r in mismatch_logs]}"
        )
        mock_alpaca.submit_limit_sell_order.assert_called_once()
        sell_kwargs = mock_alpaca.submit_limit_sell_order.call_args.kwargs
        assert sell_kwargs['qty'] == 500

    @pytest.mark.asyncio
    async def test_falls_back_to_watch_shares_on_query_error(
        self, monitor, mock_alpaca, caplog
    ):
        """If the broker position re-query raises, fall back to watch.shares
        so the exit still proceeds. Log a warning so this case is observable
        but don't block the sell."""
        import logging as _logging
        caplog.set_level(_logging.WARNING, logger='trading.stop_monitor')

        mock_alpaca.get_open_positions.side_effect = Exception(
            "alpaca API timeout"
        )
        mock_alpaca.get_order.return_value = {
            'id': 'sell-order-123', 'status': 'filled',
            'filled_avg_price': 4.22, 'filled_qty': 500,
        }
        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')
        with monitor._watch_lock:
            w = monitor._watches['PLYX']
            w.latest_bid = 4.26; w.latest_ask = 4.27
            w.latest_quote_ts = time.time()

        await monitor._execute_stop_exit(
            'PLYX', 4.25, w, exit_reason='stop_loss',
        )

        # Fell back to watch.shares=500
        mock_alpaca.submit_limit_sell_order.assert_called_once()
        sell_kwargs = mock_alpaca.submit_limit_sell_order.call_args.kwargs
        assert sell_kwargs['qty'] == 500
        # Fallback warning was logged (don't fail silently)
        fallback_logs = [
            r for r in caplog.records
            if 'position re-query failed' in r.getMessage()
        ]
        assert len(fallback_logs) >= 1
