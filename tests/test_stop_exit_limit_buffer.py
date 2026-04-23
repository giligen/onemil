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
    async def test_sl_leg_only_cancelled_after_fill_confirmed(self, monitor, mock_alpaca):
        """SL leg must survive until we've confirmed the position is flat —
        it's the catastrophic-protection backstop during market-close
        escalation. If we cancelled SL on submit and the limit failed to
        fill AND market close also failed, we'd be naked."""
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

        # SL leg ('sl-leg') cancel must come AFTER limit submit in the order log.
        # Specifically: TP cancel first, then later SL cancel — SL must not be
        # the 1st or 2nd cancel (that would put it pre-fill-confirmation).
        assert 'tp-leg' in cancel_log
        assert 'sl-leg' in cancel_log
        tp_idx = cancel_log.index('tp-leg')
        sl_idx = cancel_log.index('sl-leg')
        assert sl_idx > tp_idx, (
            f"SL leg cancelled at position {sl_idx}, TP at {tp_idx} — "
            f"SL must cancel AFTER fill is confirmed (full log: {cancel_log})"
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
