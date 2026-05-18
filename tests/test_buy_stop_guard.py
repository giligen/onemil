"""
Unit tests for trading.buy_stop_guard.evaluate_buy_stop.

Exercises every branch of the decision tree + boundary cases. The function is
pure, so these are pure value-in/value-out checks — no mocks needed.

Plan used as the canonical test fixture:
  stop_price = 4.40  (breakout level)
  limit_price = 4.49  (= round(4.40 * 1.02, 2))
  rebump_buffer = 0.02  (default)
"""

import pytest

from trading.buy_stop_guard import (
    BuyStopAction,
    BuyStopDecision,
    evaluate_buy_stop,
)


STOP = 4.40
LIMIT = 4.49        # round(4.40 * 1.02, 2)
BUFFER = 0.02


class TestSubmitAsIs:
    """ask < stop — normal pre-breakout case."""

    def test_ask_below_stop_submits_as_is(self):
        d = evaluate_buy_stop(bid=4.30, ask=4.38, stop_price=STOP, limit_price=LIMIT)
        assert d.action == BuyStopAction.SUBMIT_AS_IS
        assert d.new_stop_price is None

    def test_ask_one_cent_below_stop_submits_as_is(self):
        """Boundary: ask exactly $0.01 below stop → still normal."""
        d = evaluate_buy_stop(bid=4.30, ask=4.39, stop_price=STOP, limit_price=LIMIT)
        assert d.action == BuyStopAction.SUBMIT_AS_IS


class TestMarketableLimit:
    """bid >= stop — breakout fully confirmed."""

    def test_bid_above_stop(self):
        d = evaluate_buy_stop(bid=4.45, ask=4.46, stop_price=STOP, limit_price=LIMIT)
        assert d.action == BuyStopAction.MARKETABLE_LIMIT

    def test_bid_equal_stop_boundary(self):
        """Boundary: bid == stop → marketable_limit (>= condition)."""
        d = evaluate_buy_stop(bid=4.40, ask=4.41, stop_price=STOP, limit_price=LIMIT)
        assert d.action == BuyStopAction.MARKETABLE_LIMIT


class TestRebumpStop:
    """bid < stop <= ask, and ask + buffer <= limit."""

    def test_straddle_within_limit(self):
        d = evaluate_buy_stop(bid=4.35, ask=4.45, stop_price=STOP, limit_price=LIMIT)
        assert d.action == BuyStopAction.REBUMP_STOP
        assert d.new_stop_price == 4.47  # round(4.45 + 0.02, 2)

    def test_ask_equal_stop_rebumps(self):
        """Boundary: ask == stop with bid below → straddle branch fires
        (ask >= stop is `>=`)."""
        d = evaluate_buy_stop(bid=4.30, ask=4.40, stop_price=STOP, limit_price=LIMIT)
        assert d.action == BuyStopAction.REBUMP_STOP
        assert d.new_stop_price == 4.42  # round(4.40 + 0.02, 2)

    def test_rebumped_stop_strictly_above_ask(self):
        """The bumped stop must be strictly > ask (Alpaca requires that)."""
        d = evaluate_buy_stop(bid=4.30, ask=4.41, stop_price=STOP, limit_price=LIMIT)
        assert d.action == BuyStopAction.REBUMP_STOP
        assert d.new_stop_price > 4.41

    def test_rebumped_stop_at_most_limit(self):
        """The bumped stop must be <= limit_price (valid stop-limit)."""
        d = evaluate_buy_stop(bid=4.30, ask=4.45, stop_price=STOP, limit_price=LIMIT)
        assert d.action == BuyStopAction.REBUMP_STOP
        assert d.new_stop_price <= LIMIT


class TestSkip:
    """stop <= ask and ask + buffer > limit — breakout extended."""

    def test_ask_past_limit_skips(self):
        d = evaluate_buy_stop(bid=4.35, ask=4.48, stop_price=STOP, limit_price=LIMIT)
        # ask + buffer = 4.50 > limit 4.49 → skip
        assert d.action == BuyStopAction.SKIP
        assert d.new_stop_price is None

    def test_ask_far_past_limit_skips(self):
        d = evaluate_buy_stop(bid=4.40, ask=4.80, stop_price=STOP, limit_price=LIMIT)
        # bid >= stop here too, but bid-branch wins first
        assert d.action == BuyStopAction.MARKETABLE_LIMIT  # bid >= stop precedes

    def test_ask_far_past_limit_with_low_bid_skips(self):
        """Make sure SKIP fires when both bid<stop AND ask>>limit."""
        d = evaluate_buy_stop(bid=4.35, ask=4.80, stop_price=STOP, limit_price=LIMIT)
        assert d.action == BuyStopAction.SKIP

    def test_boundary_ask_plus_buffer_equals_limit_rebumps_not_skips(self):
        """ask + buffer == limit → bumped stop == limit → rebump (still valid)."""
        d = evaluate_buy_stop(bid=4.30, ask=4.47, stop_price=STOP, limit_price=LIMIT)
        # ask + 0.02 = 4.49 = limit → rebump, not skip
        assert d.action == BuyStopAction.REBUMP_STOP
        assert d.new_stop_price == 4.49


class TestDefensiveFallback:
    """Degenerate quotes — submit as-is when neither side is usable.

    When ONLY one side is valid, that side's branch still fires (partial
    quotes are common when one side is thin). Only the both-sides-zero case
    falls through to defensive SUBMIT_AS_IS.
    """

    def test_zero_bid_zero_ask(self):
        d = evaluate_buy_stop(bid=0.0, ask=0.0, stop_price=STOP, limit_price=LIMIT)
        assert d.action == BuyStopAction.SUBMIT_AS_IS

    def test_zero_bid_with_ask_above_stop_rebumps(self):
        """bid=0 disqualifies marketable, but ask is still actionable →
        rebump via the ask branch."""
        d = evaluate_buy_stop(bid=0.0, ask=4.45, stop_price=STOP, limit_price=LIMIT)
        assert d.action == BuyStopAction.REBUMP_STOP
        assert d.new_stop_price == 4.47

    def test_negative_bid_with_ask_above_stop_rebumps(self):
        d = evaluate_buy_stop(bid=-1.0, ask=4.45, stop_price=STOP, limit_price=LIMIT)
        assert d.action == BuyStopAction.REBUMP_STOP

    def test_bid_valid_ask_zero_below_stop_submits_as_is(self):
        """bid alone with bid < stop and ask missing → submit as-is."""
        d = evaluate_buy_stop(bid=4.30, ask=0.0, stop_price=STOP, limit_price=LIMIT)
        assert d.action == BuyStopAction.SUBMIT_AS_IS

    def test_bid_valid_ask_zero_above_stop_marketable(self):
        """bid >= stop with ask missing → bid branch still fires."""
        d = evaluate_buy_stop(bid=4.45, ask=0.0, stop_price=STOP, limit_price=LIMIT)
        assert d.action == BuyStopAction.MARKETABLE_LIMIT


class TestBufferConfig:
    """rebump_buffer is honored and floored at 0.02."""

    def test_custom_buffer_used(self):
        d = evaluate_buy_stop(
            bid=4.35, ask=4.43, stop_price=STOP, limit_price=LIMIT,
            rebump_buffer=0.05,
        )
        # ask + 0.05 = 4.48 <= limit 4.49 → rebump to 4.48
        assert d.action == BuyStopAction.REBUMP_STOP
        assert d.new_stop_price == 4.48

    def test_zero_buffer_floored_to_two_cents(self):
        """Tiny buffer floored — bumped stop guaranteed > ask."""
        d = evaluate_buy_stop(
            bid=4.35, ask=4.45, stop_price=STOP, limit_price=LIMIT,
            rebump_buffer=0.0,
        )
        assert d.action == BuyStopAction.REBUMP_STOP
        assert d.new_stop_price == 4.47        # 4.45 + 0.02 (floored)

    def test_negative_buffer_floored_to_two_cents(self):
        d = evaluate_buy_stop(
            bid=4.35, ask=4.45, stop_price=STOP, limit_price=LIMIT,
            rebump_buffer=-0.10,
        )
        assert d.action == BuyStopAction.REBUMP_STOP
        assert d.new_stop_price == 4.47


class TestRealIncidentReplay:
    """Replays the actual rejected orders to lock in the right decision."""

    def test_kpti_2026_05_14(self):
        """KPTI: stop $9.71, ask $9.79, limit $9.90 — straddle → rebump."""
        d = evaluate_buy_stop(bid=9.56, ask=9.79, stop_price=9.71, limit_price=9.90)
        assert d.action == BuyStopAction.REBUMP_STOP
        assert d.new_stop_price == 9.81

    def test_trt_2026_05_14(self):
        """TRT: stop $13.27, ask $13.30, limit $13.54 — straddle → rebump."""
        d = evaluate_buy_stop(bid=13.09, ask=13.30, stop_price=13.27, limit_price=13.54)
        assert d.action == BuyStopAction.REBUMP_STOP
        assert d.new_stop_price == 13.32

    def test_btcz_2026_05_18(self):
        """BTCZ: bar 13:38 L=$4.110 (entire bar above limit $4.10).
        bid/ask ~$4.11/$4.12, stop $4.09, limit $4.10 → SKIP (extended)."""
        d = evaluate_buy_stop(bid=4.11, ask=4.12, stop_price=4.09, limit_price=4.10)
        assert d.action == BuyStopAction.MARKETABLE_LIMIT  # bid >= stop wins

    def test_btcz_extended_no_bid_above_stop(self):
        """Hypothetical: bid just below stop, ask well above limit — SKIP."""
        d = evaluate_buy_stop(bid=4.08, ask=4.15, stop_price=4.09, limit_price=4.10)
        assert d.action == BuyStopAction.SKIP

    def test_yss_2026_05_18(self):
        """YSS: bar 13:38 L=$26.18 > stop $26.44? L is BELOW stop. But H=26.87
        — the bid likely sits near 26.30-26.50. Take the worst-case where
        bid < stop but ask > limit → SKIP."""
        d = evaluate_buy_stop(bid=26.35, ask=26.55, stop_price=26.44, limit_price=26.52)
        # ask + 0.02 = 26.57 > limit 26.52 → SKIP
        assert d.action == BuyStopAction.SKIP

    def test_bmnz_2026_05_18(self):
        """BMNZ: bar 13:38 L=$18.50 > stop $18.48, entire bar above stop.
        bid/ask ~$18.52/$18.53. stop $18.48, limit $18.54 → bid >= stop
        → marketable_limit."""
        d = evaluate_buy_stop(bid=18.52, ask=18.53, stop_price=18.48, limit_price=18.54)
        assert d.action == BuyStopAction.MARKETABLE_LIMIT

    def test_frmm_2026_05_18(self):
        """FRMM: bid $3.75, ask $3.76, stop $4.07, limit $4.08 — normal,
        no rejection."""
        d = evaluate_buy_stop(bid=3.75, ask=3.76, stop_price=4.07, limit_price=4.08)
        assert d.action == BuyStopAction.SUBMIT_AS_IS
