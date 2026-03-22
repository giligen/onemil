# Sell Limit Optimization Ideas

## Idea 1: Bid-Leading Exit (deferred — needs data)

**Concept**: Instead of waiting for a trade to confirm reversal and trigger the trailing stop, monitor the NBBO bid in real-time as a leading indicator. When the bid starts falling consecutively (3+ ticks), spreads widen rapidly, or the bid itself reaches the trailing TP level before the last trade does, exit preemptively with a limit order at the current bid.

**Why it should work**: When the bid hits the stop level, active buyers are still there at that price — selling into existing demand rather than chasing a deteriorating market. Exits 5-15 seconds ahead of trade-triggered stops, at better prices and tighter spreads.

**Conditions for implementation**:
- Bid must decline by > 1 ATR (not just 3 ticks — too noisy on low-float)
- Spread must widen > 2× the 5-minute average spread
- Bid_size must collapse below 50% of recent average
- Only fire if already in profit (never on entries)

**Why deferred**:
1. We have zero historical tick-level quote data — just started collecting (2026-03-22)
2. "3 consecutive falling bid ticks" has ~20% accuracy on $3-10 low-float stocks — constant bouncing
3. False positive exit on a +5R runner costs $10K; true positive early exit saves $150. Asymmetric risk.
4. Mathematically equivalent to tightening trail by spread width — simpler alternative exists

**Prerequisites**: 2-4 weeks of quote stream data (now collecting via StopMonitor._on_quote). Then analyze:
- `SELECT exit_pricing_method, AVG(exit_slippage), AVG(exit_fill_latency_ms), AVG(exit_quote_spread) FROM trades WHERE exit_quote_bid IS NOT NULL GROUP BY exit_pricing_method`
- How many seconds before trade-triggered stop does bid hit that level?
- What % of 3-declining-bid-tick events lead to real reversals on our stocks?

## Idea 2: Tighter Trail on Wide Spreads (testable now)

**Concept**: When the spread is wide (>$0.10), the cost of the trailing stop triggering is higher because we sell into a worse bid. Tighten the trail from 1.0R to 0.8R when conditions are deteriorating. This captures the bid-leading concept without tick-level complexity.

**Implementation**: In backtest TradeSimulator, if the current bar has wide body/wick suggesting volatility, use 0.8R trail instead of 1.0R. Or simpler: just test 0.8R globally with a fixed 0.2R slippage model (separate from the entry slippage).

**Status**: Ready to backtest.

## Idea 3: Adaptive Spread-Based Limit Pricing (live, collecting data)

**Current implementation**: StopMonitor uses spread tiers (tight <$0.05 → midpoint, medium $0.05-$0.15 → bid+$0.01, wide >$0.15 → bid). Now collecting real execution data (exit_slippage, exit_fill_latency_ms, exit_pricing_method) to validate and optimize tier boundaries.

## Idea 4: L2 Depth-Aware Sizing (future, needs $99/mo upgrade)

**Concept**: Before submitting a sell, check L2 order book depth. If total bid depth within 0.5% of price < our sell qty, reduce order size or split into tranches. Prevents blowing through the book on thin names.

**Prerequisite**: Alpaca Algo Trader Plus ($99/mo) for L2 WebSocket, or Polygon.io. Not needed for paper trading — paper fills are instant.
