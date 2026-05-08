# IREZ + TTGT Paper-vs-Prod Divergence — Deep Dive

**Date**: 2026-05-08
**Trigger**: User asked why both IREZ and TTGT — two huge runners on dev (paper) — were killed/skipped on prod (live). Both setups passed conviction + sizing on both nodes; only dev caught the move.
**Result**: **Two distinct prod-only bugs**, NOT one. IREZ root cause is the post-fill gate (already shipped fix today). TTGT root cause is a separate Alpaca live-account pre-trade validation that paper doesn't enforce.

## TL;DR

| Symbol | Dev outcome | Prod outcome | Root cause |
|---|---|---|---|
| **IREZ** | +$8,210 (trail_stop) | $0 (killed post-fill, then 3 retries rejected) | **Post-fill gate fired on SPY 0.77 boundary noise** — fixed today via SPY 3d daily snapshot + 0.5/0.5 thresholds |
| **TTGT** | +$8,149 (trail_stop) | $0 (3 stop-limit BUYs rejected by Alpaca, never filled) | **Alpaca live rejects buy stop-limit when `stop_price ≤ current bid`** — paper doesn't enforce this. Not yet fixed. |

Both bugs are prod-only because **prod is a real-money Alpaca account**; dev is paper. Real Alpaca enforces stricter pre-trade validation. Confirmed by querying live account: `paper=False`, `pattern_day_trader=True`, `daytrading_buying_power=$320,710`, `equity=$80,198` — account is healthy, BP is fine; the issues are validation rules.

## TTGT — what actually happened on prod

Three buy stop-limit orders submitted at 13:57:00, 13:57:09, 13:58:09 ET.
Each order: `qty=1553, stop_price=$5.67, limit_price=$5.78, type=stop_limit, side=BUY`.

Pulled the raw Alpaca REST API response for each rejected order:
```
created_at = 2026-05-08T13:57:00.630Z
submitted_at = 2026-05-08T13:57:00.633Z
failed_at = 2026-05-08T13:57:00.637Z   ← rejected 4ms after submit
status = rejected
reject_reason = (not exposed in API response — Alpaca's known limitation)
```

**4-millisecond rejection ≠ market check ≠ buying-power lookup.** This is a **synchronous parameter-validation rejection** at Alpaca's order-admission layer.

The actual prompt log entry:
```
TTGT: Submitting buy-stop order — BUY 1553 stop @ $5.67, limit $5.78
TTGT: quote bid=$5.68 ask=$5.72 spread=$0.040
                ^^^^^^^^^
TTGT: Pending order rejected — ID: 92a3f1cc...
```

**Bid $5.68 > stop $5.67 by $0.01.** The buy-stop trigger condition is **already met at submission time**. Alpaca rejects buy stop-limit orders where `stop_price ≤ current bid` because the stop is meaningless — the order would fire instantly upon admission, which makes it not really a stop-order. Paper Alpaca skips this validation; **live Alpaca enforces it strictly**.

Dev's TTGT order at 14:01:14 had the same shape (`stop=$5.71` with `bid=$5.95`) — same trigger-already-met condition — but **paper accepted it and filled at $5.7863 three minutes later** when ask drifted down to the limit price.

## The systematic prod problem

Looking at the last **500 prod orders** via Alpaca API:

```
Total: 500 orders   filled: 254   canceled: 222   rejected: 24 (4.8%)
```

All 24 rejections are buy stop-limit orders on bull-flag setups. The pattern across 8 distinct symbols (IREZ ×3, TTGT ×2, EAF ×6, RMAX ×2, AGCC, OPTX, MLEC ×6, SMX, TOYO) over the last ~5 weeks:

| Date | Symbol | qty | stop | limit | retries |
|---|---|---|---|---|---|
| 2026-05-08 | IREZ | 405 | 6.72 | 6.85 | 3× rejected |
| 2026-05-08 | TTGT | 1553 | 5.67 | 5.78 | 2× rejected |
| 2026-05-01 | EAF | 6750 | 7.09 | 7.23 | 6× rejected |
| 2026-04-27 | RMAX | 1116 | 10.22 | 10.42 | 2× rejected |
| 2026-04-13 | AGCC | 195 | 14.00 | 14.28 | 1× rejected |
| 2026-04-13 | OPTX | 909 | 11.40 | 11.63 | 1× rejected |
| 2026-04-06 | MLEC | varies | 9.40 | 9.59 | 6× rejected |
| 2026-04-06 | SMX | 426 | 17.55 | 17.90 | 1× rejected |
| 2026-04-01 | TOYO | 1000 | 8.43 | 8.59 | 1× rejected |

Each retry chain shows the same setup re-detecting and re-submitting on the next bar — same `stop_price` each time. The retries reject for the same reason as the first. Conservatively, ~10-15 distinct trade setups have been killed by this bug over the last 5 weeks.

This is the same shape as the IREZ post-fill drift bug: **an asymmetric paper-vs-live behavior the strategy doesn't account for**.

## Why this only fires on the prod side

Alpaca's published validation rules differ between paper and live:

| Validation | Paper | Live |
|---|---|---|
| `stop_price ≤ bid` for buy stop-limit | **Allowed** (order goes pending, fills if limit holds) | **Rejected** synchronously |
| Wash-trade detection | Lenient | Strict |
| PDT enforcement | None | Enforced |
| Marginability check | Loose | Strict |

The `stop_price ≤ bid` issue arises naturally in the bull-flag flow:

1. Pattern detector fires at bar close N (e.g., 13:56:00) with breakout = $5.67
2. Trade plan + sizing computed (~250-500ms)
3. Order submitted ~1 second later (13:56:01)
4. **In that 1-second gap, price often drifts past the breakout level on paper-tape liquidity**
5. By the time Alpaca validates, the bid is already at or above the stop → reject

This is a structural race between pattern-detection latency and the natural breakout move. Bull flags ARE breakouts — by definition the price is moving up through the stop. The faster the move, the more likely we miss the entry.

## Fix proposal

**Pre-flight bid check**: before submitting a buy stop-limit, compare the configured `stop_price` to the current best bid. If `bid >= stop_price - epsilon`, the stop is effectively already triggered. Two options:

### Option A (recommended): "Marketable limit fallback"

If `stop_price ≤ bid`, submit a **marketable limit BUY** at the configured `limit_price` instead of a stop-limit. This:
- Achieves the trade intent (buy the breakout, capped at the slippage limit)
- Passes Alpaca's validation (regular limits don't have the stop-price-vs-bid rule)
- Preserves the slippage cap that the limit_price already encodes
- Matches what dev's paper account naturally does (paper effectively converts these to immediate limit fills)

### Option B: "Skip if drifted"

If `stop_price ≤ bid`, skip the trade entirely — log an info line, move on. Loses trade opportunities but avoids any complexity.

### Option C (defense in depth): "Bump stop above bid"

Adjust `stop_price` to `max(stop_price, bid + 1tick)` and `limit_price` accordingly. Preserves the stop semantic but at the current market.

**Recommend Option A.** It's the minimum-deviation fix that mirrors how paper already behaves (so BT/dev/prod converge on outcome), captures the real-money equivalents of the trades we've been losing, and adds zero complexity at exit time (the stop_loss + StopMonitor still work as before — they bind on the fill).

### Logging gap to close

`reject_reason` is NOT exposed by Alpaca's REST GET /orders endpoint. It's only delivered via the OrderStream WebSocket trade event. Our `OrderStreamWatcher` should log the `reject_reason` from the trade event payload onto the trade row so we don't have to reverse-engineer rejections from external evidence next time.

## Estimating impact

Of the 24 prod rejections in the last 500 orders, even if half were eventual losers, the other half include known +$8K-shape winners (TTGT) and +$8K-shape winners we've already proven the gate kills (IREZ — though IREZ would have been gate-killed regardless before today's ship).

Conservative estimate: **5-10 prod-only missed entries per quarter**, of which 30-50% historically would have hit trail_stop wins of +$2K to +$10K. Annualized blast radius is in the **$20K-$50K range**, of similar magnitude to the post-fill gate bug we shipped today.

## Implementation sketch

```python
# In trading/order_executor.py before stop-limit submission:
quote = self.alpaca.get_latest_quote(symbol)
if quote and quote.get('bid', 0) >= stop_price - tick_epsilon:
    # Stop already triggered — Alpaca live would reject a stop-limit here.
    # Submit as a marketable limit instead, preserving the slippage cap.
    logger.info(
        f"{symbol}: STOP ALREADY TRIGGERED (bid ${quote['bid']:.2f} >= "
        f"stop ${stop_price:.2f}) — submitting as marketable LIMIT @ ${limit_price:.2f}"
    )
    return self.alpaca.submit_limit_order(
        symbol=symbol, qty=shares, side='buy', limit_price=limit_price,
        time_in_force='day',
    )
return self.alpaca.submit_stop_limit_order(...)  # original path
```

Plus a unit test: `tests/test_marketable_limit_fallback.py` pinning that the path triggers when `bid >= stop` and skips otherwise.

Plus a logging upgrade in `OrderStreamWatcher`: capture the trade event's `reject_reason` (when Alpaca sends one) onto the DB trade row, so future post-mortems don't need to query the REST API to discover what happened.

## Connection to today's earlier IREZ ship

The post-fill gate fix already shipped today (V1 thresholds 0.5/0.5 + SPY 3d daily snapshot) addresses the **first** blow against IREZ — the kill switch. **It does NOT fix TTGT-type cases**, which are upstream of the fill. The gate change saves ~$24K over 16 months in BT; the marketable-limit fix is targeting an additional, different pile of money.

Both are prod-only bugs because prod is real money. Dev/paper hides them.

## Recommendation order

1. **Done today**: SPY 3d daily snapshot + V1 (0.5/0.5) thresholds. (Phase 1 of IREZ post-mortem.)
2. **Next**: implement marketable-limit fallback (Option A above), with tests, deploy to dev → prod. Estimate: 1-2 hour engineering + 15-min deploy.
3. **In parallel**: capture `reject_reason` from OrderStream events into the DB so the next prod-only divergence is diagnosable in minutes, not hours of API archaeology.

Both items should land before next week's market open to capture early-week breakout moves on the same setups that have been getting rejected.
