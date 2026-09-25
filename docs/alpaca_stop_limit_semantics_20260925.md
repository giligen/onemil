# Alpaca stop / stop-limit order semantics (pulled 2026-09-25)

Doc-only research for the HOD-break resting stop-limit entry design. No orders placed, no keys used.

## 1. What triggers an equity STOP order

Source: "Order Handling Standards at Alpaca Securities LLC", `docs.alpaca.markets/docs/orders-at-alpaca`:

> "Your sell stop order will only elect if there is a trade on the consolidated tape at or lower than your
> stop price and provided the electing trade is not outside of the NBBO."

> "Your buy stop order will only elect if there is a trade on the consolidated tape that is at or above your
> stop price that is not outside of the NBBO."

Trigger = a **consolidated-tape trade print**, not last-trade-only and not a single-venue quote. The NBBO filter
excludes prints that print outside the prevailing NBBO (the usual home for bad/erroneous sub-penny or busted
prints), but no language explicitly excludes odd-lot prints — an NBBO-compliant odd-lot print reads as eligible
to elect the stop. Treat any qualifying tape print, round-lot or not, as capable of triggering.

## 2. Internal hold vs. exchange routing

No page found states plainly whether Alpaca holds stop orders server-side until elected or sends a live stop to
an exchange/market-maker for them to trigger. The only adjacent language, from the same Order Handling Standards
page: "We do not necessarily route retail orders to the exchange, but will route orders to market makers who
will route orders on your behalf to the primary market opening auction" — that is about opening-auction routing,
not stop election. **Gap: mechanism is not documented; do not assume either model.**

## 3. Stop-limit price relationship / minimum distance / increments

Source: `docs.alpaca.markets/docs/orders-at-alpaca` (Stop Limit Order / bracket sections):

> "take_profit.limit_price must be higher than stop_loss.stop_price for a buy bracket order, and vice versa
> for a sell." (bracket-order context — not necessarily binding on a standalone stop-limit).

> "Stop price >=$1.00: Max Decimals = 2; Stop price <$1.00: Max Decimals = 4"

> "The stop price input has to be at least $0.01 below (for stop-loss sell, above for buy) than the 'base
> price'."

## 4. `replace_order` semantics

Source: `docs.alpaca.markets/reference/patchorderbyorderid-1`:

> "Replaces a single order with updated parameters. Each parameter overrides the corresponding attribute of
> the existing order." → response is "The new Order object with the new order ID."

> "A success return code from a replaced order does NOT guarantee the existing open order has been replaced.
> If the existing open order is filled before the replacing (new) order reaches the execution venue, the
> replacing (new) order is rejected."

**Order id changes on every replace.** No page states queue/time priority is preserved; given a new order id is
issued, treat replace as cancel-then-new at the venue — **no queue-position guarantee**.

## 5. Partial fills

`docs.alpaca.markets/docs/orders-at-alpaca` bracket-order note: "If the take-profit order is partially filled,
the stop-loss order will be adjusted to the remaining quantity." Trade-updates stream (secondary source,
direct page fetch 404'd — lower confidence): fills are pushed as discrete `trade_updates` events; a
`partial_fill` event fires "when a number of shares less than the total remaining quantity on your order has
been filled," each carrying its own fill price/qty. **Must aggregate multiple partial_fill events per order id,
never assume one fill message = full quantity.**

## 6. Untriggered day stop-limit at 16:00 ET

Source: `docs.alpaca.markets/docs/orders-at-alpaca` (Time in Force):

> "By default, the order is only valid during Regular Trading Hours (9:30am - 4:00pm ET). If unfilled after
> the closing auction, it is automatically canceled."

A `day` stop-limit that never elects is auto-cancelled at the close — no overnight carry, no separate action
needed to clear it.

## 7. Rate limits relevant to ~50 resting orders/hour

Source: `alpaca.markets/support/usage-limit-api-calls` (Alpaca support):

> "200 requests per minute, per account" — exceeding it returns "429 - Too Many Requests."

This is a blanket per-account API limit (not order-specific): submit + replace + cancel all draw from the same
200/min budget. 50 resting orders/hour with a few calls each (place, arm-adjust, cancel) is on the order of a
few requests/min — well inside budget — but bursts (e.g. re-arming many symbols on one bar close) must be
throttled/staggered, not fired simultaneously.

## Summary — what the HOD resting stop-limit design must assume

1. Trigger = any NBBO-compliant consolidated-tape print at/through the stop, including odd lots — do not gate
   on round-lot or single-venue prints only.
2. Alpaca's internal-hold-vs-route mechanism is undocumented — build no timing/priority assumption on it.
3. Validate stop distance (≥$0.01 from base) and decimal rules (2dp ≥$1, 4dp <$1) client-side before submit.
4. Treat `replace_order` as cancel-and-new: track the new order id, handle "old filled / new rejected" races,
   assume zero queue-priority carryover.
5. Aggregate fills from possibly multiple `partial_fill` trade_updates events per order id.
6. Day TIF auto-cancels untriggered stop-limits at 16:00 ET close — no manual EOD cleanup required.
7. Stagger order placement/replace/cancel calls; stay well under the 200 req/min account-wide cap.
