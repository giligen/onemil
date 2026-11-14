# PREREG 1,777 — H1 "copy the owner": his own discretionary trades on the LIVE account (read-only, report only)

Owner consent: "Approve" 2026-11-14 (reply to the end-of-day note naming H1). Hypothesis book row H1
(`research/hypotheses/BOOK_20261113.md`). Cell count on the programme: 1,777.

## Question
Do the owner's own fills on the live account carry a measurable edge, and if so in which slice (time of day, hold,
side, price tier, in-list vs off-list) — enough to codify a rule the machine can run?

## Data (all GET, live keys `ALPACA_API_KEY/ALPACA_API_SECRET` from `.env`, `load_dotenv(override=True)`)
* `GET /v2/account/activities/FILL` paginated (`after=2024-01-01`, `page_size=100`, `page_token`=last id) — every fill.
* `GET /v2/orders?status=closed&nested=true&after=...&limit=500` paginated by `created_at` — order_id → client_order_id,
  side, type, submitted_at. Reuse `scripts/hod_owner_list_score.py::fetch_orders/fetch_fills/is_ours/OUR_PREFIXES`.
* Ownership: a fill is the OWNER's when its order's client_order_id carries none of `OUR_PREFIXES` (bracket legs
  inherit the parent, `owner_order_ids`). Everything else (ours, `micha-*`) is excluded, counted once in the report.
* No secrets in any file; account numbers never written. Writes ONLY under `research/owner_trades/`.

## Frozen construction
1. Round trip = FIFO per symbol across the whole history (not per day): an opening fill and the closing fills that
   flatten it; multi-day holds allowed; a position still open at the last fill is marked at the last daily close in
   `daily_bars` and flagged `open`. Record: symbol, side (long/short), entry_ts, exit_ts, hold_min, entry_px, exit_px,
   qty, notional, pnl_usd, ret_bp (sign-adjusted), n_entry_fills, n_exit_fills, day (ET of entry).
2. Price tier at entry: <$1, 1–5, 5–20, 20–100, >100. Entry hour ET bucket: pre (<09:30), 09:30–10:00, 10:00–11:00,
   11:00–14:00, 14:00–16:00, post. Hold bucket: <5 min, 5–30, 30–120, 120–390 (intraday), overnight+.
3. Reads (pre-committed, no others): (a) n round trips, n days, span, total $; (b) mean/median ret_bp, win rate,
   day-clustered t, ex-top-5 % and worst-5 % mean; (c) the same per side, price tier, entry hour, hold bucket —
   reported ONLY as a table, a slice is named as "edge" only if n ≥ 30 AND day-t ≥ 2 AND ex-top-5 % > 0;
   (d) top-10 / bottom-10 trades by $ with their share of the total; (e) monthly $ and n.
4. Placebo (c') for every slice named under (c): same symbol, same day, same hold length, entry at a random other
   minute of the RTH session (bars from `research/bf_zero/bars_sip.db`, read-only; Saturday, no blackout) — 200 draws,
   report the placebo mean and its p95 next to the slice mean (binding lesson 11/13). Skipped if (c) names nothing.
5. No cost model: these are real fills, cost is already in the price. Commission-free; borrow fees on any short are
   NOT in the fill price — flag shorts as "gross of borrow".

## Pass bar
A slice passes only under (c)'s rule AND beats its placebo p95. Otherwise the report is descriptive anatomy for the
owner, and H1 closes as "no codifiable slice at this n" with the MDE stated.

## Rules for the agent
Sonnet, ≤ 40 calls, READ-ONLY toward Alpaca (GET only; assert no POST/DELETE in the code), no git, no services, no
orders, no writes outside `research/owner_trades/`. Outputs: `1777_fetch.py`, `1777_score.py`, `1777_fills.csv`,
`1777_orders.csv`, `1777_roundtrips.csv`, `RESULT_1777.md` (≤ 120 lines, tables, every caveat the data shows).
Return ≤ 150 words.
