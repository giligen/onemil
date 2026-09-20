# REPORT — exec_cost entry cells (1,289 / 1,290)

Population: `analysis_results/orb_bplus_book.csv`, `entered==1`, merged to
`research/fuckup_audit/P_cost/spreads.parquet` (trigger-instant NBBO, same honesty rail
as Stage P: last quote at/before the first trade in the breakout minute with
price > range_high). TEST (>=2026-06-01) sealed, not touched. n=83 TRAIN (2025), n=39 VAL
(2026-01..05). Scripts: `fetch_and_sim.py`, `analyze.py`, `fetch_0935_buckets.py`.
Per-trade data: `per_trade_scored.csv`.

## Coverage
122/122 eligible fills fetched (0 trade-tape errors, 0 quote errors) — **100%**, clears
the 80% rail (3 of 165 filled book rows had no Stage-P coverage, dropped pre-fetch).

## CELL 1,289 — post-then-cross entry: **PASS both splits**

| split | n | passive fill rate | Δcost/trade | net Δ$ | day-clustered t | passive net_R | crossed net_R |
|---|---|---|---|---|---|---|---|
| TRAIN | 83 | 86.7% (72/83) | 0.058 R / $112 | **+$9,278** | t=5.55, 54 days | 0.613 | 0.585 |
| VAL | 39 | 82.1% (32/39) | 0.055 R / $110 | **+$4,296** | t=5.39, 31 days | 1.288 | 0.700 |

No adverse selection: passive-fill outcomes are not worse than crossed (gap +0.03R TRAIN,
+0.59R VAL, both inside the −0.05R fail line — crossed trades, if anything, do slightly
worse). Tail check: dropping the top 5% of Δ$ contributors per split still leaves net Δ$
solidly positive (TRAIN $7,036, VAL $3,448) — not a lottery-ticket result.

**Caveat**: Δ$ uses shares backed out of `pnl/pnl_pct/entry_price` (book has no share
column) — sane 1.6K–16K range, no single outlier driving the result, but not the ledger
figure.

## CELL 1,290 — spread-in-R gate: **FAIL both splits, all 3 thresholds (6 cells)**

| split | thresh | vetoed n | vetoed net$ | kept net$ (base) | kept MDD (base) | pass? |
|---|---|---|---|---|---|---|
| TRAIN | >0.05 | 33 | +$25,079 | $74,074 ($99,153) | −$6,460 (−$9,636) | N |
| TRAIN | >0.10 | 15 | +$17,624 | $81,529 ($99,153) | −$13,923 (−$9,636) | N |
| TRAIN | >0.15 | 9 | −$999 | $100,152 ($99,153) | −$11,666 (−$9,636) | N |
| VAL | >0.05 | 22 | +$87,911 | $8,647 ($96,558) | −$8,332 (−$7,337) | N |
| VAL | >0.10 | 7 | +$40,959 | $55,599 ($96,558) | −$7,337 (−$7,337) | N |
| VAL | >0.15 | 4 | +$10,230 | $86,328 ($96,558) | −$7,337 (−$7,337) | N |

Fails at every threshold, both splits: the vetoed set's net is **positive**, not negative,
at 5 of 6 cells. Wide half-spread-in-R fills are disproportionately the book's monster
winners (spread tracks the moves that make the money) — gate removes edge, not cost.

**Caveat**: MDD is a date-ordered cumsum drawdown over `pnl` (not intraday equity), a
coarse proxy — but it agrees with the net-$ verdict in every failing cell, not load-bearing.

## Spread around 09:35 by 10-s bucket (owner diagnostic, not a pre-registered cell)

NBBO quotes fetched 09:34:00–09:36:00 ET for all 122 fills (1,335 fill×bucket rows,
~160K quotes). Pooled median/P75 half-spread (bps), weighted by quote count per bucket:

| bucket start (ET) | n fills | n quotes | median bps | P75 bps |
|---|---|---|---|---|
| 09:34:00 | 115 | 15,702 | 17.18 | 20.47 |
| 09:34:10 | 114 | 12,849 | 16.78 | 19.90 |
| 09:34:20 | 114 | 13,895 | 16.81 | 20.28 |
| 09:34:30 | 112 | 13,431 | 17.91 | 21.60 |
| 09:34:40 | 114 | 12,934 | 17.78 | 20.85 |
| 09:34:50 | 112 | 9,786  | 18.52 | 22.03 |
| **09:35:00** | 115 | 17,864 | **16.96** | 20.11 |
| 09:35:10 | 113 | 17,403 | 16.92 | 21.50 |
| 09:35:20 | 111 | 14,768 | 16.70 | 20.43 |
| 09:35:30 | 110 | 14,869 | 16.99 | 20.48 |
| 09:35:40 | 104 | 11,332 | 18.94 | 22.92 |
| 09:35:50 | 101 | 10,262 | 19.56 | 23.29 |

**Answer: no.** The 09:35:00–09:35:10 bucket (16.96 bps median) is the narrowest or
tied-narrowest of the twelve, ~1.5bps *tighter* than 09:34:50 (18.52) and ~2–2.6bps
tighter than 09:35:40/09:35:50 (18.94/19.56) — spreads widen into the close of the minute,
not at the crossing instant. No evidence of an "every-algo-crosses-here" tax at :00.

Fill-time cross-check (small-n, noisy): only 53/122 actual ORB triggers land inside the
09:34–09:36 window at all — most breakouts (69/122) fire later than 09:36 (opening-range
breaks aren't clustered at the open). Of the 53, the fills that triggered in the
09:35:00–09:10 bucket show higher spread (n=21, median 22.0bps) than 09:35:20–09:35:30
(n=7–8, median 9.5–10.4bps), but those buckets have n=7–8 each — too thin to trust over
the pooled table above.

**Caveat**: the pooled table answers "how wide is the market's spread at that clock time,"
the fill-time table answers "how wide was the spread when THIS stock's breakout happened to
fire" — the two disagree in direction at n=53 because trigger timing is itself
endogenous (which names break exactly at :00 vs :20 isn't random); trust the pooled,
larger-n table for the market-wide claim.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PKSsd3LjBkXYzPECLNYR5W

## Correction (post-review) — Cell 1,289 dollar scale was wrong by 15x

The original Δ$ (+$9,278 TRAIN / +$4,296 VAL) used shares backed out of the book's raw
`pnl`/`pnl_pct` columns. `pnl_pct` checks out as a genuine price return (SMST stop-out:
-4.38% vs (range_low-entry)/entry = -4.29%, matches) — but `pnl` itself is **exactly 15x**
`_rp_pnl`/`_sized_pnl` for every one of the 122 fills (ratio 0.0666667 ± 3e-17, i.e. 1/15
to float precision) — `pnl` is computed at a fixed, much larger notional than the book's
actual sizing; `_sized_pnl` (sum $6,610 TRAIN / $6,437 VAL here) is the number that ties to
the honest book ($6,085–$6,627/21mo cited elsewhere). Shares/notional must scale with
`_sized_pnl`, not raw `pnl`.

**Corrected Δ$** (delta_cost_share × shares×(1/15)):

| split | Δ R/trade (unchanged) | net Δ$ corrected | net Δ$ (WRONG, retracted) | tail: drop top 5% |
|---|---|---|---|---|
| TRAIN | 0.058 R | **+$619** | ~~+$9,278~~ | +$469 |
| VAL | 0.055 R | **+$286** | ~~+$4,296~~ | +$230 |

Δ R/trade was never affected by the notional bug (it's a price-only ratio, no shares
term) — cell 1,289 still PASSES both splits (net Δ$ up, tail-robust, no adverse selection
finding unchanged), but at ~6.7% of the dollar magnitude first reported. At $10K-stage
sizing this rule is worth roughly $600 (TRAIN) / $290 (VAL) over the sample, not
$9K/$4K — material relative to the book's own $6,085–$6,627/21mo total, not free money.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PKSsd3LjBkXYzPECLNYR5W
