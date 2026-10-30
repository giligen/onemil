# PREREG — cell 1,721: loser anatomy of the LIVE HOD paper book (241 fills, 2026-09-29 → 10-30)

Owner 2026-10-30: "You fail turning it into a winner." This cell works bottom-up on the live fills (the only population
that carries the fill-filter information — it beats its armed population by ~+0.2 R, 10/23), not on the BT population,
which is closed (1,551 cells). Frozen before any number.

## Data
`data/trades.db`, `strategy='hod_break'`, `account='paper'`, `trade_date >= '2026-09-29'`, `pnl IS NOT NULL` (read-only;
copy to the scratchpad first). Fields: entry/fill/level (`pattern_data.level`, `consol_low`, `limit`, `target_r`),
`stop_loss_price`, `take_profit_price`, `exit_price/reason/exited_at`, `exit_slippage`, `exit_pricing_method`,
`filled_at`, quotes/spread columns when present. R = `total_risk` (nominal $50). SPY/QQQ day context from
`research/index_orb/cache/{SPY,QQQ}_1min.parquet` or `daily_bars` (read-only).

## Step 1 — decomposition (no filter yet)
Per fill: R_gross_at_levels = (exit_level − entry_level)/R using the intended levels (level, stop, target); entry slip
= fill − level (bps and R); stop slip = `exit_slippage` or exit − stop (R) on stop exits; target slip. Report sums:
net R/trade = gross-at-levels − entry cost − stop cost − target cost. If cost ≥ |gross| the lever is EXECUTION, not a
filter — say so first. Also: P&L of the 48 `eod` exits, their MFE if the bars allow (else the sign and distribution).

## Step 2 — pre-listed cuts (ONLY these 12, each with its mechanism written before the number)
1. Fill time of day (10-min buckets; mechanism: late breaks lack follow-through).
2. R as % of price (`risk_per_share/entry`; mechanism: R under the spread is noise — see [[project_r_must_exceed_spread]]).
3. Entry spread bps (`entry_quote_spread/entry`; mechanism: cost).
4. Price bucket (< 5, 5–10, 10–20, > 20).
5. Minutes from the day's first HOD arm to fill, if recoverable from `created_at`/`filled_at`; else rank of the fill in the day.
6. Fill rank within the day (1st, 2nd, 3rd+; mechanism: the first break of the day is the informed one).
7. Fills that day (1–5 / 6–10 / 11+; mechanism: broad-tape days are chop).
8. SPY return 09:30 → fill minute (mechanism: index tailwind).
9. Prior-day gap / today's gap from `daily_bars` if present (mechanism: catalyst days).
10. Stop distance in bars-of-ATR proxy = R / (level − consol_low) — the consolidation depth (mechanism: tight consolidations).
11. Symbol repeat (symbol already filled this week; mechanism: exhausted names).
12. Day of week.

## Validation (frozen)
Split the book into halves A/B by alternating trade dates (odd/even calendar day). A cut PASSES only if: the kept side
has R/trade ≥ +0.10 R above the dropped side in BOTH halves with the same sign; the kept side keeps ≥ 50 % of fills
(≥ 5 fills/week); kept ex-top-5 % R/trade > 0; the mechanism was listed above. Report every cut's A/B table, pass or
not, and the Bonferroni-adjusted p at 12 cuts. A cut that passes in A only is reported as "unconfirmed", never shipped.

## Output
`RESULT_1721.md`: decomposition table, the 12 cut tables (A, B, pooled), passes/unconfirmed/fails, MDE for a cut that
keeps 50 % of 241 fills (sd of per-fill R from the book), and a one-paragraph ship recommendation IF something passes
(the exact engine condition). No closure language: if nothing passes, state what the book can and cannot detect at 241.
`fills_1721.csv` (per fill, all derived fields). Scripts and outputs under `research/hod_live_anatomy/` only.
