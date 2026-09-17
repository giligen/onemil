# Stage I — PRE-REGISTRATION (written 2026-09-17, BEFORE any run in this directory)

Owner directive 9/17: "stack multiple strategies to maximize P&L". This stage runs NO new search for a rule. It
takes books that are already declared elsewhere and asks one arithmetic question: **does putting them in one
12/day-4-concurrent book raise weekly R and $/month versus F6 alone, and what does the combined monthly $ of the
whole portfolio look like?**

## Part A — the stacked intraday book

### Population (identical for every family, none of it re-derived here)
Source `research/fuckup_audit/C/pop_c.csv` (Stage C's scoring population, next-open fill).
Families and their `cfg` strings as they exist in that file:

| tag | fam | cfg |
|---|---|---|
| F6 | `F6` | `{}` |
| F14 | `F14` | `{"N": 15}` (the only F14 config in the file) |
| F11 | `F11` | `{"base": "F6"}` |
| F8N30 | `F8` | `{"N": 30}` |

Row filters, applied to every family identically (PLAN §1 scoring population + the ONE rule that replicated):
`next_entry` not null · `price >= 5` · `next_r_pct >= 1` · `next_entry_m <= 841` (14:01) ·
`range_so_far_pct >= 5` · **`prev_day_range_pct >= 8`** (ORB's shipped PDR veto, transferred unchanged from
`H/F6/f6_pdr_book.py`; NOT re-derived and NOT re-tuned here).

### Cost contract and book — copied, not re-invented
Contract (c) exactly as `H/F6/f6_pdr_book.py`:
`half = 0.5*(spread_cc_bps/1e4*100)/max(next_r_pct, 0.05)`;
`net = rr - 0.25*half - half*{stop 0.875, eod 0.412, target 0.875}[why]` (missing reason -> 0.875).
Exits: `hold` to 15:55 (primary) and `2r` on a bar close (secondary).
Book: `trading.hod_break.run_book(rows, 12, 4)`.
Splits: TRAIN 2025 · VAL 2026-01..05 · TEST 2026-06..09. Weeks = W-FRI periods present in that split.

### Dedupe (declared, not tuned)
One trade per (symbol, day). When two families signal the same symbol-day the higher-priority family's row is
kept. Priority **F6 > F14 > F11 > F8N30** (the order of their standalone TRAIN evidence in `H/METHOD.md`).
Within a family, if more than one row exists for a symbol-day the earliest `next_entry_m` is kept.

### Cells
Stacks: **S0** = F6 alone (the reference; must reproduce `H/F6/f6_pdr_book.md` exactly) · **S1** = F6+F14 ·
**S2** = F6+F14+F11 · **S3** = all four. × 2 exits = **8 cells**, each on TRAIN and VAL.

### Freeze rule (declared before any run)
After TRAIN and VAL are printed for all 8 cells, the stack that is carried to TEST is **the one with the highest
VAL weekly R**. TEST is then read ONCE, for that stack and for S0 (S0's TEST is already public in
`f6_pdr_book.md`, so it costs nothing new). TEST is never used to choose.

### Reported per split per cell
n, trades/week, mean net R, mean gross R, t, WR, weekly R, weeks green, worst week, MDD (on the weekly curve),
months green, and the family mix of the booked trades.

### The liquidity-capped twin (money, not R)
Reuse `H/F6_sizing`'s method exactly: fill-bar and trailing-5-minute dollar volume at `next_entry_m`, bar source
precedence `data/cache.db::intraday_bars_1min` then `research/bf_zero/bars_sip.db` (both read-only), as in
`H/F6_sizing/f6_bars.py`. At **$300 risk/trade**, drop every POPULATION row whose participation
`shares*entry / 5-min $ volume` exceeds 1% and re-run `run_book(12,4)` on the survivors, so a freed slot refills.
$/month = booked net R summed per month × $300.

### Marginal contribution
For each added family: the booked trades it contributed, their mean/sum net R, and a decomposition of where the
stack's extra trades come from — days on which the F6-alone book had a free slot at that minute (added frequency)
vs. rows that displaced an F6 trade under the priority rule (displacement).

### Sensitivities (declared, counted, NOT used to choose)
(1) spread ×1.5; (2) $150 and $600 risk on the capped twin; (3) the un-capped $/month. Tail tests
(top-1%/top-5% removed, +3R cap) are run on the frozen stack only.

## Part B — the portfolio of sleeves, monthly $

No new simulation. Each sleeve's own declared, honest artefact is read as-is:

| sleeve | file | basis |
|---|---|---|
| intraday stack | this stage's frozen stack, liquidity-capped twin | $300 risk/trade, 12/day 4-concurrent |
| QQQ noise band | `H/QQQ/final_book_days.csv` (`r1x` × $60,000) | unfiltered base book, live-convention fill, 0.5 bp/leg |
| ORB B+ | `analysis_results/orb_bplus_book.csv` (`_sized_pnl`) | `orb.yaml` B+ stage: $10,000 budget / 3 concurrent / $375 risk |
| TQQQ 1× (optional) | `Q/step3_months.csv` book `TQQQ 1x notional (0.5bp/leg)` | monthly only (no day file); $60,000 notional |

Table: month × sleeve over the window all three cover (2025-01 .. 2026-09), the sum, months green per sleeve and
combined, worst month per sleeve and combined, pairwise monthly correlations, 2025 and 2026-YTD totals.

## What this stage is NOT
Every constituent book has already FAILED its own pre-registered VAL gate (LOG.md 9/17: F6-PDR's stack failed,
F14/F8/F11 all failed, the QQQ day filters failed). Nothing here is a ship decision, and no number here is a
forecast. This is historical arithmetic on already-declared books, answering a combination question the program
has never asked. PLAN §1's phrasing rule applies to every sentence of the report.
