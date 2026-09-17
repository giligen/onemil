# Stage M — which exit shape, at the slot count that is now live

Pre-registered in `M/PREREG.md` (written before any P&L existed). Run 2026-09-17,
one `nice -n 10` process at a time under `ulimit -v 1500000`, bars read symbol by
symbol through the `(symbol, bar_date)` index on a read-only `cache.db`. Nothing
outside `research/fuckup_audit/M/` was written. No config, service, order, cron or
production artefact touched. **No threshold, z-param, quintile cutoff, veto level or
adaptive mult was re-fitted anywhere in this stage** — selection is the shipped B+
stack read out of `orb.yaml` as it stands, and only the exit moves.

---

## PAGE ONE

### The answer: **X0. The shipped exit stays.**

Six exit shapes, three slot counts, one selection stack. **None of the five
challengers clears the pre-committed gate** (improve BOTH total P&L AND max drawdown
on TRAIN *and* on VAL, at the live N = 8):

| shape | TRAIN dP&L | TRAIN dMDD | VAL dP&L | VAL dMDD | verdict |
|---|---|---|---|---|---|
| **X1** X0 + 10-min time stop < +0.25R | **-2,736** | -10 | **-2,014** | -43 | fail (both) |
| **X2** X0 + 5-min time stop < 0R | -504 | -26 | -331 | -61 | fail (both) |
| **X3** no lock, hold to 15:45 | **+299** | **-260** | **+180** | +0 | fail (drawdown) |
| **X4** X3 + 10-min time stop | -1,916 | -228 | -1,898 | -43 | fail (both) |
| **X5** X4 + breakeven at +1R | -3,945 | -200 | -2,055 | -43 | fail (both) |

X0 is also the best or second-best shape at **all three** slot counts (N=3 $7,187 /
N=8 $14,429 / N=12 $15,563; only X3 beats it, and only at N=12, by $694).

### The money line — X0 at N = 8, the config that is live

| | stage sizing (per-position $3,333) | 3x (per-position $10,000, budget $80,000) |
|---|---|---|
| **$/month over 21 months** | **$687** | **$2,061** |
| total P&L | $14,429 | $43,286 (**exactly** 3x) |
| **worst month** | **-$148** (2026-09, a partial month) | **-$444** |
| max drawdown (daily cum) | -$576 | -$1,727 |
| red months | 2 of 21 (2025-12 -$25, 2026-09 -$148) | 2 of 21 |
| $/week, weeks green | $195, 44.6% of 74 weeks | $585, 44.6% |
| mean realized risk / trade | $149 | $446 |
| capital actually needed (max 4 picks on one day) | $13,332 | $40,000 |
| participation of the 5-min range $-volume: median / p90 / >5% | 0.32% / 1.22% / **2 of 215** | 0.96% / 3.67% / **17 of 215 (7.9%)** |

`risk_per_trade_usd` is still inert: the per-position cap binds on **100%** of picks
at both sizings, so "3x" means 3x the cap *and* 3x the risk knob ($375 -> $1,125)
together, and the P&L scales exactly — **before impact**. At 3x the median pick takes
1% of its own opening-range tape and 8% of picks take more than 5%; those are the
illiquid names that carry the right tail, so the 3x row is a linear extrapolation,
not a forecast.

### Tail dependence — the number that limits every sentence above

X0 at N=8 is **$14,429**; with the **top 5% of fills removed it is $5,524** (38%), and
with every fill capped at **+3R it is $8,801** (61%). Eight fills carry $8,905 of the
book. Any exit change that protects the tail is worth more here than any change that
trims the losers — which is exactly what the grid shows.

### What changes if a shape is ever adopted (none is today)

* **X3 / X5 need NO engine change** — they are `orb.yaml` edits only.
  `trading/stop_monitor.py:3169` already gates the static lock on
  `watch.lock_arm_at_r > 0`, so `exit.lock_arm_at_r: 0` disables the lock live (X3),
  and `lock_arm_at_r: 1.0` + `lock_stop_r: 0.0` *is* a breakeven-at-+1R stop (X5's
  stop rule). The BT side needs a one-line guard:
  `study_orb_pipeline_static_lock.py::simulate_static_lock` (line ~284) and
  `::simulate_winner_stack` (line ~378) arm on `bar_high >= entry + LOCK_TRIGGER_R x R`,
  so `lock_arm_r = 0` would arm on the FIRST bar instead of never — it must become
  `if lock_arm_r > 0 and ...`. `orb_planner.py:92` reads the same yaml block.
* **X1 / X2 / X4 (the time stop) ARE an engine change.** Six locations:
  1. new shared spec `trading/orb_time_stop.py` —
     `time_stop_fires(bar_open, entry, r_unit, min_r)` (one module imported by BT and
     live, the house parity-by-construction rule);
  2. `orb.yaml` -> `exit.time_stop: {enabled, minutes, min_r}`;
  3. `trading/orb_planner.py:92` (the `exit:` reader) + the `OrbPlan` dataclass at
     line 42, carrying the knobs onto the plan the way `lock_arm_at_r` is carried;
  4. `trading/orb_engine.py::_ingest_bars` (line ~1114) — a `_evaluate_time_stop`
     beside the existing `_evaluate_touchgo`, anchored on `pos.breakout_bar_ts` (the
     BT-parity MARKET breakout bar, never the fill minute), routed through the
     `_force_touchgo_exit` seam -> `stop_monitor.force_exit` (line ~1583);
  5. `trading/exit_reasons.py` — a new `ExitReason.ORB_TIME_STOP`, added to
     `_ATTRIBUTED_EXITS` (line ~313);
  6. `trading/stop_monitor.py:1414` `_FORCE_EXIT_REASON_WHITELIST` — without it
     `force_exit` refuses the reason and the cut is silently dropped.
  **This work is not recommended.** The time stop costs $2,014-$2,736 per split at
  N=8 and its own validation on the raw fills does not survive the shipped exit (S4).

### One honest disclosure the gate deliberately ignores

**Every time-stop shape beats X0 on TEST** (2026-06+): X5 $2,695, X4 $2,229,
X1 $1,968, X2 $1,504 vs X0's $1,380 — and X1/X4/X5 also improve TEST's drawdown
(-$411 vs -$516). TEST is 4 months, 57 picks, 38 fills, and those shapes lose on the
24 months of TRAIN+VAL. The pre-committed rule reads TRAIN and VAL only; this is
reported because it was measured, and it is the one line in this stage worth
re-reading if the book runs another quarter live.

---

## 1. What was run, and the three parity checks

ONE bar walk (`M/exit_shapes.py`) over the honest entered-inclusive features
(`analysis_results/orb_features_20260916_2053.csv`: 13,033 candidates = 7,402 fills +
5,631 modeled non-fills, 427 trading days, 2025-01-02 -> 2026-09-16) computing **all
six exits per candidate**, writing six candidate dumps that differ only in
`pnl` / `pnl_pct` / `exit_reason`. The shipped selector then ran off each dump via
`ORB_BT_RESIM_CACHE`, so **the picks are identical across all six shapes by
construction** (215 picks / 162 fills at N=8 in every cell) and the diff is pure exit.

Per-position cap held at the live $3,333.33 for every slot count
(`ORB_BT_ACCOUNT = 3333.333333 x N`), `ORB_BT_RISK = 375`, `ORB_SKIP_Q1 = 1`, every
veto at its `orb.yaml` value in all 18 runs.

**Parity check 1 (code).** The parametrised walker that produces X1-X5, run with its
three new knobs at neutral, reproduced the SHIPPED `simulate_winner_stack`'s exit
price *and* reason on **all 7,402 fills** — asserted per row; the run aborts on the
first mismatch. Log: `parity-checked 7402`.

**Parity check 2 (dump).** X0's dump equals D1's `candidates_dump.csv`:
`max |dpnl| = 0.000000000000` over 13,033 rows.

**Parity check 3 (book).** X0 at N=8 reproduces D1's slot-dose-response cell to the
dollar — **$14,428.62 / 215 picks / 162 fills / MDD -$575.55 / worst month -$148.0 /
2 red months** — and X0 at N=3 reproduces the nightly production book,
**$7,186.64 / 88 / 70**.

The shipped exit physics used throughout are the ones in `orb.yaml` today: static
lock 1.75R -> +0.5R, ATR14 stop floor k = 0.25 **ON**, 40% scale-out at +3R **ON**,
touchgo Rule M/D, 15:45 ET force close, 10 bps exit slip. All six shapes keep the
floor, the scale-out, touchgo and the 15:45 flat — each shape is a **single-lever**
change (PREREG S2).

---

## 2. The 18 cells — whole window

$ at stage sizing; MDD on the daily cumulative curve; `ex-top5%` = P&L with the top
5% of fills removed; `cap3R` = every fill capped at +3R of its own risk.

| cell | picks | fills | P&L | WR fills | $/fill | R/fill | $/mo | MDD | worst mo | red mo | ex-top5% | cap3R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **X0 N=3** (nightly book) | 88 | 70 | **7,187** | 42.9% | 103 | +0.62 | 342 | -714 | -223 | 6 | 3,092 | 4,853 |
| X1 N=3 | 88 | 70 | 5,375 | 40.0% | 77 | +0.44 | 256 | -647 | -249 | 8 | 1,280 | 3,452 |
| X2 N=3 | 88 | 70 | 6,261 | 35.7% | 89 | +0.55 | 298 | -543 | -249 | 7 | 2,167 | 4,035 |
| X3 N=3 | 88 | 70 | 7,058 | 40.0% | 101 | +0.62 | 336 | -717 | -223 | 6 | 2,855 | 4,296 |
| X4 N=3 | 88 | 70 | 5,459 | 38.6% | 78 | +0.46 | 260 | -730 | -249 | 8 | 1,255 | 3,107 |
| X5 N=3 | 88 | 70 | 5,964 | 37.1% | 85 | +0.50 | 284 | -550 | -249 | 9 | 1,761 | 3,612 |
| **X0 N=8** (live) | 215 | 162 | **14,429** | 41.4% | 89 | +0.64 | **687** | **-576** | **-148** | **2** | 5,524 | 8,801 |
| X1 N=8 | 215 | 162 | 10,266 | 37.7% | 63 | +0.43 | 489 | -694 | -476 | 6 | 2,089 | 6,368 |
| X2 N=8 | 215 | 162 | 13,717 | 34.6% | 85 | +0.60 | 653 | -598 | -334 | 3 | 4,812 | 8,196 |
| X3 N=8 | 215 | 162 | **15,169** | 37.7% | 94 | +0.62 | 722 | -787 | -194 | 5 | 5,009 | 8,116 |
| X4 N=8 | 215 | 162 | 11,463 | 36.4% | 71 | +0.44 | 546 | -830 | -612 | 8 | 1,957 | 6,212 |
| X5 N=8 | 215 | 162 | 9,744 | 30.9% | 60 | +0.36 | 464 | -728 | -517 | 7 | 1,385 | 5,467 |
| X0 N=12 | 267 | 198 | 15,563 | 40.4% | 79 | +0.53 | 741 | -944 | -411 | 4 | 5,674 | 9,826 |
| X1 N=12 | 267 | 198 | 11,355 | 37.4% | 57 | +0.37 | 541 | -941 | -525 | 6 | 2,194 | 7,348 |
| X2 N=12 | 267 | 198 | 14,467 | 33.3% | 73 | +0.51 | 689 | -920 | -356 | 4 | 4,578 | 8,837 |
| X3 N=12 | 267 | 198 | **16,257** | 36.9% | 82 | +0.56 | 774 | **-1,427** | **-655** | 5 | 5,113 | 8,834 |
| X4 N=12 | 267 | 198 | 12,505 | 35.9% | 63 | +0.43 | 596 | -1,189 | -661 | 8 | 2,016 | 6,886 |
| X5 N=12 | 267 | 198 | 10,837 | 30.3% | 55 | +0.34 | 516 | -1,062 | -566 | 9 | 1,461 | 6,191 |

Two readings that do not depend on the gate:

* **Every challenger lowers the win rate on fills** (41.4% -> 30.9-37.7% at N=8) and
  every challenger except X3 lowers mean R per fill. The time stop trades a higher
  frequency of small flat exits for the trades that were going to become the tail.
* **X3 (hold to close) is the only shape that keeps R/fill** (+0.62 vs +0.64) and the
  only one that adds dollars — and it pays for them in drawdown at every slot count
  (N=3 -717 vs -714, N=8 -787 vs -576, N=12 -1,427 vs -944) and in red months
  (5 vs 2 at N=8).

## 3. Per split, at the live N = 8

| cell | picks | fills | P&L | WR | R/fill | $/wk | weeks green | MDD | worst mo | red mo | ex-top5% | cap3R | MDE (R, t=2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **X0 TRAIN 2025** | 105 | 84 | **6,662** | 44.0% | +0.62 | 167 | 50.0% | **-528** | -25 | 1 | 3,506 | 5,393 | 0.40 |
| X1 TRAIN | 105 | 84 | 3,926 | 39.3% | +0.34 | 98 | 42.5% | -537 | -44 | 3 | 973 | 3,459 | 0.33 |
| X2 TRAIN | 105 | 84 | 6,158 | 33.3% | +0.54 | 154 | 42.5% | -553 | -130 | 1 | 3,002 | 4,889 | 0.33 |
| X3 TRAIN | 105 | 84 | 6,960 | 38.1% | +0.59 | 174 | 42.5% | -787 | -194 | 4 | 2,651 | 4,695 | 0.42 |
| X4 TRAIN | 105 | 84 | 4,745 | 38.1% | +0.34 | 119 | 40.0% | -756 | -164 | 5 | 470 | 3,354 | 0.34 |
| X5 TRAIN | 105 | 84 | 2,717 | 29.8% | +0.20 | 68 | 37.5% | -728 | -212 | 4 | -237 | 2,262 | 0.24 |
| **X0 VAL 2026-01..05** | 53 | 40 | **6,386** | 37.5% | +1.10 | 355 | 38.9% | **-489** | +159 | 0 | 2,888 | 2,283 | 0.96 |
| X1 VAL | 53 | 40 | 4,372 | 30.0% | +0.73 | 243 | 33.3% | -533 | -476 | 2 | 873 | 1,198 | 0.87 |
| X2 VAL | 53 | 40 | 6,055 | 35.0% | +1.06 | 336 | 33.3% | -550 | -334 | 1 | 2,556 | 2,059 | 0.97 |
| X3 VAL | 53 | 40 | 6,567 | 35.0% | +1.11 | 365 | 38.9% | -489 | +159 | 0 | 3,068 | 2,464 | 0.98 |
| X4 VAL | 53 | 40 | 4,488 | 27.5% | +0.75 | 249 | 38.9% | -533 | -612 | 2 | 989 | 1,314 | 0.88 |
| X5 VAL | 53 | 40 | 4,332 | 25.0% | +0.71 | 241 | 33.3% | -532 | -517 | 2 | 833 | 1,195 | 0.86 |
| **X0 TEST 2026-06+** | 57 | 38 | 1,380 | 39.5% | +0.24 | 86 | 37.5% | -516 | -148 | 1 | 225 | 1,124 | 0.38 |
| X1 TEST | 57 | 38 | **1,968** | 42.1% | +0.36 | 123 | 50.0% | **-411** | -103 | 1 | 812 | 1,711 | 0.38 |
| X2 TEST | 57 | 38 | 1,504 | 36.8% | +0.26 | 94 | 37.5% | -598 | -148 | 1 | 349 | 1,248 | 0.39 |
| X3 TEST | 57 | 38 | 1,642 | 39.5% | +0.28 | 103 | 31.2% | -555 | -148 | 1 | 232 | 957 | 0.46 |
| X4 TEST | 57 | 38 | 2,229 | 42.1% | +0.45 | 139 | 37.5% | -411 | -103 | 1 | 819 | 1,544 | 0.47 |
| X5 TEST | 57 | 38 | **2,695** | 39.5% | +0.54 | 169 | 43.8% | **-411** | -103 | 1 | 1,285 | 2,010 | 0.51 |

**MDE** = the smallest mean R per fill this cell could have called significant at
t = 2, i.e. the power of the test. At n = 84 / 40 / 38 fills it is
**0.40 / 0.96 / 0.38 R**. The measured differences between shapes (0.2-0.4 R per fill)
sit at or below that bar on VAL and at it on TRAIN: the gate is separating shapes with
a ruler whose smallest tick is roughly the size of the effect. X0's own edge is above
it (whole window +0.64 R per fill, t = 4.0).

## 4. Why the time stop looked good and is not

`research/fuckup_audit/orb_timestop_validation.md` measured T10/+0.25R against each
fill's **recorded** `pnl_pct` in the features CSV — and those outcomes came from the
OLD fixed **+2R target / -1R stop** exit, where **winners are capped at +2R**. Under
the shipped exit (static lock + 40% scale at +3R + hold to 15:45) winners are
uncapped, and this book is tail-carried (top 5% of fills = 62% of P&L). A cut that is
free when the upside is capped is expensive when it is not. That validation said so
itself in its second table: on the **B+ subset** it was neutral — +0.394 -> +0.348
(TRAIN), +0.325 -> +0.543 (VAL), +0.701 -> +0.821 (TEST), n = 38 / 15 / 17. Stage M is
that same test in dollars, at n = 162, through the selector.

Measured here, at the book level (N=8):

* the 10-minute stop fires on **2,168 of 7,402 fills (29.3%)** in the candidate set
  and on **47 of 215 picks (21.9%)** in the book;
* of those 47 it **wins 27 times (+$2,091) and loses 20 times (-$6,253)** ->
  **net -$4,163**;
* the loss is concentrated: **BNAI 2026-01-23 -$1,259**, then 2025-10-02 (CRCG -$554,
  CRCA -$548, CCUP -$544 — three names on ONE day) and RGTZ 2025-10-16 -$574.
  **2025-10 alone goes $2,341 -> -$44.**
* the 5-minute/0R variant (X2) is the same mechanism, smaller: 34 picks changed,
  23 won / 11 lost, net -$712.

Trade-level it does exactly what `LIVE_LOSERS.md` predicted — it cuts the slow bleeds
(X0's `stop` bucket -$5,935 -> X1's -$2,855, 46 stops become 23) — and it charges
$1,513 of `time_stop` losses plus the runners it never lets run (`scale_eod` $16,098
-> $11,179). Net, the runners are worth more than the bleeds.

## 5. Exit mix and exit P&L (N = 8, 215 picks)

| exit | X0 n | X1 n | X2 n | X3 n | X4 n | X5 n | X0 $ | X1 $ | X2 $ | X3 $ | X4 $ | X5 $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no_fill | 53 | 53 | 53 | 53 | 53 | 53 | 0 | 0 | 0 | 0 | 0 | 0 |
| tag_bb (touchgo) | 46 | 46 | 46 | 46 | 46 | 46 | -1,295 | -1,295 | -1,295 | -1,295 | -1,295 | -1,295 |
| stop | 46 | 23 | 25 | 52 | 25 | 20 | -5,935 | -2,855 | -3,148 | -6,732 | -3,039 | -2,315 |
| lock | 14 | 7 | 9 | 0 | 0 | 0 | 919 | 395 | 523 | 0 | 0 | 0 |
| scale_lock | 11 | 11 | 10 | 0 | 0 | 0 | 2,074 | 2,074 | 1,940 | 0 | 0 | 0 |
| scale_eod | 26 | 17 | 24 | 30 | 20 | 17 | **16,098** | 11,179 | 15,162 | **19,113** | 13,790 | 11,355 |
| eod | 19 | 11 | 14 | 25 | 15 | 10 | 2,569 | 2,282 | 2,093 | 3,463 | 2,900 | 2,231 |
| scale_stop | 0 | 0 | 0 | 9 | 9 | 0 | 0 | 0 | 0 | 620 | 620 | 0 |
| time_stop | 0 | 47 | 34 | 0 | 47 | 45 | 0 | -1,513 | -1,558 | 0 | -1,513 | -1,475 |
| be | 0 | 0 | 0 | 0 | 0 | 16 | 0 | 0 | 0 | 0 | 0 | -53 |
| scale_be | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 1,297 |

The whole ORB book lives in **`scale_eod`** — a position that took 40% off at +3R and
rode the rest to 15:45. Every lever that shortens the hold takes money out of that
cell. The lock, by contrast, is cheap insurance: removing it (X3) moves $2,993 of
`lock` + `scale_lock` into `scale_eod`/`stop` for a net +$740 over 21 months and a
49% deeper TRAIN drawdown.

## 6. Per month, N = 8 (P&L; the same 215 picks in every column)

| month | picks | X0 | X1 | X2 | X3 | X4 | X5 |
|---|---|---|---|---|---|---|---|
| 2025-01 | 7 | 565 | 565 | 726 | 565 | 565 | 565 |
| 2025-02 | 9 | 501 | 73 | 220 | 298 | -130 | -8 |
| 2025-03 | 5 | 567 | 284 | 234 | 610 | 327 | 199 |
| 2025-04 | 3 | 555 | 555 | 363 | 555 | 555 | 176 |
| 2025-05 | 6 | 1,347 | 1,299 | 1,445 | 1,347 | 1,299 | 1,299 |
| 2025-06 | 11 | 209 | 369 | 209 | **1,548** | **1,708** | 53 |
| 2025-07 | 6 | 54 | -32 | -130 | -74 | -32 | -1 |
| 2025-08 | 3 | 63 | 63 | 63 | -40 | -40 | 31 |
| 2025-09 | 16 | 436 | 502 | 410 | 576 | 502 | 502 |
| 2025-10 | 10 | **2,341** | **-44** | 2,222 | 1,884 | **-164** | **-81** |
| 2025-11 | 14 | 50 | -2 | 357 | -193 | -50 | -212 |
| 2025-12 | 15 | -25 | 295 | 40 | -114 | 207 | 196 |
| 2026-01 | 12 | 750 | **-476** | 777 | 614 | **-612** | **-517** |
| 2026-02 | 13 | 1,229 | 1,053 | 1,333 | 1,229 | 1,053 | 1,053 |
| 2026-03 | 9 | 4,040 | 3,950 | 4,040 | 4,104 | 3,950 | 3,950 |
| 2026-04 | 13 | 208 | -329 | -334 | 460 | -77 | 21 |
| 2026-05 | 6 | 159 | 174 | 238 | 159 | 174 | -176 |
| 2026-06 | 16 | 292 | 361 | 525 | 740 | 809 | 969 |
| 2026-07 | 20 | 589 | 947 | 479 | 503 | 862 | 1,098 |
| 2026-08 | 17 | 648 | 762 | 648 | 547 | 661 | 730 |
| 2026-09 | 4 | -148 | -103 | -148 | -148 | -103 | -103 |

2026-03 is $4,040 of a $14,429 book in every column — one month, 9 picks. 2026-09 is
a partial month (ORB was paused 9/14 and re-enabled 9/17; its picks are BT-only).

## 7. The money line in detail

Three sizing runs of the SAME X0 book at N=8 (215 picks, 162 fills, identical
selection — only the sizer moves):

| | per-pos cap | risk knob | mean position | cap binds | mean realized risk | P&L | $/mo | MDD | worst mo |
|---|---|---|---|---|---|---|---|---|---|
| stage (live) | $3,333 | $375 | $3,333 | 100% | $149 | 14,429 | 687 | -576 | -148 |
| **3x proportional** | $10,000 | $1,125 | $10,000 | 100% | $446 | **43,286** | **2,061** | -1,727 | -444 |
| 3x literal (cap only) | $10,000 | $375 | $8,461 | 36.7% | $354 | 36,485 | 1,737 | -1,394 | -416 |

The third row is the sanity check PREREG S5 asked for, and it matters: raising only
`account_budget_usd` to $80,000 and leaving `risk_per_trade_usd` at $375 does **not**
give a 3x book. ORB stops are 4-8% of price, so `375 / stop%` falls below $10,000 on
63% of picks and the sizer becomes risk-driven instead of cap-driven — you get 2.53x,
not 3x. **Both knobs move together or neither does.**

Participation at 3x (position / the pick's own 5-minute opening-range dollar volume):
median **0.96%**, p75 2.13%, p90 3.67%, **17 of 215 picks over 5%**, 106 over 1%.
Obtainable for the median name, marginal for the top decile — and the top decile is
where the tail lives.

Capital: the slot cap almost never binds. Max **4 picks on one day**, p95 3, mean 1.41
— so N=8 really needs **$13,332** at stage and **$40,000** at 3x, not $26,667/$80,000.

## 8. Caveats

1. **TEST was computed in the same script pass as TRAIN and VAL.** The decision rule
   in code reads TRAIN and VAL only and was written before the run (PREREG S4), and
   the verdict does not use TEST — but I did not physically withhold the column, and
   the disclosure above reports what it says, including where it disagrees.
2. **This is a relative tool at stage sizing, never a forecast.** The entered-inclusive
   rebuild removed the selection look-ahead, but the fill model is still a simulator
   and the 3x row assumes zero market impact on a cap-bound position.
3. **Inherited simulator deviations**, identical in all six shapes so they cancel in
   the diff: a stop that gaps through fills at the level, not at the gap price; the
   entry fill is the recorded `entry_price`; the exit slip is a flat 10 bps.
4. **The time stop's live twin would fire one bar later than the sim.** BT reads the
   OPEN of the bar at `breakout_bar + 10 min`; live would act on that bar's event,
   i.e. at its close. Not modelled — and not worth modelling, since the shape loses.
5. **n is small where it matters.** 162 fills at N=8 over 21 months; VAL is 40 fills.
   The MDE column in S3 is the honest scale of what this stage could see.
6. **2026-09 is partial**, and ORB's 9/17 re-enable happened after the features file
   was built.
7. **The B+ short-history G1 leg is OFF** in today's `orb.yaml`
   (`short_history_veto: false`). This stage used `orb.yaml` exactly as it stands and
   did not re-litigate it.

## 9. Phrasing

No exit shape among the six was found to improve both the total and the drawdown of
the shipped ORB book in this universe (the entered-inclusive B+ candidate set), at
this book size (8 slots, $3,333 per position, 162 fills), over this window
(2025-01-02 -> 2026-09-16), at this cost model (10 bps exit slip, the shipped fill
convention) — and the smallest per-fill mean R the test could have called significant
at t = 2 is **0.40 R on TRAIN, 0.96 R on VAL, 0.38 R on TEST**. That is not a claim
that no better exit exists; it is a claim that none of these five is detectably better
here, and that two of them (the 10-minute time stop, the breakeven stop) are
detectably worse.

## 10. Cell count

* **20 decision cells**: 18 pipeline runs (6 exit shapes x 3 slot counts) + 2 sizing
  runs. All 18 declared in PREREG S3 before the walk; the 2 sizing rows in S5.
* **Descriptive cells reported**: 20 configs x 4 windows (whole + 3 splits) = 80
  summary rows; a 21 x 6 monthly display (126); an 11 x 6 exit-mix table and its
  11 x 6 P&L twin (132); 5 shape-vs-X0 diff rows; 6 participation rows; 3 tail
  columns on every summary row.
* **0 thresholds, parameters or cutoffs fitted in this stage.**
* **Prior search carried in, counted, not repeated**: the 9-cell time-stop grid
  (5/10/15 min x 0/0.25/0.5 R) of `orb_timestop_validation.md`, from which Stage M
  took the pre-declared primary cell (10 min / +0.25 R) and one flank (5 min / 0 R)
  and searched nothing further. Whole-program total for the time-stop question:
  **9 + 2 = 11 cells.**

## 11. Artefacts

```
research/fuckup_audit/M/
  PREREG.md          pre-registration, written before the run
  REPORT.md          this file
  exit_shapes.py     ONE bar walk, six exits, three parity asserts
  walk.log           its log (7,402 fills, parity-checked 7,402, PARITY OK)
  dump_X0..X5.csv    six candidate dumps (13,033 rows each, identical but the exit)
  run_grid.sh        the 18 selector-only runs   ·  grid.log, log_<cell>.txt
  run_sizing.sh      the 2 sizing runs           ·  sizing.log
  book_<cell>.csv    per-pick books              ·  monthly_<cell>.csv
  analyze.py         all tables + the gate       ·  analysis.txt (its output)
  summary.csv        80 summary rows             ·  monthly_n8.csv
  exit_mix_n8.csv    exit_pnl_n8.csv
```
