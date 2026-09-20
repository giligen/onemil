# REPORT — lev_rebalance: wrapper close-rebalance drift

**Verdict: INCONCLUSIVE — do not ship, do not cite the point estimates.** The study did
not reach adequate power. The blocking finding is a data-coverage collapse, discovered
mid-run, not a considered null about the mechanism (see phrasing rule, CLAUDE.md).

## The one caveat that alone explains everything

`intraday_bars_1min` in `data/cache.db` is populated **opportunistically** — only for
symbol-days the live system happened to scan/touch — not comprehensively. For MSTR
(the underlying that supplies 18 of 19 signal trades) the cache holds 1-min bars for
only **29 distinct days in all of 2025**. Across the three underlyings, of 381
candidate days (daily |move| >= 3%), only **47 (12.3%)** had usable 1-min bars at both
14:59 and 15:01 ET — far below the PREREG's own 80% coverage rail. The 47 covered days
are not a random subsample of the 381: they are whichever days some other process
(ORB/BF scans, manual backtests) already pulled that symbol. **The resulting trade
sample is not causally guaranteed representative of all qualifying wrapper-rebalance
days** — this is exactly the kind of selection the CLAUDE.md checklist (causality
trace) exists to catch, and it was caught, not avoided.

## Cells

| Cell | Def | n | Verdict |
|---|---|---|---|
| 1,291 | LONG, r>0 | 10 | underpowered |
| 1,292 | SHORT, r<0 | 9 | underpowered |
| 1,293 | combined | 19 (15 TRAIN, 4 VAL) | underpowered, FAILS pass bar |
| 1,294 | 2% stop diagnostic | 19 | reported, not gating |

Universe actually run: TSLA/MSTR/NVDA + `FAMILIES['tsla_leveraged'/'mstr_leveraged'/
'nvda_leveraged']` (23 wrappers total) — see FREEZE.md #1 for the scope cut from the
full offline-map universe. NVDA contributed **zero** trades (no covered day cleared the
F-threshold with usable bars — small-n artifact, not evidence NVDA is different).

## Signal vs control, both splits

| Split | Signal n / mean R / t (day-clustered) | Control n / mean R / t | Signal-Control (iid t, not clustered) |
|---|---|---|---|
| TRAIN 2025 | 15 / +0.45R / t=1.52 (n_days=14) | 59 / -0.03R / t=-0.31 | +0.50R, t~1.6 |
| VAL 2026 H1 | 4 / +0.33R / t=0.66 (n_days=4) | 21 / -0.42R / t=-2.25 | +0.75R, t~1.4 |

Point estimates are directionally consistent with the mechanism (signal > control,
same sign both splits) and exceed the +0.10R pass threshold — but **no t-stat clears
the >=2 day-clustered bar for the signal cell alone**, VAL n=4 trades makes any t
meaningless, and the TRAIN-halves check required by PREREG was NOT separately computed
(2025 H1 vs H2 not broken out — a budget cut, flagged here since it was missed in
FREEZE.md). MDE at n=4-15 with observed std ~1.0-1.1R is roughly +/-0.9-1.4R at 80%
power — an effect smaller than ~1R would not have been detectable.

## F-quintile monotonicity

Correlation of F with realized pnl_R across the 35 qualified (pre-threshold) days:
**r = -0.02** — flat, not monotone. Quintile means: 0.57, -0.34, 1.19, 0.17, 0.26R.
No support for "higher flow proxy -> more drift" in this sample; also consistent with
the sample being too small/selection-biased to see it either way.

## Reg SHO / obtainability

Of the pool of r<=-10% short candidates, **10 were excluded pre-selection** by the
uptick gate (entry-bar open not above entry-bar low) — a large fraction, consistent
with sharp down-moves gapping through the entry bar rather than ticking up into it.
Entry fill (15:01 bar open) and exit (MOC daily close) are both obtainable by
construction; the 2% stop diagnostic uses an intrabar touch (optimistic, diagnostic
only) and rarely binds in this sample — TRAIN/VAL stop-variant means (0.48R/0.37R) sit
close to the no-stop means (0.45R/0.33R).

## Cadence bar (C1-C5), live config N/A — this book has never run live

```
TRAIN: C1 fail (0 cycles) | C2 fail | C3 fail (MDD 0.48R) | C4 pass (100% green vs 51% null) | C5 fail (0.28 fills/wk) | C7 fail (0 cycles)
VAL:   C1 fail (0 cycles) | C2 fail | C3 fail (MDD 1.14R) | C4 pass (67% green vs 50% null)   | C5 fail (0.18 fills/wk) | C7 fail (0 cycles)
```
No strong week (>=5R) ever occurred in either split — arithmetically expected at
0.18-0.28 fills/week. **Fails the cadence bar outright on frequency alone**, independent
of the coverage problem.

## Coverage

12.3% of candidate underlying-days had usable 1-min bars (rail: >=80%). Winner/loser
missingness gap not meaningfully computable at n=19.

## What would fix this

Not a rule change — a data problem. Either (a) fetch full 1-min history for MSTR/TSLA/
NVDA (+ the extended offline-map wrapper universe) via a real backfill (out of scope:
cache.db is read-only in this task, and a live-API backfill needs owner sign-off per
CLAUDE.md's cache-overwrite rule), or (b) rerun using only `daily_bars` for a coarser
close-to-close proxy of the mechanism (loses the causal 15:00-cutoff precision the
PREREG deliberately specified).

## Independent check status

Full independent reimplementation (per CLAUDE.md's research-claim checklist) was
**not performed** — single-agent, budget-constrained run. This report is a first pass
only; do not relay the point estimates as a finding without a second implementation
reproducing the 19-trade signal set trade-by-trade.

---
Artifacts: `signal_days_raw.csv`, `signal_days_qualified.csv`, `signal_trades.csv`,
`control_days_raw.csv`, `control_trades.csv`, `summary.json`, `run_study.py`.
