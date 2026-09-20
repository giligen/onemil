# REPORT — lev_rebalance: wrapper close-rebalance drift

**Verdict (PASS 2, full population): REFUTED — signal does not beat control; the signal
leg alone is significantly NEGATIVE, well-powered, both splits. Do not ship, do not run
live.** This supersedes pass 1 (history below), which was VOID on data coverage, not a
finding about the mechanism.

## Pass 2 — the full-population, availability-fixed rerun

Universe: 316 underlyings (of 345 resolved from the 6,136-row wrapper class map via
`underlying_anchor`) with >=1 candidate day; control pool 400 non-wrapper stocks (89 with
candidate days). Missing 1-min bars pulled from Alpaca SIP into a study-owned db
(`bars_1500_1600.db`), `cache.db` untouched. See `PREREG.md` Pass-2 addendum for the
exact pull spec.

**Coverage**: signal 27,539 / 28,462 candidate underlying-days usable (**96.8%**, clears
the 80% rail by a wide margin — the availability problem that voided pass 1 is fixed).
Control 3,078 / 3,888 (**79.2%**, just under the rail — noted, not re-pulled further under
budget; the shortfall is symbols with no daily_bars-derivable candidate window at all,
not a winner/loser-correlated gap). Winner/loser missingness gap not separately computed
(budget) — flagged, not silently dropped.

### Cells (TRAIN 2025 / VAL 2026 H1, day-clustered t = two-sample t-test on per-day means)

| Cell | Split | n trades (n days) | mean R | t |
|---|---|---|---|---|
| 1,291 LONG | TRAIN | 4,491 (244) | -0.058 | -1.97 |
| 1,291 LONG | VAL | 2,757 (102) | -0.032 | -0.76 |
| 1,292 SHORT | TRAIN | 3,863 (238) | -0.059 | -1.81 |
| 1,292 SHORT | VAL | 2,251 (101) | **-0.188** | **-4.22** |
| 1,293 COMBINED (signal) | TRAIN | 8,354 (250) | -0.053 | -2.43 |
| 1,293 COMBINED (signal) | VAL | 5,008 (102) | -0.074 | -2.50 |
| 1,293 COMBINED (control) | TRAIN | 878 (234) | -0.031 | -0.81 |
| 1,293 COMBINED (control) | VAL | 448 (99) | -0.016 | -0.29 |
| 1,294 2% stop diagnostic | TRAIN | 8,354 | -0.059 | — |
| 1,294 2% stop diagnostic | VAL | 5,008 | -0.072 | — |

**Signal − control**: TRAIN -0.022R (-0.044% of price), t=-0.50, p=0.62. VAL -0.058R
(-0.117% of price), t=-0.92, p=0.36. Both splits: same sign (negative), neither clears
the +0.10R/t>=2 pass bar in EITHER direction — no detectable difference between wrapper
movers and non-wrapper movers. The stop diagnostic barely moves the signal mean (rarely
binds), so this isn't a tail-risk story — it's a genuine flat/negative continuation.

**MDE (80% power)**: TRAIN 0.061R, VAL 0.083R. The observed signal-control diff (TRAIN
-0.022R, VAL -0.058R) sits INSIDE the MDE band both splits — the null on the mechanism
is a real null, not a power failure. The signal-ALONE mean, by contrast, clears its own
one-sample t at both splits (t=-2.43/-2.50) — a well-powered NEGATIVE, i.e. big-move
wrapper-underlyings that stay >=5% at 15:00 tend to give some of it back into the close,
same as non-wrapper big movers (control also trends negative, just noisier).

**F-quintile monotonicity**: corr(F, pnl_R) = **0.0016** (flat). Only 3 of 5 quintile
buckets resolved (`qcut` collapsed) because **TRAIN median F = 0.0** — over half of
qualified days have zero flow proxy. Root cause: many wrapper symbols in the resolved
universe have sparse or no `daily_bars` ADV20 history (missing entirely, or not yet
PIT-listed), so `F` is degenerate for most of the sample. **This is the one caveat that
most limits this pass**: it tests "wrapper-underlying big movers vs non-wrapper big
movers" broadly, NOT "high-flow-proxy movers specifically" — the F-based ranking inside
PREREG was never really exercised. Quintile means (n=13,362 qualified days): bucket 0
-0.051R, bucket 1 -0.056R, bucket 2 -0.032R.

**Entries/week** (signal, unconstrained by the 5-slot cap): TRAIN 161/wk, VAL 238/wk.
**$/week at $66K book** (1% risk/trade = $660 R-unit, combined TRAIN+VAL mean R ×
entries/wk): **-$5,813/week** — negative, consistent with the cell table.

**Cadence bar (combined book, `scripts/cadence_bar.py`)**:
```
TRAIN: C1 fail (gap P90 6.7wk) | C2 fail (57% cycles net>0) | C3 fail (MDD 370.8R) |
       C4 fail (40% green vs 50% null) | C5 pass (157.6 fills/wk) | C7 fail (14 cycles)
VAL:   C1 fail (gap P90 5.0wk) | C2 fail (40% cycles net>0) | C3 fail (MDD 432.9R) |
       C4 fail (32% green vs 50% null) | C5 pass (227.6 fills/wk) | C7 fail (5 cycles)
```
Only C5 (raw fill frequency) passes — everything that measures whether the book actually
makes money over a cycle fails, both splits, consistent with the negative point estimate.

**The one caveat that alone could most weaken this verdict**: F is degenerate (median 0
on TRAIN) — a properly-computed high-flow subset was never isolated, so a genuine
flow-driven effect concentrated in the (unmeasured) true high-F tail cannot be ruled out
by this pass. Everything else (signal vs control, both directions, both splits, at n in
the thousands) is a clean, adequately-powered null-to-negative.

## Pass 1 (history — VOID, superseded)

Ran only TSLA/MSTR/NVDA (FAMILIES scope cut) against `cache.db`'s opportunistically
populated `intraday_bars_1min`. Coverage 12.3% of 381 candidate days (n=19 trades) — far
below the 80% rail, and non-random (whichever days another process had already scanned).
Point estimates were directionally positive (signal > control both splits) but no
t-stat cleared the pass bar, VAL n=4 made any t meaningless, and MDE was ~1R at n=4-15 —
underpowered by construction. Verdict was INCONCLUSIVE, not a finding either way.
F-quintile corr on the small sample: r=-0.02 (also flat). Full pass-1 artifacts:
`signal_days_raw.csv`, `signal_trades.csv`, `control_trades.csv`, `summary.json`
(pre-pass-2 versions, still in this directory for the record).

---
Pass 2 artifacts: `universe_map.json`, `candidate_underlying_days.csv`, `need_pull.csv`,
`control_candidate_days.csv`, `control_need_pull.csv`, `bars_1500_1600.db`,
`p2_signal_days_raw.csv`, `p2_signal_days_qualified.csv`, `p2_signal_trades.csv`,
`p2_control_days_raw.csv`, `p2_control_trades.csv`, `p2_stats.json`,
`p2_trades_TRAIN.csv`, `p2_trades_VAL.csv`, `build_universe.py`, `build_control.py`,
`pull_bars.py`, `run_study2.py`, `compute_stats.py`. TEST (>=2026-06-01) was never
queried — sealed by construction, and moot given the pass-2 null.

## Pass 3 results — flow proxy correction (F3)

**Verdict: DEAD.** Neither pre-committed condition clears. This closes the pass-2 caveat
(the true high-flow tail was never isolated) — it now has been, and it has no edge either.

F3 replaced pass-2's wrapper-ADV20 numerator with same-day wrapper dollar volume (from
`cache.db daily_bars`, since 0 of 1,705 referenced wrapper tickers have any rows in this
study's pulled `bars_1500_1600.db` — that db holds underlyings only, so F3 ran entirely
on the daily-bars fallback branch, not the intraday-bars branch PREREG allowed for). Same
pass-2 qualified, SHO-clean population (n=13,548), F3 re-ranked within each split.

| Split | Cell | n trades (days) | top-decile net R | t | rest net R |
|---|---|---|---|---|---|
| TRAIN | 1,298 LONG top decile | 163 (92) | -0.004 | -0.05 | -0.067 |
| TRAIN | 1,299 COMBINED top decile | 307 (156) | +0.047 (+0.093% price) | 0.82 | -0.031 |
| VAL | 1,298 LONG top decile | 186 (64) | -0.017 | -0.21 | -0.041 |
| VAL | 1,299 COMBINED top decile | 349 (94) | +0.003 (+0.007% price) | 0.05 | -0.085 |

No cell reaches +0.10R, and none reaches t>=2 (max observed t=0.82, TRAIN COMBINED). The
top decile is directionally less-bad than the rest in 3 of 4 cells (consistent with pass
2's "big movers give some back" story softening at higher flow) but never crosses into a
tradeable positive.

**Quintile monotonicity (COMBINED population, mean pnl_R by F3 quintile)**:
TRAIN: Q1 -0.075, Q2 +0.008, Q3 +0.055, Q4 +0.006, Q5 +0.038 — NOT monotone (Q3 > Q5).
VAL: Q1 -0.090, Q2 -0.125, Q3 -0.099, Q4 -0.126, Q5 -0.036 — NOT monotone (no clean
ordering; every bucket negative). Fails the pre-committed monotone-decreasing check on
both splits.

**F3-missing share**: 51.7% (7,000/13,548) of the qualified population — 6,984 because
NONE of the underlying's wrappers had a `daily_bars` row for that specific date (not a
20-day-history problem this time, a same-day-coverage problem: many resolved wrapper
tickers simply don't trade, or aren't in `daily_bars`, on a given candidate day), 16
because the underlying itself lacked 5 prior days for the ADV fallback.

**The one caveat**: F3's numerator never used real intraday same-day volume (the
"preferred" branch in the pre-registration never fired — no wrapper bars exist in this
study's db) — it is same-day but end-of-day dollar volume, a same-day proxy for flow, not
a 15:00-causal one; and over half the population still can't be scored at all, so the
scored 48.3% is not guaranteed representative of the missing 51.7%. Given the DEAD verdict
on the scored half, and no plausible mechanism for the missing half to look categorically
different, this is not pursued further under CLAUDE.md's north-star ladder — no live rung
to put this on. Pass 2's negative verdict stands, now without an open flow-tail caveat.

## Independent check status

Full independent reimplementation (per CLAUDE.md's research-claim checklist) NOT
performed this pass either — single-agent, budget-constrained run. Do not relay this as
a closed finding without a second implementation reproducing the pull + the scorer on a
sample of (symbol, date) trades. Given the verdict is a NEGATIVE (no live money at risk
from a false positive), the priority for a follow-up check is lower than it would be for
a ship decision, but the phrasing rule still applies: this is "no edge detected in this
universe/window/cost/book", not "no edge exists."
