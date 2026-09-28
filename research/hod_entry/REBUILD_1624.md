# REBUILD 1,624 — break breadth (the crowd) — independent rebuild from prose

Source spec: `research/hod_entry/PREREG_1623.md`, section "## Cell 1,624 — break breadth (the crowd)"
(frozen 2026-09-28 17:15 UTC). Written **without opening** `cell_1624.py`, `cell_1624_fills.csv` or
`RESULT_1624.md` (CLAUDE.md independent-reimplementation rule #1 — catches coding errors, not spec
errors). Code: `research/hod_entry/rebuild_1624.py`. Row-level output: `rebuild_1624_fills.csv`
(9,911 rows, one per causal-arming fill).

## Mechanism (verbatim from the prose)
> B30(f) = the number of ARM events (any status, all names) in the 30 minutes before f's fill
> minute; B60 likewise. Terciles set on TRAIN-H2 by minute-of-day-adjusted rank (the count rises
> through the morning — rank within the same hour bucket, so the gate is not a time-of-day gate).
> Gate: trade f only in the TOP tercile of B30 (pre-declared); report the bottom tercile and B60
> beside. Same statistics as 1,623.

"Same statistics as 1,623" (per that cell's own text): n kept, mean net R, day-clustered t,
ex-top-5%, fills/week at 12/4, the kept-vs-dropped difference. (1,623's autocorrelation-of-F table
is specific to that cell's F(f) construct; it has no 1,624 analogue and is not reproduced.)

## Headline number (this task's ask)
**Top-tercile (B30) kept, VAL: mean net R = −0.1689, day-clustered t = −3.102, n = 3,193.**

This is *negative* and does not clear the frozen pass bar for 1,623/1,624 (kept mean net R ≥ +0.15,
t ≥ 2.5): the gate fails on sign, not just magnitude.

## The one caveat that matters most: "any status" is not achievable from the sanctioned inputs
The prose's B30/B60 count "ARM events (**any status**, all names)". `causal_arming_causal.csv` has
33,852 arm rows: `fill` 9,911, `nofill` 2,010 (armed ≥1×, never filled — `n_cross` ≥ 1), `not_armed`
21,931 (`n_cross` == 0, never armed at all). Checked directly: for every `nofill`/`not_armed` row,
`fill_min`/`exit_m`/`level` are **100% null**, and `n_cross` is a per-(day,symbol) *count* with no
per-event timestamp — there is no way to place a `nofill` symbol's arm attempt(s) on the clock from
this file. The only column in any sanctioned input carrying a genuine arm **minute** is `arm_m` in
`features_1478_A.csv`, and it exists only for the 9,911 `fill` rows (row counts verified identical —
9,911 — across `features_1478_A.csv`, `model_1478_L3_predictions.csv`, `rebuild_1481_fills.csv` and
`cell_1445_features.csv`; this whole downstream research line evidently operates at fill-only
granularity). **B30/B60 here therefore count other FILLS' `arm_m` only**: "all names" is honored
(every symbol's fills count); "any status" is **not** — nofill/not_armed arm attempts (≈40% of all
arm ATTEMPTS by row count, though most of that is the zero-cross `not_armed` bucket which arguably
carries the least "crowd" information anyway) are invisible to this count. This is "breadth among
causal-arming fills," a systematic undercount of the prose's literal crowd, not an invented
shortcut — it is the most defensible reading of what timestamped data actually exists. **If the
original `cell_1624.py` used a different, broader event log not listed among this task's sanctioned
inputs, its numbers will differ from this rebuild's for a real, explainable reason — check that
file's data source first before assuming either build is wrong.**

## Other pre-declared implementation choices (fixed before any number was computed)
- **Self-exclusion**: f's own arm event does not count toward its own B30/B60. Rationale: breadth is
  the crowd *around* a fill, not the fill itself — and since arm_m precedes fill_min by a median of
  0.26 min (16 s; max 34.6 min), counting self would add a near-constant +1 to almost every row.
- **Hour bucket** = `floor(fill_min / 60)` (ET clock hour), using the fill's own `fill_min` — the
  entity being gated. Buckets present: 9, 10, 11, 12, 13, 14 (none at 15 — this population's fills
  stop by ~3pm; not investigated further, consistent with HOD-breaks concentrating early).
- **Window**: half-open `[fill_min − W, fill_min)` — strictly before the fill minute.
- **Tercile cutoffs**: per hour bucket, TRAIN-H2's 33.33rd/66.67th percentiles of B30 (resp. B60);
  `bottom` = value ≤ p33, `top` = value ≥ p66, `mid` = strictly between. Applied to **both** holdouts
  using only TRAIN-H2-derived cutoffs (VAL never contributes to its own cutoff).
- `outcome_R` = `model_1478_L3_predictions.csv`'s `outcome_R` directly (the task's routing note
  labels this "standard-cost net R"); this is the "net R" the statistics operate on.
- `exit_m`: **not present** in `model_1478_L3_predictions.csv` (header checked directly — columns
  are day, symbol, fill_min, split, why, outcome_R, L3, store_served_1438, hgb_prob_L3, hgb_kept_L3,
  lr_prob_L3, lr_kept_L3). Used `rebuild_1481_fills.csv` column `exit_m` instead (join key
  day+symbol+fill_min), needed **only** for the fills/week concurrency slotting — nowhere else in
  this cell's mechanism. 938/9,911 fills (9.5%) had no `exit_m` match and are excluded from the
  fills/week simulation only (their `outcome_R`/breadth/tercile stats are otherwise complete).
- `bars_fills_1478.db` and SPY/IWM minute bars were **not needed**: the breadth signal is built
  purely from arm/fill event minutes already on disk, and `outcome_R` is pre-computed. (SPY/IWM
  bars are cell 1,625's instrument leg, not this cell's — left untouched, `data/cache.db` never
  opened.)

## Tercile-split fidelity (discrete-count caveat)
B30 is a small-range integer (0–115, median 8), so percentile cutoffs land on exact integers and
ties at the boundary are common; with `≤ p33` / `≥ p66` both closed, ties go to bottom/top first,
shrinking `mid` below an exact third (TRAIN-H2 overall: top 38%, bottom 38%, mid 24%, vs. a target
33/33/33 — confirmed this is a per-hour-bucket tie effect, not a bug, by checking the crosstab: hours
9–12 are all within a few points of thirds). Two late hour buckets are thin: hour 13 (TRAIN n=316,
cutoffs 1/2) and hour 14 (TRAIN n=6, cutoffs 1/1) — consecutive-integer or degenerate cutoffs leave
**no room for `mid`** in either (any integer is ≤1 or ≥2, or ≤1 or ≥1), so "top tercile" there is
really a median split on a near-void sample. This affects 322/4,398 TRAIN-H2 rows (7.3%) and is
disclosed, not corrected post hoc (correcting it now, after seeing the number, would violate
"tuning the thresholds after a number" — the "Not allowed" line in the PREREG).

## Full results table

| window | split | tier    | n    | mean net R | day-clust t | ex-top-5% | cache-only % |
|--------|-------|---------|-----:|-----------:|------------:|----------:|-------------:|
| B30    | TRAIN | top     | 1675 | −0.1384    | −2.143      | −0.2498   | 21.9 |
| B30    | TRAIN | mid     | 1070 | −0.2150    | −3.611      | −0.3312   | 18.5 |
| B30    | TRAIN | bottom  | 1653 | −0.1644    | −3.674      | −0.2777   | 20.3 |
| B30    | TRAIN | dropped | 2723 | −0.1843    | −4.699      | −0.2979   | 19.6 |
| B30    | **VAL** | **top** | **3193** | **−0.1689** | **−3.102** | −0.2817 | 18.4 |
| B30    | VAL   | mid     | 1212 | −0.1610    | −3.697      | −0.2737   | 17.2 |
| B30    | VAL   | bottom  | 1108 | −0.1858    | −3.975      | −0.2983   | 20.9 |
| B30    | VAL   | dropped | 2320 | −0.1728    | −4.911      | −0.2855   | 19.0 |
| B60    | TRAIN | top     | 1584 | −0.1437    | −2.232      | −0.2546   | 21.7 |
| B60    | TRAIN | dropped | 2814 | −0.1798    | −4.630      | −0.2936   | 19.7 |
| B60    | VAL   | top     | 3266 | −0.1878    | −3.555      | −0.3010   | 18.5 |
| B60    | VAL   | dropped | 2247 | −0.1455    | −3.595      | −0.2564   | 19.0 |

kept-minus-dropped: B30 TRAIN +0.0459, B30 VAL **+0.0040**; B60 TRAIN +0.0362, B60 VAL **−0.0423**
(sign-flips — the "beside" window doesn't even agree with B30 on direction in VAL).

fills/week (12/day, 4-concurrent slotting): B30 top TRAIN 21.52/wk, VAL 38.50/wk (well above the
≥3/wk pass-bar floor — frequency was never the constraint here).

Unconditional (ungated) baseline for comparison: TRAIN mean −0.1668 (n=4,398), VAL mean **−0.1705**
(n=5,513). **The VAL top-tercile mean (−0.1689) is statistically indistinguishable from the VAL
unconditional mean (−0.1705) and from VAL-dropped (−0.1728)** — B30 does not separate outcome in
VAL. The much larger TRAIN-H2 apparent lift (kept −0.1384 vs. dropped −0.1843, Δ +0.0459) does not
reproduce out of sample; whatever produced it in TRAIN-H2 did not generalize.

Cache-only share (`store_served_1438`): B30 top-tercile VAL = 18.4%, within 5pp of the 19.5%
pass-bar reference — the one sub-check this cell does clear, though it's moot given the R miss.

## Pass bar (frozen, PREREG_1623.md "Pass bar for 1,623/1,624") — not cleared
Kept mean net R ≥ +0.15: **FAIL** (−0.169). Day-clustered t ≥ 2.5: **FAIL** (−3.10, wrong sign).
Dropped < kept on both holdouts: TRAIN yes (−0.184 < −0.138), VAL marginal-yes (−0.173 < −0.169, a
0.004R gap). TRAIN-H2 same sign: both negative, so "same sign" only in the sense both fail — not the
intended same-signed-positive-edge check. Placebo/shuffle margin ≥ +0.10R t≥2: **not computed** —
the shuffle procedure is fully specified only for 1,623 ("the OTHER holdout's days shuffled, seed
1623"); 1,624 inherits it by "same statistics as 1,623" but the exact mechanic for a tercile gate
isn't spelled out, and reproducing 1,623's day-shuffle faithfully was out of scope for this task's
explicit ask (top-tercile kept VAL mean/t/n); flagged rather than guessed. Given the kept-vs-dropped
gap in VAL is already only +0.004R, a placebo margin ≥ +0.10R is very unlikely to be met regardless.

## Not done in this task (explicitly out of scope, not silently skipped)
- No comparison against `cell_1624.py` / `cell_1624_fills.csv` / `RESULT_1624.md` — this rebuild was
  built blind to them by design; the Jaccard/mean-agreement check against the original is a separate
  step for whoever holds both builds.
- No causality/look-ahead refuter pass, no day-cohort-leakage check, no tail/day-concentration audit
  beyond ex-top-5% — PREREG_1623.md's "Independent check and consequences" section lists these for
  the *programme*, not uniquely for this rebuild task.

## Files
- `research/hod_entry/rebuild_1624.py` — rebuild code (self-contained; verbatim-copies the sanctioned
  `day_clustered_t` / `ex_top5_mean` / `fills_per_week` + `simulate_slots` helpers from
  `cell_1445.py` / `research/hod_consol/run_consol.py` rather than importing those scripts as
  modules, to avoid import-time side effects).
- `research/hod_entry/rebuild_1624_fills.csv` — 9,911 rows: day, symbol, split, fill_min, arm_m,
  exit_m, hour, b30, b60, tercile_b30, tercile_b60, kept_top_b30, kept_top_b60, outcome_R,
  store_served_1438.
- `research/hod_entry/REBUILD_1624.md` — this file.
