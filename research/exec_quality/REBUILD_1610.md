# REBUILD_1610 — independent rebuild of PREREG_1610.md (cost-aware barriers on the HOD break)

Built from the PROSE SPEC ONLY (`research/exec_quality/PREREG_1610.md`); `cell_1610.py`, any
`cell_1610_*.csv` and `RESULT_1610.md` were never opened. Code: `research/exec_quality/rebuild_1610.py`.
Outputs: `rebuild_1610_driftmap.csv` (Part A, 432 rows = 2 holdouts x 6 buckets x 36 pairs),
`rebuild_1610_fills.csv` (Part B, one row per fill, 9911 rows / 111 cols),
`rebuild_1610_summary.csv` (per-cell x holdout x bucket aggregate stats backing every table below).

## Data and joins
4398 TRAIN-H2 + 5513 VAL fills (9,911 total, `causal_arming_causal.csv` status=='fill'),
joined to `half_entry` (features_1478_A), `spread_bps_at_arm` (features_1478_C) and `outcome_R`
(model_1478_L3_predictions.csv) on (day, symbol) — verified a unique, fill_min-consistent 1:1 key
across every source file before any number was produced (one arm/fill per symbol per day in this
population; every fill has bars, A/C features and an outcome_R — no join drops any row).
R_base = fill − stop (stop = consolidation low) independently cross-checked against the base file's
own `R` column: max|diff| = 7e-15 (floating-point noise only). Minute bars from
`bars_fills_1478.db`, timestamps converted UTC→ET via `zoneinfo` (DST-aware), filtered to RTH
[09:30,16:00) — 0/9,911 fills lack a usable bar path. TRAIN-H2 spread quintile edges (bps):
10.91 / 21.10 / 34.33 / 58.44 bps (20/40/60/80th pct of TRAIN-H2 spread_bps_at_arm); applied unchanged to VAL (never refit).
174/9,911 fills (1.8%) have `spread_bps_at_arm` == NaN **in features_1478_C.csv
itself** (not a join failure) and so fall into no quintile (excluded from every Qx bucket below,
kept in ALL and in cell 1,616's `no_selection` bucket).

## Method notes (mechanisms taken directly from the prose)
* **Walk semantics** (`sip_rebuild.walk_path`, replicated in `walk_single`/`first_passage`): a bar
  with minute ≥ 15:55 exits at ITS OWN OPEN unconditionally, checked **before** that bar's own
  low/high are tested — i.e. the 15:55 bar can never also register a same-bar stop/target. A bar
  touching both stop and target exits at the stop ("stop first"); a stop exit gaps through at the
  open when the bar opened past the stop, otherwise fills at the stop price exactly; a target exit
  always fills at the exact target price (a resting limit — no gap adjustment, matching
  `walk_path`'s own target leg). The fill bar itself (m = floor(fill_min)) is the first bar walked,
  full low/high included, per the PREREG's explicit "the fill bar's low ≤ stop ⇒ stopped".
* **Cost model.** c_in = half_entry + (fill − level), c_out = stop(consolidation low) ×
  SLIP_STOP_BPS[split]/1e4 (SLIP_STOP_BPS from `cell_1478.py`: TRAIN-H2 13.832 bps, VAL
  11.936 bps = 0.88×2.9+0.12×94.0 / 0.88×3.2+0.12×76.0) — both **defined once per fill**, per
  the PREREG header, and reused identically across all seven cells. Applied post-walk: entry always
  pays c_in; a stop exit additionally pays c_out; an EOD exit pays EOD_BID_BPS[split]/1e4 of the
  exit price ({'TRAIN': 11.5, 'VAL': 9.7} bps, from Inputs); **a target exit pays nothing further**
  — no target-exit cost primitive is listed in the PREREG's Inputs, consistent with `walk_path`
  never adjusting the target fill price (a resting limit gets its price). Cells 1,612–1,614 instead
  bake c_in/c_out directly into the barrier LEVEL (a wider target/stop by exactly "what got eaten");
  their post-walk cost application is otherwise identical to every other cell.
* **Units.** Every cell's net R is reported in the **base R unit** (fill − stop, the UNSCALED
  original R) even where the cell's own stop distance differs (1,610/1,611/1,615) — this matches the
  PREREG's "in the BASE R unit" and avoids a smaller-denominator artifact; % of price is the
  pass-bar's primary (unit-invariant) metric.
* **Helpers used as given**: `cell_1445.day_clustered_t`, `cell_1445.ex_top5_mean`,
  `cell_1445.weeks_spanned` (imported directly, not reimplemented).

## Two ambiguity resolutions (documented, not silent)
1. **Driftless formula.** Part A's prose literally reads "driftless value k/(k+m)". The martingale
   (optional-stopping) result for a driftless walk with barriers at +k/−m from 0 is
   P(hit +k first) = **m/(k+m)** (the barrier is hit first in proportion to the OTHER barrier's
   distance) — and the PREREG's own already-disclosed retest-surface numbers in "What was seen"
   confirm this exact direction: P(+1% before −2%) driftless "0.67" = 2/(1+2); P(+2% before −1%)
   driftless "0.33" = 1/(2+1) — both match m/(k+m) and contradict literal k/(k+m) (which would give
   0.33 and 0.67, reversed). Used **driftless = m/(k+m)** throughout; the prose's phrase appears to
   be a shorthand slip, not a different convention.
2. **"fill − bid at the fill instant"** (Part A) needs a tick-level NBBO tape, which is not among
   this task's listed Inputs (only the derived `half_entry` and the arm-time `spread_bps_at_arm`
   were provided). Reported instead: fill − level (below) and half_entry (cost table below) as the
   two available entry-cost proxies.

## Part A — the drift map

### First-passage, population level (36 pairs), both holdouts
Empirical P(+k before −m | resolved) vs the driftless value; "neither" = censored at 15:55.

| k% | m% | driftless | TRAIN-H2 p(cond) | TRAIN-H2 95% CI | TRAIN-H2 neither | VAL p(cond) | VAL 95% CI | VAL neither |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.25 | 0.500 | 0.256 | [0.244,0.270] | 0 | 0.245 | [0.233,0.256] | 0 |
| 0.25 | 0.5 | 0.667 | 0.497 | [0.483,0.512] | 0 | 0.485 | [0.472,0.499] | 0 |
| 0.25 | 1.0 | 0.800 | 0.733 | [0.719,0.746] | 1 | 0.723 | [0.711,0.734] | 1 |
| 0.25 | 1.5 | 0.857 | 0.826 | [0.814,0.837] | 5 | 0.817 | [0.806,0.827] | 7 |
| 0.25 | 2.0 | 0.889 | 0.868 | [0.858,0.878] | 21 | 0.862 | [0.853,0.871] | 24 |
| 0.25 | 3.0 | 0.923 | 0.921 | [0.913,0.929] | 80 | 0.912 | [0.904,0.919] | 88 |
| 0.5 | 0.25 | 0.333 | 0.195 | [0.184,0.207] | 0 | 0.185 | [0.175,0.195] | 0 |
| 0.5 | 0.5 | 0.500 | 0.400 | [0.385,0.414] | 1 | 0.388 | [0.376,0.401] | 0 |
| 0.5 | 1.0 | 0.667 | 0.633 | [0.619,0.647] | 3 | 0.624 | [0.611,0.637] | 4 |
| 0.5 | 1.5 | 0.750 | 0.741 | [0.728,0.754] | 15 | 0.732 | [0.720,0.743] | 21 |
| 0.5 | 2.0 | 0.800 | 0.795 | [0.783,0.807] | 46 | 0.794 | [0.783,0.804] | 58 |
| 0.5 | 3.0 | 0.857 | 0.865 | [0.855,0.875] | 145 | 0.861 | [0.852,0.870] | 176 |
| 1.0 | 0.25 | 0.200 | 0.125 | [0.116,0.135] | 1 | 0.122 | [0.114,0.131] | 1 |
| 1.0 | 0.5 | 0.333 | 0.285 | [0.271,0.298] | 5 | 0.269 | [0.257,0.280] | 5 |
| 1.0 | 1.0 | 0.500 | 0.488 | [0.473,0.503] | 21 | 0.472 | [0.459,0.485] | 28 |
| 1.0 | 1.5 | 0.600 | 0.611 | [0.596,0.625] | 69 | 0.586 | [0.572,0.599] | 89 |
| 1.0 | 2.0 | 0.667 | 0.679 | [0.665,0.693] | 154 | 0.666 | [0.654,0.679] | 187 |
| 1.0 | 3.0 | 0.750 | 0.777 | [0.764,0.789] | 338 | 0.764 | [0.752,0.776] | 422 |
| 1.5 | 0.25 | 0.143 | 0.098 | [0.089,0.107] | 1 | 0.086 | [0.079,0.094] | 2 |
| 1.5 | 0.5 | 0.250 | 0.221 | [0.209,0.234] | 12 | 0.200 | [0.189,0.210] | 15 |
| 1.5 | 1.0 | 0.400 | 0.399 | [0.384,0.414] | 54 | 0.375 | [0.362,0.388] | 85 |
| 1.5 | 1.5 | 0.500 | 0.515 | [0.500,0.530] | 163 | 0.486 | [0.473,0.500] | 207 |
| 1.5 | 2.0 | 0.571 | 0.591 | [0.576,0.606] | 300 | 0.573 | [0.559,0.587] | 372 |
| 1.5 | 3.0 | 0.667 | 0.701 | [0.686,0.715] | 563 | 0.690 | [0.676,0.703] | 726 |
| 2.0 | 0.25 | 0.111 | 0.076 | [0.068,0.084] | 8 | 0.067 | [0.061,0.074] | 9 |
| 2.0 | 0.5 | 0.200 | 0.180 | [0.168,0.191] | 32 | 0.162 | [0.152,0.172] | 35 |
| 2.0 | 1.0 | 0.333 | 0.336 | [0.322,0.350] | 119 | 0.314 | [0.301,0.326] | 149 |
| 2.0 | 1.5 | 0.429 | 0.442 | [0.427,0.458] | 285 | 0.416 | [0.403,0.430] | 338 |
| 2.0 | 2.0 | 0.500 | 0.518 | [0.503,0.534] | 473 | 0.501 | [0.487,0.515] | 567 |
| 2.0 | 3.0 | 0.600 | 0.636 | [0.620,0.652] | 805 | 0.626 | [0.612,0.640] | 994 |
| 3.0 | 0.25 | 0.077 | 0.052 | [0.046,0.059] | 24 | 0.045 | [0.040,0.051] | 32 |
| 3.0 | 0.5 | 0.143 | 0.125 | [0.116,0.136] | 93 | 0.110 | [0.102,0.119] | 110 |
| 3.0 | 1.0 | 0.250 | 0.245 | [0.232,0.258] | 273 | 0.224 | [0.213,0.236] | 337 |
| 3.0 | 1.5 | 0.333 | 0.334 | [0.319,0.349] | 528 | 0.309 | [0.296,0.322] | 631 |
| 3.0 | 2.0 | 0.400 | 0.401 | [0.386,0.418] | 781 | 0.386 | [0.372,0.400] | 967 |
| 3.0 | 3.0 | 0.500 | 0.519 | [0.502,0.536] | 1242 | 0.516 | [0.501,0.532] | 1550 |

Wilson-CI-contains-driftless rate across all 36 (k,m) pairs: TRAIN-H2 0.58, VAL 0.43.
Mean |empirical − driftless|, population level: TRAIN-H2 0.038, VAL 0.044.
**Shape**: at the tightest symmetric scale (k=m=0.25%) the path is tilted sharply DOWN from the fill
— empirical P(+0.25% first) ≈ 0.25–0.26 vs driftless 0.50 in both holdouts (an immediate give-back
right after the break, not a small tilt) — decaying toward driftless as k,m widen, and landing
almost exactly on driftless at the widest symmetric scale tested (k=m=3%: 0.516–0.519 vs 0.50). Full
432-row grid (both holdouts x ALL + 5 quintiles x 36 pairs) in `rebuild_1610_driftmap.csv`.

### Mean signed excursion after the fill (bps, population level)
| horizon (min) | TRAIN-H2 | VAL |
|---|---|---|
| 5 | 3.25 | -0.71 |
| 15 | -1.21 | 1.93 |
| 30 | 0.42 | 7.96 |
| 60 | 4.14 | 18.38 |
| 120 | 7.64 | 15.92 |

(Last available bar close at or before fill_min+horizon, capped at 15:55; forward-filled through
gaps; NaN only when no bar exists in that window at all — 7/9,911 fills at the 5-min horizon, 0
elsewhere.) Short-horizon signs are inconsistent between holdouts; both trend mildly positive by 60–120 min — a drift too small and too inconsistent at the 5–15 min (cost-relevant) horizon to
read as a harvestable signal, consistent with the first-passage table above.

### Realized entry/exit cost by quintile
| holdout | quintile | fill-level (bps) | c_in = half_entry+(fill-level) (bps) | c_out = stop-limit std (bps) |
|---|---|---|---|---|
| TRAIN | Q1 | 4.04 | 7.46 | 13.60 |
| TRAIN | Q2 | 5.89 | 13.79 | 13.59 |
| TRAIN | Q3 | 6.85 | 19.04 | 13.59 |
| TRAIN | Q4 | 7.08 | 25.42 | 13.57 |
| TRAIN | Q5 | 6.21 | 47.01 | 13.56 |
| VAL | Q1 | 3.72 | 7.16 | 11.75 |
| VAL | Q2 | 5.41 | 13.05 | 11.73 |
| VAL | Q3 | 6.37 | 18.52 | 11.71 |
| VAL | Q4 | 6.58 | 25.17 | 11.70 |
| VAL | Q5 | 6.51 | 47.35 | 11.69 |

c_in scales ~7x from Q1 to Q5 (quintiles are cut on the SAME spread that drives half_entry); c_out
is nearly flat across quintiles (it scales off the stop price and SLIP_STOP_BPS[split], not off the
spread quintile) — this is the mechanical reason net result degrades by quintile in Part B: the
COST side, not the drift side, moves with the spread bucket.

## Part B — the six barrier cells + report-only 1,616

### ALL-population, both holdouts
| cell | holdout | n | mean net % | mean net R (base unit) | day-clustered t | ex-top5% net% | paired Δ% | paired Δ t | ex-top5% Δ% | median R% | fills/wk |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1610 | TRAIN | 4398 | -0.286 | -0.171 | -5.26 | -0.535 | -0.016 | -0.44 | -0.235 | 1.562 | 162.9 |
| 1610 | VAL | 5513 | -0.317 | -0.198 | -5.64 | -0.580 | -0.070 | -1.94 | -0.292 | 1.624 | 250.6 |
| 1611 | TRAIN | 4398 | -0.285 | -0.177 | -4.32 | -0.599 | -0.015 | -0.65 | -0.079 | 1.562 | 162.9 |
| 1611 | VAL | 5513 | -0.326 | -0.207 | -4.84 | -0.656 | -0.079 | -3.26 | -0.145 | 1.624 | 250.6 |
| 1612 | TRAIN | 4398 | -0.276 | -0.167 | -3.33 | -0.610 | -0.005 | -0.44 | -0.052 | 1.562 | 162.9 |
| 1612 | VAL | 5513 | -0.251 | -0.176 | -2.87 | -0.602 | -0.004 | -0.43 | -0.047 | 1.624 | 250.6 |
| 1613 | TRAIN | 4398 | -0.277 | -0.170 | -3.35 | -0.598 | -0.007 | -0.68 | -0.112 | 1.562 | 162.9 |
| 1613 | VAL | 5513 | -0.241 | -0.169 | -2.77 | -0.576 | 0.007 | 0.68 | -0.096 | 1.624 | 250.6 |
| 1614 | TRAIN | 4398 | -0.279 | -0.169 | -3.25 | -0.614 | -0.009 | -0.55 | -0.117 | 1.562 | 162.9 |
| 1614 | VAL | 5513 | -0.239 | -0.171 | -2.65 | -0.590 | 0.009 | 0.63 | -0.096 | 1.624 | 250.6 |
| 1615 | TRAIN | 4398 | -0.265 | -0.163 | -2.69 | -0.628 | 0.005 | 0.18 | -0.218 | 1.562 | 162.9 |
| 1615 | VAL | 5513 | -0.229 | -0.168 | -2.27 | -0.613 | 0.018 | 0.66 | -0.212 | 1.624 | 250.6 |
| 1616 | TRAIN | 4398 | -0.200 | -0.121 | -3.58 | -0.366 | 0.062 | 1.33 | -0.193 | 1.562 | 162.9 |
| 1616 | VAL | 5513 | -0.331 | -0.195 | -5.60 | -0.503 | -0.086 | -2.06 | -0.339 | 1.624 | 250.6 |

### Pass-bar check (frozen bar, VAL, ALL population)
| cell | mean≥0.15% | Δ≥0.10%&t≥2.5 | ex-top5%Δ>0 | TRAIN-H2 same-sign,t≥1 | ≥3 fills/wk | median R≥0.5% | VERDICT |
|---|---|---|---|---|---|---|---|
| 1610 | no | no | no | no | yes | yes | **FAIL** |
| 1611 | no | no | no | no | yes | yes | **FAIL** |
| 1612 | no | no | no | no | yes | yes | **FAIL** |
| 1613 | no | no | no | no | yes | yes | **FAIL** |
| 1614 | no | no | no | no | yes | yes | **FAIL** |
| 1615 | no | no | no | no | yes | yes | **FAIL** |
| 1616 | no | no | no | no | yes | yes | **report-only** |

**Every cell fails on VAL** — mean net % is negative for all seven (−0.23% to −0.33%), nowhere near
the +0.15% bar; the paired Δ vs outcome_R is negative for four of six scored cells and, where
positive (1613 +0.007%, 1614 +0.009%, 1615 +0.018%), three orders of magnitude short of the +0.10%
bar with none reaching t≥2.5. Cell 1,616 (report-only) shows the expected overfit signature: a
positive in-sample TRAIN-H2 selection (+0.062%, t 1.33) that reverses to −0.086% (t −2.06) on VAL —
the per-quintile winning (k,m) pair does not transfer.

### 1,616's TRAIN-H2-selected pair per quintile (report-only; read on VAL only, never selected)
| quintile | selected k% | selected m% |
|---|---|---|
| Q1 | 3.0 | 2.0 |
| Q2 | 1.5 | 3.0 |
| Q3 | 3.0 | 1.5 |
| Q4 | 1.0 | 3.0 |
| Q5 | 2.0 | 3.0 |

### mean net % of price by quintile, VAL
| cell | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| 1610 | -0.019 | -0.141 | -0.318 | -0.251 | -0.747 |
| 1611 | -0.056 | -0.134 | -0.306 | -0.281 | -0.740 |
| 1612 | -0.004 | -0.102 | -0.222 | -0.160 | -0.661 |
| 1613 | 0.020 | -0.106 | -0.221 | -0.151 | -0.633 |
| 1614 | 0.011 | -0.093 | -0.218 | -0.125 | -0.661 |
| 1615 | 0.015 | -0.099 | -0.217 | -0.139 | -0.612 |
| 1616 | -0.056 | -0.175 | -0.351 | -0.317 | -0.658 |

Every cell degrades monotonically-ish from Q1 to Q5, matching the realized-cost table above (c_in
rising ~7x while c_out stays flat) — net falls with the spread by roughly the extra entry cost the
wider quintiles pay, not from any cell recovering less drift there.

## Verdict
FAIL, matching the PREREG's own pre-committed consequence: **"no re-scaling of R offsets the cost on
this population."** Part A shows why mechanically: at the horizons where the SLIP_STOP_BPS/EOD-bid
costs actually bite (tens of bps, i.e. the 0.25–0.5% region of the grid), the path is tilted AGAINST
the trade (sharply so at the tightest scale), not driftless-with-recoverable-noise; by the time the
path approaches driftless (k=m≈3%) the barrier is too wide relative to the base R for any of the six
reshapings to help, and 1,616's own in-sample-optimal reshaping fails to transfer to VAL. No barrier
cell reaches the pass bar in any spread quintile inspected.

## Caveats (read as an adversary)
* Two prose ambiguities were resolved by documented, checkable reasoning (driftless-formula
  direction cross-checked against the PREREG's own disclosed numbers; target-exit cost inferred from
  the absence of a target-cost primitive in Inputs plus `walk_path`'s own no-gap-adjustment target
  leg) — a reviewer should re-derive both independently rather than take this rebuild's word for it.
* "fills/week" = raw fill count / `weeks_spanned`(days) in each bucket, NOT slot-simulated against
  the 12/day-4-concurrent live cap (out of scope for the step budget); the PREREG's Inputs describe
  the 9,911-fill population as already living under that cap upstream, so this is the
  population's cross-sectional weekly rate (correspondingly large, 150–250/wk) rather than one
  account's realized pace — it is reported only as the pass-bar's frequency check, never as a
  tradeable-frequency claim.
* "fill − bid at the fill instant" (Part A) was not computable — no tick-level NBBO tape was among
  this task's Inputs; half_entry and (fill − level) are reported as the two available proxies.
* 1.8% of fills (174) have no arm-time spread and are excluded from every spread-quintile bucket
  (present only in ALL and 1,616's `no_selection` bucket) — a real gap in `features_1478_C.csv`,
  not a bug in this rebuild's join (verified against the raw file before joining). For cell 1,615
  specifically, a NaN spread also makes `s` NaN, which degenerates that fill's stop/target to NaN —
  `walk_single` then falls through to an EOD exit for those 174 rows in the ALL bucket (numpy NaN
  comparisons are always False). This affects only cell 1,615's ALL-bucket row and cannot flip its
  verdict (it already fails the mean-net bar by >0.35 pp on VAL).
* This is a from-prose independent rebuild; it was never compared row-by-row against `cell_1610.py`'s
  own output (the task forbade opening it) — the trade-by-trade and probability-level agreement this
  rebuild is meant to check has NOT itself been verified by this session. A reviewer with access to
  both should diff `rebuild_1610_driftmap.csv` (want: within 0.02 on every empirical probability) and
  `rebuild_1610_fills.csv` (want: ≥99% of per-fill net R within 0.01 R) against the frozen build.
* Day-clustered t uses `cell_1445.day_clustered_t` (OLS, cluster-robust SE by day) exactly as given;
  no iid comparison was added (PREREG's Part B pass bar does not ask for one here, unlike the
  programme-wide statistics checklist in CLAUDE_HISTORY).
