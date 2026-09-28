# Refuter — PREREG_1610 (cells 1,610–1,616, cost-aware barriers on the HOD break)

**Verdict: REFUTED as stated. Two defects each change a verdict the result reports. The headline FAIL
(no cell passes, no re-scaling of R offsets the cost) SURVIVES every convention tested.**

Scripts: `review/refuter_1610.py` → `review/refuter_1610_out.txt`, and `review/refuter_1610_net.py` →
`review/refuter_1610_net_out.txt`. Both import `cell_1610`'s own `load_fills`, `locate`, `net_cost` and
`walk_cell`, so every difference reported here comes from a convention or estimator choice. It cannot
come from reimplementing the logic. Reproduction check: the builder's `p_up_before_down` is reproduced
exactly on all 432 rows (max diff 0.000000).

## Defect 1 (changes the Part A verdict): the "consistent downward tilt, never above driftless" is an estimator artifact

The builder reads P(+k before −m) as UNCONDITIONAL and uses the trading walk, so three biases, all
pointing the same way, push it below m/(k+m) even on a driftless path:

| bias | size (ALL, both holdouts) |
|---|---|
| 15:55 censoring counted as a failure. m/(k+m) is an infinite-horizon, resolved-path benchmark. | "neither" is 0 % at small pairs but 18–28 % at (2–3 %, 2–3 %). At (3,3) the unconditional P is 0.372, the conditional P is 0.519, and driftless is 0.5. |
| The fill bar's own low counts as a down touch. 72.6 % of fill bars OPEN below the HOD level, so that low is at least partly pre-fill. | Stopped ON the fill bar: 62–64 % of all fills at m = 0.25 %, 30–33 % at m = 0.5 %, 7–8 % at m = 1 %. In 36–39 % of fills at m = 0.25 %, the "stop" is booked at the fill bar's OPEN, a print that came **before the entry**. |
| Stop-first tie on a bar that touches both levels. | 21 % of fills at (0.25, 0.25), 10–11 % at (0.25, 0.5). |

Re-estimated with a post-fill fill-bar rule (the fill bar's high counts up, which is post-trigger for a
buy-stop; only its close counts down), with ties ordered by the open and otherwise split, and with the
censoring-exact martingale test E[X_τ] = 0 (optional stopping, day-clustered t):
* Conditional P vs driftless: TRAIN is below on 6/36 pairs (mean **+0.022**), VAL on 14/36 (mean
  **+0.009**). The builder had 35/36 (−0.048) and 36/36 (−0.061).
* E[X_τ] builder-convention: 19 / 24 pairs have t < −2, none have t > 2. Post-fill: TRAIN 4 below / 10
  above, VAL 5 / 5. Across quintiles, 19 (q,k,m) cells are positive in BOTH holdouts with t > 2, all
  at m = 0.25 % (ALL, k ≥ 0.5 %: +0.03 to +0.09 %, t 2.2–6.2). 11 are negative in both (k = 0.25 %
  with wide m, and Q5).
* The m ≤ 0.5 % region is **not identified from minute bars**. The pessimistic bound (builder) is
  −0.21 to −0.25 % and the optimistic bound (post-fill) is +0.03 to +0.09 %. The within-bar order is
  unknown, and only the tape can resolve it.
* The builder's Wilson intervals (29 TRAIN / 36 VAL pairs "entirely below") are iid binomial on
  day-clustered fills, over 36 pairs that share the same paths. They are not 36 confirmations.
* "Below driftless inside every quintile, so no bucket hides a positive drift" falls for the same
  reason.

**What the map legitimately says:** no deviation is large enough to pay the cost, at either bound.
The cost is about 0.26 % of price. Net of the PREREG standard cost (half_entry + stop 13.8/11.9 bps +
EOD 11.5/9.7 bps), only **2 of 216** (q,k,m) barriers are positive in both holdouts at the optimistic
bound: Q1 (2 %, 3 %) and Q1 (3 %, 3 %), at +0.02 to +0.04 %, t 0.17–0.39. The same 2 appear at the
pessimistic bound. The best in Q2–Q5 is negative. This is a multiplicity-level null, far below the
+0.15 % bar.

## Defect 2 (changes the paired-Δ verdict): "1,612–1,614 post a real, significant paired improvement (t up to 6.7)" is a cost-convention offset

The builder prices its cells with `net_cost`, which is the PREREG standard: `half_entry` from
features_1478_A, SLIP_STOP_BPS and SLIP_EOD_BPS. The paired base it uses is `outcome_R` from
model_1478_L3, which follows the cell_1457/1445 standard, where half_entry is recovered from
`cost_R` by `corrected_cost`. The two conventions differ:
* Re-walking the base rule with the builder's own `net_cost` gives **+0.054 % TRAIN / +0.052 % VAL**
  more than `base_outcome_pct`. The walk agrees with the causal base on 98.6 % of rows. On target
  rows the entry cost is 0.098 R vs 0.129 R.
* **Identical-exit test:** on the 5,765 rows where cell 1,612 and the base exit at the same stop, the
  reported Δ is **+0.042 %** where it must be 0. Against the same-convention base it is −0.0000 %.
* The same-convention paired Δ on VAL: 1,612 **+0.003 % (t 0.36)**, 1,613 +0.014 (t 1.59), 1,614
  +0.016 (t 1.22), 1,615 +0.022 (t 0.81). Cells 1,610 and 1,611 come out at **−0.064 (t −1.81)** and
  **−0.073 (t −3.07)**, so the tight stop is worse than the base, not neutral. On TRAIN, 1,612–1,614
  are all within ±0.003.
* 1,616 (report-only) against the same-convention base: TRAIN +0.074 (t 1.55) → VAL −0.080 (t −1.92).
  The TRAIN-selected pair reverses on VAL. Selection is TRAIN-only in the code, and the VAL read is
  not selected.

## Gross beside net (the "no double charge" lens)

| VAL | base | 1610 | 1611 | 1612 | 1613 | 1614 | 1615 | 1616 |
|---|---|---|---|---|---|---|---|---|
| gross % | +0.068 | +0.002 | +0.002 | +0.075 | +0.081 | +0.087 | +0.093 | −0.038 |
| net % (builder) | −0.195 | −0.259 | −0.268 | −0.192 | −0.181 | −0.179 | −0.170 | −0.273 |

Re-scaling moves gross by at most ±0.07 %, against a cost of 0.23–0.27 %.
* The builder does not double-charge. `c_in` and `c_out` only move the barrier levels, and net charges
  half_entry once plus the exit slip.
* **The REBUILD does double-charge.** `rebuild_1610.net_result` returns `(exit − fill) − c_in − …`
  with `c_in = half_entry + (fill − level)`, so the entry slippage already embedded in the fill price
  is charged a second time. That is ≈ 0.06 % of price: the rebuild's 1,612 VAL is −0.251 vs the
  builder's −0.192. Its paired Δ ≈ 0 is a coincidence of two different offsets.
* Also confirmed from source: the builder's `net_R` uses the cell's OWN stop distance
  (`R_new = fill − stop_price`), not the BASE R unit the PREREG requires. Its `mean_net_R` and
  `ex_top5_R` are therefore mislabeled for 1,610/1,611/1,613–1,616. There is no verdict effect,
  because the bar is in % of price.
* Together with the rebuild's EOD fallback on the 174 no-spread rows (1,615/1,616), these three
  convention differences explain the Part B row-agreement failure (0.2–13 %) in `1610_compare.md`.
  The walk itself is not the cause.

## Other lenses (clean or immaterial)
* **Quintile edges** use TRAIN only: recomputed as [0, 10.91, 21.1, 34.33, 58.44, 558.23] bps, an
  exact match. VAL's own edges would have been [0, 12.7, 23.7, 37.8, 63.0, 496.5]. VAL falls
  17/18/20/22/23 % into the TRAIN quintiles, so they are applied causally.
* **Cost fields.** `half_entry` is joined on (day, symbol, fill_min), i.e. at the fill instant. The
  1,615 scale uses `spread_bps_at_arm`, the causal field. The builder drops the 174 no-spread fills
  from 1,615/1,616 and logs a WARNING; this is correct. The rebuild instead walks them to a fabricated
  EOD exit.
* **Fill-bar and gap-through in Part B.** Stops booked at the pre-fill open of the fill bar are 0.8 %
  of rows in 1,610/1,611 and ≤ 0.16 % elsewhere, costing ≤ 0.002 % of price. Re-walking Part B
  entirely under the post-fill rule leaves VAL net at −0.17 to −0.25 % and the same-rule Δ at ≈ 0
  (1,611 VAL −0.060, t −2.6). The FAIL is robust to the convention.
* **Positive groups** are VAL Q1 of 1,610 and 1,612–1,615, and TRAIN Q1 of 1,616. They show +0.02 to
  +0.06 % with t 0.25–0.60 and ex-top-5 % of −0.12 to −0.24 %. The top 5 days carry 2.5–5.2× the
  group total, and the other holdout is negative in every case. This is tail and day concentration,
  not edge.
* **Driftless formula.** The builder's correction to m/(k+m) is right (gambler's ruin) and reproduces
  the PREREG's disclosed 0.67/0.33.
* **PREREG freeze.** The PREREG was committed at 15:36:10 UTC (d7593ca), before both builds
  (rebuild 15:47, builder 15:55), and is unchanged since. The header's "16:35 UTC" is a clerical
  error: it is later than this review's clock, 16:15 UTC.

## What must change before this is relayed
1. Part A goes on record with the censoring-exact E[X_τ] test and both fill-bar bounds. The
   "downward tilt" wording is withdrawn. m ≤ 0.5 % is marked unidentified from minute bars; tick data
   around the fill is the only resolution.
2. The paired Δ is recomputed against the base re-walked under the same `net_cost`. Delete "real,
   significant improvement". Report 1,611's significantly negative Δ.
3. The FAIL branch stands: no barrier reaches +0.15 % net at either bound, and gross moves ≤ 0.07 %
   against a ≈ 0.26 % cost. The rebuild's double charge should be fixed before it is used as the
   independent row-level check.
