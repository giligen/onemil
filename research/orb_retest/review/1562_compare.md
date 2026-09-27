# Cell 1,562 independent-rebuild comparison

Builder: `research/orb_retest/cell_1562_fills.csv` (821 lines incl. header, 820 rows = 410 (symbol,date)
signals x {15-min, 30-min} window variant, `cell` 1562/1563).
Rebuild: `research/orb_retest/rebuild_1562_fills.csv` (same 820 rows, identical symbol/date/split order —
keyed here on symbol+date+within-day-occurrence since neither file carries an explicit window-length
column).

**Column-name trap found first**: builder's `r_prime` is the RISK UNIT (R' = fill_price − range_low,
confirmed algebraically: `r_prime == fill_price - range_low` and `r_base_signal == trigger - range_low`
on every row checked), not the trade outcome. The outcome in R-multiples is `retest_r` (builder) /
`net_R` (rebuild); `base_r` (builder) / `base_R` (rebuild) is the shared zero-latency-chase baseline.
Comparing `r_prime` to `net_R` directly (an easy first mistake) produces a nonsense ~0.6% same-value
rate — the correct pairing is `retest_r` vs `net_R`.

## Exit rule parity — SAME (confirmed in source, not just prose)
Both scripts hard-code identical constants and logic:
- `LOCK_TRIGGER_R = 1.75`, `LOCK_STOP_R = 0.5`, initial stop = `range_low` (cell_1562.py:121-122,
  rebuild_1562.py — arm/lock levels computed off fill_price + R', matching
  study_orb_pipeline_static_lock.py:143-145).
- Stop/lock cost = `SLIP_STOP_BPS = {'TRAIN': 0.88*2.9+0.12*94.0, 'VAL': 0.88*3.2+0.12*76.0}` — byte-identical
  formula in both files.
- EOD cost: builder `EOD_BPS = {'TRAIN': 11.5, 'VAL': 9.7}`, rebuild `EOD_BID_BPS` — same values, different
  name only.
- Force-close at 15:55 ET (PREREG override of the BT's 15:45), same in both.

`base_r` (builder) == `base_R` (rebuild) exactly on every one of the 517 jointly-scored fills (max abs
diff 0.0), which confirms the shared upstream signal/population code is solid and the two runs are
walking the same bars for the baseline leg.

## Fill-set agreement
"Scored fill" = a retest limit fill that produced a computed exit R (builder: `retest_r` not-null,
equivalently `excluded == False`; rebuild: `filled == True`, which is exactly co-extensive with `net_R`
not-null in this file).

| | n |
|---|---|
| Builder scored fills | 518 |
| Rebuild scored fills | 578 |
| Intersection | 517 |
| Union | 579 |
| **Jaccard** | **0.893** |

Discordant fills:
- **142 builder-only** (retest_fill=True but excluded, i.e. never scored): 141 tagged
  `no_bars_for_exit_walk`, 1 unlabeled — the builder is stricter about requiring exit-walk bars.
- **61 rebuild-only** (rebuild scored a fill the builder excluded): breaks down by rebuild's own exit
  path as eod 34 / lock 20 / stop 7, split across `status` filled 37 / skipped_guard 24 — the rebuild
  found tradeable exits on cases the builder's bar-availability gate dropped.
This is a real, reportable disagreement about *which* retests are even walkable, not just about their
value — 62 of 579 union fills (10.7%) exist on only one side.

## Value agreement on the 517 jointly-scored fills
- **Share within 0.01 R (net R′, i.e. `retest_r` vs `net_R`): 0.679** (352/517). Not tight — roughly a
  third of commonly-scored fills disagree by more than a rounding/cost-convention amount.
- **VAL mean difference** (builder `retest_r` − rebuild `net_R`, VAL-split common fills, n=425):
  **+0.0406 R** — small aggregate bias, builder trends fractionally higher.
- **Paired ΔR difference** (builder's own `paired_dr` = `retest_r`−`base_r` vs rebuild's equivalent
  `net_R`−`base_R`, same 517 fills): **mean +0.0471 R, mean |diff| 0.104 R** — the aggregate bias is
  small but the typical per-fill disagreement (0.10 R) is not; individual gaps run into the 1-2.7 R
  range (below), which is large relative to a book whose live edge estimate is ~0.02-0.1 R/fill.

## Dominant cause of the 10 largest differences
| symbol | date | split | builder exit / rebuild exit | Δ(retest_r − net_R) |
|---|---|---|---|---|
| MLTX 2026-01-09 (both windows) | VAL | eod / stop | +2.73, +2.71 |
| ASST 2025-05-22 (both windows) | TRAIN | eod / eod | +2.39, +2.05 |
| BMNZ 2026-03-06 (both windows) | VAL | eod / stop | +2.21, +2.03 |
| CRWU 2026-01-27 (both windows) | VAL | lock / eod | +1.74, +1.66 |
| IRE 2025-10-23 | VAL | eod / lock | +1.65 |
| AAOX 2026-07-09 | VAL | lock / stop | +1.50 |

**8 of the 10** are pairs where the two implementations pick a **different exit path** for the identical
trade (builder says `eod`, rebuild says `stop` or `lock`, or vice versa) — not a different rule, a
different resolution of the same-bar stop/lock-vs-still-open ordering ambiguity. Both scripts' own
docstrings disclose this exact limitation ("ordering (fill vs. same-bar stop/target) is a bar-low/high
approximation, not tape truth"), so this is the known, named cause, not a latent bug. All 10 rows carry
`withdrawn_15min = True`, but that flag is the population norm here (507/517 = 98% of jointly-scored
fills), so it does not itself explain why these particular 10 diverge.

The remaining **2 of 10** are the ASST 2025-05-22 pair: **both sides agree the exit was `eod`** yet the
magnitude differs by ~2.0-2.4 R. Same exit path, different value, points to an entry-fill-price or
EOD-reference-price divergence for that specific symbol/day rather than the general exit-path ambiguity —
flagged for a targeted trace, not yet root-caused (out of budget for this pass).

## Verdict
Aggregate statistics look clean (VAL mean diff +0.04 R, paired ΔR bias +0.05 R) but row-level agreement is
not: 10.7% of the union fill set is scored by only one implementation, and of the fills both score, only
68% match within 0.01 R, with a same-bar exit-path ambiguity producing individual gaps up to ~2.7 R.
Per the project's independent-check protocol (row-level rebuild agreement required before a mechanism
ships), this pass does **not** clear the bar as-is — the exit-path tie-break (stop/lock vs eod on
same-bar ambiguous data) needs a single shared resolution rule (parity by construction) before cell
1,562/1,563 numbers go in front of the owner, and the 142 vs 61 fill-set asymmetry needs the
exit-walk bar-availability gate reconciled between the two implementations.
