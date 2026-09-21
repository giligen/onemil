# S1 creative pass — cells 1,322–1,328. VERDICT: S1 standalone closed; the UNION rung (1,328) is a LEAD

PREREG `PREREG_S1_CREATIVE.md` (8ec4adf; addendum 4069ab7 before its scorer ran). Book `out/runS1_true.csv`,
R = $375, cost M, TEST sealed. Scorers `score_s1_creative.py`, `score_comb.py`, `score_priority.py`.

## 1. Filter cells on the S1 book (1,322–1,326) — none passes, none reaches the exploration tier
| cell | TRAIN kept R (n) | VAL kept R (n) | VAL dropped R | read |
|---|---|---|---|---|
| 1,322 GAP-RANK top-10 | n = 1 | n = 0 | | inert: ~50 gappers/day in the universe, a 3–5 % gap is never on the list |
| 1,323 PM-$VOL ≥ TRAIN median $740K | −0.067 (102) | +0.068 (92) | −0.013 | sign flips; coverage 98 %, missingness gap 1.8 pp (valid) |
| 1,324 RVOL-5m ≥ median | −0.057 (105) | +0.087 (72) | −0.009 | sign flips |
| 1,325 HMM-CALM | −0.045 (179) | +0.051 (123) | −0.038 | keeps 86 %; TRAIN non-calm was +0.084 |
| 1,326 SPY-GAP-UP (peeked) | +0.009 (145) | +0.032 (90) | +0.054 | dropped cohort positive on VAL |
Diagnostics: price $10–30 VAL +0.118 / TRAIN −0.006; float (current) nothing. Pre-market bars now cached for all
351 S1 symbol-days (`scripts/orb_premarket_backfill.py`).

**Read:** every conviction feature flips sign between 2025 and 2026 on this stratum; the 2026 +0.04 is noise-shaped.
The attention frame explains the stratum instead of fixing it (never on a top-10 list, PM participation does not
rescue it). S1 as a standalone book is closed at this pass: flat under 3 selection cuts, 4 exits, 5 features.

## 2. The rung: production ∪ S3 ∪ gap 4–5 % ($3–30)
### 1,327 shared-pool walk — FAIL (pre-registered bar), informative
| split | production $ / R / fills/wk | combined $ / R / fills/wk | prod picks preserved | wk MDD |
|---|---|---|---|---|
| TRAIN | 6,561 / +0.206 / 2.5 | 6,902 / +0.129 / 4.2 | 78 % | −1.41 → −3.49 R |
| VAL | 6,398 / +0.406 / 2.3 | 8,260 / +0.229 / 5.3 | 67 % | −1.32 → −1.16 R |
Displacement is NOT slot competition (max 6 picks/day of 8) and NOT anchor dedup (4 of 45 share an anchor); the
missing production picks are lower-composite Q2/Q3 rows, so a pool-dependent step in the pipeline (candidate for
the Q1/quintile stage, drop counts 1,152 → 1,480) changes with the pool. Unresolved; irrelevant to the union
design below, but it means a single shared pool is NOT how to widen the seed.

### 1,328 UNION of independently evaluated pools — passes VAL, fails TRAIN on MDD only
| split | production $ / R / fills/wk / wkMDD | union $ / R / fills/wk / wkMDD | added n / R / ex-top-5 % | bar |
|---|---|---|---|---|
| TRAIN | 6,561 / +0.206 / 2.5 / −1.41 R | **9,190** / +0.151 / 4.8 / **−2.89 R** | 77 / +0.091 / +0.033 | FAIL: MDD 2.05× (bar 1.25×) |
| VAL | 6,398 / +0.406 / 2.3 / −1.32 R | **10,207** / +0.247 / 6.1 / −1.16 R | 68 / +0.149 / +0.101 | PASS (all) |
Added cohort by slice: gap 4–5 % TRAIN +0.069 (n 63, halves +0.13 / +0.03), VAL +0.084 (n 58); S3 TRAIN +0.192
(n 14), VAL +0.529 (n 10). Largest added trade +2.25 R; 9 of 145 ≥ +1 R; added weekly P&L max +4.4 R. Not
tail-dependent. Cadence block on the union book: VAL passes C1–C5 (one cycle, C7 fails on power); TRAIN passes C5
only.

## 3. Adequacy and what stands
* The union rung raises $ by 40 % (TRAIN) and 60 % (VAL) and doubles fills/wk; the added cohort is positive on
  both splits, both TRAIN halves, and ex-top-5 %. Its per-trade edge (+0.07–0.15 R) is a third of production's.
* It fails ONE pre-registered criterion, TRAIN weekly MDD 2.05× production (−$1,084 vs −$529 at the $26.7K stage).
  That is the price of 2.3 extra fills/wk at a third of the edge: more trades, wider weeks. The owner's cadence bar
  wants fills; the MDD bar wants the base untouched; they conflict here by construction.
* Owner's exploration tier (9/18): positive point estimate on every split and half, mechanism = the same ORB
  breakout on adjacent strata, downside bounded by the stop and the stage size, ~3 added fills/wk → ~40 fills in a
  quarter. **Qualifies for the tier; does not clear the research bar.**
* Not yet done (required before any live PREREG): independent rebuild of the union book from a prose spec; ≥ 3 R
  obtainability audit is moot (no added trade ≥ 3 R); the pool-dependent displacement step must be identified so
  the live implementation evaluates the add-on pool separately (two universe queries, one slot pool).

Cell count: **1,328**.
