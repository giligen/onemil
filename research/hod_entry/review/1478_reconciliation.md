# Reconciliation: cell 1,478 sign flip (builder vs independent rebuild), L2

Builder (`cell_1478.py`, `RESULT_1478.md`): L2 kept-VAL mean net R = **-0.0375 (HGB) / -0.0250 (LR)**.
Rebuild (`rebuild_1478.py`, `rebuild_1478_compare.csv`): L2 kept-VAL mean = **+0.3568 (HGB) / +0.3703 (LR)**, Jaccard 0.51-0.69.

**Verdict up front: the rebuild is wrong. It contains an accidental label leak. The builder is correct.**

## Step 1 — feature-matrix diff (join on day, symbol, fill_min; 9,911 rows)

Full table + 3-row examples: `review/_step1_2.txt`. Highlights:

| builder col | rebuild col | share differ >1e-6 |
|---|---|---|
| atr14_pct (causal, shift(1).rolling(14) TR / prev_close) | **f_atr_pct_old** | 1.0000 (values are 0/1, not %) |
| range_to_j_pct | range_to_j | 0.8547 (different bar store per amendment; both still causal, up-to-j) |
| rv_j | rv_at_j | 1.0000 (different realized-vol window) |
| arm_index | arm_index | 1.0000 (rebuild's index ~2.5-3x builder's — different bar-count start, e.g. premarket-inclusive store) |
| prev_day_volume | prior_volume | 0.0000 (agrees) |
| dist_from_open_pct / ask_distance / spread_at_fill / time_of_day / R_pct | f_dist_from_open / f_ask_distance / f_spread_at_fill / f_time_of_day / f_R_pct | ~0.96-1.00 (all mismatched) |

Root cause of the mismatches: `rebuild_1478.py`'s `rename_1445` / `rename_1457` dicts assume `cell_1445_features.csv` / `cell_1457_features.csv` columns `flag_1445, flag_1447-1450, flag_1457-1462` are the continuous PREREG item-1 feature *values* (dist-from-open %, ATR%, float, etc.). They are not: every `flag_NNNN` column in those files is a **binary pass/fail condition for a numbered HOD research cell** (confirmed: `cell_1445_features.csv.flag_1445/1447/1448/1449/1450` are `{0,1}`; `cell_1457.py:590`, `out['flag_1457'] = ceiling_cond.astype(int)`). The rebuild's own code comment (rebuild_1478.py:288-294) discloses this mapping is "an INFERENCE... not confirmed against the builder's code."

**The critical one: `flag_1457` = `ceiling_cond` = cell 1,457's full-day-range ≥ 10% flag — i.e., (essentially) the L1 label itself, computed from the whole day's high/low.** The rebuild renamed it `f_atr_pct_old` and fed it into `FEATURE_COLS` as if it were a causal ATR%. This is a direct post-arm-bar-j leak: it uses information (the day's full high/low) unavailable until market close.

Separately, `rebuild_1478.py:235-243` computes its own NEW `atr_pct`: `tr` uses the **current day's own** high/low/prev_close (not shifted), `atr14 = tr.rolling(14).mean()` (current day's TR enters its own window), divided by the **current day's own close**. This is a second, independent leak channel — not a PREREG-specified feature at all (PREREG's item 2 "NEW" list has no second ATR feature) and it duplicates + corrupts the one already-causal ATR14% that should have been carried over from `cell_1457_features.csv.flag_1458` (masked by `atr14_pct.isna()` — genuinely the ATR-conditioned flag) but wasn't (mis-mapped to `f_prior_day_range_old` instead).

## Step 2 — rebuild's kept VAL set scored on the builder's own outcome_R

| label | model | n_kept | mean(builder outcome_R) | kept cache-only share | base 19.5% | share already realized (range_to_j_pct≥10) |
|---|---|---|---|---|---|---|
| L2 | hgb | 2004 | **+0.3568** | 0.2515 | 0.195 | 0.3468 |
| L2 | lr | 1973 | **+0.3703** | 0.2544 | 0.195 | 0.3599 |

Confirms the +0.36/+0.37 is real on the builder's own P&L column (not a rebuild-side R/cost bug) — the rebuild's kept set really does earn that on the shared outcome column. Cache-only share (25%) is close to the base rate (19.5%, +5.6pp) — not the SIPZERO/cache-cohort artifact. Only ~35-36% of the kept set already has realized full-day range ≥10% at bar j — so `range_to_j_pct` alone doesn't explain the picture; the kept set is **the ceiling_cond / prior-day-volume leak cohort in disguise**, not the SIPZERO cohort and not simply the "already realized" cohort.

## Step 3 — permutation importance audit (HGB, VAL, ROC-AUC drop; refit with the saved-best grid params)

Original (leaking) model, L2/HGB, top-10:

| feature | importance | postdates arm bar j? |
|---|---|---|
| **f_atr_pct_old** | 0.2358 | **YES — is `ceiling_cond`, the full-day range≥10% flag (day's own high/low)** |
| prior_volume | 0.2242 | No — prior day's volume, causal, restates one of L2's AND-clauses (same as builder's finding for its `prev_day_volume`) |
| range_to_j | 0.0012 | No — up-to-j only |
| rv_at_j | 0.0005 | No |
| gap_pct | 0.0004 | No — prior close vs today's open |
| prior_rvol | 0.0002 | No |
| vwap_distance_pct | 0.0002 | No |
| sector_peers_j | 0.0002 | No |
| arm_index | 0.0001 | No |
| higher_lows_ct | 0.0001 | No |

`f_atr_pct_old` and `prior_volume` alone carry >99% of the model's rank power. `f_atr_pct_old` is the confirmed leak (full-day-derived, computed after bar j and after the market close); `prior_volume` is causal and matches the builder's own disclosed non-leak finding. The independently-recomputed `atr_pct` (also full-day/unshifted) does not surface in the top-10 here but is a second confirmed leak channel by construction (see Step 1) and was removed alongside `f_atr_pct_old` for Step 4.

No other top-10 feature uses the fill bar j+1, the day's close, the daily high/low, a store-served bar count, or the fill price itself in a way that postdates bar j — `f_dist_from_open`, `f_ask_distance`, `f_spread_at_fill`, `f_time_of_day`, `f_R_pct` etc. are mis-mapped (Step 1) but are all thresholds on already-causal cell-1445/1457 quantities, not confirmed look-aheads themselves; they just add noise, not signal (importance ≈ 0 for all of them here).

## Step 4 — corrected refit (leak columns `f_atr_pct_old`, `atr_pct` removed; same feature list otherwise, same seed 1478, mini-grid over the frozen HGB grid)

| | AUC (VAL) | n_kept | corrected kept-VAL mean net R |
|---|---|---|---|
| HGB | 0.9288 | 2316 | **-0.0744** |
| LR | 0.8205 | 2309 | **-0.0248** |

vs. builder's original: HGB -0.0375, LR -0.0250. **Same sign, same order of magnitude, LR essentially identical (-0.0248 vs -0.0250).** The residual AUC (0.93/0.82) still leans on `prior_volume`, the disclosed-but-legitimate near-tautological feature that both sides agree is causal, not look-ahead.

## Verdict

The rebuild is the defective side. Its `rename_1457` mapping (an inference the rebuild's own comments flagged as unconfirmed) assigned `flag_1457` — cell 1,457's `ceiling_cond`, i.e. the day's full-range ≥10% condition, computed from the whole day's high/low — to a feature it called `f_atr_pct_old`, and additionally introduced its own second, independently-leaking `atr_pct` column computed from the current day's own unshifted high/low/close. Together these hand the model a near-direct read of the L1/L2 label, which is why the rebuild's kept-VAL set shows AUC 0.94-0.99 and a spurious +0.36/+0.37 R kept mean. Removing both columns and refitting on the identical grid/seed collapses the rebuild's number to -0.07 (HGB) / -0.02 (LR), matching the builder's -0.0375 / -0.0250 in sign and (for LR) almost exactly in magnitude. **Corrected L2 number both sides agree on: kept-VAL mean net R ≈ -0.03 to -0.07 R — cell 1,478 fails the +0.15 R pass bar on label L2, as the builder originally reported.** This does not change RESULT_1478.md's overall verdict (already VOID on the decoy-AUC and cache-only-share grounds, and failing on profitability); it closes the sign-flip discrepancy in the builder's favor and identifies a concrete, fixable bug (a wrong flag-to-feature name mapping plus a stray unshifted ATR calculation) for anyone re-running the rebuild.
