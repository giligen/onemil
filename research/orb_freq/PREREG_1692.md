# PREREG_1692 -- reversed protocol: SELECT on 2026, TEST on 2025 (FROZEN 2026-10-01)

**Owner's ask (verbatim intent):** "can we tune on 2026 and test/eval on 2025?" This is a diagnostic
on regime-dependence, not a new discovery process: every per-fill book scored here was already built
and scored (cells 1684, 1685, 1690) under the production TRAIN(2025)->VAL/HELDOUT(2026) split. This
cell re-slices the SAME per-fill R values under the OPPOSITE split and reports both directions
side by side. No new R is derived except re-walking 1690's own (b)/(c) exit variants with its own
unmodified `walk_all`/`apply_cost_sizing` code (same bars, same formulas -- only the date mask differs).

## Candidates (24, + 1 contextual baseline not counted in the tally)
- **P1 family, 8 (research/orb_freq/1690_reads.csv, 1684_pool_books.csv pool=='idea1'):**
  a_F1, a_F3, a_F4, a_F5, a_F6 (feature-gated sub-populations of P1), b_scale50_1R,
  b_noexit2R_half3R_trail1R (alternate exits, re-walked on cache.db bars), c_cost_sizing
  (range%-based size-down). `P1_plain` and `b_live_rule` are shown for context ONLY -- 1690's own
  script excludes both from its candidate set (plain = baseline, live_rule = sanity check that the
  re-walked i0/stop reconstruction matches the base book, not an improvement attempt).
- **15 pools:** idea1, idea2, idea10, idea11 (research/orb_freq/1684_pool_books.csv, 4 add-on pools,
  window=='in_regime') + AF6, BF1, BF3, BF4, BF5, BF6, CF1, CF3, CF4, CF5, CF6 (1685_pool_books.csv,
  11 scored sub-pools, window=='in_regime').
- **Production reference:** analysis_results/orb_bplus_book.csv, entered==1 (same loader as
  1684_score.py's `load_book`).

## Windows
- **SELECT (tune) = 2026-01-01..2026-09-26.** Disclosed gap (same as PREREG_1690.md): the in-regime
  build for every one of these books stops at 2026-09-18 (universe/bar fetch not run past that date,
  not fetched here either -- no new Databento pull without owner GO per memory
  `feedback_databento_spend_convince_first`). SELECT therefore evaluates on the data that exists
  through 09-18 and is reported as such; the 8 missing trading days (09-19..09-26) are a known,
  disclosed shortfall, not silently patched.
- **TEST (eval) = 2025-01-01..2025-12-31.** Identical range to the ORIGINAL protocol's TRAIN window
  (`v1690.TRAIN`) -- used as a built-in parity check: for the P1 family, TEST2025 computed fresh here
  must closely match the existing `{variant}__TRAIN` row already on disk in 1690_reads.csv (same
  population, same r_col, same dates). Any mean_r discrepancy > 0.01 R is logged as a WARNING.
- **Original-direction numbers** (for the side-by-side): P1 family pulls `{variant}__TRAIN` and
  `{variant}__VAL` straight from the existing 1690_reads.csv (no recompute -- VAL there is
  2026-01-01..2026-06-30, `v1690.VAL`, NOT the same as this cell's SELECT). Pools have no pre-existing
  TRAIN/VAL split (RESULT_1684.md / RESULT_1685.md scored them on the full in-regime window only) --
  their "original" TRAIN2025/VAL2026 numbers are computed fresh here, by the identical method, and are
  flagged as such (not previously published).

## Selection rule (pre-committed, mirrors 1690's own `is_candidate`)
Pass SELECT2026 iff `mean_r >= +0.05` AND `dc_t >= 1.5` on the SELECT window. Pass TEST2025 under the
same two thresholds. Classification:
- **robust** = passes SELECT2026 AND passes TEST2025.
- **regime_specific** = passes SELECT2026, fails TEST2025.
- **fails_both** = fails SELECT2026 (TEST2025 result irrelevant to the tally; reported anyway).

## Stats convention (reused verbatim, not invented)
`stats()` in 1692_reverse.py merges 1684_score.py's formula (iid_t, MDE = 2.802*sd/sqrt(n), 80% power
2-sided 5%) with 1690_variants.py's `r_col` parameter and green-week count -- both already exist in
this codebase; copied per 1690_variants.py's own documented convention (digit-prefixed filename,
<20-line function, kept single-file/auditable) rather than imported, so this script stays self-con-
tained and diffable against both sources. R_USD=$375, week = `date.dt.to_period('W')` (weeks with zero
fills are not counted in n_weeks -- inherited convention, not a new bias).

## Known limitations (stated up front, not discovered after the fact)
1. **Multiplicity.** 24 candidates x 2 windows x 2 directions = continues the programme's cell ledger
   (cells 1684/1685/1690 already spent multiplicity scoring these same populations under the original
   split). A "robust" verdict here is necessary, not sufficient -- it is the SAME population re-cut,
   not a fresh out-of-sample population. Any pass is read as "survives a second cut," not "proven."
2. **SELECT2026 is truncated** at 2026-09-18 (8 trading days short of the nominal 09-26 end) -- see
   above.
3. **Causality / fill realism / price-scale** for every population here were established in
   PREREG_1684.md / PREREG_1685.md / PREREG_1690.md and are NOT re-verified in this cell (reused,
   unmodified per-fill books). The 80% bar-walk availability rail for (b)/(c) variants is re-checked
   (same `walk_all` warning) since the date mask changes which fills are walked.
4. **ex-top-5%, green weeks, weekly P10** are reported for TEST2025 per the owner's ask; tail risk on
   SELECT2026 (the tuning window) is NOT separately gated here beyond the dc_t>=1.5 bar -- a pool that
   passes SELECT2026 on a thin tail is still flagged by its own TEST2025 ex-top-5% column.
5. No placebo/green-week-null decomposition run here (budget: read-only, <=30 tool calls). Any
   "robust" result should get that treatment before it is treated as a ship decision, not a research
   finding.

## Outputs
research/orb_freq/RESULT_1692.md (<=100 lines, side-by-side table first), 1692_reads.csv (long-format
every stat, every window, every direction), 1692_reverse.py (this cell's only new code), 1692_reverse.log.
