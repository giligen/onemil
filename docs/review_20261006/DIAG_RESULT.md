# DIAG_RESULT 2026-10-06 — DFDV never ordered (10/5) + nightly P1 book failed (read-only diagnosis, nothing edited)

## 1. Engine: DEFECT (slot double-count in the post-preplace selection; not a "design = provisional set only" rule gap)
Timeline (journal 13:34-13:36 UTC, `logs/orb_selection_audit.jsonl` 13:35:25, file:line = trading/orb_engine.py):
- 13:34:57.727 `_preplace_provisional_rank` (:3960-4066). budget = N(8, orb.yaml:336) - |entered|vetoed|open| = 8 (:4018-4020); top-8 of the PROVISIONAL
  (snapshot dailyBar) ranking = EWZS BCYC PURR SUPV HOG ALVO JAGX CRCG. The veto loop (:4029-4033) runs `_pdr_veto_reject(temp)` on throwaway copies,
  but the veto SIDE EFFECT is on self: `self._pdr_vetoed_today.add(sym)` (:4896/4930/4959/4984). 6 names vetoed -> set = {EWZS,BCYC,PURR,SUPV,HOG,ALVO}; 2 planned: JAGX, CRCG.
- 13:35:07 `_preplace_submit_at_close` (:4071-4139): both submits TypeError -> "no refill"; plan_submitted never set, no DB row, not open. `_reconcile_preplaced` (:3233,
  latched) skips them (not submitted); grace defer returned [] (:3306) so the PARITY line was lost (spec limitation, docs/orb_preplace_spec_20260928.md:103-108).
- 13:35:25 full scoring, `_run_pool_selection` (:3432-3720): `budget = N - |entered | _pdr_vetoed_today | open|` = 8 - 6 = **2** (:3614-3616).
  Ranked (audit, Q4 first per `ranking_order` :866): SUPV .4208, HOG .4114, BNC .4025, ELPC .3392, GGB .3377, EWZS .333, **DFDV .3216 (rank 7)**, BCYC .3116, PAX, UGP, JAGX ...
  top_syms = dedup(ranked, max_keep=2) = [SUPV, HOG] = audit `picks`; both already in the vetoed set, re-vetoed ("PDR VETO: SUPV/HOG" 13:35:25); add-on pass picks [GDS, BABX], vetoed.
  Nothing else runs (no refill). a) The code path exists and ran; it submitted nothing because budget=2 was spent on two names already counted. No "selection done" flag involved.
- b) DFDV IS in the engine's ordered list (rank 7 of 11 Q4) and inside the BT's top-8, but outside the engine's top-2. SUPV and HOG each consumed a slot TWICE (provisional veto set,
  then re-ranked as the budget's top-2); PURR/ALVO (provisional Q5, final ranks 12-13) consumed 2 more slots they would never hold in the BT's top-8. Provisional top-8 overlaps BT top-8 on 4/8.
- c) Rule: spec docs/orb_preplace_spec_20260928.md (and no preplace line in research/orb_machine_rules.md or CLAUDE_HISTORY.md) says vetoes are evaluated ONCE at provisional time
  (:96-99) and reconcile compares only the PREPLACED symbols (`_reconcile_preplaced` :4175-4196: scored built from preplaced_syms only, final_top = top-len(preplaced) OF THEM).
  BT (`study_orb_pipeline_static_lock.py`) has no 09:34:57 snapshot: one-shot top-8 of the final ranking, vetoes drop slots, no refill. Gap = provisional ordering != final ordering and
  the preplaced set is never checked against the full final top-N. Even with a successful submit JAGX(rank 11)/CRCG(Q3, ~15) would have been kept and DFDV never entered (budget 0).
- d) SPEC (smallest fix; do not edit before Monday's boot): (1) `_preplace_provisional_rank` must not mutate `_pdr_vetoed_today` (snapshot+restore the set around :4029-4033, or a
  `record=False` arg to the four veto methods); provisional vetoes live in `_preplace_state['vetoed']`. (2) `_reconcile_preplaced` ranks ALL production candidates with final ranges, takes
  top-N via the same dedup, cancels preplaced names outside it (slot consumed, `_pdr_vetoed_today.add`), and re-runs the four vetoes on the final top-N to rebuild `_pdr_vetoed_today`
  from FINAL top-N names only. (3) :3614-3621: take `dedup(ranked_all, max_keep=N)` FIRST, then drop entered/open/vetoed/plan_submitted (BT semantics: slots consumed once), not
  `max_keep = N - consumed` over a list that still contains the consumed names. Parity test: replay the 10/5 audit record (ranked list above, provisional vetoed set {EWZS,BCYC,PURR,
  SUPV,HOG,ALVO}, JAGX/CRCG failed) and assert picks == BT top-8 minus vetoes (DFDV); plus a test that a provisional veto never changes the final budget, and that a reconcile with
  successful JAGX/CRCG cancels them (not in final top-8) and the normal pass enters DFDV. Integration: dry-run day via `[ORB DRY] WOULD PREPLACE` ledger vs the nightly BT picks.

## 2. Nightly P1 book failed: fetch and RESIM read the SAME store, but the fetch covers 1 day and the features CSV carries many
- `CACHE_DB = ORB_CACHE_DB or data/cache.db` (orb_backtest.py:51); nightly sets no env -> data/cache.db for the fetch (`fill_intraday_for_pairs`, :499-503), the features subprocess
  (:504) and RESIM (`ORB_BT_BARS_DB=CACHE_DB`, :~540). Same store. The fetch is scoped to `dates` = marker_dates = computed days + today = {2026-10-05}: "27 qualifying pairs over 1 day(s)", 9,478 bars.
- But `study_orb_features.py` is incremental over `analysis_results/pool_P1/orb_features_20261003_1013.csv` (Saturday acceptance run, SIDE cache) and wrote a 21:16 CSV (120 candidates) with
  earlier days; `cands[-1]` feeds the pipeline. Those earlier P1 pool-days' bars were only ever written to Saturday's side cache. Verified read-only: data/cache.db has 0 `intraday_bars_1min` rows
  for MARO 2026-10-02. RESIM: 28/44 entered missing (0.6364 > 0.02) -> `sys.exit(1)` (study_orb_pipeline_static_lock.py:898-903) BEFORE `_write_markers` (:1138/1199/1596) -> no marker -> EOD NO-DATA.
  Answer: the nightly never wrote the pool bars for any day but 10/5; it will fail every night until backfilled.
- 872 s (21:06:25 -> 21:20:36): fetch 270 s (21:06:25-21:10:55), features ~320 s (to the 21:16 CSV mtime), pipeline ~260 s (inferred; its stdout is buffered to the end).
- FIX SPEC (nightly service = the only writer of data/cache.db; agents never write it): in `build_pool_books` after the features build (:511-513) read `cands[-1]`, take its
  (symbol,date) rows, `_intraday_bars_cached_pairs(CACHE_DB, pairs)`, and `fill_intraday_for_pairs` the missing ones (one-time backfill ~120 pairs, then 1 day/night), THEN run the pipeline.
  Test: features CSV with a date not in the cache -> fetch called for exactly those pairs; second run fetches 0.
- FAILED MARKER SPEC: on `feat.returncode != 0` (:513-515) and `res.returncode != 0` (:546-547) call one helper `_write_failed_markers(dates, pid, reason)` -> upsert
  `date,pool_id,candidates,picks,status` with status=failed (reason in a `note` column), `_write_markers` keeps `status` (default ok for old rows); `scripts/eod_sections.bt_marker`
  (:844) returns it and the P1 line prints `P1 BT: FAILED (<reason>)` (ERROR log) instead of NO-DATA. Tests: stub subprocess rc=1 -> marker row status=failed; EOD text has FAILED, not NO-DATA.
