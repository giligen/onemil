# DIVE ranked set 10/1 - RESULT (2026-10-03)

## Verdict
No rule difference between engine and BT. The 10/1 mismatch was a MONITOR defect: the engine scores TWO pools in one
session (production gap >= 5 %, and add-on pool P1 `addon_gap35_range5`, gap 3-5 %, PAPER orders since 10/1,
orb.yaml:55-61), and `scripts/eod_sections.parse_orb_log` treated every `ORB SCORED` line as the production ranked
set. The BT features universe is production only (orb_backtest.py:55 MIN_GAP_PCT 5.0).

## Step 1-2 filter chains (identical after admission)
| stage | BT (study_orb_features / study_orb_pipeline_static_lock) | engine (trading/orb_engine.py) |
|---|---|---|
| universe | gap >= 5 %, prev vol >= 500K, open $3-30 (orb_backtest.py:55-58, `_qualifying_pairs_for_dates`) | same thresholds, production pool; P1 pool adds gap 3-5 % (orb.yaml add-on) |
| features | 09:30-09:34 bars (study_orb_features.py:239+) | `_compute_features` (range_open = 09:30 bar open) |
| composite >= thr 0.01208, quintile | `composite_score`, `assign_quintile` (shared study_orb_*) | `composite_score` ~L3540; SCORED line logged AFTER the threshold, BEFORE skip_q1 |
| skip_q1, order Q4,Q5,Q3,Q2, dedup, N=8, PDR/G1/range vetoes | pipeline | `_run_pool_selection`, same helpers (orb_pdr_veto, orb_g1_veto, orb_range_size_veto) |

## Steps 3-5 the five names + scores
Production engine SCORED (7) = BT rows >= threshold (7) exactly: WVE, MEDS, CBRX, LPA, IBX, RKLX, EFXT. Comp and quintile
equal to 4 decimals on all 7 (CBRX .5883 Q5, LPA .5181 Q5, IBX .4326 Q5, WVE .4010 Q4, RKLX .1752 Q2, EFXT .0843 Q1,
MEDS .0345 Q1). The other 5 BT rows (NOWL, INFY, NWCL, SDEV, PSQL, comp < 0.0121) are below threshold in the BT, and the
engine did not log them as scored either.
- APPS 4.865, CRMG 4.943, KD 4.618, TDAY 4.452 (and CNXC 3.49, ADBG 4.01, BRZE 4.22, DXC 4.52): all gap < 5 % = pool P1
  candidates; absent from the BT because the BT features universe has no add-on pool. Not a data or rule difference.
- RKLX: BT rank 5 (Q2 .1752), engine SCORED it Q2 .1752, then G1 VETO (rv20 6.708 < 7.106). It was never BT-only: the old
  engine "top-8" computed from the mixed set put 8 P1/production names ahead of it, so the engine's top-8 dropped RKLX.

## Step 6 root cause
scripts/eod_sections.py `parse_orb_log` (scored/detail) + `engine_top_n`: no pool separation. Rules identical; not a
snapshot-vs-SIP data difference.

## Step 7 fix
- trading/orb_engine.py ORB SCORED log: appends `pool=<pool_label>` (logging only).
- scripts/eod_sections.py: `_is_production_scored` (pool tag; old archives fall back to gap >= 5.0); production feeds
  `scored`/`detail`, add-on names go to `parsed["addon"]`.
- tests/test_orb_ranked_set_parity.py (10/1 engine lines + BT ranked set): fails before, passes after. tests/test_eod_sections.py
  fixture gap=1 -> gap=10 (production-like).

## After
10/1 ranked-set: engine top-8 = [WVE, CBRX, LPA, IBX, RKLX] vs BT 5 -> match 5/5 (8/8 of the BT's ranked set), engine-only none, BT-only none.
Replay `--parity`: 9/30 57 = 57 (ASTX/AEHG excluded both), 10/1 35 = 35, 10/2 128 = 128.

## Left over
- Engine's first scoring on 10/1 was 09:44 ET (late decision flag), outside the 09:34-09:40 window: separate timing issue.
- Pool P1 has no BT counterpart in the nightly book; its forward parity needs its own BT replica.
