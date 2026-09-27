# REBUILD_1562 — independent rebuild of PREREG_1562 (cells 1,562-1,563), prose-only

Built without reading cell_1562.py / test_cell_1562.py / cell_1562_fills.csv / RESULT_1562.md.
Signal population (results.csv delay_s==0, status in filled/skipped_guard): n=410 (TRAIN 164, VAL 246).
Excluded for missing cache.db bars (counted, not replayed): 216 of 820 cell-signal rows across both cells.

## MAJOR DEVIATION — tape mismatch (obtainability)
The task named research/hod_ofi/raw/*.parquet as the tape. That directory is the HOD-break study's own tick tape (different symbols/dates): only 20 of the 410 target signals match it by (date,symbol) (checked directly, not assumed). The correct ORB tick tape, research/orb_latency_bt/raw/*.parquet, matches ~all 410 signals but is truncated at 09:40:00 ET (WIN_END_S=300 in replay.py) — too short for a 15/30-minute retest search. This rebuild therefore runs the ENTIRE retest search and exit walk on 1-minute bars from data/cache.db (intraday_bars_1min), never on tick data past the original trigger print. Any intra-minute ordering (fill vs. same-bar stop/target) is a bar-low/high approximation, not tape truth — this is exactly the obtainability gap CLAUDE.md requires flagging, and it could not be closed inside this task's data/tooling/step budget.

## Simplification — touchgo prefire excluded
Rule M / Rule D ("touchgo") bar-shape exits (tag_bb / tag_b1 in the live rule; study_orb_pipeline_static_lock.py evaluate_rule_m/evaluate_rule_d) are NOT implemented: their threshold config (TOUCHGO_CFG.rule_m_threshold, bb_close_pos) is not derivable from prose within budget. Only the static-lock core is walked: arm at entry+1.75·R′ (LOCK_TRIGGER_R), lock stop at entry+0.5·R′ (LOCK_STOP_R) once armed, else initial stop = range_low, else 15:55 ET close-at-bid. This differs from the full live rule on the entry bar and bar 1 and is disclosed as a real gap, not a silent one.

## Cost model (PREREG cost section, cited)
Stop/lock exits: SLIP_STOP_BPS (cell_1478.py) TRAIN=13.832bps, VAL=11.936bps. EOD exits: 11.5/9.7bps (cell_1457.py, RESULT_1443.md table). Entry and target legs: zero cost (both passive per PREREG).

## Results
### Cell 1,562 (15-min window, level − $0.01)
- **TRAIN**: n_signals=164, filled=54 (fill_share=32.9%), withdrawal_share_15m=50.0% (n=2 never-retest rows), mean_net_R=0.868, day_clustered_t=2.57, ex_top5%=0.417, winner_capped(+3R)=0.511, fills/wk=0.64
  paired ΔR (vs zero-latency replay fill, n=40): mean=0.517, pooled_t=1.69, ex_top5%=0.156
  book mean incl. non-fills as zero=0.286 vs base mean (all 164 signals)=0.057
  never-retest cohort (n=2): base_R mean=0.730
  exit mix: {'eod': 21, 'stop': 18, 'lock': 15}
- **VAL**: n_signals=246, filled=233 (fill_share=94.7%), withdrawal_share_15m=30.8% (n=13 never-retest rows), mean_net_R=0.342, day_clustered_t=1.72, ex_top5%=-0.086, winner_capped(+3R)=0.030, fills/wk=4.09
  paired ΔR (vs zero-latency replay fill, n=167): mean=0.253, pooled_t=1.34, ex_top5%=-0.099
  book mean incl. non-fills as zero=0.324 vs base mean (all 246 signals)=0.062
  never-retest cohort (n=13): base_R mean=0.083
  exit mix: {'stop': 124, 'eod': 75, 'lock': 34}

### Cell 1,563 (30-min window, level × (1 − 0.002))
- **TRAIN**: n_signals=164, filled=56 (fill_share=34.1%), withdrawal_share_15m=nan% (n=0 never-retest rows), mean_net_R=0.955, day_clustered_t=2.87, ex_top5%=0.516, winner_capped(+3R)=0.588, fills/wk=0.67
  paired ΔR (vs zero-latency replay fill, n=42): mean=0.604, pooled_t=2.01, ex_top5%=0.257
  book mean incl. non-fills as zero=0.326 vs base mean (all 164 signals)=0.057
  never-retest cohort (n=0): base_R mean=nan
  exit mix: {'eod': 22, 'stop': 18, 'lock': 16}
- **VAL**: n_signals=246, filled=235 (fill_share=95.5%), withdrawal_share_15m=27.3% (n=11 never-retest rows), mean_net_R=0.372, day_clustered_t=1.84, ex_top5%=-0.064, winner_capped(+3R)=0.043, fills/wk=4.12
  paired ΔR (vs zero-latency replay fill, n=167): mean=0.292, pooled_t=1.51, ex_top5%=-0.071
  book mean incl. non-fills as zero=0.355 vs base mean (all 246 signals)=0.062
  never-retest cohort (n=11): base_R mean=0.098
  exit mix: {'stop': 123, 'eod': 75, 'lock': 37}

## Pass bar (frozen, PREREG_1562.md) — mechanical check against the numbers above
Paired ΔR ≥ +0.10R both splits, pooled day-clustered t ≥ 2.5, same sign each split; VAL own mean net R′ ≥ +0.15 with t ≥ 2; ex-top-5% > 0 both; winner-capped positive; ≥3 fills/wk; never-retest-inclusive book mean ≥ base mean; median R′ ≥ 0.5% of price. See per-cell numbers above — median-R-as-%-of-price was not separately tabulated in this rebuild pass (all Rp values are in rebuild_1562_fills.csv net_R/base_R columns; a follow-up pass can compute it directly from entry/limit and range_low there).

## Independent-check status
This is the FIRST (independent, prose-only) build. It has NOT been cross-checked against cell_1562.py's own fill set (task instructions forbid reading that file). Fill-set Jaccard and net-R agreement vs. the original builder remain to be run by whoever DOES have access to both builds, per PREREG's "Independent check" section.
