# ORB PIT pre-history seed - RESULT 2026-11-11

Seed `research/orb_exit_resim/orb_refit_history_seed.jsonl` (read with the refit's own yaml keys `filter.features.<f>.{mean,std,sign}` + `quintile_cutoffs`):
| ts (UTC) | set | snapshot | evidence |
|---|---|---|---|
| 20260419_085604 | z 8c558a06, cutoffs .1082/.1959/.2893/.4081 | pre_stage_0 | commit 47d6e1d ORB launch (same literals in tests/test_orb_filter.py) |
| 20260815_071706 | z d4e1c57f, cutoffs .1074/.1942/.2876/.3970 | pre_winner_stack_20260824 = pre_shorthist_drop_20260913 = refit_20260913 | commit ab7fcc2 B+ restart (frozen clean-2025H1 TRAIN refit) |
Snapshot mtimes are all copy times (9/20), useless; dates come from git. Sets 3-9 are the live history. Unknown: manual edits of untracked orb.yaml between 4/19 and 8/15 (git shows none in the template). Days before 4/19/2026 remain `current_fallback`.

Consistency: every `bak.refit_<ts>` (9/20 ... 11/08) equals the previous history entry (8/8 True); orb.yaml equals the last entry; pre_buffer50/catoff/pause/resume equal the 9/13 entry. No mismatches, no manual edits.

Code: `trading/orb_pit_params.py` `load_history(history, seed_path)`, `params_for_day(..., seed_path)`, source 'seed'; overlap raises; pipeline counts 'seed' as PIT (`study_orb_pipeline_static_lock.py:657`). Tests: 24 pass (19 pit incl. 7 new + 5 refit).

Book `research/book_review_20261111/orb_bplus_book_pit_full.csv` (days: 84 seed, 29 pit, 203 fallback, all pre-4/19). $ = `_sized_pnl`.
| window | PIT fills | PIT $ | ex-top-1 | ex-top-5% | published fills / $ / ex1 / ex5% |
|---|---|---|---|---|---|
| 4wk (10/14) | 25 | +1,139 | +679 | +679 | 24 / +1,347 / +886 / +886 |
| 6wk (9/30) | 30 | +800 | +340 | -76 | 29 / +1,007 / +547 / +547 |
| 6mo (5/11) | 186 | +1,533 | +832 | -3,090 | 177 / +3,557 / +2,856 / -992 |
Source mix: 4wk and 6wk 100 % pit; 6mo 70 seed + 29 pit days.
