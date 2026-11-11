# ORB BT point-in-time params BEFORE the first recorded refit (seed the history from the orb.yaml snapshots) — spec 2026-11-11

`docs/orb_bt_pit_params_20261111_RESULT.md`: `logs/orb_refit_history.jsonl` starts with the 2026-09-13 refit, so
605 of the book's 666 rows (everything before 9/13) still score with TODAY's params (`params_source =
current_fallback`, look-ahead). The parameters in force before 9/13 are recoverable: the refit script snapshots
orb.yaml before it writes (`orb.yaml.bak.refit_20260913_200003` = the file as it stood BEFORE the 9/13 refit, i.e.
the set in force up to 9/13; likewise `orb.yaml.bak.refit_20260920_200003` … `_20261108_200004` — verify each
equals the history entry that follows it, a free consistency check of the history itself), and earlier manual
snapshots `orb.yaml.bak.pre_winner_stack_20260824`, `pre_stage_0`, `pre_shorthist_drop_20260913`, `pre_pause_20260914`,
`pre_resume_20260917`, `pre_buffer50_20260918`, `pre_catoff_20260919` (repo root, untracked). Before the weekly refit
existed the fit was FROZEN (rulebook doctrine 1), so the oldest snapshot's z-params/cutoffs are the set in force from
their fit date (find it: `docs/CLAUDE_HISTORY.md` grep "z-params" / "quintile_cutoffs" / "frozen fit" / the
composite's ship date) onward.

Rules: no git (no commit/stash), no service action, no orders, never write orb.yaml / config.yaml / .env /
data/*.db / crontab / analysis_results/*. **`logs/orb_refit_history.jsonl` is a logs state file: do NOT write
it** — write the seed to `data/research/orb_refit_history_seed.jsonl` (or `research/book_review_20261111/` if
that path is refused) and make `trading/orb_pit_params.load_history` read an optional seed file BEFORE the live
history (a `seed_path` argument defaulting to that file; entries must be strictly older than the first live entry,
else raise). pytest only on files that exist, `--ignore=tests/integration`, one run at a time, never piped through
head/tail. Never Read a file > 300 lines in full (orb.yaml snapshots: grep the `filter:` / `features:` /
`quintile_cutoffs` blocks with -n and sed ranges). Budget ≤ 35 calls. This task IS the owner's request (an honest
PIT book for 4 wk / 6 wk / 6 mo); do not pivot. Do not change any parameter value; you are recording history.

## Steps
1. Extract (params, cutoffs) from each snapshot with the SAME reader the refit uses to write them (grep
   `scripts/orb_weekly_refit.py` for the yaml keys: `filter.features.<f>.{mean,std,sign}`, `quintile_cutoffs`,
   anything else the history entry stores) → a table: snapshot file, mtime (UTC), params hash, cutoffs. Collapse
   consecutive identical sets. Each seed entry = {ts: the moment that set came INTO force (the previous refit's /
   change's timestamp, format `YYYYMMDD_HHMMSS` UTC like the history), params, cutoffs, source: 'snapshot:<file>'}.
   The oldest set gets `ts` = its fit date from the history docs (state the evidence; if unknown, use the oldest
   snapshot's mtime and say the book before that date is still `current_fallback`).
2. Consistency check: for every `bak.refit_<ts>` whose next history entry exists, assert snapshot(before) ==
   previous history entry's params (or the seed's), report mismatches — a mismatch means a manual edit between refits
   (name it and add a seed entry at that snapshot's mtime).
3. `trading/orb_pit_params.py`: `load_history(history_path, seed_path=DEFAULT_SEED)`; `params_for_day` unchanged;
   `source` = 'seed' for seeded days. The pre-history WARNING now names only days before the oldest seed.
4. Tests `tests/test_orb_pit_params_20261111.py` (extend): seed + live history merge order, overlap → raise, a day in
   the seed range → source 'seed' with the snapshot's params, a day before the oldest seed → fallback + WARNING.
   Run `python3 -m pytest tests/test_orb_pit_params_20261111.py tests/test_orb_weekly_refit*.py -q
   --ignore=tests/integration` (files that exist) → 0 failures.
5. Rebuild the PIT book 2025-05-01 → 2026-11-10 into `research/book_review_20261111/orb_bplus_book_pit_full.csv`
   (pipeline output flag; never analysis_results) and report: realized $ and fills for the last 4 wk (since 10/14),
   6 wk (since 9/30), 6 mo (since 5/11), each with ex-top-1 fill and ex-top-5 %, and the `params_source` mix per
   window; beside each the published (current-params) book's number from `analysis_results/orb_bplus_book.csv`.

## Output
`docs/orb_pit_prehistory_seed_20261111_RESULT.md` ≤ 250 words: the seed table, mismatches, file:line, test counts,
the window table. Return ≤ 120 words with the window table.
