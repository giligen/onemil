# ORB pick parity: a composite drift caused by a vendor bar revision is a NAMED reason, not `unexplained` (2026-11-20)

## The incident (verified read-only: journal/archive, analysis_results/orb_features_20261119_2141.csv, Alpaca SIP bars)
* The ORB ramp has been FROZEN since 2026-11-13 (`logs/ramp_freeze.json`). The 11/13 breach (DAMD, KOLD, WSE) no
  longer exists: a re-run of `scripts/orb_pick_parity.py --date 2026-11-13` on 11/20 gives `matched 7/8 | diffs 2
  (slot_displaced 1, universe_live_only 1) | unexplained 0` (the features CSV has been rebuilt since). The freeze was
  re-raised on 2026-11-16 and that one still reproduces:
  `ORB pick parity 2026-11-16: matched 8/8 | diffs 1 (unexplained 1)` —
  `MUZ both unexplained live comp 0.4531 vs BT 0.4538; refit 20261115_200002`.
* MUZ was in BOTH final top-8 sets (side `both`). The only difference is the composite, Δ = 0.0007 > COMP_TOL 1e-4,
  with matching parameter timestamps. Line ~258-262 of `scripts/orb_pick_parity.py` classes that `unexplained`.
* Cause, feature by feature. Live `ORB SCORED` line (session archive 2026-11-16 14:35:11 UTC):
  `MUZ comp=0.4531 Q5 | gap=11.482 rtv=125552 rabr=1.213 rs=3.933 p20h=-26.849 pdcp=0.373 rcp=0.939`.
  BT feature row (orb_features_20261119_2141.csv, MUZ 2026-11-16): gap_pct 11.482, **range_total_volume 125652**,
  range_avg_bar_range_pct 1.2132, range_size_pct 3.9326, price_vs_20d_high_pct -26.849, prev_day_close_position
  0.3725, **range_close_position 0.9405**. Two of the seven composite inputs differ: range volume by exactly 100
  shares and the range close position by 0.0015. Alpaca SIP 1-minute bars for MUZ 14:30–14:34 UTC fetched on 11/20
  sum to 125,652 (24643+30676+14570+27882+27881) and give (11.02−10.625)/(11.045−10.625) = 0.9405 — the BT values.
  So a 100-share late print landed in the SIP bars AFTER the engine took its 09:35 snapshot; the engine cannot wait
  for it (it must act at 09:35), and the nightly feature build sees the revised bars. The decision did not change:
  MUZ was ranked, picked and ordered on both sides.
* Precedent: `docs/reporting_fixes_20261017.md` §A — a composite drift is a hard breach only when it changes a
  decision. The pick parity (cell 1,733 spec `docs/orb_selection_parity_test_spec_20261111.md`) has no reason for
  this case yet, so an unavoidable data revision reads as "we do not know what we are running".

## The fix (TDD: failing tests first). Plumbing in the parity classifier; no engine, strategy or constant change.
### 1. `scripts/orb_pick_parity.py`
* Add reason `bar_revision` to `REASONS`.
* Parse the live features from the `ORB SCORED` line for every production-pool name (the engine format,
  `trading/orb_engine.py` ~4010-4022; key → features-CSV column, logged precision):
  `gap`→`gap_pct` (%.3f), `rtv`→`range_total_volume` (%.0f), `rabr`→`range_avg_bar_range_pct` (%.3f),
  `rs`→`range_size_pct` (%.3f), `p20h`→`price_vs_20d_high_pct` (%.3f), `pdcp`→`prev_day_close_position` (%.3f),
  `rcp`→`range_close_position` (%.3f). Put it in `parse_live` as `"features": {sym: {column: value}}` (a module
  constant `LIVE_FEATURE_KEYS` holding the mapping and the precision). Use the same production-pool filter the
  existing parsing uses (`es._is_production_scored`); take the FIRST production line of a symbol inside the
  decision window the way `scored_any` does (read `es.parse_orb_log` ~616-690 and reuse its helpers, do not
  duplicate the window logic — if the simplest correct way is to add the features to `scored_any`'s tuple in
  `eod_sections.parse_orb_log`, do that instead and keep every existing caller working; `scored_any` is consumed at
  `eod_sections.py` ~845 and in `orb_pick_parity.parse_live`).
* Expose the BT feature row per symbol from `eod_sections.bt_ranked`: add `"features": {symbol: {column: float}}`
  for the seven composite columns of every row of the day (the frame `g` already holds them; `FILTER_FEATURES` in
  `study_orb_filter.py` lists the columns). No other change to `bt_ranked`.
* New helper `feature_revisions(live_feats, bt_feats) -> List[str]`: the composite inputs whose live value differs
  from the BT value by more than the live log's rounding (half a unit of the last logged decimal, i.e. 0.0005 for
  %.3f, 0.5 for %.0f, plus 1e-9) — returns strings like `range_total_volume 125552 vs 125652`,
  `range_close_position 0.939 vs 0.9405`. Empty list when every input matches to the logged precision.
* Classification, in the shared-name loop (~258-262) AND in `_bt_only_reason` (~183-188) AND in the `live_only`
  branch (~271): when `ts_ok` and `|Δcomp| > COMP_TOL`:
  - `bar_revision` with detail `live comp X vs BT Y; inputs revised after the 09:35 snapshot: <revisions joined by
    ', '>` when `feature_revisions` is non-empty AND `|Δcomp| <= HARD_COMP_DRIFT` (new constant 0.05 — the
    rollout plan's "composite drift held < 0.05", `docs/orb_rollout_plan.md`), AND — for the shared-name case — the
    name is in both final sets (it is, by construction of that loop);
  - otherwise stays exactly as today (`unexplained` with the current detail; a drift with every input matching is a
    formula/parameter bug and must keep alarming; a drift above HARD_COMP_DRIFT is a different data source).
  For `bt_only` / `live_only` names a revision that moved the name across the top-K boundary IS a decision change:
  keep those `unexplained` unless the existing `slot_displaced` logic already names them; do not widen that. State
  in your return if you found the `bt_only`/`live_only` composite branches never reach the comp check in practice.
* `summary_line` / alarm: `bar_revision` is an explained difference → INFO, never ERROR (the existing rule
  "ERROR only for `unexplained` / fills outside the final set" already does this; verify with the test).
### 2. Tests — `tests/test_orb_pick_parity_bar_revision_20261120.py` (style of `tests/test_orb_pick_parity_20261111.py`;
read its builders first and reuse them, do not copy the fixture)
1. The MUZ pair (live 0.4531 with the live features above; BT 0.4538 with the CSV values above; same refit ts; both
   in final) → one difference, reason `bar_revision`, detail names `range_total_volume 125552 vs 125652` and
   `range_close_position`; `n_unexplained == 0`; `emit`/alarm logs INFO, no ERROR record in caplog.
2. Same composites, every feature equal to the logged precision → `unexplained` (the existing detail text).
3. A revision present but Δcomp = 0.06 → `unexplained`.
4. `feature_revisions`: rounding inside half a unit is NOT a revision (rtv 125652 vs 125652.4; rcp 0.940 vs 0.9404),
   one unit is.
5. Parser: the real MUZ line → the seven features with the right columns; a line without the feature block (old
   format) → no features and no exception; an add-on-pool line is ignored.
6. Integration (marked like the existing real-log tests, skipped when the files are absent): run
   `scripts/orb_pick_parity.py` logic for 2026-11-16 against the real session archive and the real newest features
   CSV → `n_unexplained == 0` and the MUZ difference is `bar_revision`. Keep it read-only.
### 3. Docs
* `docs/orb_selection_parity_test_spec_20261111.md`: append an "Amendment 2026-11-20" paragraph: the 1e-4 rule
  stands; a shared-name drift ≤ 0.05 whose composite inputs differ in the live `ORB SCORED` line vs the rebuilt
  features row carries the reason `bar_revision` naming the inputs (a vendor bar revision after the 09:35 snapshot,
  MUZ 2026-11-16); drift with identical inputs, or > 0.05, stays `unexplained`.
* Append a short "Result" section to THIS file: what you changed, test counts, the 11/16 re-run summary line.

## Run (these files only; the main session runs the full suite — never start it yourself)
`python3 -m pytest tests/test_orb_pick_parity.py tests/test_orb_pick_parity_20261111.py
tests/test_orb_pick_parity_bar_revision_20261120.py tests/test_eod_sections*.py tests/test_daily_green_check*.py
-q -p no:cacheprovider --ignore=tests/integration` → all green; then
`python3 scripts/orb_pick_parity.py --date 2026-11-16` → `diffs 1 (bar_revision 1) | unexplained 0`, INFO.

## Rules
Sonnet, ≤ 60 calls. No `git commit`/`git stash`, no service restart, no orders, no writes to `data/*.db`,
`logs/*`, `orb.yaml`, `config.yaml`, `.env`, `analysis_results/*`. Files you may edit: `scripts/orb_pick_parity.py`,
`scripts/eod_sections.py` (only `parse_orb_log` features and `bt_ranked` "features"), the new test file,
`docs/orb_selection_parity_test_spec_20261111.md`, this file. Do not touch `COMP_TOL`, the engine, the BT pipeline
or any frozen constant. Docstrings on every new function; every fallback logs WARNING. Return ≤ 150 words:
files changed, test counts, the 11/16 re-run line, anything you could not do.

## Result
* `scripts/orb_pick_parity.py`: reason `bar_revision`, `HARD_COMP_DRIFT`, `feature_revisions`, `_shared_drift`, `parse_live` "features".
  `scripts/eod_sections.py`: `LIVE_FEATURE_KEYS`, `parse_scored_features`, `parse_orb_log["scored_feats"]` (separate key, `scored_any` tuple untouched), `bt_ranked["features"]`.
* Tests: 160 -> 168 green (8 new in `tests/test_orb_pick_parity_bar_revision_20261120.py`). Amendment added to the 11/11 spec.
* 11/16 re-run: `ORB pick parity 2026-11-16: matched 8/8 | diffs 1 (bar_revision 1) | unexplained 0` (INFO).
* bt_only/live_only composite branches left as they were (a revision crossing the top-K boundary is a decision change).
