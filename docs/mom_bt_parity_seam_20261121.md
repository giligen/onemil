# MOM BT parity: the adjustment=ALL extension breaks the panel seam, and the replay's week window drops the weekend gap (2026-11-21)

## The incident (verified read-only: logs/mom_bt_parity.log 11/14 vs 11/21, panel + extension parquet, cache.db raw daily bars, broker GETs)
* Saturday 14:20 UTC `scripts/mom_bt_parity.py` sent `[MOM] MISMATCH BT parity wk 11/16: names 19/20 | BT -11.4 % | paper -19.6 % | live -15.1 % | SPY -8.0 % | resid -8.2 pp`
  (BT-only HPE, paper-only SNDK). Last Saturday every week was 20/20; today EVERY week since 10/5 is 19/20 with the same HPE/SNDK swap.
* SNDK split 8-for-1, ex-date 2026-11-20 (broker announcement; live account activity SPLIT 0.4619 -> 3.6956 sh).
  `scripts/mom_panel_refresh.py` fetches with adjustment=ALL, so Alpaca restated SNDK's WHOLE extension history to the post-split
  scale (ext 2026-09-29 close 216.22) while the frozen base panel `research/momentum_weekly/panel_2016_2026.parquet` keeps the raw
  scale (2026-09-29 close 1729.76). `mom_bt_parity.extend_data` appends ext bars after each symbol's last panel bar (2026-09-29 for
  13,262 symbols) WITHOUT rescaling -> the replay sees SNDK fall 87.5 % on 10/5 and drops it from every top-20; HPE fills the slot.
  Seam check today: 13,266 symbols overlap at 2026-09-29; 1,513 differ by > 0.5 % (dividend restatements), 186 by > 5 %
  (splits / reverse splits: SNDK x0.125, BUUU x0.1, ELME x0.017, AXG x200, RCT/LICN/SXTC/TWG x100 ...). The refresh itself is
  correct (it replaces the restated symbols' extension rows, 275 today); only the parity's seam is wrong.
* Window defect: `bt_weeks` measures the BT week Monday OPEN -> Friday close, while `paper_week_ret` / `live_week_ret` measure
  account equity from the last session BEFORE Monday -> Friday close. The weekend gap is dropped from the BT side. Week 11/16 on
  the raw cache, equal weight, the BT's 20 names: Mon open -> Fri close -11.45 % (= the BT -11.4), Fri close -> Fri close -14.48 %;
  gap 3.03 pp. The real BT chain in `research/momentum_weekly/REBUILD_1700tu.py` (lines 154-160) is open -> next open, i.e.
  complete; only the parity replay is lopsided.
* Decomposition of the -8.2 pp: 3.0 pp weekend-gap window, 0.7 pp HPE-for-SNDK (seam artefact), 4.5 pp PAPER-account artefact:
  Alpaca PAPER did not apply the split to the fractional SNDK position (still 0.827562 sh @ $79.24 = $65.58 instead of 6.62 sh =
  $524.6; no SPLIT activity on the paper account; the live account applied it). The live book (-15.1 %) equals the true equal-weight
  close -> close return of the held names (-15.14 %). Paper is understated by ~$459 (3.3 % of the slice) until Alpaca fixes it;
  nothing of ours to change there. The live sleeve resyncs quantities from the broker at every run (`sync_state_from_broker`), so
  Monday's rotation sells the post-split quantity; `data/momentum_sleeve/state_live.json` is only stale until then.

## The fix (TDD: failing tests first). `scripts/mom_bt_parity.py` only. No change to REBUILD_1700tu, mom_panel_refresh.py,
## the parquet files, NAMES_MIN / RESID_MAX_PP / N_WEEKS, or the paper/live return functions.
1. Seam rescale. New helper `rescale_seam(data: Dict, ext: pd.DataFrame, tol: float = SEAM_TOL) -> Dict[str, float]`
   (`SEAM_TOL = 0.005`, module constant) called by `extend_data` BEFORE appending: for every symbol with extension rows that is in
   `data`, take the extension bar dated exactly the symbol's last panel date `d[-1]`; `ratio = ext_close / panel_close`; when
   `abs(ratio - 1) > tol`, multiply the panel's open/high/low/close arrays by `ratio` and divide volume by `ratio` (in place), and
   record `{sym: ratio}`. Log ONE INFO summary: count rescaled, count > 5 %, and the five most extreme ratios by symbol
   (`SNDK x0.1250`). A symbol with extension rows but NO bar at `d[-1]` is appended unscaled with a WARNING naming it (fallback).
   `extend_data` keeps its signature and return value (the appended row count) -- `tests/test_mom_bt_current_20261031.py` must
   still pass unchanged. Docstrings state why (adjustment=ALL restates history; the frozen panel is raw).
2. Window. In `bt_weeks` the week return (names and SPY) runs from the CLOSE of the last session before the rebalance Monday
   (`cal[cal.searchsorted(reb) - 1]`, close) to the end close -- the same window as `paper_week_ret`. The signal date, the names,
   `end` and `complete` are unchanged. Update the docstring ("last close before Monday -> last close before the next Monday").

## Tests -- `tests/test_mom_bt_parity_seam_20261121.py` (import the module the way `tests/test_mom_bt_current_20261031.py` does)
1. SNDK replay: a synthetic panel dict with one symbol at the 1700 scale, daily bars through 2026-09-29, and an extension frame at
   1/8 scale from 2026-09-01 to 2026-11-20 -> after `extend_data`: the panel part is multiplied by 0.125 (close on 2026-09-29 equals
   the extension's), volume x8, the series is continuous across the seam (no bar-to-bar move > 20 %), the return value equals the
   number of extension rows after 2026-09-29, the INFO summary names the symbol and ratio.
2. A symbol off by 0.2 % at the seam -> not rescaled, no ratio returned.
3. A symbol with extension rows but no bar on its last panel date -> appended unscaled, exactly one WARNING naming it.
4. A symbol absent from the panel -> skipped silently (existing behaviour pinned).
5. Integration with the real files (skip with a reason when either parquet is missing): read ONLY SNDK from the base panel
   (`pd.read_parquet(..., filters=[('symbol', '==', 'SNDK')])`) and from the extension; build a two-symbol `data` (SNDK + a copy
   as 'SPY' is NOT needed -- `extend_data` does not touch SPY) -> after `extend_data` the SNDK seam ratio is within 0.5 % and the
   2026-11-20 close is 79.24.
6. Window: a stub rule object (`TOPN`, `features` returning the same eligible rows for each date, `px` copied from REBUILD_1700tu's
   semantics) with SPY and TOPN+1 symbols where Monday open != previous close -> `bt_weeks(...)[i]['ret']` equals
   mean(close_end / close_prev_session - 1) and `spy_ret` likewise; a value computed from the Monday open must NOT match.
7. `format_line` / `is_mismatch` pinned: a row with 19 names and resid -8.2 is MISMATCH; 20 names and -0.7 is not.

## Run (these files only; the main session runs the real script and the full suite -- NEVER run `scripts/mom_bt_parity.py` yourself)
`python3 -m pytest tests/test_mom_bt_parity_seam_20261121.py tests/test_mom_bt_current_20261031.py
tests/test_mom_panel_refresh_gate_20261107.py -q -p no:cacheprovider --ignore=tests/integration` -> all green.

## Rules
Sonnet, <= 50 calls. No `git commit`/`git stash`, no service action, no orders, no writes to `data/*`, `logs/*`, `research/*`,
`config.yaml`, `orb.yaml`, `.env`. Never load the full 2 GB panel (`R.load()`) -- the integration test reads SNDK only with a parquet
filter. Files you may edit: `scripts/mom_bt_parity.py`, the new test file, and a "Result" section appended to THIS file. Docstrings on
every new function; every fallback logs WARNING. Return <= 120 words: files changed, test counts (red before / green after), anything
you could not do, verified facts separated from guesses.

## Result (2026-11-21)
* Implemented in `scripts/mom_bt_parity.py`: `SEAM_TOL`, `rescale_seam` (called first by `extend_data`, signature/return unchanged; INFO summary, WARNING fallback for no seam bar), and `bt_weeks` now measures last close before Monday -> end close.
* New `tests/test_mom_bt_parity_seam_20261121.py` (7 tests). Red before the fix: 5 failed / 2 passed (the pinned ones). Green after: spec Run command = 28 passed.
* Real-file integration test ran (SNDK-only parquet filter): seam within 0.5 %, 11/20 close 79.24 verified. Real script not run, full suite not run (main session).

## Review (main session, 2026-11-21 17:4x UTC)
Diff read: `rescale_seam` (overlap-bar ratio, OHLC x ratio, volume / ratio, WARNING when no overlap bar) and the close -> close
window match the spec. Real-script dry run after the fix: 316 symbols rescaled (68 by > 5 %), every week 20/20, residuals -0.1 /
-0.3 / -0.4 / +0.9 (10/26, pre-existing 17/20 names) / -0.9 / -0.2 / -4.5 pp; week 11/16 BT -15.1 % = live -15.1 %, the -4.5 pp is
the paper account's un-split fractional SNDK (broker artefact, re-check Monday). Full suite then commit; the corrected [MOM] line
is sent by the real run after the commit.
