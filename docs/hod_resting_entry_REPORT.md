# HOD resting stop-limit entry — implementation report (2026-09-25)

Spec: `docs/hod_resting_entry_spec_20260925.md`. Dry only; no order is ever submitted in `resting_stop_limit`
mode regardless of `dry_run`.

## Files
* `trading/hod_break.py` — shared spec module: `HodBreakParams.entry_limit_pct` (default 0.0015), `arm_state()`
  (arming at bar j's close for bar j+1, closed-data only), `resting_entry_fill()` (fill iff ask <= limit).
* `trading/hod_break_engine.py` — `entry_mode` cfg (`next_open` default, byte-identical; `resting_stop_limit`),
  `dry_ledger_path` cfg (default `logs/hod_dry_entry_ledger.csv`), `Candidate.resting_arm/resting_scanned_idx/
  resting_filled`, `_evaluate_resting()` (bar-by-bar arm/resolve walk, reused by backfill batches), `_append_dry_ledger()`.
  No print stream exists in this engine, so every cross uses the documented fallback: bar high >= trigger + the
  current quote, logged WARNING and `tape_accurate=0` in the ledger.
* `tests/test_hod_resting_entry.py` — 13 tests: arming boundaries (consolidation, min_dist_open_pct, rv band,
  last_entry_minute), re-arm after a no-cross bar, fill-at-limit / no-fill-above, one-fill-per-symbol-day, ledger
  row shape, `entry_mode=next_open` unchanged (submits as before, resting fields untouched), and a PARITY test vs
  `research/hod_entry/causal_arming.py`'s `arm_state` on a synthetic day. `tests/test_hod_*.py`: 122/122 pass.

## Config keys (orb.yaml-style `hod_break:` block)
```yaml
hod_break:
  entry_mode: resting_stop_limit   # or next_open (default, unchanged)
  dry_run: true                    # MUST stay true for this rollout
  dry_ledger_path: logs/hod_dry_entry_ledger.csv
  params:
    entry_limit_pct: 0.0015
```

## Grep lines for the dry sessions
```
journalctl -u onemil-trader --since today | grep '\[HOD DRY\] ARMED'
journalctl -u onemil-trader --since today | grep '\[HOD DRY\] CROSS'
journalctl -u onemil-trader --since today | grep 'no live print stream'   # tape_accurate=0 rows
```
`logs/hod_dry_entry_ledger.csv` — one row per cross (filled or not); `tape_accurate` column flags the fallback rows.

## Rehearsal plan (owner GO required before any config change on the live node)
1. Weekend boot rehearsal on the exact `ExecStart` with `hod_break.entry_mode: resting_stop_limit`,
   `dry_run: true`; confirm `[HOD DRY] ARMED`/`CROSS` lines appear and `logs/hod_dry_entry_ledger.csv` grows.
2. Real-API probes: `_quote()` against live Alpaca data during RTH, confirm no exceptions, confirm WARNING fires
   on every fallback cross (expected on every row today — no print stream is wired yet).
3. A real report run: `python3 scripts/hod_dry_ledger.py` extended (not done here) or an ad-hoc read of the CSV.
4. Grep the boot log for ERROR/exception; zero tolerated before the owner enables the node for the week.

## Daily comparison (main session runs)
Compare live dry fill rate and mean fill R against TEST's 27.8% / +0.33 R (`research/hod_entry/REPORT_1427.md`):
`fills / crosses` from the ledger CSV, `(fill_px - stop) → outcome` once `scripts/hod_dry_ledger.py` is extended to
read this CSV (not done in this change — out of the 40-call budget; the CSV schema is stable and ready for it).
