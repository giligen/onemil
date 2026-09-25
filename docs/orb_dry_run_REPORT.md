# REPORT — production ORB dry-run mode (2026-09-25)

Implements `docs/orb_dry_run_spec_20260925.md`. ORB live stays PAUSED
(`orb.yaml strategy.enabled: false`); this mode is the pre-relaunch instrument,
not a relaunch.

## Files changed
- `trading/orb_engine.py` — `self.strategy_dry_run` (from `strategy.dry_run`,
  default False) added next to `self.enabled`; production's
  `_run_pool_selection` call now passes `dry_run=self.strategy_dry_run`
  instead of a hardcoded `False`. A new `pool_label == 'production' and
  dry_run` branch inside `_run_pool_selection` (same submit call as the
  addon-pool dry branch — one code path, dry branch at the point of
  submission, no forked path) logs
  `[ORB DRY] WOULD BUY {sym} stop ${trigger} limit ${limit} shares {n}
  risk ${r} | quote {bid}/{ask} at {HH:MM:SS.mmm} ET`, sends the same text as
  a `[ORB DRY]` Telegram, appends a row via the new `_append_dry_ledger_row`
  helper, and calls `_check_first_submit_latency()` so the tripwire fires on
  the would-be submit instant exactly as live. No order object, no
  StopMonitor watch, no `trades` row.
- `orb.yaml.template` — `strategy.dry_run: false` documented next to
  `enabled`. (Live node's real `orb.yaml` is gitignored and untouched — flip
  it manually before the rehearsal.)
- `scripts/orb_selection_observer.py` — new pure function
  `_live_submitted_symbols(submitted_lines, dry_lines)` unions
  `ORB ENTRY SUBMITTED` and `[ORB DRY] WOULD BUY` picks so the dry week's
  live-vs-BT pick set is non-empty; `[ORB+ DRY]` (addon pool) lines are
  never passed to it.
- `tests/test_orb_dry_run.py` — 7 tests, all green: flag default/on parsing,
  zero-orders + WOULD BUY log content, ledger row content, tripwire fires in
  dry mode, flag-off takes the real submit path (ledger writer never
  called), and the observer's dry/live line parsing.

## Grep lines for the dry week
```
journalctl -u onemil-trader --since today | grep '\[ORB DRY\] WOULD BUY'
journalctl -u onemil-trader --since today | grep 'LATENCY TRIPWIRE\|first order submit'
tail -20 logs/orb_dry_ledger.csv
python3 scripts/orb_selection_observer.py   # picks up dry lines via _live_submitted_symbols
```

## Rehearsal plan
1. Weekend boot with `strategy.enabled: true`, `strategy.dry_run: true`,
   `execution.prewarm_seed: true` in the live node's `orb.yaml` (owner
   action — not committed here).
2. Monday–Friday: zero orders, dry ledger + logs accumulate.
3. Pass bar (owner word required before flipping `dry_run: false`):
   - >= 80% pick agreement with the BT book on the same days
   - first would-be submit <= 10s after 09:35:00 ET every day (tripwire log)
   - would-be entries within 25 bps of the BT's entry price
4. Only after all three hold for the full week: real orders resume.

`tests/test_orb_*.py`: 967 passed, 0 failed (127.6s). Not run: the weekend
rehearsal itself (owner-scheduled; requires editing the live node's real
`orb.yaml`, which this task did not touch).
