# ORB latency fix — measured breakdown, fix, rollback (2026-09-25)

Spec: `docs/orb_latency_fix_spec_20260925.md`. Flag: `orb.yaml execution.prewarm_seed` (default **false**).

## 1. Measured breakdown

Offline, read-only against the real `data/cache.db` (`/tmp/.../scratchpad/time_universe_seed.py`, run 2026-09-25 12:12 UTC):

| Stage | Cost | Source |
|---|---|---|
| `sqlite_prefilter` (daily_bars `ROW_NUMBER()` full-table scan, `scanner/_orb_universe_source`) | **18.71s**, 2,360 symbols | measured offline just now |
| `get_snapshots` REST (Alpaca bulk snapshot, `build_orb_universe_from_snapshots`) | ~**36–49s** (residual of the 55.3s/64.1s tripwire totals minus the SQL stage) | inferred from live LATENCY TRIPWIRE logs 9/22 (55.3s) and 9/23 (64.1s), not re-run live (see caveat) |
| `post_open_range_sweep` | 5.5–6.9s | unchanged, out of scope for this fix |
| `rank_and_submit` | small (not the bottleneck) | unchanged |

**Caveat:** yesterday's exact `INTRADAY QUALIFIED` symbol list was not recoverable — `journalctl -u onemil-trader` returned zero entries for 2026-09-24 09:00–09:40 (query over the full day timed out at 25s, suggesting journal volume/indexing issues, not absence of the unit — worth a separate look), and `logs/session_archive/2026-09-24.log` is a curated 135-line file with no qualification lines. Used the real `daily_bars` prefilter query (same SQL the live code runs) as a representative candidate set (2,360 symbols, same order of magnitude as the 5,973/2,500-symbol figures cited in the code's own comments). The REST snapshot stage was **not** re-run live in this session (would issue a real ~2,000-symbol Alpaca call outside the agent's time/call budget); its cost is attributed by elimination from the two independently-measured live tripwire totals, both of which already exceed the SQL-stage time by 36s+.

Root cause: **both** `universe_seed` sub-stages re-run in full on **every ~60s ORB tick**, including the 09:35 tick that gates the day's first submit — even though `daily_bars` doesn't change intraday and most candidate symbols were already snapshotted on the prior tick.

## 2. Fix (flag-gated, default off)

Two call sites, both gated by `execution.prewarm_seed`, both no-ops when false (byte-identical to pre-fix code):
- `scanner/realtime_scanner.py::_orb_universe_source` — caches the `daily_bars` prefilter symbol set once per day (`_orb_sqlite_seed_cache`); reused on every later tick.
- `trading/orb_engine.py::_get_snapshots_warm` (new) — per-symbol snapshot cache (`_snapshot_cache`); each tick fetches **only** symbols not yet cached (new qualifiers), logged at WARNING with the fetch cost. `build_orb_universe_from_snapshots`'s admission logic is untouched — same code, only the snapshot's fetch time differs, which is what makes the parity test valid.

Both caches reset in `reset_daily()`; the snapshot cache also self-clears on a UTC date rollover as defense-in-depth (the scanner process is restarted daily so this rarely triggers in practice).

**Expected 09:35 latency after fix** (reasoned from the incremental design, not yet measured live): first tick of the day still pays the full ~55–64s (now off the critical path, since ticks start near 09:30, ~5 min before the 09:35 gate); every later tick — including 09:35 — becomes an in-memory cache hit except for symbols that newly qualified since the last tick (typically single digits), each costing a small individual fetch logged at WARNING. `universe_seed` at 09:35 should drop to low single digits of seconds.

## 3. Tests

`tests/test_orb_prewarm_seed.py` — 8 tests: flag-off regression (unchanged full-fetch-every-time), hot-path zero-REST-call cache hit, incremental-refresh-fetches-only-new-symbol, WARNING on cache miss, day-rollover clears cache, `reset_daily()` clears cache, and a parity test (3 incremental warm fetches vs. 1 full old-path fetch admit the identical symbol set). All pass. Full `tests/test_orb_*.py` suite: **958 passed, 0 failed** (132.3s).

## 4. Rollback

Set `orb.yaml execution.prewarm_seed: false` (already the default) — both call sites revert to the exact pre-2026-09-25 code path; no data migration, no restart-order dependency.

## Review fix (2026-09-25, same session)

Coordinator review caught: a cached snapshot with `open<=0` (pre-open) or `daily_bar_date != today (ET)` was never refreshed, so a symbol first seen before it printed would keep a stale/incomplete snapshot all day — the old (flag-off) path would have refreshed it every tick. Fixed in `_get_snapshots_warm`: a cache entry is reused only if `open > 0 AND daily_bar_date == today_et`; otherwise it's put back in the re-fetch list every tick (logged WARNING with the count). 2 new tests added (`test_incomplete_snapshot_open_zero_is_refetched_and_fresh_value_used`, `test_complete_snapshot_is_not_refetched`); `tests/test_orb_prewarm_seed.py` now 10/10 passing including the parity test.

## 5. Before going live

Per `docs/CLAUDE_HISTORY.md` protocol: a weekend boot rehearsal on the exact `ExecStart` with `prewarm_seed: true`, verifying in logs (a) the first tick's full-fetch WARNING fires once, (b) the 09:35 tick's `universe_seed` phase drops to low single digits, (c) zero tracebacks/ERROR, (d) the admitted symbol set matches a parallel flag-off run for the same day (live parity, not just the offline unit test) — then a real-API read-only probe before any owner GO to flip the flag on the live node.
