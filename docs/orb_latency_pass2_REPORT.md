# ORB latency pass 2 — implementation report, 2026-09-25

Spec: `docs/orb_latency_pass2_spec_20260925.md`. Flag: `execution.fast_submit` (default false,
byte-identical when off — verified: full `tests/test_orb_*` suite has the identical 15 pre-existing
failures with the diff stashed in or out, all traced to `data/guardrail_state.json` marking ORB
PAUSED in this environment, unrelated to this change).

## Done (items 1-5)
1. **Instrumentation** (always on, not gated): new `_record_latency_phase` calls for
   `bar_arrival` (`drain_bar_events` — WS queue drain + `_ingest_bars` cost), `scoring` and
   `ranking` (both in `_run_pool_selection`, timed with local checkpoints, not the old
   cumulative-from-`t_rank` pattern), and `vetoes` (the four post-ranking veto calls, timed as one
   `or`-chained block — same short-circuit order as before, so identical behavior). Existing
   `rank_and_submit` phase and the per-order `[ORB] SUBMIT LATENCY` line (G2) were reused, not
   duplicated.
2. **`post_open_range_sweep`**: under `fast_submit`, the REST fallback (`get_1min_bars_multi` /
   `get_1min_bars`) is skipped entirely — logs WARNING with the still-missing symbols and returns,
   relying only on WS bars already ingested via `drain_bar_events`. Off-flag path unchanged.
3. **Vetoes/ranking purity**: `tests/test_orb_fast_submit.py::TestVetoesNoRestNoSql` calls all four
   veto methods with `alpaca=MagicMock(spec=[])` / `db=MagicMock(spec=[])` (raise on any attribute
   access) — confirms zero REST/SQLite calls in the hot path.
4. **Concurrent submit**: ranked, veto-cleared, buying-power-cleared picks are collected in order
   into `_pending_submits`; under `fast_submit` the REST `_submit_entry` calls run through a
   `ThreadPoolExecutor` (`min(8, n)` workers), results applied back in ranking order — a failed
   submit (`order_id` falsy) leaves its slot empty, no refill. Off-flag path is the original serial
   per-symbol call, untouched.
5. New `tests/test_orb_fast_submit.py` (7 tests, all green): no-REST/SQLite vetoes, ranking-order
   preservation + no-refill-on-failure, 8-worker cap, flag default/on, sweep REST-skip (clock
   pinned to 09:40 ET so the window gate doesn't mask the assertion). Full `tests/test_orb_*.py`:
   969 passed, 15 pre-existing failures (unrelated — see above), 0 new failures.

## Not done — item 6 (offline replay)
Blocked by the spec's own guard: "no cache.db reads 13:25–20:05 UTC" — this task ran entirely
inside that window (`date -u` = 2026-09-25 15:1x UTC). No `data/cache.db` read was performed.
**Next step**: rerun a replay of a recorded day's 09:30-09:36 bar events through the engine with
`prewarm_seed` + `fast_submit` both on, after 20:05 UTC (or before 13:25 UTC any day), print the
measured breakdown, and append the timing + rehearsal plan to this file before any owner GO.

## Expected live effect (analytical, not yet measured)
`post_open_range_sweep` REST calls were the dominant remaining cost after pass 1 (5.5-6.9s
observed) — removing them should bring that phase to ~0s (bounded by whatever stragglers exist,
logged not blocked). Concurrent submit should cut the up-to-8×(100-300ms) serial `rank_and_submit`
tail to roughly one round-trip (~100-300ms) for a full slate. Combined with pass 1's prewarm fix,
first-submit should land near the ≤3s target — **this must be confirmed by the item-6 replay, not
assumed**, before `fast_submit: true` goes anywhere near the live config.

## Rehearsal plan (unchanged pattern from pass 1)
Weekend boot rehearsal on the exact `ExecStart` with both `prewarm_seed` and `fast_submit` true,
real-API probes (paper or read-only), a full report run, grep for ERROR/exception — required before
either flag flips on the live node, per CLAUDE.md's pre-deploy rehearsal protocol.
