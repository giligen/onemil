# SPEC — ORB latency pass 2: first order within ~3 s of 09:35:00 (after the pre-warm fix), 2026-09-25

## Context
Pass 1 (`docs/orb_latency_fix_REPORT.md`, flag `execution.prewarm_seed`) removes the 55–64 s universe seed. The
tripwire's own breakdown still shows `post_open_range_sweep 5.5–6.9 s` plus `rank_and_submit` (serial REST submits,
each 100–300 ms × up to 8 picks) plus whatever the entry-drain thread adds. Owner 9/25: "if you need to reduce latency
to near zero, do it". Target: first order submitted ≤ 3 s after 09:35:00 ET on a normal day; every stage measured and
logged in the existing `measured:` line.

## Task (Sonnet; ≤ 40 tool calls; TDD; do not commit; do not start/restart any service; no DB writes; no cache.db
reads 13:25–20:05 UTC — check `date -u`)
1. Instrument first: extend the `measured:` breakdown so EVERY stage between the 09:35:00 bar close and the first
   submit is timed (bar arrival → range sweep → scoring → ranking → vetoes → per-order submit), and log the per-order
   submit latency (already specced in `docs/live_guardrails_spec_20260925.md` G2 — reuse, don't duplicate).
2. `post_open_range_sweep`: it must use the 1-minute bars already delivered by the websocket (StopMonitor bar handlers /
   the scanner's bar cache) — never a REST bars call at 09:35. If a symbol's 09:34 bar has not arrived by 09:35:02,
   log it at WARNING and use what is there (do not block the sweep on stragglers). Time it; target ≤ 0.5 s.
3. Ranking/vetoes: pure in-memory (they already should be); assert no REST/SQLite call in the hot path with a test
   using `MagicMock(spec=...)` collaborators that raise on unexpected calls.
4. Submits: submit the ranked picks concurrently (a small thread pool, ≤ 8 workers, one REST call each) instead of
   serially; preserve the ranking order for slot arithmetic (assign slots before submitting; a failed submit frees its
   slot without refill — the no-refill rule stays). Time it; target ≤ 1 s for 8 picks.
5. Flag: everything behind `execution.fast_submit` (default false; byte-identical when off). Tests in
   `tests/test_orb_fast_submit.py`: no REST/SQLite in the hot path when on; concurrent submits keep ranking order and
   slot accounting; a failed submit does not refill; flag off unchanged (parity on a recorded tick). Run the full
   `tests/test_orb_*` set — zero failures.
6. Offline timing: replay a recorded day's 09:30–09:36 bar events (from `data/cache.db` before 13:25 or after 20:05
   UTC) through the engine with both flags on and print the measured breakdown; write it to
   `docs/orb_latency_pass2_REPORT.md` with the expected live first-submit time and the rehearsal plan. Return ≤ 150 words.
