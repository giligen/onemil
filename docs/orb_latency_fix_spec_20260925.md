# SPEC — ORB first-order latency: the 09:35 hot path must not take 26–49 s (live defect, 2026-09-25)

## Evidence (journalctl -u onemil-trader, `LATENCY TRIPWIRE`)
* 9/18 first submit 09:35:48.9 ET (48.9 s late); 9/22 09:35:26.5 (universe_seed 55.3 s, post_open_range_sweep 6.9 s);
  9/23 09:35:28.1 (universe_seed 64.1 s, range sweep 5.5 s). The tripwire threshold is 10 s.
* Consequence 9/22: `BIAF ENTRY SKIPPED — breakout extended past limit (ask 8.55 + buf 0.02 > limit 8.51)` — the
  backtest's best fill of the week (+$576 at $375 risk) was missed because we arrived 26 s after the breakout. The
  verification (`research/orb_verify/E_parity.md`) found live captured 3 of the BT's 5 fills; fast breakouts — the
  winners — are exactly the ones a late arrival loses. The backtest assumes the order is resting at 09:35:00.

## Task (Sonnet; ≤ 40 tool calls; TDD; do not commit; do not start/restart any service; no DB writes)
1. Instrument, don't guess: find the code behind the `measured: universe_seed …, post_open_range_sweep …, rank_and_submit …`
   line in `trading/orb_engine.py` and the scanner's `orb_engine.build_universe(source_loader=…)` call
   (`scanner/realtime_scanner.py` ~line 200 and the per-cycle re-seed). Enumerate every call inside `universe_seed`
   that touches the network (Alpaca REST: bars, quotes, asset names, news), SQLite (`daily_bars`, cache), or
   per-symbol Python loops over the whole scanner universe. Time each stage offline: a script that runs the seed
   function against the real `data/cache.db` (read-only) with yesterday's qualified list (journal `INTRADAY
   QUALIFIED` symbols for 2026-09-24) and prints per-stage seconds. Write the breakdown to
   `docs/orb_latency_fix_REPORT.md`.
2. Design the fix on the principle: the 09:35:00 hot path is pure in-memory. Everything the seed needs (prev close, ADV,
   ATR, PDR, anchor/asset class, quintile inputs) is pre-computed in a WARM phase that runs from 09:30 to 09:34:30 ET on
   the list as of then and is refreshed incrementally (only for symbols that qualified after the warm) — never a full
   rebuild at 09:35. Any call that cannot be pre-warmed is logged at WARNING with its cost. Keep ONE spec for BT and
   live: the selection inputs must be byte-identical to what the seed computes today (add a parity test that the warm
   path and the old path produce the same ranked list on a recorded day).
3. Implement behind a config flag `orb.yaml execution.prewarm_seed` (default false until the owner's GO), with unit
   tests (`tests/test_orb_prewarm_seed.py`: warm phase populates the cache; the hot path makes zero REST/SQLite calls
   — assert via `MagicMock(spec=...)` collaborators; incremental refresh handles a symbol that qualifies at 09:34:50;
   parity of the ranked output) and an integration test on a recorded day. Run the full `tests/test_orb_*` set — zero
   failures.
4. Report in `docs/orb_latency_fix_REPORT.md`: the measured breakdown, the expected 09:35 latency after the fix
   (measured offline), what the flag changes, the rollback (flag false), and what must be rehearsed (a weekend boot
   rehearsal per `docs/CLAUDE_HISTORY.md` protocol) before it goes live. Return ≤ 150 words.
