# Journal flood on the live node (Fable, 2026-11-02) — 2.3M journal lines/day, 1.9 GB of journald

Found while building `/status` (journalctl over today took 16–35 s). Ranked by lines today (`journalctl -u
onemil-trader --since today -o cat`):
1. **1.06M** `scanner.criteria:175` DEBUG `CLOSE CALL: {sym} | Met 5/6 | Missing: ['intraday_change']` — every
   symbol that passes 5 of 6 criteria and simply has not moved is a "close call": ~3,900 symbols per cycle.
2. **1.1M** plain `print()` rows from `scanner/realtime_scanner.py:1593-1602` ("Close calls (N):" table under
   `self.verbose`, same population, 272 cycles today).
3. **46** entries of several hundred KB each: the Alpaca stream subscription ack (`..., corrections: ['SPY'],
   cancelErrors: ['SPY']`) with the full ~6,000-symbol list — a third-party logger (alpaca-py / websockets) at DEBUG
   because `--verbose` sets the root level to DEBUG.
4. `trading.orb_gap_gate:155` 76K, `scanner.criteria:167` 29K, `trading.trading_engine:1351` 26K — look at each: if it
   is a per-symbol-per-cycle line with no state change, same treatment as 1.

The service runs with `--verbose` by design (the owner reads the journal); the fix is the DEFINITION and the cap, not
the level.

## Fixes (production, each with unit tests)
A. **Close call = a real near-miss.** In `scanner/criteria.py` define `is_close_call(candidate)` once: exactly one
   criterion missing AND, when that criterion is `intraday_change`, `intraday_change_pct >= 0.5 × min_intraday_change`
   (the configured threshold; find its name). The DEBUG line at :175 and the scanner's `close_calls` list (find where
   it is built, ~`realtime_scanner.py` before :1593) both use that one predicate. The verbose table prints at most
   the top 20 by `intraday_change_pct` and says `(+N more)`. Expected: ≤ 20 lines per cycle, < 10K/day.
B. **Third-party DEBUG off under --verbose.** Where the log level is configured (grep `setLevel`/`logger.add`/
   `basicConfig` in `main.py`, `utils/`, `monitoring/`), pin `alpaca`, `websockets`, `urllib3`, `httpx`, `asyncio`
   loggers to INFO regardless of `--verbose` (our own loggers keep DEBUG). Test: after setup with verbose=True,
   `logging.getLogger('alpaca').getEffectiveLevel() == logging.INFO` and ours == DEBUG.
C. Item 4: report each line's purpose and count in the RESULT; apply a transition-only or per-symbol-session dedupe
   ONLY where the line carries no state change (say which, with the before/after count estimate). Leave anything that
   records a decision.

Not in scope: journald retention (`SystemMaxUse`) is a system config = owner's line, note it in the RESULT.

## Deliverables
`docs/journal_flood_spec_20261102_RESULT.md`: per item before (today's count) / after (estimated from the predicate on
today's journal: replay the new predicate over today's `CLOSE CALL` lines is not possible — instead count today's
cycles (272) × the cap), the exact code locations changed, tests added, suite count. Return ≤ 150 words.

## Rules
No commit/stash, no restart, no orders, no writes to data/*.db, config.yaml, orb.yaml, .env, crontab, logs/*; one
pytest process at a time, the suite once at the end in the background with a scratchpad log, never piped, always
`--ignore=tests/integration`; never Read > 300 lines at once; nothing deleted from the journal. The service is LIVE
and running — do not touch it; the change is picked up by tomorrow's boot. This task IS the owner's request.
