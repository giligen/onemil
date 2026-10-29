# Exit-coverage invariant + alert flood cap — spec (Fable, 2026-10-29)

Owner: "stop losing money to bugs." LABX 10/27: a partial-fill stall cancel killed the bracket legs; 423 shares sat
with NO broker exit for 6 hours; nothing noticed until a scale-out hit a dead leg id 1,600 times. Four ORB execution
defects in ten days were each found by an incident. This spec adds the one mechanism that catches the whole class.
Scope = exactly the two parts below. Agents never commit / stash / restart services / submit orders (`--submit`
forbidden) / write cache.db, bars_sip.db, config.yaml, orb.yaml, .env, crontab, logs/. `tests/integration` is a REAL
account: every pytest run `--ignore=tests/integration`, one pytest at a time, foreground. The live trader runs from this
working tree during market hours: keep the tree importable after every edit (slice tests after each file).

## Part 1 — `trading/exit_coverage.py` (new, shared) + a 60 s task in StopMonitor
Invariant: **for every WatchEntry in StopMonitor, the broker holds a position of `qty` shares on that watch's account
(`_client_for(watch.strategy)`), and the open SELL orders on that symbol (stop / stop_limit / limit / trailing, any
order class, status in new/accepted/held/partially_filled/pending_*) cover ≥ qty shares.** Covered = fine. Under-covered
= NAKED → act.

Design (one helper module, no engine-specific code outside it):
* `exit_coverage.check(client, symbol, qty_expected) -> CoverageResult(position_qty, covered_qty, open_sell_orders,
  naked_qty)` — pure REST reads: `get_position`-style call (absent position → position_qty 0) and open orders for the
  symbol (the client already has an orders query; reuse it, never a new SDK path). One REST pair per watch per cycle;
  with ≤ 40 watches that is ≤ 80 calls/min on the paper accounts — acceptable; if the watch count exceeds 60, check only
  the oldest 60 per cycle, rotating, and log that once.
* `exit_coverage.repair(client, watch, result, place_stop_fn) -> str` — when `naked_qty > 0` AND the watch has a stop
  price: cancel nothing; place ONE plain stop-sell (`submit_stop_sell_order`, or the existing client method the SNDQ
  emergency path uses — grep `EMERGENCY stop-market` in stop_monitor.py and reuse that call) for `naked_qty` at the
  watch's current stop (lock stop if armed). Record its id on the watch as `sl_leg_id` when the watch had none or a dead
  one (read the old leg: if its status is terminal, replace the id). Log ONE `logger.error` (this IS Telegram-worthy):
  `EXIT COVERAGE: {strategy} {symbol} NAKED {naked_qty}/{position_qty} sh — placed stop-sell @ {stop} (order {id[:8]})`.
  If the placement fails: `logger.critical(... NAKED and the repair failed: {e} — VERIFY ON ALPACA)`.
* Position_qty < watch.shares (broker holds fewer than we think) or 0 → not this spec's business beyond ONE warning per
  symbol per session (`EXIT COVERAGE: {symbol} broker qty {p} < watch qty {w}`) — the orphan / over-exit paths own it.
* Position_qty > watch.shares → treat the EXCESS as naked too (covered must be ≥ broker qty), same repair.
* StopMonitor: a new coroutine `_exit_coverage_loop` started where the other `while self._running` loops start, period
  `_EXIT_COVERAGE_PERIOD_S = 60`, first run 90 s after start (lets boot adoption finish), skipping any watch younger
  than 20 s (leg placement in flight) and any watch currently inside a stop-exit / scale-out / force-close sequence
  (find the flags the monitor already keeps for "exit in progress"; if none exists, add `watch.exit_in_progress` and set
  it at the start of `_execute_stop_exit` and the scale-out, cleared at their end). Exceptions in the loop are caught
  per watch (WARNING with the symbol) — the loop never dies. A per-symbol repair cooldown of 120 s (never two repairs
  in a row for the same symbol without a fresh check confirming the first one did not take).
* Session summary: at the end of every cycle with ≥ 1 naked repair, log INFO `EXIT COVERAGE: {n_watches} watches
  checked, {n_naked} naked, {n_repaired} repaired`. With 0 naked, DEBUG only.
* Config: `stop_monitor.exit_coverage_enabled` (default **true**) read the same way other StopMonitor knobs are read; do
  NOT edit config.yaml — the default lives in code.

Tests (TDD, `tests/test_exit_coverage_20261029.py`, real `StopMonitor` instance with `spec=` AlpacaClient doubles):
covered → no action; dead SL leg + live TP for full qty → covered (any open sell counts); no open sells → ONE stop-sell
for the full qty at the lock stop when armed else the stop, ERROR logged once, `sl_leg_id` set; partial coverage (OCO for
200 of 423, the LABX shape) → stop-sell for 223; placement failure → CRITICAL, no retry inside the cycle, retry next
cycle after the cooldown; broker qty < watch qty → one WARNING, nothing placed; watch younger than 20 s skipped; watch
with `exit_in_progress` skipped; loop survives an exception on one watch and checks the next; disabled flag → loop
exits immediately. Integration-style test: feed a WatchEntry sequence through two cycles and assert exactly one order.

## Part 2 — alert flood cap in `notifications/telegram_notifier.py`
Today: exact-text dedup 60 s + 20/min cap. LABX produced one Telegram line per minute for 3+ hours because the text
was identical but the gate only looks 60 s back. Add a PATTERN cap in the same gate chain (`_dedup_gate` →
`_rate_cap_gate` → new `_pattern_cap_gate`): the pattern key = message with every run of digits, every hex id ≥ 8 chars
and every `$` amount replaced by `#`. If a pattern has been SENT ≥ 3 times inside the last 10 min, suppress; when the
window rolls over (or at the first message after suppression > 10 min later) send ONE line
`⏸ suppressed {n} further "{pattern[:80]}" in the last 10 min`. Never suppress messages whose text contains
`EXIT COVERAGE`, `NAKED`, `EMERGENCY`, `CRITICAL`, `[MOM LIVE]` or `FILL` (money events are never silenced). Unit tests
in `tests/test_telegram_pattern_cap_20261029.py`: 4th identical-pattern message in 10 min suppressed; differing ids
same pattern; summary line sent once; protected words bypass; window reset.

## Validation
Slice: `python3 -m pytest tests/test_exit_coverage_20261029.py tests/test_telegram_pattern_cap_20261029.py tests/test_stop_monitor*.py tests/test_telegram*.py -q -x --no-header -p no:randomly --ignore=tests/integration`
then full `tests/` same flags, zero failures. Write `docs/exit_coverage_invariant_spec_20261029_RESULT.md` (≤ 40 lines:
files, the client calls used for position / open orders / stop-sell, test counts, anything not done). Return ≤ 150 words.
