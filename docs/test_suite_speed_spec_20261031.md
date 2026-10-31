# Spec — the test suite must run in minutes, not eleven (Fable, 2026-10-31; owner: "11 min test suite means there are unnecessary delays, should take 1 min for 3000 tests")

Today: `timeout 2700 python3 -m pytest tests -x -q --ignore=tests/integration -p no:randomly` = 5,470 passed in
577–750 s on a 2-vCPU t3.large. Target: **≤ 180 s wall** for the same command, no test deleted, no assertion weakened,
no test skipped to get there. Root causes, not work-arounds (CLAUDE.md).

## Step 1 — measure before touching anything (Sonnet; ≤ 40 calls; one pytest process at a time)
`python3 -m pytest tests -q --ignore=tests/integration -p no:randomly --durations=80 --durations-min=0.5
-o cache_dir=/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/pytest_cache > …/scratchpad/durations_before.txt 2>&1`
(background, log file, never piped). Also time the collection alone (`--collect-only -q`) and the import of the heavy
modules (`python3 -X importtime -c "import trading.hod_break_engine, trading.orb_engine, scripts.eod_sections"` →
scratchpad). Write the top-80 table and the sum of the top 80 vs the total to the RESULT first.

## Step 2 — fix the classes of delay, largest first (expected, verify each by the durations table)
* **Real sleeps / waits**: `time.sleep`, `asyncio.sleep`, poll loops with wall-clock timeouts (`order_timeout_s`,
  `FC_LEG_POLL_S`, StopMonitor waits, `wait_for_sends`, retry back-offs in the Alpaca client). Patch the clock or the
  sleep in a conftest fixture (autouse, `monkeypatch.setattr(time, 'sleep', lambda s: None)` is NOT acceptable
  globally — do it per module/fixture so tests that assert on timing keep a fake clock), or inject the timeout as a
  parameter and pass 0 in tests.
* **Network**: any test that reaches alpaca/telegram/yfinance/EDGAR (grep `requests.`, `httpx`, `urlopen`, `yf.`,
  `TradingClient(`, `StockHistoricalDataClient(` in tests and in code paths tests exercise without a mock); block the
  network in conftest (a socket guard that raises) and mock what fails.
* **Heavy fixtures**: per-test SQLite schema creation, pandas parquet reads, config/YAML loads, logging handlers →
  session-/module-scoped fixtures, `tmp_path_factory`, copy-on-write of a prebuilt DB file.
* **Import time**: modules importing pandas/pyarrow/sklearn at import for a function the test never calls → lazy
  imports inside the function (production-safe).
* **Collection**: duplicate test discovery (tests importing other test modules), `conftest` doing work at import.
* Run with `-p xdist -n 2` ONLY as a measurement of the remaining parallelisable share; do not make xdist the fix —
  the owner's bar is about unnecessary delays, and the suite must still be green serially.

## Step 3 — verify
Same command as today, twice (cold and warm), wall time and `--durations=20` into the RESULT; then the suite exactly as
today's command once more: zero failures, same test count (5,470 ± new tests). No test removed or skipped; any test
whose assertion changed is listed with the before/after and why it is equivalent.

## RESULT `docs/test_suite_speed_spec_20261031_RESULT.md`
Before/after wall time, the top-20 before vs after, each fix with file:line and the class it belongs to, the remaining
top offenders if the target is not met and what they need. Return ≤ 150 words.

## Rules
No commit/stash, no service restart, no orders, no writes to data/*.db, config.yaml, orb.yaml, .env, crontab, logs/*;
never Read a file > 300 lines in full; one pytest process at a time (never two suites); production code changes only
for lazy imports and injectable timeouts, with the behaviour unchanged. This task IS the owner's request.
