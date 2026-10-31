# Test-suite speed, pass 2 (Fable, 2026-10-31) — from 371 s to ≤ 180 s

Pass 1 (`docs/test_suite_speed_spec_20261031_RESULT.md`, commit 8feb581): 649 s → 371 s. Remaining offenders named
there: (a) `tests/test_momentum_sleeve.py` 6 real-panel parity tests = 103 s, cause `hygiene_events` in
`trading/momentum_sleeve.py` (per-symbol groupby + Python lambda); (b) fixture setup 99 s in total; (c) thousands of
20–40 ms tests (`test_trading_engine.py` 25 s, `test_scanner.py` 17 s).

## Owner-level decision (Fable, acting as owner): the production change in (a) is APPROVED under an equivalence test
1. Vectorise `hygiene_events` (pandas group ops / numpy, no per-group Python lambda). Equivalence test FIRST:
   `tests/test_hygiene_events_equivalence_20261031.py` runs the OLD implementation (keep it as
   `_hygiene_events_reference` in the test module, copied verbatim) and the new one on the real-panel fixture the six
   parity tests use AND on three synthetic edge panels (a symbol with one bar; duplicate (symbol, date) rows; NaN
   closes; a split-day) — identical rows, order, dtypes (`pd.testing.assert_frame_equal`, `check_like=False`). The
   six parity tests stay untouched and must still pass.
2. Fixtures: find the 99 s with `--setup-show`/`--durations` on setup phases; session-scope read-only fixtures
   (panels, parquet, YAML, DB schemas) with copy-on-write for tests that mutate (`tmp_path` copy of a prebuilt DB
   file instead of re-running the schema). A fixture that a test mutates in place must NOT be widened — list each
   fixture changed with its scope before/after and why it is safe.
3. The 20–40 ms class: profile one representative (`python3 -m cProfile -s cumtime -m pytest tests/test_scanner.py
   -q -p no:randomly` → scratchpad); if the cost is a shared import/setup (logging handler install, config load, DB
   connect) fix it once at session scope; if it is intrinsic, say so and stop.
4. Verify exactly as pass 1: the daily command cold and warm, `--durations=20` into the RESULT, zero failures, same
   test count (+ the new equivalence tests). Target ≤ 180 s; if not met, the RESULT ranks what is left with numbers.

## RESULT `docs/test_suite_speed_spec_20261031_pass2_RESULT.md`. Return ≤ 150 words.
Rules as pass 1: no commit/stash, no restart, no orders, no writes to data/*.db, config.yaml, orb.yaml, .env, crontab,
logs/*; one pytest process at a time, suite runs in the background with a log file, never piped; never Read > 300
lines at once; production changes only (a) and lazy imports / injectable timeouts with unchanged defaults. This task
IS the owner's request.
