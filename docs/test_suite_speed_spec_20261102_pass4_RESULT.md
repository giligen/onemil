# Test-suite speed RESULT, pass 4 (2026-11-02) - bar (<= 70 s) NOT met: 239 s single-process, 200 s with `-n 2`

Command (background, log in scratchpad, never piped): `python3 -m pytest tests -x -q --no-header --ignore=tests/integration -p no:randomly [-n 2] --durations=25`.
| Run | Wall (pytest / `time`) | Load before | Peak RSS | Result |
|---|---|---|---|---|
| pass-3 AFTER single (reference) | 283 s | 0.6 | 2.03 GB | 5,510 passed, 6 skipped |
| pass-4 mid-run single (items 2-4 only) | 252 s | 0.63 | 2.47 GB | 5,531 passed |
| **pass-4 final single** | **239 s** (4:04) | 1.10 (no other pytest; live trader running) | 2.47 GB | 5,531 passed, 6 skipped, 0 failed |
| **pass-4 final `-n 2`** | **200 s** (3:23) | 1.27 | 2.35 GB (largest process) | 5,531 passed, 6 skipped, 0 failed |
+21 tests = the new equivalence/cache/config tests. `pgrep -af pytest` showed no other pytest before every run. Saved: 44 s single, 23 s `-n 2` vs pass 3.

## Changes
1. Momentum (spec 1). Profile of `test_parity_with_build_a` (24 s for 3 targets, cumtime): parquet decode 5.1 s (1.7 each), `risk_adjusted_momentum` 12.9 s (6 calls: the test calls it and `eligible_universe` calls it again), `hygiene_events` 3.9 s, sort_values 3.4 s. Fixes: (a) `tests/real_panel_cache.py::real_panel_window` - the decoded pyarrow Table per (lo, hi) is cached per process (compact, ~100 MB/window), each caller gets its own pandas frame with float32 closes; used by the 2 parity tests + the 2 equivalence files; (c) `memoised_ramom(reb)` context manager in the 6 parity tests serves the second `risk_adjusted_momentum` call (inside `eligible_universe`) from the first; window/assertions unchanged; the equivalence files do NOT use it. (b) no production change: the profile is flat (sort/factorize of 3.7M string rows). Parity 6 tests 43 s -> 28 s. Tests: `test_real_panel_cache_20261102.py` (cached == inline read, mutation isolated).
2. `eod_report` (spec 2): the 20 s was `journalctl -g` over the real verbose journal (14 s of 20, cProfile). It is the only subprocess test; it stays a real-script smoke (real DB/checkers) with a stub `journalctl` on PATH (empty journal): 20 s -> 4 s. The journal query itself stays covered by `test_journal_error_count_*` and the live EOD run.
3. `Database._migrate` (spec 3): fresh `Database()` costs 17 ms (measured), 2.5 ms when the schema file is pre-copied. `tests/conftest.py`: session autouse fixture builds the template once with the REAL constructor and copies it to every single-path `Database(path)` whose file does not exist (split cache/trades paths, existing files, ':memory:' untouched); the real `__init__` still runs. `tests/test_database_template_equivalence_20261102.py` (4 tests): schema dump + pragmas identical to the real constructor, existing file never overwritten, split not templated, migration still adds a dropped column.
4. `os.environ` 284k reads (spec 4): found with an `__iter__` tracer: `dotenv.main` interpolation, called from `Config.__init__` -> `load_dotenv`, copies the environment once per .env variable (4,384 per 143 tests). PRODUCTION change `config.py::Config._load_env_file`: `load_dotenv(path, interpolate="$" in file_text)` - identical result (interpolation can only change a value containing `$`), files with `$` keep the full path. `tests/test_config_env_loading_20261102.py` (3 tests: applied, existing env wins, `${}` still interpolates, interpolate flag false/true).
5. Sleeps >= 1 s (spec 5) not touched (all are test-paced waits already <= 5 s; listed below).

Production files changed: `config.py` only (loader helper). Tests/conftest otherwise. No commit, no restart, no writes to data/*.db, config.yaml, orb.yaml, .env, crontab, logs.

## Left, ranked (seconds in the 239 s run)
1. Hygiene reference tests ~31 s: the verbatim OLD per-symbol-lambda `_guard_frame_reference` (16.2 s full 2026-06-29 + 8.3 + 7.1 sampled) - the cost IS the reference; cut only by caching the reference per window across sessions on disk (owner call: a golden file replaces a live reference).
2. Parity tests 28 s + ramom equivalence ~14 s (eligible_universe sorts 3.7M string rows: factorised codes + lexsort in `trading/momentum_sleeve.py`, est. -30 %, needs its own equivalence).
3. ~5,000 small tests floor ~110 s: pytest/pluggy + mock construction + deepcopy of the YAML cache (0.7 s/143 tests) + gc.collect; only xdist helps (`-n 2` 1.2x; node has 2 vCPUs shared with the live trader).
4. Mid sleeps ~25 s: test_trading_integration orphan 5.0, orb_morning_e2e 4.4, stop_monitor BracketSLRace 3.0, preplace lock-wait 2.0+1.5, alpaca_client timeout 2.0, scanner drain 1.5 x2, entry-drain 1.25, stop_exit_ladder 1.1 - need injectable intervals with unchanged defaults, each a separate small change.
The 70 s bar needs xdist with >= 4 workers (not available on 2 vCPUs) or deleting the real-panel/reference tests; not reachable by this node's CPU alone.
