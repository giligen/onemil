# Test-suite speed RESULT, pass 3 (2026-11-02) - bar (<= 70 s) NOT met: 283 s single-process, 223 s with `-n 2`

Daily command measured (log in scratchpad, never piped): `timeout 2700 python3 -m pytest tests -x -q --no-header --ignore=tests/integration -p no:randomly [-n 2] --durations=20`.

## Wall, peak RSS, counts
| Run | Wall | Peak RSS (whole tree, `/usr/bin/time -v`) | Result |
|---|---|---|---|
| BEFORE, single | 509 s (8:30; **contended**: a second project's pytest, `micha-signals`, ran on the node and my own `--collect-only` overlapped the first minute) | 1.97 GB | 5,498 passed, 6 skipped, 0 failed |
| AFTER, single (node quiet, load 0.6) | **283 s** (4:48 wall) | 2.03 GB | 5,510 passed, 6 skipped, 0 failed |
| AFTER, `-n 2` | **223 s** (3:45 wall) | 1.63 GB (time -v reports the largest process, not the sum; two workers about 2 x 1.6 GB is the bound, under 3 GB measured by `free` at start/end only) | 5,510 passed, 6 skipped, 0 failed, identical set |
The before number is not comparable to pass 2's 258 s (that run was on a quiet node). Honest read: pass 3 so far saves about 40-60 s of the 283 s on a quiet node (momentum 54 -> ~34 s incl. the new equivalence tests, hygiene reference 43 -> ~32 s, drain 5.5 -> ~0.5 s, emergency-close 3 s); NOT the 4x the owner asked for. `-n 2` gives only 1.27x because the six momentum parity tests + two full-window equivalence tests (~85 s of CPU) are single long tests and the node has 2 vCPUs shared with the live trader.
xdist installed (`pytest-xdist 3.8.0`, `pip3 install --user`); `requirements.txt` is the only requirements file and has no dev section - NOT edited (add `pytest-xdist>=3.5` under a "dev" comment when Fable adopts `-n 2`). No test failed under xdist (no shared-state bug found). Daily command is NOT changed (pytest.ini untouched).

## Production changes (all with evidence)
1. `trading/momentum_sleeve.py::_grouped_pct_change` (new) replaces `groupby('symbol')['close'].pct_change()` in `risk_adjusted_momentum` (numpy shift masked at symbol change + groupby.ffill to keep pandas' default ffill semantics; float32 stays float32). Equivalence: `tests/test_risk_adjusted_momentum_equivalence_20261102.py` (12 tests) keeps the OLD function verbatim as `_risk_adjusted_momentum_reference`; `assert_frame_equal(check_like=False)` on: full 2026-09-21 real window, sampled 2021-02-08 / 2024-06-03 / 2026-06-29 windows, float32+float64, one-bar symbol, duplicate rows, NaN closes (leading, run of two, on the as-of bar), zero close, 4:1 split, history gaps, categorical + unsorted + duplicate-index panels, empty-after-asof (ERROR log + empty frame). The 6 parity tests are untouched and pass. Per call ~4 s -> ~3 s (the rest is sort/factorize of 3.7M string rows: flat profile, no hot spot left).
2. `trading/trading_engine.py`: `_CLOSE_FILL_POLL_S = 0.5` class attribute replaces the hard-coded `time_mod.sleep(0.5)` in `_emergency_close_position` (production default unchanged); autouse `_fast_emergency_close_poll` in `tests/conftest.py` sets 0.0 (profile: 6 calls = 3.0 s in test_trading_engine.py alone).
3. Tests only: `tests/test_hygiene_events_equivalence_20261031.py` computes the slow OLD `_guard_frame_reference` once per window instead of twice (every assertion and window kept); `tests/test_hod_break_replay.py::test_drain_thread_evaluates_without_the_scan_cycle` sets `HodBreakEngine.SETTLE_S` (existing class attribute, default 0.4) to 0.02 and paces 0.06 s per bar instead of 0.5 s.

## Step-4 attribution (tests/test_trading_engine.py, 143 tests, cProfile, 16.0 s profiled / 10.0 s plain)
(a) `--collect-only` 2.2 s (import cost, once per process; not the problem). (c) `-q` 12.3 s vs default `-v` 11.0 s: no gain from `-q`. (d) logging 0.05 s, no log file written by tests.
| Category | s of 16.0 |
|---|---|
| `time.sleep` (all `_emergency_close_position`, now fixed) | 3.0 |
| sqlite `execute` (2.2 s of it = `Database._migrate`, 18 statements x 512 Database() creations, ~4 ms per fresh DB) + executescript 0.25 | 2.6 |
| `copy.deepcopy` (session YAML cache copy-on-read + mock internals) | 0.7 |
| `unittest.mock` construction (`_mock_set_magics` 3,600 calls) | 0.7 |
| `gc.collect` (5 calls) | 0.4 |
| `os.environ.__getitem__` (284k calls) | 0.8 |
| pytest + pluggy internals | 1.0 |
| autouse fixtures in conftest (`_isolated_*`, poll shrinkers) | ~0.0 |
So the autouse fixtures and YAML cache are NOT hot; the per-test ~14 ms floor is sqlite DB creation + mock construction + `os.environ` lookups (what reads env 284k times?).

## What is left, ranked (seconds in the 283 s run)
1. tests/test_eod_report.py::test_script_runs_end_to_end_no_send_no_llm 20 s (50 s contended): a real `scripts/eod_report.py` subprocess against the real DB, journalctl and checkers. It is the only subprocess of its kind; make the report's journal queries injectable (or mark it opt-in with the integration tests) = owner/Fable decision.
2. Momentum real-panel work ~85 s: 6 parity tests (~50 s, each reads a 3.7M-row parquet window 2.5 s + 2 x `risk_adjusted_momentum` 3 s + `eligible_universe` ~5 s) + 2 full-window equivalence tests (16 s + 9.5 s) + 3 sampled hygiene/ramom windows (~20 s). Next step: a session-scoped cache of the 5 parquet windows (~150 MB each; careful under `-n 2`) and `eligible_universe` (4.9 s, calls `risk_adjusted_momentum` again plus `hygiene_events`) sharing one feature computation; sort via factorized symbol codes + `np.lexsort` instead of `sort_values` on object strings (est. -50 %).
3. `Database._migrate` on every fresh DB (~4 ms x thousands of Database() in the suite): build the schema once into a template file/`sqlite3.Connection.backup` per test, or skip migration statements already satisfied by the base schema. Candidate 10-20 s suite-wide; needs its own equivalence test (schema dump identical).
4. `os.environ` reads (284k in 143 tests): find the caller (likely `Config`/`load_dotenv` per construction) - candidate 1-2 s per 143 tests = ~30 s suite-wide if it generalises.
5. Mid-size pacing still >= 1 s: test_trading_integration orphan 5.0 s, test_orb_morning_e2e 4.3 s, test_stop_monitor BracketSLRace 3 s, test_hod_failure_signal boot 2.9 s (model load), orb_preplace lock-wait 2.0 + 1.5 s, alpaca_client timeout 2.0 s, scanner drain tests 1.5-1.8 s; other `time_mod.sleep(0.5)` sites in trading_engine.py (lines ~1699, 2410, 2473, 2581, 4635) and stop_monitor.py:3161, orb_engine.py:7103, hod_break_engine.py:2405 were not hit by the profile of this file but are the same idiom.
6. The ~5,000 sub-50 ms tests: ~120-130 s total; item 3 and 4 are the only shared costs found.
Reaching 70 s needs 1 + 2 + 3 + 4 (est. -100 s) AND `-n 2` (est. 1.6x on the balanced remainder); one more pass, not achievable by sampling or deleting tests.

## Process note
The first `--collect-only` was started while the baseline run was in progress (one-pytest-at-a-time rule slip, 2 s of overlap); the baseline's 509 s was mostly the other project's pytest competing for the 2 vCPUs.
