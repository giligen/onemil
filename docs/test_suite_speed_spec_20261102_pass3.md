# Test-suite speed, pass 3 (Fable, 2026-11-02) — owner: "should be less than a quarter of what it is now"

Pass 1 649 → 371 s, pass 2 → 286 s (`docs/test_suite_speed_spec_20261031_pass2_RESULT.md`; integration tests are
now opt-in, so the daily command starts from ≈ 258 s). Owner's bar: ≤ 70 s wall for the daily command
`timeout 2700 python3 -m pytest tests -x -q --ignore=tests/integration -p no:randomly` on this 2-vCPU node.
Nothing is deleted, skipped, weakened or sampled away; same test count + the new equivalence tests.

## Known remaining cost (pass 2 ranking) and the approved fixes
1. `tests/test_momentum_sleeve.py` 54 s — `risk_adjusted_momentum` (`trading/momentum_sleeve.py:61`, groupby.pct_change
   over 63,666 groups). APPROVED production change under the pass-2 recipe: keep the old function verbatim as
   `_risk_adjusted_momentum_reference` in `tests/test_risk_adjusted_momentum_equivalence_20261102.py`, vectorise the
   production one (sorted frame, `np.concatenate`/shift-by-group masks, no per-group Python), `assert_frame_equal`
   (`check_like=False`, exact dtypes) on the real-panel windows the parity tests use + the same 4 edge panels as the
   hygiene test (one-bar symbol, duplicate rows, NaN closes, split day) + a panel where a symbol has gaps. The 6 parity
   tests stay untouched and must pass. Any other groupby-with-Python-lambda in that module found by the profile: same
   treatment, same equivalence test.
2. `tests/test_hygiene_events_equivalence_20261031.py` 43 s — the OLD reference is the slow part. Keep every assertion
   and every window: compute the reference ONCE per session (session-scoped fixture returning a dict window → frame;
   the test functions compare against it). Not sampling, not fewer windows.
3. Mid-size pacing: `test_stop_exit_limit_buffer` 7 s, `test_trading_integration` 5.7 s, `test_hod_break_replay` drain
   5.5 s (0.5 s pacing), `test_orb_morning_e2e` 4.6 s, `test_eod_report` subprocess 3.7 s, and anything else ≥ 1 s in
   `--durations=0`. Each: find the sleep/poll/retry/subprocess, make it injectable with an UNCHANGED production default
   (class attribute or config key, as pass 1 did for `fc_leg_poll_s`), shrink it in the test via the existing autouse
   poll-shrink fixture. A subprocess test may stay a subprocess if it is the only one of its kind; say so.
4. The ~5,000 sub-50 ms tests ≈ 70 s = the floor if their ~14 ms each is really intrinsic. Pass 2 did not prove that.
   Measure it: (a) `python3 -m pytest tests/test_trading_engine.py -q -p no:randomly --collect-only` wall (import cost);
   (b) cProfile of a whole mid-size file sorted by cumtime, attributed to pytest internals vs `tests/conftest.py`
   autouse fixtures (`_isolated_guardrail_state`, `_isolated_hod_state`, `_isolated_orb_state`, the poll-shrink
   fixture, the session YAML cache's `copy.deepcopy` per `safe_load` call) vs the tests' own mock construction;
   (c) `addopts = -v --tb=short` in pytest.ini — measure `-q` vs `-v`; (d) logging: check whether every test run
   installs the loguru/Telegram handler or writes a log file per test (`logs/` must not be written by tests at all).
   Fix what is shared: a `deepcopy` of the 25 KB parsed config per call is a candidate (return a frozen structure
   and deepcopy only when a test mutates? — only if the copy-on-read guarantee from pass 2 is preserved, state how).
5. Parallelism: install `pytest-xdist` (`pip3 install --user pytest-xdist`, add it to the dev requirements file the
   repo uses — find it; if none, `requirements-dev.txt` with a comment). Measure the daily command with `-n 2`
   (2 vCPUs) AFTER steps 1–4, with `/usr/bin/time -v` for peak RSS of the whole process tree (the node has 8 GB + 4 GB
   swap; the run must stay under 3 GB total). Any test that fails only under xdist is a shared-state bug: fix it (tmp
   paths, env vars, module globals), never mark it serial without saying why.

## Deliverables
`docs/test_suite_speed_spec_20261102_pass3_RESULT.md`: cold + warm wall for the daily command single-process AND
with `-n 2`, `--durations=20`, the step-4 attribution table (seconds by category), peak RSS, test counts (passed /
skipped / failed = 0), the list of production changes with their equivalence or injectable-default evidence, and a
ranked list of what is left with seconds. If `-n 2` passes identically and meets the bar, the RESULT states the new
daily command verbatim (it is adopted by Fable, not by the agent — do not edit pytest.ini addopts to add `-n`).

## Rules
No commit/stash, no restart, no orders, no writes to data/*.db, config.yaml, orb.yaml, .env, crontab, logs/*; one pytest
process at a time (xdist workers inside one command are fine), suite runs in the background with a log file in the
scratchpad, never piped; never Read > 300 lines at once; `--ignore=tests/integration` always. Production changes only
as listed (vectorisation under equivalence tests, injectable timeouts with unchanged defaults, shared-fixture cost).
This task IS the owner's request. Return ≤ 150 words.
