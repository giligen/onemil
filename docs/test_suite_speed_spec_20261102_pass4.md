# Test-suite speed, pass 4 (Fable, 2026-11-02) — the ranked leftovers from pass 3

Pass 3 (`docs/test_suite_speed_spec_20261102_pass3_RESULT.md`, 38c511d): 283 s single-process, 223 s `-n 2`; owner's bar
≤ 70 s. Pass 3's own ranking of what is left, in order; each item = the fix approved here. Same rules as pass 3
(nothing deleted/skipped/weakened; equivalence or unchanged-default evidence for every production change).

1. **Real-panel momentum tests ~85 s** (`tests/test_momentum_sleeve.py` parity tests + the two equivalence files).
   Profile ONE of them (cProfile, cumtime) and attribute: parquet load / `risk_adjusted_momentum` / `hygiene_events` /
   the reference implementations / assertions. Fixes in order: (a) the panel parquet loads once per session
   (session-scoped fixture, read-only; tests that mutate get a copy); (b) any remaining per-group Python in
   `trading/momentum_sleeve.py` or `scripts/momentum_sleeve.py` on the hot path → vectorise under an equivalence test
   (same recipe); (c) the reference implementations run once per session per window (cache in a session fixture).
   A parity test's WINDOW and assertions never change.
2. **`eod_report` subprocess 20 s** (`tests/test_eod_report*.py`): find what the subprocess does for 20 s (imports?
   DB? network?). If it is a `python3 scripts/eod_report.py --no-send` run, keep ONE subprocess smoke test and make the
   others call the module's functions in-process with the same arguments and assertions.
3. **`Database._migrate` per fresh DB**: build the schema once per session into a template file, `shutil.copy` it per
   test (`tmp_path`) instead of re-running migrations; a test that asserts on migration behaviour keeps the real path.
   Count how many tests construct a Database and the seconds saved.
4. **284k `os.environ` reads in 143 tests**: find the reader (profile shows the frame); a hot loop reading
   `os.environ` on every call is a production smell too — read once at init / module level with an explicit
   refresh hook if tests need it, unchanged behaviour, and a test that the value is honoured.
5. **Mid-size sleeps**: everything still ≥ 1 s in `--durations=0` after 1–4; injectable with unchanged defaults.
6. Re-measure: cold + warm, single and `-n 2`, `--durations=20`, peak RSS, counts. **Check `uptime` load and
   `pgrep -af pytest` before each timing run** — on 11/2 another project's pytest on the same 2 vCPUs doubled the
   "before"; if the node is contended, say so and re-run when it is quiet.

RESULT `docs/test_suite_speed_spec_20261102_pass4_RESULT.md`, ranked leftovers with seconds. Return ≤ 150 words.
Rules: no commit/stash/restart/orders; no writes to data/*.db, config.yaml, orb.yaml, .env, crontab, logs/*; one pytest
process at a time, background with a scratchpad log, never piped, `--ignore=tests/integration`; never Read > 300
lines. This task IS the owner's request.
