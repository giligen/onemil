# Fix spec — ORB paper engine, from the independent review (docs/review_20261003/A_orb.md), 2026-10-03

Scope: `trading/orb_engine.py`, `scanner/realtime_scanner.py`, `data_sources/alpaca_client.py` (only if a timeout
parameter is needed), tests. No change to selection, gap gate thresholds, spread gate, tilt edges, add-on rule, caps.

## A1 — the 09:30-bar fetch must respect the open-tick budget INSIDE a chunk (`_fetch_today_open_bars`, orb_engine.py ~1226–1255)
Today the deadline is checked only between 200-symbol chunks; one chunk call can take the client's default 90 s timeout
plus one retry. Required: compute `remaining = deadline - time.time()` before each chunk; if `remaining < 5 s` stop (the
existing aggregated WARNING); call the multi-bar fetch with a per-call timeout of `min(remaining - 1, 20 s)` and NO
retry on this path (add an optional `timeout_s`/`retries` parameter to `get_1min_bars_range_multi` if it lacks one; the
default behaviour for every other caller must not change). A chunk that times out marks its symbols as misses (60 s
retry) and counts in the WARNING.

## A2 — take only the bar whose timestamp IS 09:30 ET (orb_engine.py ~1248–1252)
`df.iloc[0]['open']` is used without checking the bar time; with an inclusive end a symbol lacking a 09:30 print gets the
09:31 open and it is cached all day (the DB path already checks via `_bar_open_at_0930`). Required: select the row whose
timestamp equals `s0` (handle tz-aware/naive index), else treat as a miss. Unit test with a frame holding only a 09:31 row.

## A3 — exits before the universe build in the tick (realtime_scanner.py `_orb_tick` ~775–800)
`build_universe(...)` runs before `check_exits()`; a slow build starves exits. Required: `check_exits()` first, then the
build when due. Document in the docstring why. Test: a build that raises/sleeps must not prevent `check_exits` being called.

## A4 — no silent failure in admission or tilt
Admission loop `except` logs at DEBUG (~1437): aggregate per build into ONE WARNING `ORB admission: n symbol(s) skipped on
exception (first: …)`; if EVERY candidate failed, log ERROR once. RVOL tilt fail-open (~4444): keep the fail-open (multiplier
1.0, BT default) but log WARNING with the reason once per day. Tests for both.

## A5 — second build must not overlap the first (review note: the 50 s tick timeout lets a second build start)
If a build is in progress (flag set in `build_universe`, cleared in `finally`), a new tick skips the build with one INFO.
Test: re-entrancy guard.

## Done means
`python3 -m pytest -q tests/test_orb_gap_input_parity_20261002.py tests/test_orb*.py -x --ignore=tests/integration` green
(name the counts), then the replay parity check on Friday:
`cd /home/ec2-user/onemil && bash scripts/research_run.sh -m 2500M python3 scripts/orb_open_tick_replay.py --date 2026-10-02 --parity`
(read the script's --help first for the exact flags) must still report 128/128 admissions vs the BT. Write
`docs/review_20261003/FIX_A_result.md` (≤ 40 lines: per item what changed, tests, the replay line). Never start the
service, never submit orders, never git, never touch config/.env/data.
