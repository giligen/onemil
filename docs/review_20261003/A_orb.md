# Track A review — ORB paper (reviewer: independent, 2026-10-03)

Scope read: trading/orb_engine.py (universe_build_due, build_universe, _get_snapshots_warm, _fetch_today_open_bars,
build_orb_universe_from_snapshots, rvol tilt hooks), trading/orb_gap_gate.py, scanner/realtime_scanner.py `_orb_tick`,
alpaca_client.get_1min_bars_range_multi, orb_add_on.py, orb_rvol_tilt.py, orb.yaml. Tests not run (`python` absent from PATH;
use the venv interpreter). No code edited.

## Findings (ranked)

### 1. [loses the day, medium likelihood] Open-tick budget is not enforced on the REST open fetch
trading/orb_engine.py:1226-1241 + data_sources/alpaca_client.py:1262 (no per-call timeout): `_fetch_today_open_bars` checks
`deadline` only BETWEEN 200-symbol chunks; each chunk uses `_call_with_timeout` with the DEFAULT 90 s timeout and 1 retry
(alpaca_client.py:50,62). One hung SIP request = up to 180 s inside `build_universe`, far over `open_tick_budget_sec` (20 s) and
over the 50 s ENGINE_TICK_TIMEOUT_S (realtime_scanner.py:~558). Worse, `_orb_tick` runs build_universe BEFORE `check_entries()` and
`check_exits()` (realtime_scanner.py:795-801), so a stalled build also starves exits for that tick; the timed-out future keeps
running and the next tick starts a second build_universe concurrently on the same engine (race on `_open_bar_cache`, `_gap_gate_inputs`).
Trigger: any data-API slowness around 09:31-09:40 (Friday had 279 TIMEOUTs from a related cause).
Fix: pass `timeout=min(8, remaining budget), timeout_retries=0` (as NEWS_API_TIMEOUT does) for this call; and wrap
build_universe in try/except in `_orb_tick` (or run check_exits first) so exits never depend on the seed build.

### 2. [wrong vs BT, silent] REST open fetch can adopt the 09:31 bar as "the 09:30 open"
orb_engine.py:1240-1255: window s0=09:30, s1=09:31 (end inclusive on Alpaca), then `df.iloc[0]['open']` is cached for the day
without checking the bar timestamp. If the symbol had no 09:30 print (common for thin $3-30 gappers) the 09:31 bar's open is used
as the official open and cached permanently (`_open_bar_cache`, never revalidated). The DB path does check this
(`_bar_open_at_0930`, :1330) — the REST path does not. Also a fresh-snapshot symbol uses the daily-bar open (parity doc shows
RUM/NAIL/CDE-type disagreements of 0.3-1 pt of gap, 1-3 names/day) — known, but those flip prod<->P1 classification.
Fix: require `df.iloc[0].timestamp == 09:30 ET` (reuse `_bar_open_at_0930`), else treat as miss.

### 3. [silent failure] Per-symbol except in the admission loop logs at DEBUG
orb_engine.py:~1437 `except Exception as e: logger.debug(...snapshot parse failed...); continue`. Any systematic bug in the
newly added gap-input branch (e.g. a missing key, dtype change in `get_snapshots`) yields an EMPTY universe with no WARNING
and no ERROR — the exact class of failure this project's rules forbid. Fix: count exceptions, log one aggregated ERROR
(count + first exception) per build.

### 4. [risk / sizing, not a crash] Stacked risk amplifiers on one paper account, all changed 10/1-10/2
orb.yaml 330-361 + orb_add_on.py: add_on `stop_mode: original`, units 1.0, at +1R (R = OPENING RANGE size, not entry-stop
distance): the add doubles the share count at a price 1 range-unit above entry while the stop stays at the original level, so
stop-out after an add loses ~base(1R) + add(~2R) ≈ 3R. It sits OUTSIDE the Q5 1.5x total cap (the cap lives in the planner, the
add is sized at fire time, orb_engine.py:2364). RVOL tilt low tercile 1.5x x add-on x scale_out multiplies further (worst case
~4.5R vs the $375 nominal). Also three mechanics changes + add-on pool P1 live at once (tilt, add-on, pool P1 with real orders),
which defeats the stated "ONE mechanics change per session" attribution; the tilt was shipped despite failing its own ordering
robustness clause (docstring of orb_rvol_tilt.py). Paper, so bounded, but fills cannot be attributed. Fix: log combined-position
worst-case $ risk at add time and cap it at the Q5 1.5x * risk_per_trade; or run add-on one week later.

### 5. [minor] RVOL tilt fail-open is silent
orb_engine.py:4444-4478: unknown rel_volume (missing `volume_20d`, adv20<=0) returns mult 1.0 / tercile None and
`_log_rvol_tilt` returns without any log. A day where daily_stats is empty would silently run untilted and look like "mid
tercile" in attribution. Fix: WARNING once per symbol when tilt is enabled and rvol is None. Tercile edges use strict `<`
(low if rvol < 2.898) — check the BT convention (`1693_pool_exits.py::tercile_edges`) uses the same side at ties (negligible).

### 6. [minor] `bar_date` missing treated as stale (orb_engine.py:1396-1405)
`_fresh` requires `bar_date == today`; a snapshot lacking `daily_bar_date` (fail-open elsewhere, :1346) would take the stale
branch and use `close` as prev_close (today's close, wrong gap). Only with an older client; log if the key is absent.

## Verified OK
- `universe_build_due()` (orb_engine.py:1007): plumbing only, after 10:00 ET skips seed/gap gate; `check_entries` cutoff
  (:3219) is the same `_past_last_entry_time`; exits/force-close still run after the cutoff (realtime_scanner.py:799-803).
  One INFO per day. Friday's 210K-WARNING loop is closed.
- `gap_input_needs_today_open` and `resolve_gap_input` logic are consistent with the spec (never gate on a prior day's open;
  prev_close = stale bar's close). `_qty_num` handles fractional qty. max_spread_bps 300 (verdict doc ok, not below 150),
  skip_q1 true, catalyst veto off per config, Q5 cap clamp is in the planner (not touched).
- No refill after veto: nothing in the paths read refills the slot.
