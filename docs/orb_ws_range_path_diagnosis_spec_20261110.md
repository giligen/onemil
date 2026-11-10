# Why does the websocket bar path never complete an ORB opening range? — diagnosis spec 2026-11-10

READ-ONLY diagnosis (no code changes except a failing test that reproduces the defect, if you find one). Budget ≤ 40
calls. No git, no service action, no orders, no writes outside `docs/orb_ws_range_path_diagnosis_20261110_RESULT.md`
and `tests/test_orb_ws_range_path_20261110.py`. pytest only on that new file, `--ignore=tests/integration`. Never
Read a file > 300 lines in full — grep -n, offset/limit. This task IS the owner's request; do not pivot.

## Facts
- ORB opening range = 5 one-minute bars 09:30–09:34 ET; `ORBEngine._ingest_bars` (trading/orb_engine.py ~2091)
  completes it when `_first_session_open_ts_utc(bars_df)` finds a bar at exactly 14:30Z (EST) / 13:30Z (EDT) in the
  frame it is handed AND ≥ `range_minutes` bars fall in [open, open+5 min). It logs "ORB: {sym} range complete — …".
- Two sources feed `_ingest_bars`: (a) the websocket: `StopMonitor.register_bar_handler('orb', self._on_bar_close)`
  (~1939), `_on_bar_close` (~1945) queues events, `drain_bar_events` (~2062) ingests them (scanner calls it at
  realtime_scanner.py:682/1679; a drain thread at ~2007); (b) the REST post-open sweep `_post_open_sweep` (~1726) at
  09:35:00 ET via `alpaca.get_1min_bars_multi`, which logs "(n applied, m superseded by the stream)".
- Journal evidence since 2026-10-01 (every session day): **m superseded is ALWAYS 0** and every "range complete" line
  of every day is stamped inside the sweep's apply window (e.g. 11/9: all 148 at 14:35:08–14:35:15 UTC). On 11/10 the
  sweep applied 0/59 (vendor bars late) and **no range completed all day** although the scanner's websocket subscribed
  bars for the 2,368-symbol seed at 12:30 UTC and HOD's stream reconcile reported "0 had missing bars". The REST bars
  re-fetched at 15:00 UTC complete the range for NVD/BEZ (5/5, open ts found) → the range logic itself is fine.
- Hypotheses to test, in order: (1) the frame `_on_bar_close` receives (StopMonitor's per-symbol bar buffer) does not
  contain the 14:30Z bar — buffer starts at subscribe-or-first-bar, is per-watch (positions/quote-watch only), or is
  trimmed; (2) `_on_bar_close` is only registered/fired for symbols the StopMonitor is "watching" (positions), not
  for the universe `bars:` subscription (which lives in the scanner's websocket, not StopMonitor's); (3) the object-
  dtype timestamp path returns None; (4) the drain only runs after the sweep has marked `range_data` for the day or
  is gated by `_post_open_range_sweep_done`; (5) candidates are admitted (gap gate) AFTER the 09:30 bar and the WS
  window handed over does not include history.

## Method
1. Trace the WS path end to end with file:line: which websocket (scanner's `alpaca.data.live` subscription vs
   StopMonitor's) delivers bars to which handler, what frame shape/length reaches `_ingest_bars`, and whether the
   universe symbols (not only positions) ever reach `_on_bar_close`. Quote the code.
2. Check the journal for one day (11/9, 14:29–14:40 UTC) for the drain's own logging (`drain_bar_events`, "bar events",
   "ingest", per-symbol DEBUG) — bounded `journalctl -u onemil-trader --since "2026-11-09 14:29" --until
   "2026-11-09 14:40" --no-pager -o cat | grep -E "<pattern>"`; never without --since/--until.
3. Write ONE failing test (unit, mocked StopMonitor/scanner with `spec=`) that feeds the engine the 5 WS bars the way
   production does and asserts the range completes — it should FAIL for the reason you found (if production's shape
   makes it pass, say so and show what differs live).
4. Rank the hypotheses with the evidence for/against each; name the root cause with confidence; propose the minimal
   fix (file:line, ≤ 10 lines of pseudo-code) and the test that proves it. Do NOT implement the fix.

## Output
`docs/orb_ws_range_path_diagnosis_20261110_RESULT.md` ≤ 350 words + the trace table. Return ≤ 150 words: root cause,
confidence, the failing test's name, the proposed fix.
