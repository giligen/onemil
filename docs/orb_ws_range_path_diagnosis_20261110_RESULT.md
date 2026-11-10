# WS range path diagnosis — RESULT 2026-11-10

Spec: `docs/orb_ws_range_path_diagnosis_spec_20261110.md`. Read-only; no code changed except the reproduction test.

## Root cause (confidence: high — reproduced with the real StopMonitor)
ORB's per-symbol bar window in StopMonitor **starts at ORB's subscribe time**, and ORB subscribes only when the gap
gate admits a candidate, which by construction is **after the 09:30 bar** (the gate needs today's open: 11/9 admit
14:31:54Z, 11/10 14:33:40Z / 14:34:32Z). The 14:30Z bar was on the wire (HOD bulk-streams the seed from 12:30Z), but
bulk symbols get **no window**, and the bulk→windowed conversion starts an **empty** list. Every frame `_ingest_bars`
ever sees starts ≥ 14:31Z → `_first_session_open_ts_utc` = None all day. The WS path has never completed a range;
"0 superseded" is structural, and the REST sweep is the only source. **`execution.fast_submit` (WS-only) must stay
false** — it would produce zero ranges.

## Trace
| Step | file:line | Behaviour |
|---|---|---|
| HOD seed subscribe 12:30Z | hod_break_engine.py:523 → stop_monitor.py:1015-1017 | `_bulk_bar_symbols.add(s)` — "NO rolling window" |
| 09:30 bar arrives | stop_monitor.py:1142-1144, 1182-1183 | bulk → not appended; window handlers skipped |
| ORB `_on_bar_close` | orb_engine.py:2044 | `symbol not in self.universe` → dropped anyway (universe empty) |
| Gap gate admits ≥ 09:31 | orb_engine.py:1152-1154 → 1616 | `subscribe_bars(sym)` |
| bulk → windowed | stop_monitor.py:999-1002 | `discard(bulk)`; `_bar_windows.setdefault(symbol, [])` → empty |
| later bars | stop_monitor.py:1186-1189 → orb_engine.py:2047 → 2175 → 2240 | frame starts 14:31Z → open ts None → `return` |

## Hypotheses
1. **CONFIRMED** — window lacks 14:30Z (starts at subscribe, not trimmed).
2. Rejected — windowed universe symbols do reach `_on_bar_close` (control test).
3. Rejected — tz-aware dicts → datetime64[UTC]; control passes.
4. Rejected — drain is not gated by the sweep flag.
5. **CONFIRMED (the trigger)** — admission after the open bar; same mechanism as 1.

## Failing test
`tests/test_orb_ws_range_path_20261110.py::test_ws_range_completes_when_candidate_admitted_after_the_open_bar`
FAILS ("window … starts at 14:31:00Z"); `test_control_ws_range_completes_when_window_holds_the_open_bar` PASSES.

## Proposed minimal fix (not implemented)
```
# stop_monitor.py _on_bar (~1143): keep a session head for bulk symbols
if not windowed:
    head = self._bulk_head.setdefault(symbol, [])
    if len(head) < 15: head.append(bar_dict)           # ~15 dicts x 2.4K syms, reset daily
# stop_monitor.py subscribe_bars (~999): seed the window on bulk→windowed
if symbol in self._bulk_bar_symbols:
    self._bulk_bar_symbols.discard(symbol)
    self._bar_windows[symbol] = list(self._bulk_head.pop(symbol, []))
# orb_engine prewarm (pre-09:30): sm.subscribe_bars_many(seed) so non-HOD names (11/10 BTCZ, ADTN…) are covered
```
Proof: the failing test turns green; add a case for a symbol not in HOD's seed (must be bulk-subscribed pre-open).
