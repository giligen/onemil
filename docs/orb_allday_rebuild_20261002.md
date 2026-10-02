# ORB all-day universe rebuild + 210K-WARNING journal (2026-10-02)

## Diagnosis
1. The build is BY DESIGN every ~2 min all day: `RealtimeScanner._orb_tick` -> `ORBEngine.build_universe(source_loader=_orb_universe_source)`
   (idempotent, extends the candidate set). Normal day 9/30 (journal 12-22 UTC): 268 "ORB universe seed" + 268 "snapshot universe".
   Nothing stopped it after the entry window. ORB's own cutoff is `orb.yaml entry.last_entry_submit_time_et: "10:00"`
   (`_past_last_entry_time`, trading/orb_engine.py); past it `check_entries` returns [] all day, so a candidate added then
   can never trade - the build was pure cost. It is NOT only restart-after-window: 10/01 (no restart) had 144,215 fallback
   WARNINGs and 80 cycle overruns; 10/02 209,937 WARNINGs, 279 tick TIMEOUTs. 9/30 had 0 only because the gap-gate WARNING
   shipped that evening (docs/orb_parity_20260930.md).
2. Cost at HEAD in the restart state, replay (cache.db ro, REST stubbed 5.5 s/3,041): one cycle = 5.8 s and 3,044 log lines
   (1 aggregated WARNING after this fix; 3,041 DEBUG `[ORB] GAP_GATE` lines; before the fix +3,041 WARNINGs). The 120-233 s
   live cycles were the N+1 SQL (docs/orb_open_tick_stall_20261002.md), already fixed at HEAD.
3. 09:30 bars are "not available" for ~every symbol because nothing in live ever writes today's bars to `intraday_bars_1min`:
   the only `save_intraday_bars` callers are backtests and `batch/intraday_bars_backfill.py` (completed days, EOD). The batched
   lookup at HEAD reads the same table, so it has the SAME ~100 % miss rate - it only made the miss cheap. Consequence: the
   9/30 "prefer the settled 09:30 bar open" rule is inert live (gap is always gated on the snapshot open). Owner decision, NOT
   changed here (would change which symbols are admitted).

## Fix (plumbing only)
* `trading/orb_engine.py`: `universe_build_due()` - False past `last_entry_submit_time_et`, ONE INFO per ET day
  ("universe build + gap gate skipped for the rest of the day"). `trading/orb_engine.py` aggregation: one WARNING per build
  (count, first 10 symbols, reason) via `fallback_sink` in `trading/orb_gap_gate.py::resolve_gap_input` (default keeps the
  per-symbol WARNING for single-symbol callers; resolved input identical).
* `scanner/realtime_scanner.py::_orb_tick`: build only `if orb_engine.universe_build_due()`. Entries/exits/force-close untouched.
* `build_universe` itself is unchanged (many tests call it at wall-clock time).
* Not changed: selection, thresholds, sizing, exits, the 09:35 path, the per-symbol DEBUG line.

## Replay (`scripts/orb_open_tick_replay.py --date 2026-10-02 --restart-at-et HH:MM`, never sends orders)
| state | seconds | log lines |
|---|---|---|
| before (10:03 or 16:01 restart, live) | 120-233 | ~6,450 (3,224 WARN + 3,224 DEBUG + ...) |
| after 09:36 (in window) | 5.8 | 3,044 (1 aggregated WARN, 3,041 DEBUG) |
| after 10:03 / 16:01 | 0.00 | 1 INFO (once per day) |

## Tests
`tests/test_orb_open_tick_stall_20261002.py` (+5: skip + one INFO, in-window builds, one WARNING per build, legacy per-symbol
WARNING kept, scanner skips build) ; 398 passed across every ORB/scanner test that uses build_universe + replay + add-on.

## Monday journal proof (restart not needed on a normal boot)
`journalctl -u onemil-trader --since 12:30 | grep -c "ORB GAP_GATE: .*no 09:30"` -> <= 1 per build (~1 per 2 min before 10:00 ET,
then 0); `grep -c "universe build + gap gate skipped"` -> 1 at ~10:00 ET; `grep -c "ORB universe seed"` after 14:00 UTC -> 0.
