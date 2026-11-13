# ORB paper 2026-11-13 — DAMD: scale-out vs exit-coverage conflict, 3-hour retry loop — weekend fix spec

Status: INCIDENT CLOSED by hand (owner-approved order changes); ENGINE FIX OPEN (this spec). Paper book only
(ORB paper key); no live money touched.

## Timeline (journal, UTC)
- 14:35:07 ORB entry DAMD 222 sh stop-bracket @ 7.47/7.50, broker legs TP $22.50 (placeholder) / SL $6.75;
  fill 7.50; StopMonitor watch stop 7.30, lock 7.80→7.58, SCALE OUT armed 88 sh @ 8.02 (trade_id 783).
- later: add-on pool fill 9837d612 222 sh @ 7.7777 → position 444, avg 7.6373. The bracket-leg resize for the
  add-on FAILED; `trading/exit_coverage.py` placed an independent stop-sell 222 @ 7.30 (9aa1761d) for the
  naked half. DB row 783 became shares=444 but entry_price stayed 7.50 (avg not updated).
- ~15:00 → 18:11: price ≥ 8.02 → `StopMonitor._scale_submit_core` every tick: the watch carried no live leg
  ids (the bracket SL leg was dead/dropped; the TP leg + coverage stop were not on the watch), so no resize ran
  and `submit_limit_sell_order(88)` failed with `held_for_orders 444 / available 0`. The submit-failure path
  (`stop_monitor.py` ~L2229) returns False WITHOUT touching `scale_fail_count` → re-triggered on every price
  tick, bursts of 4 submits/s, ≈ 120 REST calls/min on the ORB paper key for ~3 h (Alpaca cap 200/min — the
  EOD flatten shared that key). Every attempt logged 2 WARNINGs.
- 18:11:00 Fable cancelled the orphan TP leg (fa299c27, 222 @ 22.50). Within 6 s exit_coverage re-covered the
  freed shares (ERROR "NAKED 222/444 — placed stop-sell @ 7.58", 4f8e467e) and StopMonitor adopted that id as
  a leg: 3 resize failures → "scale-out DISARMED" (ERROR) — the 3-strike counter exists on the RESIZE path only.
- 18:12:14 Fable cancelled 4f8e467e + 9aa1761d and closed the position: 444 @ 8.44 market (≈ +$357 paper).
  Loop lines after 18:12: 0.

## Defects (each needs a unit test + an integration test, CLAUDE.md bug protocol)
D1 `stop_monitor._scale_submit_core`: the submit-failure branch must count toward `scale_fail_count` and
   DISARM at 3 like the resize branch (one ERROR, then silence); additionally rate-limit to ONE submit attempt
   per symbol per 30 s regardless of tick rate (the broker state cannot change faster than that).
D2 Ownership of shares: `exit_coverage` covers every share not under a broker stop, StopMonitor exits by
   cancel-and-place on its own watch. For a symbol with a StopMonitor watch, coverage MUST register its order
   id on the watch (`sl_leg_id`) so scale-outs resize it — or coverage must skip StopMonitor-managed shares and
   log why. Decide by reading both modules' docstrings; parity test: after a coverage placement, a scale-out
   trigger must succeed on a paper replay (resize → submit → runner covered).
D3 Add-on merge: `trades.entry_price` must become the share-weighted average (7.6373, not 7.50) when shares
   go 222 → 444; P&L of row 783 is wrong otherwise. Find the merge site (orb_engine add-on fill handling).
D4 Investigate from the journal 14:35–15:30 why the watch had no live leg ids after the add-on resize failed
   (which call dropped them, and whether the original SL leg 6.75 was cancelled at the broker).
D5 Reconciliation — CONFIRMED a defect at 18:40 UTC: the only engine reaction to the external close was
   `stop_monitor:897 WARNING EXIT COVERAGE: DAMD broker qty 0 < watch qty 444` (18:13:08); no exit was booked,
   row 783 still has exit_price NULL 28 min later, the watch (444 sh, stop 7.30) is still armed on a position
   that no longer exists. Required: broker qty 0 on a watched symbol → book the exit from the broker's last fill
   (cfae5a8c, 444 @ 8.44, reason `external`), remove the watch, one INFO line. Owner manual exits on the live
   book would be mis-booked the same way — this one is live-relevant.

## Rules for the fixing agent
Sonnet, ≤ 60 calls, TDD, no git commit, no service restart, no orders. Tests: `pytest tests/ -q -x
--ignore=tests/integration`; Fable runs the full suite before commit. Weekend boot rehearsal before Monday.
