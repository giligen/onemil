# OPS 2026-10-02 — HOD paper account, after-hours flatten (same procedure as 10/1)

The instance rebooted at 14:03 UTC; the service restarted 14:03:54 and re-adopted resting orders. After the 15:55 ET
flatten two positions were still at the broker (HOD paper PA39QSZR60WC):
* SPCF 22 sh — remainder: the registry sold its own quantity at the +2R target, the broker held more (registry /
  broker quantity mismatches all afternoon: NEBX 21 vs 44, COHX 50 vs 120).
* VIRT 24 sh @ 59.42 — a resting stop-limit submitted 14:05 and adopted at 14:08 filled AFTER the daily kill rail
  had switched the arm to tape-only: an unmanaged real position (no stop, no exit).
Flattened 20:06 UTC with extended-hours DAY limits at the bid: SPCF 22 @ 18.41 (+$12.08), VIRT 24 @ 58.06 (−$32.64).
Account flat. Engine day P&L at the flatten: −$150 on 16 exits since the restart.

Reading: today's HOD paper fills are NOT a parity sample (restart mid-session, doubled quantities).
Defects to fix before the next session reads as evidence:
1. Kill rail trips → resting entry orders at the broker must be cancelled (VIRT).
2. Restart mid-session → registry quantity must be rehydrated from the broker (NEBX / COHX / SPCF / CRCA).
