# TOM sleeve: the 15:58 late fallback cancels every live MOC — spec 2026-11-05

## What happened (2026-11-04 exit, paper, ORB key)
- 20:45 UTC (15:45 ET, MOC window): `_process_exit` submitted **MOC (CLS) sells for SPY, QQQ and IWM**
  (`use_moc = moc_window`, l.806 — ignores `MOC_RELIABLE_SYMBOLS`, which the ENTRY leg honours at l.643/676).
- 20:58 UTC (15:58 ET, late window): all three were `live` (a MOC cannot fill before 16:00), so l.791–793 called
  `_late_replace` → cancelled the three MOCs and sold MARKET at 15:58. Fills 766.36 / 691.57 / 283.82 vs closes
  766.81 / 691.95 / 284.02 = **−$37 vs the rule's MOC fill** (cycle still +$642). Telegram said "unfilled at the close —
  cancelled, SELL MARKET …" which the owner read as a failed exit.
- Same conflict on the ENTRY leg 10/30: `tom-202610-QQQ-in` (MOC) cancelled at 19:58 → `QQQ-in-mkt`. And on 10/7 the
  1-share `QQQ-out-r2`. **Every MOC leg since the 10/7 late-fallback fix has been converted into a 15:58 market order.**
  The sleeve is PREREG 1,649's close-to-close rule; a 15:58 market fill is a rule drift, now measured.

## Fix (root cause, both legs)
1. `_late_replace` must never touch a live **CLS** order: in both callers (entry l.649–654, exit l.791–793) skip
   `_late_replace` when the live order's `time_in_force == 'cls'` (expose it from `chain_state`, e.g. `st['tif']`,
   read from the broker order; fall back to "coid was submitted as MOC" only if the API object lacks it). Log INFO
   "MOC live — fills at the close, left alone". An EXPIRED MOC is already handled next session by the catch-up path
   (`owed_exits`, `-rN`) — do not add a second mechanism.
2. Exit leg mirrors the entry leg: `use_moc = moc_window and symbol in MOC_RELIABLE_SYMBOLS`; a non-MOC symbol in the
   MOC window does nothing (wait for the late window), and in the late window with no live order submits MARKET (as
   now). So SPY/IWM exit MARKET at 15:58 by design (the 9/30 evidence: their paper MOCs expire unfilled), QQQ exits MOC.
3. Message text: `_late_replace` says "unfilled before the close (15:58 ET)", not "at the close".

## Tests (TDD, `tests/test_tom_sleeve*.py` — extend the existing file)
- unit: late window + live CLS order → no cancel, no market order, INFO line (entry AND exit).
- unit: late window + live DAY limit (entry) → cancel + market (unchanged behaviour).
- unit: exit in the MOC window → MOC only for QQQ; SPY/IWM no order; late window SPY/IWM → MARKET.
- integration (real ledger file in a tempdir, AlpacaClient `MagicMock(spec=AlpacaClient)`): the 11/4 sequence
  (15:45 tick then 15:58 tick) ends with ONE order per symbol: QQQ MOC still live, SPY/IWM one MARKET each, ledger rows
  consistent. Rerun the existing tom tests; zero failures.

## Rules
Never submit orders, never run the script against the API (use `--dry-run` only if you must, and only in tests).
No git commit, no service/cron changes, no writes to logs/ or data/. `--ignore=tests/integration`. Budget ≤ 35 calls.
RESULT → `docs/tom_sleeve_moc_fallback_spec_20261105_RESULT.md` (≤ 150 words); return ≤ 100 words.
