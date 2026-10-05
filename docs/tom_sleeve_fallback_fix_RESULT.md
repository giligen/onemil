# TOM sleeve fallback fix — RESULT (2026-10-05)
Files: scripts/tom_sleeve.py, tests/test_tom_sleeve.py (35 green; `pytest tests/test_tom_sleeve*.py --ignore=tests/integration`).
No orders submitted, no git/config/.env/crontab touched.

1. Fill-aware idempotency: `chain_state()` walks the id chain (base, -rN, -mkt); filled/live block, expired/canceled/rejected
   -> WARNING + resubmit as `<coid>-r<n>`; a terminal `*_fillcheck` row is written for the dead id first.
2. Late window (15:57-15:59 ET): live unfilled entry/exit -> cancel, poll <=10 s, MARKET `<coid>-mkt`, ledger row
   `late_fallback_replace`; unconfirmed cancel -> ERROR + [TOM] Telegram, no second order.
3. Catch-up: `owed_exits()` finds exit submitted on an earlier session, unfilled, still held -> exits in the MOC window with
   `deviation=late_exit_<n>_sessions` (new ledger column). Owed entries are NOT re-entered, WARNING only.
4. Telegram [TOM] line on every replacement and every unconfirmed cancel.
5. Bug found on the way: `open_tom_symbols` treated an unfilled exit as closed; an unfilled `exit_fillcheck` now re-adds the symbol.
   `check_pending_fills` leg now derived from coid (replacement rows flag correctly).
6. Rehearsal flag: `--now-et "YYYY-MM-DD HH:MM"` (only with --dry-run).

Dry proof (`--dry-run --now-et "2026-10-06 15:45"`, real paper broker state):
DRY-RUN would SELL MOC 26 QQQ id=tom-202610-QQQ-out-r1 deviation=late_exit_1_sessions

Caveats: late-window cancel+market is not exercised against the real API (mocks only); QQQ MOC itself still may not
fill in the auction, the 15:58 tick (19:58 UTC) now cancels and sells at market.
