# TOM sleeve: unfilled MOC must be replaced, expired ids must not block (fix spec, 2026-10-05)

Incident: `scripts/tom_sleeve.py` submitted `tom-202610-QQQ-out` SELL MOC 26 QQQ at 19:45 UTC on the ORB PAPER account
(`ALPACA_ORB_*`, `assert_paper_account`). At the 19:58 UTC tick the order was still `new`; the script logged
"already exists — skip (idempotent)" and did nothing; Alpaca paper expired the MOC at the close — unfilled. The 9/30
entries for SPY and IWM expired the same way (QQQ's entry filled). The position (26 QQQ) is still open.

## Required (read the script first; keep its structure, windows and paper guard)
1. Fill-aware idempotency: an existing order with the same client_order_id blocks a resubmit ONLY while it is live
   (`new`, `accepted`, `pending_new`, `partially_filled`) or `filled`. `expired` / `canceled` / `rejected` → the symbol
   is still owed; log WARNING `tom_sleeve: <coid> <status> unfilled — resubmitting as <coid>-r<n>` (n = retry count).
2. Late fallback window (15:57–15:59 ET): if the tracked exit (or entry) order is live but unfilled, CANCEL it, wait for
   the cancel to confirm (poll ≤ 10 s), then submit a MARKET order (`<coid>-mkt`), ledger row `late_fallback_replace`.
   If the cancel does not confirm, log ERROR and do not double-submit.
3. Next-session catch-up: on a tick inside the MOC window, a tracked position whose exit was owed on an earlier session
   (ledger shows exit submitted, no fill, position still at the broker) is exited that session; ledger row carries
   `deviation=late_exit_<n>_sessions`. Same for an owed entry: NOT re-entered (the entry window is gone), logged WARNING.
4. Telegram: one `[TOM]` line on any unfilled exit at 15:59 ET and on any replacement — never silent.
5. Tests (`tests/test_tom_sleeve*.py`, extend the existing file): expired id → resubmit; live-unfilled at 15:58 → cancel +
   market; filled id → skip; owed exit next session → exits with the deviation tag; no-op outside the windows unchanged.
   `python3 -m pytest -q tests/test_tom_sleeve*.py -x --ignore=tests/integration` green.
6. Proof without orders: a dry invocation that prints what it WOULD do for 2026-10-06 15:45 ET given the current
   ledger/broker state (QQQ 26 owed since 10/5) — paste the line in the result. No orders are submitted tonight.

## Rules
Never submit orders, never touch config/.env/crontab/services, no git. python via `bash scripts/research_run.sh -m 1500M`.
Write `docs/tom_sleeve_fallback_fix_RESULT.md` ≤ 30 lines; return ≤ 100 words. This task IS the owner's request.
