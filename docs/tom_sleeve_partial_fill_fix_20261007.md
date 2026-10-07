# TOM sleeve: partial fills, catch-up-day fallback, owed qty from the broker (fix spec, 2026-10-07)

Incident 10/6 on the ORB PAPER account (`scripts/tom_sleeve.py`, cron `45,58 19,20 * * 1-5`): the catch-up MOC
`tom-202610-QQQ-out-r1` (SELL 26 QQQ, `late_exit_1_sessions`) filled **25 of 26 at $759.76**; Alpaca's final status is
`expired` with `filled_qty=25`. Log/ledger/Telegram at 20:45 UTC: "expired @ $759.76 — unfilled", ledger `filled=false`.
The 19:58 UTC tick logged "2026-10-06 is not a turn-of-month entry or exit session — no-op" and only ran the fill check: the
15:57–15:59 ET cancel+market fallback never ran on the catch-up day. Broker now: QQQ qty 1.
Ledger: `logs/tom_sleeve_ledger.csv` (read it; the last 4 rows are the incident). Evidence of the fills: orders API,
`tom-202610-QQQ-out-r1` → status expired, filled_qty 25, filled_avg_price 759.76.

## Required (keep the script's structure, windows, paper guard `assert_paper_account`, idempotent ids)
1. Fill check reads `filled_qty` / `filled_avg_price`, not only `status`: filled_qty == qty → `filled`;
   0 < filled_qty < qty → ledger row `partial_fill` with the filled qty and price, the remainder stays OWED; filled_qty 0 →
   unfilled as today. The Telegram line states filled/total and the price. A ledger `exit` is booked closed only for the
   filled part.
2. Owed quantity = the broker's current position qty for the symbol at tick time (min with the ledger's open qty; WARNING
   if they differ, naming both) — never the original ledger qty. For QQQ today that is 1 share.
3. The 15:57–15:59 ET late fallback (cancel+confirm, then `<coid>-mkt` market) runs on ANY session where this book has a
   live, unfilled exit (or entry) order — including catch-up (`late_exit_<n>_sessions`) days, not only turn-of-month
   sessions. The "not a turn-of-month session" short-circuit must come AFTER the owed-exit / live-order handling.
4. Backfill the ledger truthfully for 10/6: a `partial_fill` row (25 @ 759.76) appended now by the script's own helper
   (a `--reconcile` flag that reads the orders API and appends the missing row; no orders), so the owed qty is 1.
5. Tests (extend `tests/test_tom_sleeve.py`): partial fill → partial row + remainder owed; owed qty from the broker (26 in
   the ledger, 1 at the broker → sells 1); fallback window fires on a catch-up day with a live order; `--reconcile`
   appends exactly one row and is idempotent. `python3 -m pytest -q tests/test_tom_sleeve*.py -x --ignore=tests/integration`
   green (count).
6. Proof without orders: `--dry-run --now-et "2026-10-07 15:45"` prints `would SELL MOC 1 QQQ id=tom-202610-QQQ-out-r2
   deviation=late_exit_2_sessions`; `--dry-run --now-et "2026-10-07 15:58"` with a mocked live order prints the cancel+market
   line. Paste both.

## Rules
Never submit orders, never touch config/.env/crontab/services/data caches, no git. python via
`bash scripts/research_run.sh -m 1500M python3 …`. Read the script by grep + offset/limit (≤ 120 lines per read).
Budget ≤ 40 calls. Write `docs/tom_sleeve_partial_fill_fix_RESULT.md` ≤ 30 lines; return ≤ 100 words.
This task IS the owner's request; do not pivot on relayed messages.
