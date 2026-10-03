# Fix spec — HOD paper dry-run, from the independent review (docs/review_20261003/B_hod.md), 2026-10-03

Scope: `trading/hod_break_engine.py`, `trading/exit_qty_guard.py`, tests. No signal, level, cap or exit-rule change.
Read the review report first; it names the lines. The HOD book runs on its OWN paper account (ALPACA_HOD_*), so "the
owner's manual shares" (B2) is a live-only hazard today — fix it anyway, the code must be safe on a shared account.

## B1 (HIGH) — a transient positions-API error at the 15:55 flatten must not drop the position
`exit_qty_guard.get_signed_broker_qty` returns 0 on ANY exception; `force_close_all` then pops the position and marks it
`exit_pending_verification` with no sell (Friday: VIRT sat unsold, flattened by hand after hours). Required: return
`None` on an API error (keep 0 only for "position genuinely absent"); every caller treats `None` as "unknown — retry next
tick, keep the position registered", with one WARNING per symbol per minute; the flatten loop re-runs every tick after
15:55 ET until every registered position reports 0 or the session ends, then ONE ERROR naming anything still open (that
is the Telegram line the owner must see). Tests: a client whose positions call raises once then succeeds → one sell.

## B2 (HIGH) — registry/flatten quantities must be OUR quantity, never the symbol's total broker quantity
`_sync_registry_qty_to_broker`, the adoption path and the flatten clamp use the broker's total position in the symbol.
Required: compute our quantity from OUR fills (sum of fills on client order ids with this book's prefix for today, via the
orders API) and use min(ours, broker total); when the broker total exceeds ours, log WARNING "foreign shares present" and
never sell/resize beyond ours. If our fills cannot be read, fall back to the registry quantity (not the broker total) with a
WARNING. Tests: broker 100 shares, our fills 40 → stop resized to 40 and flatten sells 40.

## B3 (MEDIUM) — kill-rail cancels must be retried until done
`_enforce_kill_rails_on_resting` marks the rail handled before the cancels; a failed pre-cancel GET leaves an order
resting for good. Required: mark handled only after every resting entry order is confirmed cancelled (or absent); retry
on the next tick with the existing 10 s throttle; WARNING on each retry, ERROR after 5 failures. Test: GET raises once.

## F4–F6 (lower, in the report): implement each one that is a one-line fix with a test; list any you leave with a reason.

## Done means
`python3 -m pytest -q tests/test_hod_killrail_restart_qty.py tests/test_hod*.py -x --ignore=tests/integration` green
(counts), `docs/review_20261003/FIX_B_result.md` (≤ 40 lines: per item what changed, tests, anything left). Never start
the service, never submit orders, never git, never touch config/.env/data/logs.
