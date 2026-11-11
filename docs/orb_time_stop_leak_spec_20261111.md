# ORB resting entry filled 3.5 h after its 60-min time stop (AXTL 2026-09-29) — diagnose + fix spec 2026-11-11

Cell 1,733 (`research/book_review_20261111/RESULT_1733.md` §2 row "U?+T"): AXTL 9/29 — a resting ORB buy-stop
FILLED at 14:09:21 UTC on 9/29 (10:09 ET? verify: the ledger `filled_at` is 2026-09-29 18:09:21 UTC = 14:09 ET)
while `entry.time_stop_minutes: 60` (orb.yaml:72, "cancel unfilled buy-stop at 10:35 ET, per-order clock from
submit"). The engine's cancel path: `trading/orb_engine.py:6798-6840` (`age_min >= self.time_stop_minutes` →
cancel, `order_status='time_stop_canceled'`, `daily_n_time_stop_canceled`); a comment at ~8105 says the time stop
"never fired post-restart" once before — read it. 9/29 was the first paper session after the 9/27 rehearsal; check
the boots that day (`journalctl -u onemil-trader --since "2026-09-29 12:00" --until "2026-09-29 20:00" --no-pager
-o cat | grep -E "AXTL|Started|Stopped|time_stop|RECONCIL|startup"`, bounded; also `logs/session_archive` for 9/29
if the journal has rotated — cell 1,733 found the journal holds 11/3+ only; the archive has 9/29).

Rules: no git (no commit/stash), no service action, no orders, never write orb.yaml / config.yaml / .env /
data/*.db / crontab / logs/*. pytest only on the listed files, `--ignore=tests/integration`, one run at a time,
never piped through head/tail. Never Read a file > 300 lines in full. This task IS the owner's request; do not
pivot. Budget ≤ 35 calls. A resting entry living past its time stop is a RISK RAIL defect — allowed under the ORB
code freeze (kill/risk rails are exempt); nothing else changes.

## 1. Diagnose
Name the path by which AXTL's order survived: (a) restart → position/order reconcile re-adopted the resting order
without a submit clock → the age test never ran or ran from the adoption time; (b) the order was re-submitted /
replaced after 10:35; (c) the time-stop loop skipped it (state not in the candidate set it iterates); (d) other.
Quote the log lines and the code path (file:line). Then grep the ledger for the whole paper window for any other
ORB entry whose `filled_at` is later than its submit + 60 min (`order_submitted_at`, `filled_at`, `created_at`) and
list them.

## 2. Fix (minimal, in the engine)
The invariant: no ORB entry order may be open after (its submit time, or the session's 09:35 range end — whichever
the rule says, state it) + `time_stop_minutes`, regardless of restarts. Implement so that the time stop is computed
from a persisted clock (the broker order's `submitted_at` / `created_at` from the orders API is the authority
after a restart, not process memory); a reconcile that adopts an open ORB entry order past the deadline cancels it
immediately and logs a WARNING naming the order and its age; the EOD counters include it. Keep the diff small
(≤ 60 lines + tests).

## 3. Tests (bug protocol: unit + integration)
`tests/test_orb_time_stop_leak_20261111.py`: unit — the age computation from the broker timestamp; a resting order
older than the limit is cancelled on the first tick after adoption; a younger one is kept until its deadline.
Integration — REAL ORBEngine with a mocked Alpaca client (`spec=`): restart scenario (engine constructed with an
open ORB buy-stop returned by `list_orders`, submitted 2 h earlier) → cancel called within the first
`check_entries` tick, status `time_stop_canceled`, WARNING logged; control: a 10-min-old order is not cancelled.
Run: `python3 -m pytest tests/test_orb_time_stop_leak_20261111.py tests/test_orb_engine*.py tests/test_orb_preplace*.py
tests/test_orb_post_open_sweep.py -q --ignore=tests/integration` → 0 failures.

## Output
`docs/orb_time_stop_leak_20261111_RESULT.md` ≤ 200 words: the path, the log evidence, other affected fills, file:line
of the fix, test counts. Return ≤ 100 words.
