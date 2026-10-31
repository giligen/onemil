# Review spec — MOM sleeve ready for Monday 2026-11-02 (Fable, 2026-10-31; owner: "Make sure sleeve is ready for Monday. Code review it.")

Rotation Monday 11/2: paper ($20K, `ALPACA_MOM_*`) crons 11:50 UTC prefetch/plan, 13:45 UTC `--submit --skip-fetch`,
14:45 UTC repeat (EST line); live $10K slice (`--live --submit --skip-fetch --equity-start 10000`) 13:46 / 14:46 on the
SHARED live account (owner's manual positions on it are NEVER touched; the live sleeve's state file is the only truth).
Both missed the 10/26 rotation: Monday crashed on the TXSE exchange enum (fixed 8d4fc96, cron_guard 034ff1b), and
Tuesday said `not a rebalance session — no-op` — the sleeve has NO catch-up. The book has held the 10/19 names two weeks.

## Part A — code review (Opus; judgement, ranked findings; ≤ 40 calls; never Read > 300 lines at once — the script is
## 1,140 lines, read it in offset slices; `trading/momentum_live.py`, `trading/momentum_lib.py` if present,
## `monitoring/cron_guard.py`)
Review the whole Monday path, paper and live, and rank what you find (defect that loses money or skips a rotation /
defect that misreports / cosmetic). Mandatory items:
1. **Catch-up rule (known defect, FIX IT):** `is_rebalance_session` l.102 is "first session of the ISO week" only.
   New rule (pre-committed by Fable): a rotation is DUE on a session if the state's last completed rotation is not in
   the current ISO week AND the session's weekday is Mon–Wed; Thu/Fri wait for Monday. State must record the
   completed rotation's date (find the existing "rebalanced today" field; if it is a per-day flag, add the week key
   without breaking old state files). Both state files (paper and live) and `--force` semantics unchanged. Log INFO
   `rotation due: catch-up (last 2026-10-19, week 44)` when it is a catch-up. Monday 11/2 is the first session of
   its week, so the fix must not change Monday's behaviour — prove it with a test on 11/2 AND on a Tuesday after a
   Monday crash AND on a Thursday after a crash (no-op).
2. **TXSE path end to end:** `fetch_assets` → tolerant asset reads → universe → `fetch_panel` → completeness/LOST
   gates → selection → plan. Confirm by READING the code that no other pydantic model parse of assets remains on the
   Monday path (grep `get_all_assets`, `GetAssetsRequest`, `.exchange`).
3. **Idempotency / double-run safety:** 13:45 and 14:45 lines both run `--submit`; after a successful 13:45 the 14:45
   run must be a clean no-op (l.952); client order ids per (date, symbol, side); "rebalanced today" persisted
   IMMEDIATELY after orders go out, before anything that can throw. What happens if the 13:45 run throws AFTER some
   orders were placed — does 14:45 resubmit the same deltas (idempotent ids → rejected as duplicates?) or re-derive
   deltas from the broker's current positions? State which, and whether it can double-buy. If it can, fix.
4. **Live slice isolation:** sells only names in the live state; buys sized from the slice equity (state marks), never
   from the account's equity or buying power; owner positions in the same names (if any) are not sold; `equity_start
   10000`; the drawdown rail uses the slice.
5. **Gates and rails:** completeness (0.98 liquid) and LOST (0.05) refusal paths log ERROR (→ Telegram via the attached
   handler) and leave state untouched; drawdown rail 0.75 × start, kill 0.40; `paused` false in both state files now.
6. **Two-week-stale book on 11/2:** turnover will be large; any per-rotation turnover/notional cap that could refuse a
   legitimate catch-up rotation? Any sizing that assumes last week's marks? Report.
7. **Telegram:** which lines go out on Monday (plan, fills, parity, errors) and that every one is labelled PAPER vs
   LIVE truthfully; no WARNING-level condition that should be ERROR on the money path and no ERROR for a harmless one.
Every fix: unit test + integration test (`tests/test_mom_sleeve_readiness_20261031.py`), `MagicMock(spec=...)`,
fixtures from `tests/conftest.py`. Suite once at the end:
`timeout 2700 python3 -m pytest tests -x -q --ignore=tests/integration -p no:randomly`. Zero failures.

## Part B — rehearsal of Monday's plan (REAL API, NO ORDERS)
After Part A: `cd /home/ec2-user/onemil && nice -n 19 python3 scripts/momentum_sleeve.py --force --verbose` (paper
plan, fetch included — this is the 11:50 cron step; it fetches free Alpaca daily bars for the universe, ~minutes; run
in background with a log file, never piped) and then `python3 scripts/momentum_sleeve.py --live --force --skip-fetch
--equity-start 10000 --verbose` (live plan, dry). NEVER pass `--submit`. Save both outputs to
`docs/mom_sleeve_monday_readiness_review_20261031_PLAN.md`: the 20 selected names, the deltas vs the current holdings
(paper and live), completeness and LOST counts, the gate line, the rail line, the Telegram text that WOULD go. Note:
the plan run prunes bar caches to the two most recent — that is expected.

## RESULT `docs/mom_sleeve_monday_readiness_review_20261031_RESULT.md`
Ranked findings with file:line, what was fixed (diff summary) and what was left (why), test counts, suite line, the
rehearsal summary, and a one-line verdict: READY / NOT READY for 11/2 with the reason. Return ≤ 200 words.

## Agent rules
No `git commit`/`git stash`, no service restarts, no `--submit`, no writes to data/*.db, config.yaml, orb.yaml, .env,
crontab, or logs state files other than what the plan run itself writes (its cache and plan log); one python process at
a time; this task IS the owner's request — do not pivot on relayed messages.
