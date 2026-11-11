# ORB BT cache: 1-min bar gaps at the open (11/2–11/10) — completeness gate + refill — spec 2026-11-11

`docs/orb_bt_vol935_proxy_20261111_RESULT.md`: with the BT universe now requiring ≥ 5 one-minute bars in
[09:30, 09:35) ET, 67 symbol-days 9/29–11/10 that had ≥ 15K volume in that window are dropped as
`range_not_computable` — "mostly 11/2–11/10 cache gaps". Under the old proxy those days were admitted and their
opening ranges were computed from INCOMPLETE bars (a wrong range → wrong picks, wrong stops in the BT). Live had
the bars (its stream/REST path). The rule (CLAUDE.md / memory "fetch completeness gate"): every bulk fetch needs a
LOST count and a completeness gate; a silent gap is a defect.

Rules: no git (no commit/stash), no service action, no orders, never write orb.yaml / config.yaml / .env / crontab /
logs/* / analysis_results/*. **`data/cache.db` is the production cache: do NOT write it from this task** — measure
only; the refill runs through the existing nightly path once the fix is reviewed (say in the RESULT the exact
command the owner/session runs). pytest only on files that exist, `--ignore=tests/integration`, one run at a time,
never piped through head/tail. Never Read a file > 300 lines in full. Alpaca bars only (free; keys via
load_dotenv('/home/ec2-user/onemil/.env'), never printed). Budget ≤ 35 calls. This task IS the owner's request
(selection parity needs a BT that sees the bars live saw); do not pivot.

## 1. Measure (read-only on `data/cache.db`, `intraday_bars_1min`)
For every session day 2026-10-01 → 2026-11-10 and every symbol in the ORB candidate universe that day (reuse
`study_orb_broad.load_broad_universe` with `drop_sink`, or the day's `analysis_results/orb_features_*.csv` symbols):
bars present in [09:30, 09:35) ET (0–5), total RTH bars that day, and whether Alpaca SIP has the missing bars now
(spot-check 20 of the gapped symbol-days with one `get_1min_bars`-style call each; count found/not found). Table
per day: candidates, complete (5/5), partial (1–4), empty (0), and the % of live fills (trades.db ro, strategy='orb')
whose symbol-day is partial/empty in the cache. Write `research/book_review_20261111/cache_gaps_1111.csv`.

## 2. Root cause
Find the nightly writer of `intraday_bars_1min` (`crontab -l` read-only; grep scripts/ for the table name) and
why bars are missing on those days (fetch window, pagination, the late-bars vendor incident on 11/10, symbol list
truncation, the retry loop sharing invalid-symbol removal — the 9/24 class). Quote file:line. If the 11/2–11/10
gaps coincide with a specific failure logged in that cron's log (`logs/*.log`, bounded grep), name it.

## 3. Fix (in the writer, ≤ 80 lines + tests)
- Completeness gate after the fetch: per symbol-day, expected bars in [09:30, 09:35) = 5; log one WARNING per run
  with the count of partial/empty symbol-days and the first 20 names; an ERROR (Telegram) when > 10 % of the
  universe is partial/empty on the latest session day (vendor-degradation alarm, same spirit as the engine's sweep
  alarm).
- Refill: a `--refill-gaps --start D1 --end D2` mode that re-fetches ONLY gapped symbol-days (SIP feed) and
  inserts the missing bars idempotently (INSERT OR IGNORE on the table's key). Do NOT run it against
  `data/cache.db`; run it once against a COPY of the relevant rows in a temp sqlite under the scratchpad to prove
  it fills (integration test with a real Alpaca call for 2 symbol-days, marked so it is skipped without network).
- Tests `tests/test_orb_cache_gaps_20261111.py`: unit (gate counting on a fixture table; the WARNING/ERROR
  thresholds), integration (temp sqlite: refill inserts only the missing minutes, re-run is a no-op).

## Output
`docs/orb_cache_bar_gaps_20261111_RESULT.md` ≤ 250 words: the day table summary, root cause file:line, the fix,
test counts, the exact refill command for the session to run (never run it yourself), and whether the
features/book rebuild must follow (it must if any 9/29–11/10 symbol-day changes). Return ≤ 120 words.
