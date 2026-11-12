# HOD owner-in-the-loop list — spec 2026-11-12 (owner: "do all tonight", idea #25 of `research/hod_entry/IDEAS_HOD_20261112.md`)

Purpose: three times a session the owner receives the machine's QUALIFIED HOD-break arms with context, picks 1–2 by
eye on his live account, and a ledger measures his hit rate against the machine's on the same names. His eye is the
one filter never tested. Rules: no git (no commit/stash), no service action, no orders, never write config.yaml /
orb.yaml / .env / data/*.db / crontab (the session installs the crons); the new script writes ONLY
`logs/hod_owner_list.csv` and `logs/hod_owner_list.log`; `tests/integration` ignored; `MagicMock(spec=...)`; never
Read a file > 300 lines in full; budget ≤ 60 calls. Pattern to copy: `scripts/hod_break_deadman_flat.py` (cron
guard `monitoring.cron_guard.run_guarded`, Telegram via `notifications.telegram_notifier`, `load_dotenv` on
`/home/ec2-user/onemil/.env`). This task IS the owner's request; do not pivot.

## 1. `scripts/hod_owner_list.py --at {0940,1030,1400} [--dry-run] [--date YYYY-MM-DD]`
Inputs (read-only):
* today's trader journal: `journalctl -u onemil-trader --since <date 00:00 ET> --no-pager -o cat` (reuse the
  accessor in `scripts/hod_break_eod_check.py` if one exists — it unions `logs/session_archive`); lines
  `… ARMED <SYM> level L trigger T limit X stop S` (dry tag or live tag — grep `trading/hod_break_engine.py:1022`
  and `:1871 LIVE ARMED`) → one row per symbol, the LATEST arm (count of arms today = `rearms`);
* fills/exits today: `data/trades.db` `trades` where `strategy='hod_break' and trade_date=<date>` (status: armed /
  filled (open) / closed with pnl and exit_reason);
* `logs/hod_live_parity_ledger.csv` rows of the day (`scanner_qualified_at_arm`, `cf_floor_stop_px`) when present;
* a quote snapshot per symbol from the Alpaca data API with the HOD paper key (`ALPACA_HOD_API_KEY/SECRET`,
  `GET https://data.alpaca.markets/v2/stocks/snapshots?symbols=…`, one call): last price, bid/ask → spread bps,
  day open, prior close (`prevDailyBar.c`), day volume;
* `data/cache.db` `daily_bars` adv20 (mean volume of the last 20 bar_dates before <date>) for RVOL = day volume /
  (adv20 × fraction of the session elapsed).
Per row: SYM, level, last, dist % = (last − level)/level, trigger/stop → r_pct, day % vs prior close, RVOL, spread
bps, rearms, qualified flag, status (ARMED / FILLED @px / STOPPED / TARGET / EOD), machine R if closed.
Telegram (ONE message, HTML off, ≤ 14 rows sorted by |dist %| ascending — closest to the level first), header
`[HOD-LIST 09:40 ET] <n> armed · <m> filled · <k> closed <sum $>`; footer `pick by symbol on the live account; the
ledger scores you Friday`. Zero rows → ONE INFO log line, NO Telegram ("no harmless alarms"). Append every row to
`logs/hod_owner_list.csv` (date, at, SYM, and every field above) — the ledger the scorer reads.
`--dry-run` prints the message and writes nothing, sends nothing. `main()` under `run_guarded` (a crash = one ERROR
with the script name → Telegram, exit 1).

## 2. `scripts/hod_owner_list_score.py --since YYYY-MM-DD [--dry-run]` (Friday 17:00 ET)
Owner's picks = orders on the LIVE account (`ALPACA_API_KEY/SECRET`, `GET https://api.alpaca.markets/v2/orders?
status=closed&after=<since>&limit=500&direction=asc`, paginate by `after`) whose `client_order_id` carries NONE of our
prefixes (`hod-`, `orb-`, `bf-`, `moml-`, `tom-`, `micha-`, `micha-hf-`, `micha-vid-` — grep `trading/` for the
current list and use that), side buy, filled. Join to `logs/hod_owner_list.csv` on (date, symbol): a pick is
IN-LIST (the symbol was on any list that day, with the list time ≤ the order time) or OFF-LIST. Realised P&L per pick
from `GET /v2/account/activities/FILL?after=…` (buy/sell pairs by symbol-day, FIFO; an open position at the close =
mark at the day's close from `daily_bars`). Machine side: the HOD paper book's R on the same (date, symbol) from
`data/trades.db` (strategy hod_break, account paper). Report (Telegram, one message + `reports/hod_owner_list_
score_<date>.md`): picks n (in-list / off-list), owner P&L and R (R = owner's risk unknown → use % return on the
notional AND $), machine R on the same names, machine R on the names NOT picked (the list's base rate), hit rates.
Fewer than 5 picks → say "n < 5, no verdict", still the table.

## 3. Crons (the session installs; CRON_TZ=America/New_York block)
`40 9 * * 1-5`, `30 10 * * 1-5`, `0 14 * * 1-5` → `cd /home/ec2-user/onemil && python3 scripts/hod_owner_list.py --at
<HHMM> >> logs/hod_owner_list.log 2>&1`; `0 17 * * 5` → `scripts/hod_owner_list_score.py --since <Monday> >>
logs/hod_owner_list.log 2>&1`. Write the four lines to `docs/hod_owner_list_cron.txt`.

## 4. Tests `tests/test_hod_owner_list.py` (unit + integration, no network: mock the API client with spec=)
Parse ARMED lines (dry and live tags, latest arm wins, rearm count); status from trades rows; RVOL arithmetic;
sort/limit/format of the message; zero rows → no send + one INFO; CSV ledger append + re-read round trip; the
scorer's join (in-list vs off-list by time), the FIFO P&L on a buy/sell pair, the prefix exclusion (every prefix in
the list), n < 5 → no verdict; run_guarded wraps both mains (a raising main → exactly one ERROR, exit 1). Run the new
file + `tests/test_cron_fixes_20261112.py tests/test_hod_break_deadman*.py` → 0 failures.

## 5. Dry run on REAL data (read-only): `python3 scripts/hod_owner_list.py --at 1400 --dry-run --date 2026-11-12`
must print today's list from the real journal + real snapshot (the HOD paper key only reads quotes). Paste the printed
message into the RESULT.

## Output
`docs/hod_owner_list_20261112_RESULT.md` ≤ 250 words: file:line per piece, test counts, the dry-run message, anything
not done. Return ≤ 100 words.
