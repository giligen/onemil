# Hourly Telegram summary per strategy (spec, 2026-10-09 — owner: "Add hourly summary for each strat. Test it well! Launch today")

One cron script, `scripts/hourly_summary.py`, sends ONE Telegram message per run with today's state of every book.
Read-only: it never submits, cancels or modifies orders, never writes the DB. Pattern-copy `scripts/tom_sleeve.py`
(`Config()` loads .env, `AlpacaClient(key, secret, paper=True)`, `TelegramNotifier(cfg.telegram_bot_token,
cfg.telegram_chat_id, enabled=True)`, `send_message_sync`). Never print or log a key.

## Books (accounts from .env, all PAPER; `assert_paper_account`-style check: refuse any account whose number does not
start with "PA" — reuse/import the helper from tom_sleeve if importable, else copy it)
| Book | Keys | Attribution |
|---|---|---|
| ORB | `ALPACA_ORB_API_KEY/SECRET` | all symbols except QQQ (TOM shares the account) |
| TOM | same account | QQQ only; show only when a QQQ position or a QQQ order today exists |
| HOD | `ALPACA_HOD_API_KEY/SECRET` | all |
| MOM (momentum sleeve) | `ALPACA_MOM_API_KEY/SECRET` | all; weekly book → show day P&L, open P&L, n positions, market value |
| LIVE (owner's manual account, report-only) | `ALPACA_API_KEY/SECRET`, paper=False | ONE line: day P&L and equity; nothing else, never per-position |

## Numbers per book (computed from the orders API + positions, like the scratch script the owner already saw)
- Day P&L = `equity − last_equity` for the account (for ORB/TOM split: realized from today's closed fills per symbol
  (buy VWAP vs sell VWAP × min qty) + unrealized_intraday_pl of open positions; the account total is the cross-check —
  log a WARNING if |sum of books − account day P&L| > $5).
- Per book: day P&L in $ and % (ORB on $10,000 stage; HOD on gross notional bought today; MOM on `equity`; TOM on
  the QQQ cost basis), fills today (count), open positions with symbol/qty/unrealized, closed trades with symbol and
  realized $, biggest winner / loser. Orders API query: `GetOrdersRequest(status=CLOSED, after=<today 00:00 UTC>, limit=500)`;
  use `filled_qty`/`filled_avg_price` (never `qty`), ignore unfilled.
- Week to date: sum of the account's daily `profit_loss` from `get_portfolio_history(period='1W', timeframe='1D')` for
  sessions since Monday (note: Alpaca stamps each daily bar with the NEXT UTC date — verify against today's
  `equity − last_equity` and document what you found in the RESULT).

## Message (HTML, one message, ≤ 25 lines; `📊 HOURLY hh:mm ET` header)
```
📊 HOURLY 12:00 ET (Fri 10/9)
ORB   $0 (0.0%) | 0 fills | flat                      wk +$164
HOD   −$79 (−0.3%) | 9 fills | open 4: SOXS +16 GRAL +7 DDOG +3 IONZ −8 | worst UNHG −54   wk −$30
MOM   +$241 (+0.4%) | open −$450 on $19.5K, 20 names   wk −$466
TOM   (hidden when idle)
LIVE  +$0 | equity $65,0xx
```
No ERROR text in the message. If one account fails, the line says `ORB   n/a (api error)` and the others still send;
the failure is logged WARNING. If Telegram is not configured, print the message to stdout and exit 0.

## Flags
`--dry-run` (print, do not send), `--now-et "YYYY-MM-DD HH:MM"` (header/day boundaries for tests), `--once` is the
default. Exit 0 always after a send attempt; non-zero only on a programming error.

## Tests — `tests/test_hourly_summary.py` (unit, mocked clients with `spec=`; ≥ 12 tests)
ORB/TOM split; partial-fill P&L uses filled_qty; open-position lines; week-to-date date stamping; API failure on one
account still sends the others; LIVE line never lists positions; paper guard refuses a non-PA account number;
message ≤ 25 lines and contains no key/secret substrings; dry-run sends nothing; Telegram unconfigured → stdout.
Then the REAL run: `python3 scripts/hourly_summary.py --dry-run` against the real paper accounts (read-only) and paste
the printed message into the RESULT; then one REAL send `python3 scripts/hourly_summary.py` (the owner wants it
launched today) and confirm `send_message_sync` returned True in the log.

## Cron (do NOT install — write the exact line in the RESULT for the owner)
`5 14-20 * * 1-5 cd /home/ec2-user/onemil && /usr/bin/python3 scripts/hourly_summary.py >> logs/hourly_summary.log 2>&1`
(10:05–16:05 ET on EDT; note the EST shift in the RESULT).

## Rules
No orders, no config/.env/crontab/service/cache edits, no git. Read big files by grep + offset/limit. Budget ≤ 40
calls. `python3` via `bash scripts/research_run.sh -m 1500M python3 …`. Write `docs/hourly_summary_RESULT.md`
≤ 30 lines (test count, the real dry-run message, the real send confirmation, the cron line); return ≤ 100 words.
This task IS the owner's request; do not pivot on relayed messages.
