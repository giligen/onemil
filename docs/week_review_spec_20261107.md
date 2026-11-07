# Week review 2026-11-02..11-06, all books, with market context — spec 2026-11-07

Owner asks for an in-depth but crisp review of the week across every book, against a bad market week. Read-only:
no git, no orders, no service, no writes to data/*.db / config / logs. `load_dotenv('/home/ec2-user/onemil/.env')`,
never print keys or account numbers beyond 4 chars. Budget ≤ 35 calls. Run python as script files.

## Accounts (one book per key except ORB)
- LIVE `ALPACA_API_KEY/SECRET` (`https://api.alpaca.markets`): MOM live only (moml- positions; stupid-money stopped 10/22).
- PAPER MOM `ALPACA_MOM_API_KEY/SECRET`; PAPER HOD `ALPACA_HOD_API_KEY/SECRET`; PAPER ORB `ALPACA_ORB_API_KEY/SECRET`
  (also hosts the tom sleeve `tom-*` and third-party `micha-*` orders — split them out by client_order_id prefix).
  All paper: `https://paper-api.alpaca.markets`.

## 1. Daily P&L per book (table: Mon..Fri + week)
`GET /v2/account/portfolio/history?period=1M&timeframe=1D` per key → equity per session close (timestamps are 20:00 ET
= that session). Day P&L = equity(d) − equity(d−1); week = Fri 11/6 − Fri 10/30. For the ORB key subtract the tom
sleeve: tom fills from `/v2/account/activities/FILL` (client ids `tom-`), Nov cycle sold 11/4 (SPY/QQQ/IWM).
Cross-check each book against `data/trades.db` (`file:...?mode=ro`): per strategy per `trade_date`, fills, realized
`pnl` (`order_status in ('filled','closed')`, `fill_price not null`, `exit_price not null`), wins/losses, stops
(`exit_reason like 'stop%'`). Report any day where DB realized and account delta disagree by > $50 and say why
(open positions marked, partial fills, the CRDU 10/27 unledgered 861 sh is known — not this week).

## 2. Market context (same five sessions, IEX daily bars via any key, close-to-close)
SPY, QQQ, IWM, MTUM (momentum factor ETF), VIXY (vol proxy), and the equal-weight of the MOM sleeve's 20 LONG names
in `data/momentum_sleeve/state_live.json` positions vs its SHORT names if any (qty < 0) — show long leg, short leg,
net. Daily % table Mon..Fri + week. Say which day was the crash day and how each book did that day.

## 3. Per-book anatomy (short)
- MOM paper + live: week % (from the weekly CSVs `logs/momentum_sleeve_weekly.csv` 11/2 row vs Friday mark and
  `logs/momentum_sleeve_live_weekly.csv`), live-vs-paper gap, 3 worst and 3 best names by $ (positions endpoint
  `unrealized_pl`, `change_today` is not enough — use cost basis vs Friday close), distance to the live $7,500 rail and
  the paper −25 %/−40 % lines, and the Saturday BT parity result: `tail -40 logs/mom_bt_parity.log` (does the BT rule
  on the same names/dates show the same week? quote its residual line verbatim).
- HOD paper: fills/day, win rate, stop rate, best/worst trade, $ at risk per trade; vs its reference −0.026 R net.
- ORB paper: fills/day, the 11/3 weekly-kill trip and the rescale on 11/4, days with no entries and why (grep
  `journalctl -u onemil-trader --since 2026-11-02 --until 2026-11-07 --no-pager -o cat | grep -E "\[ORB\].*(kill|veto|no entries|FILLED)"`
  — run directly, bounded by --since/--until, do not pipe through head).
- TOM: Nov cycle P&L from `logs/tom_sleeve_ledger.csv` (entered 10/30, exited 11/4) — one line.

## Output
`docs/week_review_20261107_RESULT.md` ≤ 350 words + the tables. Return ≤ 150 words: the per-book week numbers, the
crash-day line, the MOM parity residual verbatim, and anything that does not reconcile.
