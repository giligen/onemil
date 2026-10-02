# Momentum sleeve - paper, dry run 2026-10-02

## What it does
Weekly risk-adjusted 12-1 momentum (close[t-21]/close[t-252]-1 divided by the 252-day daily-return std), top 20
of US stocks with close >= $10, 20-day ADV >= $200M, >= 273 rows of history, word-boundary name exclusions
(ETF/ETN/fund/trust/warrant/unit/preferred/right, `^Z[A-Z]ZZT$`). Every first trading day of the week ALL 20
names are reset to 1/20 of the sleeve's own equity (state cash + sum qty x price; starts at $20,000). Sells
(fractional qty, market DAY) go first and are polled to fill, then buys (notional, market DAY). Selection is in
`trading/momentum_sleeve.py` (pure; parity test = 20/20 holdings vs build A on 2021-02-08, 2024-06-03,
2026-09-21). Runner: `scripts/momentum_sleeve.py`; default is dry-run; `--submit` needs the PAPER account
(ALPACA_ORB keys) and 09:31-15:30 ET on a session day.

## Cron (main session installs; escape % if ever used)
```
31 13 * * 1,2 cd /home/ec2-user/onemil && nice -n 19 python3 scripts/momentum_sleeve.py --submit >> logs/momentum_sleeve.log 2>&1
31 14 * * 1,2 cd /home/ec2-user/onemil && nice -n 19 python3 scripts/momentum_sleeve.py --submit >> logs/momentum_sleeve.log 2>&1
```
13:31 UTC = 09:31 EDT; the 14:31 line covers EST (Nov-Mar). The wrong-DST line is refused by the 09:31-15:30 ET
window guard. Tuesday covers a Monday holiday: the run no-ops ("not a rebalance session") unless it is the first
session of the week, and a second run the same day exits on `state.last_rebalance`; client_order_ids
(`mom-<YYYYMMDD>-<SYM>-<s|b>`) make a retry idempotent at the broker.

## Files
data/momentum_sleeve/daily_YYYYMMDD.parquet (latest 2 kept), data/momentum_sleeve/state.json,
logs/momentum_sleeve_ledger.csv (fills + slippage vs the day's open), logs/momentum_sleeve_weekly.csv
(equity, cash, turnover, names in/out, SPY close), logs/momentum_sleeve.log. One `[MOM]` Telegram line per run.

## Stop
Remove the cron lines. Positions stay at the broker; to flatten, sell the symbols in state.json by hand (the
paper account is shared - do not touch other strategies' positions).

## Kill rule
Sleeve drawdown > 40 % from its peak equity -> halve the sleeve. The weekly Telegram line prints dd and appends
"KILL RULE" when breached; the halving itself is a manual step (reduce `--equity-start`/state cash by half).

## Known limits
Active tradable assets only (delisted names never entered a live universe anyway). Equity uses the signal-date
close as "last price". Unfilled/timeout orders are logged ERROR and not booked in state.

## Dry run output (`--force`, asof 2026-10-01, nothing submitted)
```
2026-10-02 15:39:28,065 INFO momentum_sleeve: COMPLETENESS: requested 13515 symbols, 13196 with bars, 13191 with an 2026-10-01 bar (98%), LOST 318
COMPLETENESS: requested 13515 symbols, 13196 with bars, 13191 with an 2026-10-01 bar (98%), LOST 318
asof 2026-10-01  today 2026-10-02  mode DRY-RUN
sleeve equity $20,000.00; target per name $1,000.00
TOP:
   1 SNDK   signal 174.762  close 1787.69
   2 AXTI   signal 118.688  close 81.81
   3 TXG    signal 90.213  close 88.76
   4 MU     signal 89.480  close 1097.39
   5 RVMD   signal 83.996  close 205.89
   6 TWST   signal 76.571  close 188.05
   7 LITE   signal 70.427  close 1045.78
   8 ASX    signal 67.715  close 44.73
   9 ATI    signal 56.339  close 191.14
  10 WDC    signal 54.783  close 462.56
  11 STX    signal 52.535  close 945.57
  12 VLO    signal 50.142  close 408.46
  13 FTI    signal 48.820  close 68.85
  14 TGT    signal 46.513  close 156.70
  15 PSX    signal 45.935  close 264.23
  16 MPC    signal 45.857  close 420.15
  17 GH     signal 44.362  close 174.75
  18 MRK    signal 44.147  close 143.81
  19 TRGP   signal 43.853  close 279.31
  20 ROIV   signal 43.812  close 35.50
ORDERS (20), sells first:
  BUY  ASX    $1,000.00
  BUY  ATI    $1,000.00
  BUY  AXTI   $1,000.00
  BUY  FTI    $1,000.00
  BUY  GH     $1,000.00
  BUY  LITE   $1,000.00
  BUY  MPC    $1,000.00
  BUY  MRK    $1,000.00
  BUY  MU     $1,000.00
  BUY  PSX    $1,000.00
  BUY  ROIV   $1,000.00
  BUY  RVMD   $1,000.00
  BUY  SNDK   $1,000.00
  BUY  STX    $1,000.00
  BUY  TGT    $1,000.00
  BUY  TRGP   $1,000.00
  BUY  TWST   $1,000.00
  BUY  TXG    $1,000.00
  BUY  VLO    $1,000.00
  BUY  WDC    $1,000.00
```
