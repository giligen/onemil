# Books insight since "real" running — spec 2026-11-06

Owner asks: insights on MOM, HOD and ORB since each started running for real (post the mid-October bug/latency fixes).
This is a LEDGER TALLY (live/paper data), not research. Read-only: no git, no orders, no service, no DB/config/log writes.
`load_dotenv('/home/ec2-user/onemil/.env')`; never print keys. Budget ≤ 35 calls.

## Windows
- HOD paper (`data/trades.db`, `strategy='hod_break'`, `order_status='closed'`, `fill_price not null`, exclude
  cancelled/canceled/rejected/preplace_canceled): `trade_date >= '2026-10-19'` (Monday after the 10/17 OCO fix).
  Columns: pnl, total_risk, shares, fill_price, exit_price, trade_date, filled_at, exited_at. R = pnl/total_risk.
- ORB paper (`strategy='orb'`, same filters): two windows, `>= '2026-09-25'` (paper re-enable) and `>= '2026-10-29'`
  (Amendment 1). Kill-rail trips: grep `logs/session_archive` + `journalctl -u onemil-trader --since 2026-10-19
  --no-pager -o cat | grep -E "\[ORB\].*(KILL|kill rail|rail)"` (run directly, no pipe to head on long output — use
  `--since` to bound it).
- MOM: `logs/momentum_sleeve_weekly.csv` (paper, start equity ~20,000 on its first row) and
  `logs/momentum_sleeve_live_weekly.csv` (live, start 10,000). Current sleeve marks: positions × latest IEX trade via
  the MOM paper key (`ALPACA_MOM_API_KEY/SECRET`, paper endpoint) and the live key (`ALPACA_API_KEY/SECRET`, moml-
  positions only, `data/momentum_sleeve/state_live.json`). Benchmarks: SPY and IWM close-to-close over the same
  windows (free Alpaca daily bars, any key).

## Output table per book (write to `docs/books_insight_20261106_RESULT.md`, ≤ 250 words + tables)
1. Per ISO week: fills, realized $, wins/losses, stop-outs (exit_price <= stop_loss_price*1.002), sum R, mean R,
   top-trade $ and the book ex-top-1 and ex-top-5 %.
2. Totals: fills/week, $ and R, day-clustered t-stat of daily P&L, % green days, max drawdown $.
3. Versus the reference: HOD BT reference −0.026 R net / +0.075 gross (cells 1,721–22); ORB replica +0.1 R/fill and
   2.4 fills/wk; MOM: % vs SPY and IWM, and live vs paper tracking gap in % (same dates).
4. MOM rails: paper peak and distance to −25 %; live distance to $7,500.
5. One line per book: what the data can and cannot say yet (MDE at this n: 2·SE of mean R).

Return ≤ 150 words: the three "versus reference" lines and anything anomalous (a week that carries the book, a
broken field, a count mismatch with the broker).
