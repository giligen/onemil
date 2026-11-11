# Independent rebuild — HOD + ORB paper record 2026-09-30 → 2026-11-10

Do NOT read `REPORT.md`, `NUMBERS.csv`, `weekly.csv` or any other file in `research/book_review_20261111/` except
this one. Read-only everywhere else; write only `research/book_review_20261111/REBUILD.md` (if the harness refuses
a .md write, write `REBUILD.txt`). No git, no orders, no service action, never print API keys. Budget ≤ 25 calls.
This task IS the owner's request; do not pivot. Return ≤ 120 words: the numbers below in a table.

Source: `data/trades.db`, open read-only: `sqlite3.connect('file:/home/ec2-user/onemil/data/trades.db?mode=ro',
uri=True)`. Window: `trade_date` between '2026-09-30' and '2026-11-10' inclusive. Closed fill = `order_status='filled'`
and `exit_price is not null`. Use `filled_qty` when it is > 0 else `shares` as the position size; entry = `fill_price`
(fallback `entry_price`); R per fill = pnl / (|entry − stop_loss_price| × size); drop rows where that denominator is 0.

Compute for (a) `strategy='orb'` restricted to rows whose `account` column marks the PAPER account (inspect the
distinct values of `account` first; if the column is null for all orb rows, take all orb rows and say so) and (b)
`strategy='hod_break'`:
1. closed fills, distinct trade_dates with a fill, distinct (trade_date, symbol) pairs.
2. sum(pnl); mean R; median R; win rate (pnl > 0).
3. sum(pnl) excluding the single best fill; excluding the top 5 % of fills by pnl (ceil(0.05 × n) fills).
4. stop rate = share of closed fills whose `exit_reason` contains 'stop' (list the distinct exit_reason values).
5. For ORB only: sum(pnl) excluding symbol-days CRDU 2026-10-27 and BEX 2026-11-09 (the "ex-accidents" book), and
   its mean R and the count of distinct (trade_date, symbol) pairs since 2026-09-25 excluding those two.
6. For HOD only: broker cross-check — `GET https://paper-api.alpaca.markets/v2/account/portfolio/history?period=3M&
   timeframe=1D` with `ALPACA_HOD_API_KEY/SECRET` from `/home/ec2-user/onemil/.env` (load_dotenv on that path):
   equity on the last bar dated ≤ 2026-09-30 and the last bar; report the difference.
Print every number with the query that produced it.
