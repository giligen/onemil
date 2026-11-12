# PREREG 1,741 — option-flow before the break as a HOD-break filter (idea #42 / F4: dealer gamma, call crowding)

FROZEN 2026-11-12 23:15 UTC before any number. Programme count on the HOD line: 1,740 → 1,741. Rules as PREREG_1736
(no git / service / orders; read-only everywhere; never print keys; write only `research/hod_entry/1741_*` and
`RESULT_1741.md`; `nice -n 19`; never Read a file > 300 lines in full; budget ≤ 60 calls per agent). This task IS
the owner's request ("run it"); do not pivot. Two agents: (A) builds and STARTS the fetch in the background and
returns; (B) scores after the fetch log says DONE.

## Why
Historical per-strike open interest is not free, so dealer gamma (GEX) cannot be rebuilt. What IS free and verified
(cell 1,738): Alpaca 1-min option TRADE bars back to 2025-01 incl. expired contracts, chains via
`GET paper-api.alpaca.markets/v2/options/contracts` (`status=inactive` for expired). Option trades before the break
are the causal shadow of dealer positioning and of call crowding (F4). Cost of this cell: API calls only.

## Population
`research/hod_exit_lab/signals.parquet` TRAIN + VAL (12,135 signals; TEST sealed) joined to `b0_trades.csv` net_R
(the live rule's outcome at the true level). Fetch is per (symbol, day), de-duplicated.

## Fetch (agent A: `1741_fetch.py`, cache `research/hod_entry/1741_cache/`, atomic writes, resume, ≤ 180 req/min,
`df` ≥ 5 GB before starting, `nohup nice -n 19 … > research/hod_entry/1741_fetch.log &`, prints a progress line
every 200 symbol-days and `DONE n_symbol_days=… n_requests=…` at the end; reuse `1738_fetch.py` helpers)
1. Chain: contracts on the underlying with expiration in (day, day+45], strike within ±20 % of `entry`, both types.
   None listed → `covered = False`, no further requests (expected ≈ 44 % of symbol-days, as 1,738).
2. Bars: 1-min option bars for those contracts from 09:30 ET to the minute BEFORE the break bar (`entry_m − 1`, as
   in 1,740: the break bar is entry_m − 1 of the stored entry minute — state the convention used), and the PRIOR
   trading day's full session (09:30–16:00) for the same contracts (one request set).
3. Per (symbol, day) write: covered, n_contracts, call_vol, put_vol (contracts), call_notional, put_notional
   (Σ v·c·100), n_prints, prior_day_vol (contracts, full session), first_print_minute.

## Features (computed by agent B, strictly pre-break)
F1 call share = call_vol / (call_vol + put_vol) (NaN if both 0). F2 relative activity = (call_vol + put_vol) /
minutes_elapsed ÷ (prior_day_vol / 390) (NaN if prior_day_vol = 0). F3 covered-with-print = any print before the
break (yes/no). F4 call_notional / (call_notional + put_notional).

## Availability rail (first number reported)
Coverage = share of signals with ≥ 1 option print before the break, per split and by B0 winner/loser; the
winner/loser coverage gap. Coverage < 15 % → VOID (the filter cannot carry a book); gap > 5 pp → VOID.

## Scoring (TRAIN H1, H2, TRAIN, VAL; TEST sealed)
On the covered set: tercile cuts of F1, F2, F4 from TRAIN; B0 net R mean per tercile, top − bottom spread, day-
clustered t, iid t, n, ex-top-5 %, fills/week in the top tercile; permutation null (300 shuffles of the feature within
day) → percentile of the observed spread. Also the F3 split: covered vs not-covered B0 net R (the "optionable"
filter itself), day-t. Primary (pre-named): F1 top tercile. Others exploratory (5 reads → Bonferroni noted).

## Pass bar (pre-committed, primary only)
F1 top − bottom ≥ +0.15 R with day-t ≥ 2.0 on TRAIN AND VAL, same sign both TRAIN halves, top-tercile mean net R
≥ 0 on VAL, ex-top-5 % of the spread ≥ 0, permutation percentile ≥ 95 on VAL, ≥ 3 fills/week in the top tercile.
Pass → sealed TEST read ONCE with the identical code; then an independent rebuild. Fail → stop; TEST untouched.

## Output
`1741_fetch.py`, `1741_flow.csv` (one row per symbol-day), `1741_score.py`, `1741_summary.csv`, `RESULT_1741.md`
≤ 350 words (rail, F3 split, tercile tables, PASS/FAIL, caveats: trade bars not quotes, chain as listed today for
expiries, no OI). Agent A returns ≤ 80 words (started, PID, expected requests, ETA). Agent B returns ≤ 120 words.
