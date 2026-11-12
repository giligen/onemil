# PREREG 1,738 — calls instead of shares on the HOD break: Stage 0 availability gate, Stage 1 paired book (owner 2026-11-12 "do all tonight")

FROZEN 2026-11-12 19:55 UTC before any number. Programme count on the HOD line: 1,737 → 1,738. Rules: no git, no
service action, NO ORDERS (the options endpoints used are read-only: contracts + historical bars), read-only on every
DB/CSV; write only `research/hod_entry/1738_*` and `RESULT_1738.md`; never Read a file > 300 lines in full; `nice -n
19`; the data API is rate-limited — ≤ 200 requests/min, back off on 429, cache every response to
`research/hod_entry/1738_cache/` (gitignored path under research/, JSON per request) so a re-run costs nothing.
Budget ≤ 70 calls. This task IS the owner's request; do not pivot.

## Why
The HOD long pays its edge away in stop slip and the chase. A call has no stop (max loss = premium), the exit is a
sale at a price, and leverage replaces size. Both accounts are approved for options level 3 (verified 2026-11-12,
read-only). What is unknown: whether the names in this book HAVE liquid enough options for a 1-min entry and exit.
Verified today (read-only probes): `GET https://paper-api.alpaca.markets/v2/options/contracts?underlying_symbols=X&
expiration_date_gte=…&expiration_date_lte=…&type=call&status=inactive` returns expired contracts;
`GET https://data.alpaca.markets/v1beta1/options/bars?symbols=<contract>&timeframe=1Min&start=…&end=…` returns
1-minute TRADE bars for them (FTAI250117C00170000 on 2025-01-10: prints at 14:38, 14:41, 14:49 … UTC, v 1–8);
historical options QUOTES are NOT available (404). Therefore every fill below is a PRINT — a price someone traded —
and the spread is unknown: it is modelled conservatively and the result is reported at two spread assumptions.

## Population
`research/hod_exit_lab/signals.parquet` + `b0_trades.csv` (TRAIN 7,390 / VAL 4,745; TEST sealed): every signal has
day, symbol, entry_m, entry, stop, target, R, exit_m, exit_price, why, net_R. Stage 0 runs on a RANDOM sample of
400 signals (200 TRAIN, 200 VAL; seed 1738) — enough to measure coverage to ± 5 pp.

## Contract rule (per signal, no look-ahead)
Expiration: the nearest expiry with DTE ≥ 2 and ≤ 14 calendar days from `day` (contracts listed via the endpoint
above with `expiration_date_gte = day + 2`, `_lte = day + 14`, `status=inactive` or `active` for recent days — the
listing is as of today; a contract that existed then but is absent now is a disclosed miss). Strike: the lowest call
strike ≥ `entry` (first OTM or ATM). One contract per signal.

## Stage 0 — availability gate (the rail: ≥ 80 % coverage, ≤ 5 pp winner/loser missingness gap, else VOID)
For each sampled signal: `has_contract`; `entry_print` = the first 1-min trade bar of the contract with t ≥ the
signal's entry minute and t ≤ entry + 5 min (its `vw`); `exit_print` = the last trade bar with t ≤ the B0 exit minute
and t ≥ exit − 15 min (fallback: the first bar after the exit minute within 15 min, flagged). Coverage = share of
signals with contract AND both prints. Report coverage per read, by stock price bucket (20–50 / 50–100 / 100+), by
B0 outcome (winner net_R > 0 vs loser) — the missingness gap. If coverage < 80 % or the gap > 5 pp: write the Stage 0
table, verdict **VOID (availability)**, and STOP — no Stage 1 number, no fill in the gaps.

## Stage 1 — only if Stage 0 passes, on the FULL TRAIN + VAL population (the cache makes re-runs free)
Call P&L per signal = (exit_print − entry_print) / entry_print, with the spread model: A (10 % of premium round-trip)
and B (20 %). Sizing: premium at risk = 1 R of the stock trade (so a total loss = −1 R, same risk unit as the stock
book): call R = pct_pnl × (1 R) / 1 → net_R_call = pct_pnl − spread; compare PAIRED to the stock's B0 net_R on the
same signal (ΔR = call − stock). Exits identical in time to B0 (stop/target/15:55 minute). Also report an
"exit-at-15:55 only, no stop" option variant (max loss = premium): the no-stop book the stock cannot have.
Scoring as PREREG_1735: paired ΔR mean, day-clustered t, ex-top-5 % of ΔR, winner-capped, own mean net R / day-t /
ex-top-5 %, green-week share, weekly P10, per read; placebo = the same contract bought at a random print AFTER the
signal window on the same day (300 draws, VAL).

## Pass bar (pre-committed)
Stage 1 PASSES only if, under spread model B: paired ΔR ≥ +0.10 R with day-t ≥ 2 on TRAIN AND VAL; own VAL mean net
R ≥ +0.10 with day-t ≥ 2; ex-top-5 % ≥ 0 on both; placebo ≥ 95; coverage from Stage 0 ≥ 80 %. A pass = a PAPER options
instrument spec, never live.

## Output
`research/hod_entry/1738_fetch.py` (contracts + bars with cache), `1738_score.py`, `1738_stage0.csv` (one row per
sampled signal with the fields above), `1738_per_signal.csv` (Stage 1, if run), `RESULT_1738.md` ≤ 350 words: the
Stage 0 coverage tables, VOID or the Stage 1 tables with PASS/FAIL, request counts, caveats (prints not quotes,
listing as of today, spread model). Return ≤ 120 words: coverage per read and bucket, VOID or the Stage 1 verdict.

## Independent rebuild (only if Stage 1 runs)
A second agent recomputes contract choice + entry/exit prints for the Stage 0 sample from the cache directory and this
prose; compare on (day, symbol): contract symbol identical, prints to 1e-6; `REBUILD_1738.md`.
