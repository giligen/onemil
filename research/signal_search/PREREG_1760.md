# PREREG 1,760 — G1: the ORB / HOD breakout structure on Alpaca CRYPTO (BTC, ETH, SOL), hourly, 24/7 — Stage 1 rejection-only

FROZEN 2026-11-13 19:35 UTC before any number. Programme cell count: 1,759 → 1,760. Rules: write only
`research/signal_search/1760_*`, `RESULT_1760.md`, cache `1760_cache/`; no git/service/orders; network allowed
ONLY for the Alpaca crypto bars endpoint (free, `https://data.alpaca.markets/v1beta3/crypto/us/bars`, keys from
`.env` — never print them; ~80 K rows total, < 20 MB; ≤ 2 requests/s); `ulimit -v 2500000; nice -n 19`; budget
≤ 50 calls. Parent: F4 (frequency is the gating quantity; crypto trades 168 h/wk, no PDT, no borrow for shorts).
Prior (survey rule): crypto intraday momentum is documented around the US equity open and on weekends; breakout
strategies on BTC decayed after 2021 in most published backtests; fees (Alpaca taker 0.25 %, maker 0.15 % per side)
are the known killer at hourly horizons.

## Data
Hourly bars (open/high/low/close/volume/trade_count/vwap) BTC/USD, ETH/USD, SOL/USD, 2021-01-01 → 2023-12-31
(TRAIN 2021-01 → 2022-12, halves 2021 / 2022; VAL 2023 once); TEST 2024-01 → sealed (never fetched). Store
`1760_cache/bars.parquet`. Report gaps (missing hours) per symbol; > 5 % missing in any split → VOID that symbol.

## Signals (long AND short — crypto shorts have no borrow)
S1 ORB-13:30: the range = the hour 13:00–14:00 UTC (US equity open); break of the range high (long) / low (short)
   in hours 14:00–21:00 UTC; stop at the opposite side of the range; target 2 R; time-out 21:00 UTC → exit at the
   close of that hour.
S2 HOD-24: running high (long) / low (short) of the UTC day, set ≥ 3 hours earlier; break → entry at the break
   bar's close; stop = the prior 3-hour low (high); target 2 R; time-out at the end of the UTC day.
S3 Weekend variant of S2 (Sat–Sun only) vs weekday — the published weekend effect.
Entry at the bar close after the break (obtainable, 1 bar late — say so); exits as stated; R = entry − stop.

## Cost (charged, every read)
Taker 0.25 % per side (primary) and maker 0.15 % (sensitivity) + 5 bp spread; stop slip 10 bp. R per trade ≈ the
range (report the median R in %); cost in R units beside gross.

## Numbers (per signal × side × symbol; TRAIN halves, TRAIN, VAL once)
n/wk, gross and net mean R, t (iid trades; day-clustered), ex-top-5 %, winner-capped +3 R, weekly P10, green-week
share, month signs, MDE at 80 % power.

## Kill (primary = S2 long, BTC + ETH pooled, taker cost)
VAL net ≤ 0 or t < 2; TRAIN halves opposite sign; VAL ex-top-5 % ≤ 0; < 3 trades/wk pooled; maker-cost read not
materially different in sign. Survive → TEST once → independent rebuild → PAPER sleeve on the Alpaca crypto paper
endpoint (24/7, $50 R, 2 names).

## Output
`1760_fetch.py`, `1760_score.py`, `1760_summary.csv`, `RESULT_1760.md` ≤ 350 words ending with "CHILDREN:" (≥ 2).
Return ≤ 120 words: coverage/gaps, median R %, cost in R, per signal the VAL net / t / trades-wk, long vs short,
weekend vs weekday, KILL/SURVIVE, children.
