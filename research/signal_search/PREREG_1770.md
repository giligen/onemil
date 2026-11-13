# PREREG 1,770 — J8 cost audit: the SAME 1,769 trades re-charged with MEASURED NBBO at 09:35 (entry), 10:30 (cover) and the stop minute — never a band (CLAUDE.md check 4)

FROZEN 2026-11-13 21:20 UTC before any number. Programme cell count: 1,769 → 1,770. Parent: 1,769 (VAL gross
+61 bp, band cost ≈ 117 bp → net −57; the agent flagged the 1748 price-band half-spread as likely overstated at
09:35). Rules: write only `research/signal_search/1770_*`, `1770_cache/`, `RESULT_1770.md`; no git/service/orders;
network ONLY for Alpaca historical quotes (`/v2/stocks/<symbol>/quotes` or the multi-symbol `/v2/stocks/quotes`,
feed=sip, a 60-second window around each minute, limit small; ≤ 5 req/s; checkpoint per day into `1770_cache/`);
`ulimit -v 2500000; nice -n 19`, peak RSS < 1 GB; budget ≤ 60 calls. TEST sealed (no 2024+ quotes fetched).

## Sample (frozen)
ALL VAL 2021–23 events of `1769_events.parquet` (2,498) plus a 1,000-event random sample of TRAIN (seed 1770) —
the quotes at 09:35:00–09:35:59 (entry: the LAST quote before the 09:35 bar close → sell at the BID), at
10:30:00–10:30:59 (cover: buy at the ASK), and, for stop-hit trades, the first quote at or after the first minute
whose high ≥ stop (cover at the ASK, plus the measured drift from the stop level to that ask as the slip).
Coverage ≥ 90 % of events with all required quotes, else VOID; missingness gap winners vs losers ≤ 5 pp.

## Numbers (VAL primary; TRAIN sample beside it)
Measured half-spread at 09:35 and 10:30 (median, mean, by price tier) vs the band used in 1,769; per-trade net
with measured costs + 2 bp per leg + borrow 50 bp/day (100 bp sensitivity); n/wk, mean net bp, day-clustered t,
hit, stop-hit, worst-5 %, ex-top-5 % of winners, 2022 sign, weekly P10, MDE. The 1,769 band-cost net beside it,
trade by trade (paired Δ).

## Kill (any one — this is the FINAL read on J8)
VAL measured-cost net ≤ +50 bp or day-t < 2; ex-top-5 % of winners ≤ 0; worst-5 % day mean < −8 %; 2022 negative;
net at borrow 100 bp ≤ 0; TRAIN sample opposite sign. Survive → TEST once → independent rebuild → owner short-list
line (report-only).

## Output
`1770_fetch_quotes.py`, `1770_score.py`, `1770_quotes.parquet`, `1770_summary.csv`, `RESULT_1770.md` ≤ 350
words ending with "CHILDREN:" (≥ 2). Return ≤ 120 words: quote coverage, measured half-spread 09:35 / 10:30 vs
the band, VAL measured-cost net / day-t / stop slip / worst-5 % / 2022, TRAIN sample sign, KILL/SURVIVE, children.
