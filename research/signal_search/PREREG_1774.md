# PREREG 1,774 — gap-fade short, MOO entry, 10:30 ET cover at the MEASURED NBBO (child 2 of 1,773)

FROZEN 2026-11-14 before any number. Cell 1,774. Rules: write only `research/signal_search/1774_*`, `RESULT_1774.md`; read-only DB
URIs; `ulimit -v 2500000; nice -n 19`; budget ≤ 60 calls. Network: ONLY the free Alpaca historical quotes endpoint
(`GET https://data.alpaca.markets/v2/stocks/{symbol}/quotes`, feed sip, keys `ALPACA_HOD_API_KEY/SECRET` from .env, never print
them), ≤ 200 requests/min, cache every response in `research/signal_search/1774_cache/quotes/<day>.parquet` (atomic write, resume).
Saturday: no RTH blackout; paid data (Databento) forbidden.

## Why
1,773: MOO short → MOC cover is VAL +80 bp net (t 1.6) with a −21 % worst-5 % tail (afternoon squeezes); the 10:30 cover is GROSS
t 8.6 on the same book (1,767: 82–92 % of the fade is done by 10:30). The 10:30 cover is a TAKER fill → the cost must be the
measured per-trade NBBO at that minute (rule 4; the 1,769 band → 1,770 measured lesson). Entry at the opening auction has no
quoted spread. Same population, same drops (gap > 200 %), same locate / borrow / fee lines as 1,773.

## Data
1773_events.parquet (reuse: open, close, prev close, tiers, filter (c), split, capacity fields) + the 10:30 price from
`1762_cache/bars/<day>.parquet` (the 10:30 bar OPEN = the first print of the cover minute) + NEW: the NBBO at 10:30:00 ET from
Alpaca quotes: fetch quotes for [10:29:30, 10:30:30] ET per symbol-day (limit 1000, feed sip), take the LAST quote at or before
10:30:00; cover price = ASK (buy to cover); half-spread = (ask − bid)/2/mid in bp; drop quotes with half-spread > 1,000 bp or
mid outside 0.5–2× the bar price (state the count; same cleaning as 1,770, declared here, not added later). Coverage rail: a
quote for ≥ 80 % of executable rows and winner/loser missingness gap ≤ 5 pp, else VOID.

## Definitions
r = open(D) / ask_1030 − 1 (bp, positive = paid). Net = r − 4 bp fees − locate (0 / 9 / 38 bp) − borrow (rate 0 / 100 / 300 %,
one day) ; the ask already carries the half-spread. Equal-$ $300, ≤ 10 names/day by gap rank (declared), no stop. R unit as 1,773.

## Reads
PRIMARY: VAL all tiers at locate 9 + borrow 100: mean net, day-t, ex-top-5 % / ex-top-1 % of winners, worst-5 % / worst-1 %,
weekly P10 (R), green-week share vs the count-matched null, median names/day, measured half-spread quantiles at 10:30 by tier.
Secondary (TRAIN first, same sign on both halves required): (a) tiers $1–5 / 5–20 / > 20; (b) gap terciles; (c) the 1,755
filter (prior range ≥ 7 % AND prior-day return ≥ 0) with its count-matched placebo p95; (d) cover at 10:00 and 11:00 (same
quote method, two cells, declared); (e) ask-side cost sensitivity: cover at MID (what a resting bid at mid would earn if filled —
report fill-share unknown, informational only).
Placebos (binding): (i) random universe symbol same days 09:30 open → 10:30 ask-equivalent (use bar open at 10:30 + the
symbol's measured half-spread median for the placebo only, say so), 200 draws p95; (ii) same name +5 trading days, same windows.

## Pass bar
VAL primary net ≥ +50 bp, day-t ≥ 2, TRAIN halves same sign, ex-top-5 % > 0, worst-5 % ≥ −12 %, weekly P10 ≥ −2 R, beats BOTH
placebo p95s, quote coverage rail met, ≥ 3 names/day median. PASS → paper instrument spec (locate at 09:20, MOO short, 10:30
marketable-limit cover at ask + 10 bp, $300/name, ≤ 10 names, HOD paper account), owner GO before any real locate dollar.
FAIL → FACT + CHILDREN ≥ 2.

## Output
`1774_fetch.py` (quotes, resumable, progress every 200 symbol-days), `1774_score.py`, `1774_events.parquet`, `1774_summary.csv`,
`1774_placebo.csv`, `RESULT_1774.md` ≤ 350 words ending "CHILDREN:". Return ≤ 150 words: quote coverage + cleaning count, measured
half-spread median by tier, primary (net, t, ex-top-5 %, worst-5 %, weekly P10, names/day), tiers, both placebo p95s, PASS/FAIL,
children. Independent rebuild follows before any number reaches the owner.
