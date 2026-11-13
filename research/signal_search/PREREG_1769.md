# PREREG 1,769 — J8: short the ≥ 10 % gapper at 09:35, stop = the first-5-minute high, cover 10:30 — PRE-REGISTERED primary on 2016–2023 minute bars (child of 1,768)

FROZEN 2026-11-13 21:10 UTC before any number. Programme cell count: 1,768 → 1,769. Parent: 1,768 found E1 at
VAL +142 bp (day-t 4.7, 16.5 names/wk, stop-hit 34 %, worst-5 % −5 %) on 2026 only, as a NON-primary read among
8 cells, with 2025 as the only training year. This cell makes E1 the single primary on eight years. Rules: write
only `research/signal_search/1769_*`, `1769_cache/`, `RESULT_1769.md`; no git/service/orders; network ONLY for
Alpaca minute bars 09:30–10:30 ET (SIP, raw, ≤ 5 req/s, 50 symbols per request, per-day checkpoints); `ulimit -v
2500000; nice -n 19`, peak RSS < 1 GB; budget ≤ 60 calls. TEST 2024 → sealed (never fetched).

## Population (daily bars on disk, built exactly as 1,754)
Universe ranks 1–3,000 at t−1, gap = open_t / close_{t−1} − 1 ≥ 10 %, RAW open ≥ $5 (borrowable proxy — say so),
attention yesterday as 1,754 (prior-day volume ≥ 1.5× its 20-day mean OR prior-day range ≥ 5 %). Report the
event count per year and the fetch coverage (≥ 90 % or VOID); the share of events whose 09:30–09:35 bars are
missing (gap-through / halted at the open) is a bucket of its own.

## The ONE rule (E1, frozen)
Short at the 09:35 bar close; stop = max high of 09:30–09:35 + 50 bp slip (filled at the stop when any later
bar's high ≥ stop — minute bars cannot time intrabar; say so); cover at the 10:30 bar close. Cost 2 bp per leg +
the measured half-spread (1748_cost.csv price-band table; say which) + borrow 50 bp/day (100 bp sensitivity).
Secondary reads (reported, not searched): cover at 10:00 and 15:55; gap 5–10 %; open $1–5 (unborrowable — the
fade's home, for the record only).

## Numbers (TRAIN 2016–20 halves 2016–18 / 2019–20, TRAIN, VAL 2021–23 once)
n/wk, mean net bp, day-clustered t, hit rate, ex-top-5 % of winners, worst-1 % and worst-5 % day means, max
single-trade loss, stop-hit share, weekly P10, green-week share vs count-matched null, year signs (8), MDE at
80 % power, net at borrow 100 bp.

## Kill (primary only; any one kills)
VAL net ≤ +50 bp or day-t < 2; TRAIN halves opposite sign; VAL ex-top-5 % of winners ≤ 0; worst-5 % day mean
< −8 %; 2022 negative; < 3 names/wk; net at borrow 100 bp ≤ 0. Survive → TEST 2024–26 once → independent rebuild
from this prose → owner short-list line (report-only, 10:30 list carries "faded-open shorts" with the stop) —
the engine never shorts.

## Output
`1769_fetch.py`, `1769_score.py`, `1769_events.parquet`, `1769_summary.csv`, `RESULT_1769.md` ≤ 350 words ending
with "CHILDREN:" (≥ 2). Return ≤ 120 words: events/yr and coverage, VAL net / day-t / n-wk / hit / stop-hit /
worst-5 % / ex-top-5 % / 2022, TRAIN halves, borrow-100 net, KILL/SURVIVE, children.
