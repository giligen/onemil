# PREREG 1,773 — the gap-fade SHORT re-opened by Alpaca HTB locates: auction-to-auction (MOO short → MOC cover), NO stop, diversified, full price range incl. $1–5

FROZEN 2026-11-14 (Saturday) before any number. Programme cell 1,773. Rules: write only `research/signal_search/1773_*`,
`RESULT_1773.md`; no git / service / orders / network (all data on disk); read-only DB URIs; `ulimit -v 2500000; nice -n 19`;
budget ≤ 50 calls. Saturday: no RTH blackout.

## Why now (new fact, 2026-11-14)
Alpaca launched HTB short selling with API locates (2026-06-24; `GET /v1/locates/quotes` advisory, `POST /v1/locates`
reserves). Read-only sweep today on the 300 most recent population gappers (`scratchpad/locate_quotes_20261114.csv`, copied to
`1773_locate_quotes_20261114.csv`): 148 easy-to-borrow (fee 0), 116 HTB quoted — locate fee median 8.6 bp of price, p25 2.8,
p75 38, p90 113 — 23 threshold-list (cannot locate), 13 not found. Every prior short cell assumed "$1–5 = unborrowable" and
used px ≥ $5 (1,755, 1,767 read 4, 1,768, 1,769/1,770). The ≥ $5 book is dead at zero spread (+7 bp, 1,770). The $1–5 tier
was NEVER fetched (1,769 caveat) although 1,754/1,767 put the biggest fade there (population open→15:55 −431/−437 bp, t −7).
Second new design element: every prior short died on the STOP (stop-hit 19–48 %, worst-5 % −18 %, "the stop costs 38–65
bp and does not fix the tail"). This cell has no stop: tail risk is carried by small equal-$ positions across many names
and a portfolio rule, which is how the fade's mean can be collected if it exists.

## Data (on disk)
`research/hod_entry/cell_1428_population.csv` (7,487 PIT gapper symbol-days, TRAIN 2025 / VAL 2026); `data/cache.db`
`daily_bars(symbol, bar_date, open, high, low, close, volume)` for open(D), close(D), prev close, prior-day range/return;
`research/hod_entry/1762_cache/bars/<day>.parquet` for the 09:30 and 16:00 bar volumes (auction capacity) and the 10:30 price
(gross-only secondary). SPY from daily_bars for the same-day excess.

## Definitions
Short at the opening auction: entry = open(D). Cover at the closing auction: exit = close(D). r = open/close − 1 (bp, positive
= the fade paid). Auction fills carry no quoted spread (CLAUDE.md rule 4); fees 2 bp/leg. Borrow: HTB daily fee = notional ×
rate/360 for one day — sensitivity at rate 0 / 100 % / 300 % (28 / 83 bp per day). Locate fee sensitivity 0 / 9 / 38 / 113 bp
(today's quantiles; the true same-day fee is unknown historically — the forward ledger `scripts/locate_ledger.py` measures it
from Monday). Threshold-list haircut: drop every symbol in today's threshold set (`1773_locate_quotes_20261114.csv` errors)
as a proxy, report n dropped. Capacity: $300 position ≤ 2 % of the 09:30-bar $vol AND ≤ 2 % of the 16:00-bar $vol.
Availability rail ≥ 80 %, missingness gap ≤ 5 pp (winner = day fade > 0), else VOID. SSR note: a name that trades ≤ −10 % vs
prev close intraday becomes SSR for the NEXT day only; irrelevant to a same-day MOO short — state it.

## Reads (1 primary + 5 secondary, declared)
PRIMARY: equal-$ book, all tiers, no stop: mean r net (fees + locate 9 bp + borrow 100 %), day-clustered t on VAL, ex-top-5 %
and ex-top-1 % of WINNERS, worst-1 % / worst-5 % trade, winner-capped at +50 %, weekly series → weekly P10, green-week share vs
the count-matched null, cadence bar (`scripts/cadence_bar.py` if importable, else the spec's rules by hand) on R = $300
position × r / $150 unit risk… state the R unit used.
Secondary: (a) price tier $1–5 / 5–20 / > 20 (the tier question); (b) gap tercile; (c) the ONE pre-declared causal filter from
1,755 — prior-day range ≥ 7 % AND prior-day return ≥ 0 — with the count-matched placebo p95 (binding lesson 11/13);
(d) TRAIN halves H1/H2 sign; (e) 10:30 cover GROSS only (fact, taker cost not measured here).
Placebos: (i) short a count-matched random symbol from the daily_bars universe (close ≥ $1, $vol ≥ $1 M) on the same days,
200 draws → p95; (ii) the same name 5 trading days later.

## Pass bar (pre-committed)
VAL primary net ≥ +50 bp/trade at locate 9 bp + borrow 100 %, day-t ≥ 2, same sign in TRAIN H1 and H2, ex-top-5 % of winners
> 0, worst-5 % mean ≥ −12 % (no-stop tolerance, equal-$ $300), weekly P10 ≥ −2 R, beats both placebo p95s, capacity share ≥
60 %, ≥ 3 names/day median. PASS → paper instrument spec: locate at 09:20 ET (POST, real money ≤ $5/day locate budget — owner
GO first), MOO short, MOC cover, $300/name, ≤ 10 names/day, HOD paper account first, cadence-bar scored.
FAIL → FACT; CHILDREN ≥ 2.

## Output
`1773_score.py`, `1773_events.parquet`, `1773_summary.csv`, `1773_placebo.csv`, `RESULT_1773.md` ≤ 350 words ending "CHILDREN:".
Return ≤ 150 words: coverage, n, primary (net, t, ex-top-5 %, worst-5 %, weekly P10), tier split, filter (c) vs placebo p95,
PASS/FAIL, children. Independent rebuild from this prose follows before any number reaches the owner.
