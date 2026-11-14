# Independent rebuild of cell 1,776's PRIMARY (rule: no research number reaches the owner without it)

You have NOT seen 1774_score.py / 1776_score.py and must not open them. Build from prose only. Write `1776R_build.py`,
`1776R_events.parquet`, `1776R_compare.md`. Budget ≤ 50 calls; read-only DB URIs; nice 19; HOD paper keys from .env with
`load_dotenv(override=True)`, never printed; Alpaca sip quotes cached under `research/signal_search/1776R_cache/` (resumable,
≤ 200 req/min, background run + one until-loop poll).

## Spec (prose)
Population: `research/hod_entry/cell_1428_population.csv` (day, symbol, adv20, split: TRAIN = 2025, VAL = 2026). Drop rows whose
gap (daily_bars open / previous daily_bars close − 1) exceeds 200 %. For each day take the TOP-10 rows by gap. Liquidity L1 =
pre-market dollar volume 04:00–09:25 ET on day D from `research/hod_entry/1762_cache/premarket/<day>.parquet` (sum of close × volume
per symbol over bars ending ≤ 09:25; rows with no premarket data → excluded from terciles and counted). Terciles of L1 computed on
TRAIN rows only, applied to VAL. The book = L1 top tercile.
Trade: short at the opening auction = `daily_bars.open` of day D (`data/cache.db`, table daily_bars, columns symbol, bar_date,
open, close). Cover at 10:30:00 ET at the ASK: fetch Alpaca historical quotes (feed sip) for [10:29:30, 10:30:30] ET, take the last
quote at or before 10:30:00; drop a quote if half-spread > 1,000 bp of mid or mid outside 0.5–2× the day's daily_bars open (count
them). r_bp = (open / ask − 1) × 1e4. Net = r_bp − 4 (fees) − 9 (locate) − 27.8 (borrow 100 %/360 for one day).
Stats on VAL: n, mean net, day-clustered t (cluster by day), ex-top-5 % of WINNERS (drop the top 5 % of trades by net), worst-5 %
trade mean (in %), weekly P10 in R where R = $300 × net / $150 with ≤ 10 names/day, median names/day. Same on TRAIN and on TRAIN
halves (2025H1 / 2025H2).
Placebo (ii), properly sized: for EVERY VAL book trade, the same symbol 5 trading days later (next 5th trading day present for
SPY in daily_bars), same open → 10:30 ask computation with a fresh quote fetch; report n, mean, and p95 of 200 count-matched
bootstrap means. Placebo (i): other population symbol-days on the same days in the same L1 tercile not in the book, same pricing
(fetch), 200 draws, p95.
## Compare
Join your VAL book trades to `research/signal_search/1776_events.parquet` on (day, symbol) — read ONLY its day/symbol/net
columns AFTER your numbers are final — and report: n matched / only-yours / only-theirs, mean |net diff| on matched, your primary
line vs theirs (their claim: VAL n 342, +159 bp net, day-t 2.87, ex-top-5 % +17, worst-5 % −17 %, weekly P10 −0.51 R, ~3.4
names/day; TRAIN +312, halves +445/+180). Return ≤ 150 words: your primary line, the match counts, the two placebo p95s with n,
and whether the claim is CONFIRMED (every number within 15 % / 0.3 t) or NOT, with the largest discrepancy named.
