# PREREG 1,747 — dividend ex-day / dividend-month price pressure (candidate G3) — Stage 1, rejection-only

FROZEN 2026-11-13 12:25 UTC before any number. Programme cell count: 1,746 → 1,747. Rules as PREREG_1746 (write
only `research/signal_search/1747_*`, `RESULT_1747.md`, cache `1747_cache/`; no git / service / orders; budget ≤ 60
calls; never print keys). This task IS the owner's request; do not pivot.

## Claim under test (Hartzmark–Solomon JFE 2013 "The dividend month premium"; "Predictable price pressure" NBER
w30688 — re-read the G3 block of CANDIDATES_20261113.md and record the URLs in RESULT)
Names in their predicted dividend month earn +37 bps/month abnormal vs the same names in other months; the ex-day
itself +26 bps; announcement→ex +17 bps; −72 bps over the 40 days after the ex-day. Mechanism: dividend-seeking
demand before the ex-date, reinvestment flow after payment. Liquid payers → round trip 5–10 bps.

## Data (free; state any that turns out not to be)
1. Dividend events: Alpaca corporate actions `GET /v2/corporate_actions?types=cash_dividend&start=&end=` (ex_date,
   record_date, payable_date, rate, symbol) — probe its history depth FIRST (report the earliest ex_date returned);
   if it does not reach 2016, fall back to EDGAR-free sources only if they are free and documented, else shorten
   TRAIN and say so. Regular cash dividends only (rate > 0, exclude specials > 3 × the trailing regular rate).
2. Daily bars: Alpaca `/v2/stocks/bars?timeframe=1Day&adjustment=split` (NOT dividend-adjusted — the ex-day drop
   must be in the price) for every payer, 2015-12 → 2026-11; PIT bars for the sealed window as in PREREG_1746.
3. Universe: common stocks, 20-day median dollar volume ≥ $20 M at t−1, price ≥ $5, non-ETF/fund (same filters
   as 1,746). Report the $5–20 M tier separately.

## Books (gross, equal weight, each a pre-named read)
B1 ex-day: buy at the close of ex−1, sell at the close of ex-day; return measured WITH the dividend added back
(close_ex + dividend − close_{ex−1}) / close_{ex−1}. The paper's +26 bps.
B2 run-up: buy at the close of (announcement day) or, if the announcement date is unavailable, ex−10 (state which),
sell at the close of ex−1. The paper's +17 bps.
B3 post-ex short: short at the close of ex-day, cover at the close of ex+40 (the paper's −72 bps; report the long
side too — a short book needs ETB, report the ETB share from the live assets file).
Primary = B1 (highest frequency, cleanest). Each read vs a matched control: the same name on the same calendar
day ± 1 month (no dividend event within 20 days), and vs the universe mean.

## Splits
TRAIN 2016-01 → 2020-12 (halves 2016–18 / 2019–20); VAL 2021-01 → 2023-12 (2022 inside); TEST 2024-01 → 2026-11
sealed (read once only if TRAIN and VAL survive).

## Numbers (per split, per book)
n events, events/week, mean gross excess return vs control, event-clustered and day-clustered t, ex-top-5 % of
events, the share of events with |return| > 5 % (tail dependence), monthly green share, year-by-year sign, cost
from the measured NBBO half-spread at the 15:55–16:00 minute on a 400-event sample (+ 2 bp per leg; MOC fills
pay no spread but report both), net, MDE at 80 % power.

## Kill (pre-committed — any one kills the primary; B2/B3 reported, not promoted)
VAL B1 gross excess ≤ +10 bps or |t| < 2; TRAIN halves opposite sign; ex-top-5 % ≤ 0 on VAL; 2022 negative;
net of cost ≤ 0 on VAL; < 3 events/week in the ≥ $20 M tier. Survive → TEST once on PIT bars → independent
rebuild → paper-sleeve spec (long at ex−1 close, MOC exit ex-day, 8 names/day, $150 R).

## Output
`1747_fetch.py`, `1747_events.parquet`, `1747_score.py`, `1747_summary.csv`, `RESULT_1747.md` ≤ 350 words.
Return ≤ 120 words: data depth, n events/yr, B1 VAL gross / t / net, year signs, B2/B3 one line each, KILL/SURVIVE.
