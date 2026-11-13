# PREREG 1,748 — no-news extreme-loser reversal BELOW the liquid top-1000 (candidates D4 + D5) — Stage 1, rejection-only

FROZEN 2026-11-13 12:30 UTC before any number. Programme cell count: 1,747 → 1,748. Rules as PREREG_1746 (write
only `research/signal_search/1748_*`, `RESULT_1748.md`, cache `1748_cache/`; no git / service / orders; budget ≤ 60
calls; never print keys). This task IS the owner's request; do not pivot.

## Claim under test (Chan JFE 2003; Tetlock RFS 2011; Nagel RFS 2012 — URLs in the D4/D5 blocks of
CANDIDATES_20261113.md, record them in RESULT)
Extreme one-day losers WITHOUT news reverse (an uninformed liquidity shock), losers WITH news drift; the reversal is
larger in small illiquid names and when VIX is high (the price of liquidity provision). Distinct from the closed
cell 1,717–1,719 (UNCONDITIONAL reversal on the liquid top-1000): here conditioned on no news, ranks 1,001–3,000.

## Data (free)
1. Daily bars: Alpaca `/v2/stocks/bars?timeframe=1Day&adjustment=all` for the universe 2015-12 → 2026-11; PIT bars
   for the sealed window as in PREREG_1746. Survivorship caveat as 1,746 (rejection-only).
2. News: Alpaca `GET /v1beta1/news?symbols=SYM&start=&end=&limit=50` (Benzinga history; probe its depth FIRST and
   report the earliest item date). A name-day is NEWS if any item tagged to the symbol has created_at in
   [16:00 ET of t−1, 16:00 ET of t]; else NO-NEWS. Cache every query. Report the share of events with ≥ 1 item.
3. Universe at t−1 (point-in-time from the bars): rank 1,001–3,000 by 20-day median dollar volume among common
   stocks (filters as 1,746), price ≥ $3, 20-day median dollar volume ≥ $2 M.
4. VIX close from `data/cache.db daily_bars` if present, else Alpaca `VIXY`/`^VIX` proxy — state which; terciles
   over TRAIN.

## Event and book
Event: close-to-close return on day t ≤ −8 % (secondary: the sector-ETF-residual version, residual ≤ −8 %).
Entry at the OPEN of t+1 (auction), exits at the CLOSE of t+1 (primary) and the CLOSE of t+3 (secondary). GROSS.
Placebo: the same name on a random non-event day drawn AFTER the event window (`feedback_placebo_window_selection`).
Reads: NO-NEWS leg, NEWS leg, NO-NEWS − NEWS, by VIX tercile, by rank bucket (1,001–2,000 / 2,001–3,000).

## Splits
TRAIN 2016-01 → 2020-12 (halves 2016–18 / 2019–20); VAL 2021-01 → 2023-12; TEST 2024-01 → 2026-11 sealed.

## Numbers (per split)
n events/yr and per week, mean gross return per leg and the difference, day-clustered t, placebo percentile,
ex-top-5 %, winner-capped +10 %, the share of events that gap further down at the open (obtainability of the
auction fill), the R of a 1-ATR stop as % of price, cost from the measured NBBO half-spread at 09:30–09:35 on a
400-event sample (+ 2 bp per leg), net, MDE at 80 % power (state it beside the kill threshold).

## Kill (pre-committed — any one kills)
VAL NO-NEWS leg gross ≤ +40 bps (t+1 close) or |t| < 2; NO-NEWS not above NEWS on VAL; high-VIX tercile not the
largest on TRAIN and VAL (mechanism fails); TRAIN halves opposite sign; ex-top-5 % ≤ 0 on VAL; net of measured
cost ≤ 0 on VAL; < 3 events/week. Survive → TEST once on PIT bars → independent rebuild → paper-sleeve spec.

## Output
`1748_fetch.py`, `1748_events.parquet`, `1748_score.py`, `1748_summary.csv`, `RESULT_1748.md` ≤ 350 words.
Return ≤ 120 words: news depth and news share, events/yr, VAL no-news gross / t / net, VIX-tercile pattern,
KILL/SURVIVE.
