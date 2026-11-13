# PREREG 1,746 — FINRA daily short-volume ratio, liquid tier (candidate E1) — Stage 1, rejection-only

FROZEN 2026-11-13 13:05 UTC before any number. Programme cell count: 1,745 → 1,746 (new line: signal search).
Rules: no git / service / orders; read-only on every existing store; write only `research/signal_search/1746_*`
and `RESULT_1746.md`; caches under `research/signal_search/1746_cache/` (atomic, resume); `df` ≥ 5 GB first; ≤ 180
requests/min; `nice -n 19`; background scripts with log polling; never Read a file > 300 lines in full; budget ≤ 60
calls; never print keys. This task IS the owner's request ("go wide and find it"); do not pivot.

## Claim under test (source: Diether–Lee–Werner RFS 2009; independent replication 2022–26 reported by the search
agent as [m] — re-read `CANDIDATES_20261113.md` E1 and record the source URL in RESULT)
Names with a HIGH daily short-volume share underperform over the next 5 days (−26.7 bp / 5 d, t −4.1, never
positive 2022–26 [m]); a bottom-minus-top decile L/S is positive gross. Mechanism: informed short sellers.

## Data (free)
1. FINRA Reg SHO daily short-volume files, consolidated NMS: `https://cdn.finra.org/equity/regsho/daily/CNMSshvolYYYYMMDD.txt`
   (fields Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market). Fetch 2016-01-04 → 2026-11-11, one file
   per trading day; LOST count per `feedback_fetch_completeness_gate`; coverage < 95 % of trading days → VOID.
2. Daily bars: Alpaca `/v2/stocks/bars?timeframe=1Day&adjustment=all` for the universe (open, close), 2015-12 →
   2026-11 (survivorship-biased before 2024-07 — Stage 1 is rejection-only, a reject stands, a pass must be re-read
   on the point-in-time bars `data/research/databento/equs_daily_*.parquet` for 2024-07 → 2026-09).
3. Universe: common stocks (Alpaca assets `class=us_equity`, not ETF — use `data/research/databento/alpaca_assets_all_20260905.csv`
   and the live `trading/bf_universe_filter.is_bf_eligible` name rule), 20-day median dollar volume > $100 M as of
   t−1 (point-in-time from the bars themselves), price ≥ $5; the $10–100 M tier reported separately. Exclude
   `^Z[A-Z]ZZT$` and anything absent from the bars.

## Signal and book
SVR_t = ShortVolume / TotalVolume on day t (consolidated), known after the close of t. Deciles of SVR within the
universe each day. Entry at the OPEN of t+1 (auction), exit at the CLOSE of t+5 (MOC). Equal weight. GROSS.
Secondary reads: exit t+1 close; SVR averaged over 5 days; the long-only bottom decile minus the universe mean.

## Splits
TRAIN 2016-01 → 2020-12 (halves 2016–18 / 2019–20); VAL 2021-01 → 2023-12 (contains 2022); TEST 2024-01 → 2026-11
SEALED (read once only if TRAIN and VAL pass, on PIT bars where available).

## Numbers (per split)
Mean L/S (bottom − top decile) return over the 5-day hold, Newey–West (lag 5) t on the daily series of overlapping
portfolios, iid t; the long and short legs separately vs the universe mean; ex-top-5 % of name-days; the top-decile
share of the L/S; monthly green share; the year-by-year sign; MDE at 80 % power. Cost: measured NBBO half-spread at
09:30–09:35 for a 400-fill sample via Alpaca `/v2/stocks/quotes` (cache it), + 2 bp per leg; report net.
Fills/week of the long leg at 8 names/day.

## Kill (pre-committed — any one kills)
VAL gross L/S ≤ 0 or |NW t| < 2; TRAIN halves opposite sign; ex-top-5 % ≤ 0 on VAL; net of measured cost ≤ 0 on
VAL; 2022 negative. Survive all → TEST read ONCE on PIT bars, then an independent rebuild before the owner sees a
number, then a paper-sleeve spec (long bottom decile + short top decile, 8 names a side, $150 R).

## Output
`1746_fetch.py`, `1746_svr.parquet`, `1746_score.py`, `1746_summary.csv`, `RESULT_1746.md` ≤ 350 words (coverage,
universe size by year, TRAIN/VAL tables, year-by-year, cost, KILL/SURVIVE, TEST if read). Return ≤ 120 words.
