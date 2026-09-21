# PREREG + SPEC — The overnight effect on index ETFs (close-to-open), cells 1,347–1,350

Owner 2026-09-21 "sure": test the one daily-frequency, unlimited-capacity line the literature supports — nearly
all of the equity premium accrues between the close and the next open (Cooper, Cliff & Gulen 2008; Kelly & Clark
2011; Lou, Polk & Skouras 2019). This is market beta concentrated in the overnight session, NOT a day-trading
edge; the test decides whether it is a book by the cadence bar. The implementing agent reads ONLY this file.

## Data
Alpaca daily bars (`feed=sip`, raw) for SPY, QQQ, TQQQ, UPRO from 2010-03-01 to 2024-12-31 (TEST ≥ 2025-01-01
is NOT fetched). Cache as `research/overnight/cache/<SYM>_daily.parquet`. Report days missing per symbol.
Use `config.Config` for keys and the repo's Alpaca client or `alpaca-py` directly.

## Rule (frozen)
Buy at the official close (MOC fill at the daily bar's close), sell at the next session's official open (MOO fill
at the daily bar's open). Every session, no filters. Overnight return r_on = open_t / close_{t-1} − 1; the
complement r_id = close_t / open_t − 1 is reported beside it (the mechanism's signature is r_on > 0 and r_id ≤ 0).
Cost: 0 at the auctions (base), plus a sensitivity line at 1 bp per side. Notional 1× (no margin): $ figures at
$66,000 notional. R for the cadence bar = 0.5 % of notional per night (pre-set risk unit; ≈ one overnight sd of SPY).

## Cells and splits
1,347 SPY · 1,348 QQQ · 1,349 TQQQ · 1,350 UPRO. TRAIN 2010-03..2019-12 (halves 2010–2014 / 2015–2019),
VAL 2020-01..2024-12 (contains the 2020 crash and the 2022 bear on purpose).

## Report (`research/overnight/REPORT.md`, plus `nights_<SYM>.csv` with date, r_on, r_id)
Per cell per split: nights, mean r_on (bp) and r_id (bp), sd, iid t and month-clustered t, hit rate, annualized
r_on, total return of the overnight-only strategy vs buy-and-hold of the same ETF over the same span, worst night,
max drawdown of the overnight-only equity curve vs buy-and-hold's, ex-top-5 % nights mean, year-by-year mean r_on,
day-of-week table (report-only), weekly series → `scripts/cadence_bar.py --trades nights_<SYM>_weeks.csv --split
TRAIN|VAL` (columns date, pnl_R with R = 0.5 % of notional).

## Pass bar (all)
1. VAL mean r_on > 0 with month-clustered t ≥ 2;  2. TRAIN halves both > 0;  3. r_id ≤ 0 on TRAIN and VAL (the
signature);  4. overnight-only max drawdown ≤ 60 % of buy-and-hold's on VAL and total return ≥ buy-and-hold's on
VAL (else the book is just "hold it");  5. cadence C3 and C4 on VAL;  6. the 1 bp line keeps 1–4.
Pass → independent rebuild (Haiku) → owner decision: this is market exposure as a book; live = MOC buy + OPG sell
each day on the chosen ETF at a fixed notional, kill = weekly loss ≤ −4 R or a night ≤ −6 R.

## Multiplicity
4 cells; programme count 1,350 after this pass. Nothing added after results.
