# PREREG — multiday_catalyst: cells 1,285 (long continuation) / 1,286 (matched control)

Pre-registered 2026-09-20, written by the main session, copied verbatim below before any scoring is run.
Cell count for this program: **1,285** (long continuation) and **1,286** (matched control). Any additional
cell looked at during implementation is counted and disclosed in REPORT.md.

## Universe

Point-in-time (`research/scripts/pit_listings.py`), common stocks AND 2x wrappers flagged, price >= $5 at the
signal close, ADV20 >= 500K, exclude `^Z[A-Z]ZZT$` and any symbol absent from cache.db `daily_bars`.

## Daily bars

`data/cache.db` `daily_bars` (Alpaca, raw). **SPLIT RAIL**: for every candidate symbol-window, if
|close-to-open overnight return| > 40% on any day in the window OR the PIT definition feed marks a corporate
action, the window is EXCLUDED and counted; report the count. Also the **price-scale check**: the entry close
from `daily_bars` must equal the 15:59/16:00 1-min close in `intraday_bars_1min` within 0.5% on a 200-window
sample (report the share that disagrees).

## Signal day D

(a) own-ticker pre-market news present in `data/research/orb_news_catalyst_nightly.csv` (the has_news flag per
symbol-day), AND (b) day-D dollar volume >= the 90th percentile of the universe that day, AND (c)
close_D > open_D (a green catalyst day). All three known by the close of D.

## Entry

At the CLOSE of D (the 16:00 auction print; charge the minute-of-day half-spread for 15:55 from the frames14
table — auction fills do NOT pay a quoted spread, so charge the 15:55 half-spread ONCE as a conservative proxy
and say so) — if the 16:00 print is unavailable use the next day's open and report the share.

## Stop / Exit

Stop: low of day D, checked on daily lows (a gap through the stop fills at the open, not at the stop). Exit:
the close of D+3 (third session after D), or the stop. No target. One position per symbol; a symbol
re-signalling while held is skipped.

## R

R = close_D − low_D. Skip if R < 1% of price (report the share).

## Book

10 concurrent max, first-come by signal order then symbol; 1% of $66K risk per position, notional cap 1x
equity total; report positions/day and the unlimited-slot per-trade numbers separately.

## Control (cell 1,286)

Same universe, same day D, the SAME (b) and (c) but NO news — walked identically. The claim is signal minus
control.

## Splits

TRAIN 2025 (both halves), VAL 2026-01-01..2026-05-31, TEST >= 2026-06-01 **SEALED (never query)**. Also report
2024-07..2024-12 as a THIRD out-of-time split if daily_bars covers it (report coverage).

## PASS BAR (VAL)

Signal net >= +0.15 R with day-clustered t >= 2; signal minus control >= +0.10 R with t >= 2; TRAIN halves
same-signed; >= 3 entries/week at 10 slots; cadence bar C1-C5 pass on the 10-slot book (run
`python scripts/cadence_bar.py --trades <csv> --split VAL` and `--split TRAIN`, columns date,pnl_R,symbol;
paste both blocks). Diagnostics: gross, cost in R, WR, avg win/loss, ex-top-1%/5%, top-5 share, MDD in R and $,
holding-day P&L profile (D+1, D+2, D+3 closes), wrappers with/without, MDE beside every null, iid and clustered
SE.

## Caveat

The ONE caveat that alone could explain the headline, named in the report. Any mid-run change recorded.

---

## Pre-flight facts gathered before implementation (this session, 2026-09-20)

These do not change the spec above; they record the API/data facts the spec told me to grep, frozen before
any scoring code ran.

- **`research/scripts/pit_listings.py`** (`PitListings`): `.coverage` -> `('202407','202609')` (27 monthly
  parquets, Databento EQUS.SUMMARY definition feed, bought 2026-09-18). `.listed_symbols(date)`,
  `.common_stock_symbols(date)`, `.was_listed(symbol, date)`, `.screen(symbols, date)` ->
  `(kept, {dropped: reason})`. Test tickers (`^Z[A-Z]ZZT`) are stripped inside every accessor already. A date
  outside 2024-07..2026-09 raises `KeyError` rather than silently returning an empty/present-day universe.
- **2x wrapper flag**: `trading/orb_asset_class.py::load_class_map()` reads the offline dump
  `data/research/orb_asset_class_map_20260711.csv` (symbol,asset_class,name) -> `{symbol: 'stock'|'wrapper'}`.
  Universe keeps symbols mapped to `stock` or `wrapper`; symbols absent from the map are `unknown` and
  excluded (count reported — this is IN ADDITION to the pit_listings drop, not instead of it).
- **`data/cache.db` `daily_bars`**: `(symbol, bar_date, open, high, low, close, volume, fetched_at)`, PK
  `(symbol, bar_date)`. Coverage measured this session: `bar_date` 2024-06-03 .. 2026-09-18, 13,620 distinct
  symbols, 5,022,976 rows. 2024-07..2024-12 is fully inside this range -> the third out-of-time split is
  runnable for the CONTROL cell; the SIGNAL cell needs news (next bullet) which only starts 2025-01-02, so
  2024H2 signal is NOT runnable — reported as a coverage gap, control-only for that split.
  `intraday_bars_1min`: `(symbol, bar_date, timestamp, open, high, low, close, volume)`, PK `(symbol,
  timestamp)` — used for the 200-window price-scale check against the daily close.
- **`data/research/orb_news_catalyst_nightly.csv`**: columns `symbol,day,n_articles,earliest,latest,headlines`.
  `has_news` per symbol-day = `n_articles > 0`. Coverage measured this session: `day` 2025-01-02 ..
  2026-09-18. Pre-market window is prev-day 15:00 ET -> fetch time (per the CLAUDE.md PM-mult section); this
  file is read as-is, own-ticker only, no interpretation of headline content (per the CLAUDE.md rule against
  an LLM/keyword catalyst-quality filter).
- **ADV20 definition (causal)**: mirrors `trading/hod_break_engine.py::load_adv20_from_daily_bars` — mean
  `volume` over the trailing <=20 `daily_bars` rows strictly BEFORE day D (D-1 back to D-20 trading days
  present in the table, min 5 rows), never day D's own (provisional) volume.
- **15:55 half-spread (frames14)**: `research/mature_method/frames14/f45_minute_table.csv`, row
  `clock_m=955, label='15:55*HOD/BF flat'`: median **0.1725 %** of price, mean **0.2853 %** of price (n=208,
  98.0% coverage of the frames14 HOD/BF sample). Per the PREREG instruction this session charges the median,
  0.1725% of price, ONCE per trade (entry leg only; the D+3/stop exit is a close/gap-through fill, not an
  auction, and is charged the same 0.1725% of price as a conservative single-leg proxy on that leg too, since
  the PREREG names only the entry explicitly — this choice is disclosed as the ONE-of-several-reasonable
  cost-model decisions in REPORT.md, not hidden).
- **`scripts/cadence_bar.py`**: `SPLIT_RANGES = {'TRAIN': (2025-01-01, 2025-12-31), 'VAL': (2026-01-01,
  2026-05-31)}` — matches this PREREG's split boundaries exactly. Expects `--trades CSV` with columns
  `date,pnl_R,symbol` (a `date` per trade, i.e. per closed trade's exit or entry date — this run uses the
  ENTRY (signal) date D, consistent with "entries/week" framing in the pass bar).

No mid-run spec change. This file is committed alone, before any scoring code, per the research-claim
checklist in CLAUDE.md ("no research claim ships without an independent check") and the north-star discipline
of pre-committing the rule before looking at the outcome.
