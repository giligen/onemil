# frames21 — 2024H2 extension data for the mirror short (DATA ONLY, no scoring)

Goal: extend the mirror-short signal sample backwards in time (never into the sealed TEST period,
2026-06-01+) so the next pre-registered pass has more signal fills. Current sample: TRAIN 2025,
VAL Jan-May 2026, TEST sealed (frames15/16, `common15.py::TEST_FROM = '2026-06-01'`).

This file reports DATA FACTS ONLY — counts, coverage, cost. No cells, no t-stats, no verdicts.

## 1. Pipeline read, and the extension window

**frames15/common15.py**: shared loaders. `TEST_FROM='2026-06-01'`; `split_of` puts everything
`< 2026-01-01` in TRAIN. `daily_fields()` reads `daily15_{2025,2026}.parquet`; `hourly_fields()`
reads `hourly15.parquet`.

**frames15/daily.py**: builds the dense daily panel from `research/multiday/data/prices_by_year`
(the point-in-time daily panel, delisted included — same source frames13 F40 used), columns
symbol/date/open/high/low/close/volume/(adjusted close). Hardcodes `YEARS=(2024,2025,2026)`,
`FROM='2024-11-01'` — the 2024 slice is loaded ALREADY TRUNCATED at Nov 1 before the 20-session
trailing stats (adv20, interest5, ret5) are computed, so the shipped `daily15_2024.parquet`
(2024-11-01..2024-12-31, 360,714 rows) is COLD for its own first ~20 sessions (adv20 NaN into early
Dec) — a pre-existing warm-up gap in frames15's own artifact, not something this pull touches.
The SOURCE panel (`prices_by_year/raw/year=2024.parquet`) actually covers the FULL year,
2024-01-02..2024-12-31 (2,119,382 rows, open/high/low/close/volume/trade_count/vwap) — daily.py's
own `FROM` constant is the only thing truncating it to Nov 1.

**frames15/hourly.py -> profile.py -> armB_intra.py** (the actual mirror-short signal producer):
1. `hourly.py` reads `research/bf_zero/bars_sip.db` (1-min bars, table `bars(symbol,day,t,o,h,l,c,v)`
   + `fetch_log`) and aggregates to one row per (symbol, session, ET session-hour): hour RTH volume,
   first open, last close, day volume. `hourly_state.json` shows only 2025-01..2026-09 processed.
   `bars_sip.db`'s own `fetch_log` confirms **zero 2024 rows** (min day 2025-01-02).
2. `profile.py` turns that into `hourly15.parquet`: `hrv = hour volume / (adv20 * pshare)`, where
   `pshare` is a rolling (window 20, min_periods 3, stale after 60 sessions) mean of the symbol's
   OWN prior hour-share-of-day-volume, and `adv20`/`close` are merged in from `daily15_*.parquet`.
3. `armB_intra.py` is the STANDALONE mirror-short detector: filters `hourly15.parquet` to hours
   with `hrv>=3` or a 3-hour `sus2_3` streak, hours 9-14 only; per signal hour it reads 1-min bars
   from `bars_sip.db` directly to compute `gate5` (session high, UP TO the signal hour's close,
   already >= session open x 1.05 — the causal membership gate, avoiding the `bf_zero` REPORT §6b
   full-day look-ahead) and to walk three declared exits (bare stop, +2R target, ORB-style static
   lock) from a next-bar-open entry under a 0.6% no-chase cap. `f_mir` = `hrv>=3` AND
   `|hour_ret| > 2%` (the mirror flag). This is the code short_walk.py's own docstring calls "frames15
   B12 re-derived." **It never reads `daily_fields()`/`daily15` directly — only through profile.py's
   `adv20`/`pshare` merge — and needs no `load_breaks4` HOD-break machinery.**

**frames16/pull.py + pull_log.csv**: a ONE-DAY (2026-05-11), 3-symbol (AAPL/FSLY/USAX) Databento
schema/venue COMPARISON (`EQUS.MINI`, `XNAS.ITCH`, `XNAS.BASIC`, `XNYS.PILLAR` x mbp-1/bbo-1s/tbbo)
used only to calibrate the imputed-spread cost model (`calib.py`, `cost_check.py`) — it is **NOT**
the source of the actual 1-min walk or NBBO data. `short_walk.py` (the actual 1-min walk) reads
`research/bf_zero/bars_sip.db` directly (the same SIP store frames15 uses, sourced via Alpaca REST,
see below). `nbbo.py` (the actual entry/exit NBBO) pulls via
`alpaca.data.requests.StockQuotesRequest(feed=DataFeed.SIP)` — also Alpaca, not Databento.
`research/bf_zero/refetch_thin_tape.py` (docstring, `bars_sip.db`'s actual builder) confirms:
`bars_sip.db` = "the consolidated SIP tape ... re-fetched from Alpaca REST (1-min, feed SIP,
adjustment raw, 04:00-20:00 ET)". So **the frames16 1-min walk and NBBO pulls are BOTH Alpaca, zero
Databento dollar cost** — Databento only entered frames16 as a same-day schema comparison.

**research/scripts/pit_listings.py**: point-in-time listing facts from the Databento EQUS.SUMMARY
`definition` feed, bought 2026-09-18, monthly parquets `data/research/databento/pit_definition/
def_202407.parquet` .. `def_202609.parquet` — **starts 2024-07**, nothing before.

**Extension window** = max(PIT listings start, daily-panel-source start) .. 2024-12-31
= max(2024-07-01, 2024-01-02) .. 2024-12-31 = **2024-07-01 .. 2024-12-31** (bounded below by the
PIT listing feed, not by the daily panel — the daily panel source already has all of 2024).

**Missing input identified**: `bars_sip.db` (the 1-min SIP store both the hourly-volume signal and
the walk depend on) has NO rows before 2025-01-02. This is the one real gap. It was filled by
re-running `bars_sip.db`'s OWN documented build method (Alpaca REST, feed SIP, adjustment RAW,
04:00-20:00 ET, `refetch_thin_tape.py`'s `fetch_day`, copied unchanged into `frames21/fetch_ext.py`)
against a NEW candidate universe for 2024H2, built from data already on disk (no pull needed):
`research/multiday/data/prices_by_year` (open/high/low/close/volume, full 2024) filtered to
`(high/open - 1) >= 0.05` (the day-range-from-open rule `bars_sip.db`'s docstring states — "session
high reached open x 1.05"), `close >= $1`, `adv20 (20-session trailing, strictly prior) >= 100,000` —
the same three thresholds `build_candidates.py`'s docstring gives for `universe.csv`
("range >= 5%, price >= $1, ADV20 >= 100K"). The ORIGINAL script that built `universe.csv` itself
was not found in the repo (only its consumers), so this is a faithful reconstruction of the
documented rule, not a byte-identical rerun — flagged as a methodology note, not hidden.
`frames15/daily.py`'s dense-panel formulas (the `G` class, all field definitions) were copied
UNCHANGED into `frames21/build_daily_ext.py`, loading the FULL 2024 raw panel (proper >=20-session
lookback at 2024-07-01, unlike the shipped `daily15_2024.parquet`) and slicing the OUTPUT to
2024-07-01..2024-12-31. `frames15/hourly.py`'s `one_day()` and `frames15/profile.py`'s hrv/pshare
formulas were likewise copied UNCHANGED into `frames21/build_hourly_ext.py` /
`build_profile_ext.py`. `frames15/armB_intra.py`'s `signals_by_day()`/`walk()` were copied UNCHANGED
into `frames21/build_signals_ext.py`, pointed at the extension's own store and hourly parquet.

**Caveat (declared, not fixed)**: `pshare`/`pvmean` (profile.py's rolling mean over a symbol's PRIOR
sessions at that hour, window 20, min_periods 3) have no pre-2024-07 hourly history to draw on in
this extension, so the first few sessions of July for any given symbol/hour are colder than a
mid-window session would be — the same shape as frames15's own `daily15_2024.parquet` truncation
gap at its Nov-1 edge. Monthly mirror-signal counts (gate5 & f_mir) INCREASE across the window
(Jul 326 -> Aug 668 -> Sep 724 -> Oct 987 -> Nov 2,242 -> Dec 2,003) rather than decaying from a
cold start, consistent with a real H2-2024 small-cap volatility regime (not a warm-up artifact,
though this is an observation, not a scored claim).

## 2. Counts

| quantity | value |
|---|---|
| extension window | 2024-07-01 .. 2024-12-31 (128 trading days in the daily source; 118-128 days served bars, see below) |
| candidate universe symbol-days (day high/open-1>=5%, close>=$1, adv20>=100K) | 41,975 (128 days, 3,492 symbols, avg 328/day) |
| 1-min bars fetched (Alpaca SIP, feed SIP, adjustment RAW) | 13,026,905 bars, 41,975/41,975 symbol-days attempted, 41,963 served (12 empty — Alpaca has no data, likely delisted/renamed tickers), 100% of days (2024-07-01..2024-12-31) |
| dense daily panel rows (frames21 own build, `daily21_2024h2.parquet`) | 1,103,512 rows, 2024-07-01..2024-12-31 |
| hourly volume table rows (`hourly_ext_2024-{07..12}.parquet`) | 292,120 symbol-hours, 3,492 symbols, 128 sessions |
| hrv coverage (non-NaN) on the hourly table | 78.1% (`hrv_raw` identical, 78.1%) |
| signal-hour rows fired (hrv>=3 or sus2_3, hours 9-14) | 20,688 signal-hours over 125 sessions |
| **signals_ext.csv rows** (found a valid next-bar entry under the no-chase cap) | **18,860** |
| of which gate5 (causal membership) & f_mir (mirror flag) | **6,950** |
| **2025 TRAIN reference** (frames15 `intra_2025-*.csv`, gate5 & f_mir, 12 months) | 2,238 |
| NBBO population (gate5 & f_mir & price>=$5 & ex-wrapper via `attach_instrument`) | 2,844 rows |
| NBBO legs requested (entry_m + 3 exit legs, deduped) | 6,500 distinct (day, symbol, minute) |
| NBBO legs with a measured quote | 6,352 / 6,500 = **97.7%** |
| **coverage: signal rows (NBBO population) with ALL 4 legs quoted** | **2,713 / 2,844 = 95.4%** (>= the 80% availability rail) |
| bars coverage of signals_ext.csv rows | 100% by construction (a row cannot exist in `signals_ext.csv` without its own 1-min bars already present) |
| total Databento cost this task | **$0.00** of the $80.00 cap (see `pull_log.csv`) |

**Schema/venue note vs frames16**: identical. Both the 1-min bars and the NBBO quotes are Alpaca
REST, feed `SIP`, `Adjustment.RAW` — the same calls `refetch_thin_tape.py::fetch_day` and
`frames16/nbbo.py::main` make, copied into `frames21/fetch_ext.py` and `frames21/nbbo_ext.py`
respectively with only the output paths and the source population changed. No Databento schema or
venue choice was needed for either, since neither frames15/16's actual walk/NBBO used Databento —
only the frames16 ARM-1 same-day comparison did, and that comparison is not part of this extension.
The single Databento-sourced input used (EQUS.SUMMARY `ohlcv-1d`, 2024-07-01..2024-12-31, ALL
symbols) was already purchased by an unrelated prior task
(`research/fuckup_audit/N_databento/N2`, 2026-09-18) and sits on disk at
`data/research/databento/equs_daily_2024H2.parquet` — read, not re-bought; it was used only to
size/sanity-check the candidate-universe count, not as a hard input into `signals_ext.csv` itself
(the candidate screen and adv20 in this pull both come from `prices_by_year`, matching frames15's
own daily-panel source).

## 3. Files

- `fetch_ext.py`, `raw/bars_sip_ext.db` (gitignored, 2.1 GB) — the 1-min bars store
- `raw/candidates_ext.csv` (gitignored) — the 41,975-row candidate universe fed to the fetch
- `build_daily_ext.py`, `daily21_2024h2.parquet` — the dense daily panel, frames15 formulas unchanged
- `build_hourly_ext.py`, `hourly_ext_2024-{07..12}.parquet` — the per-symbol-hour volume table
- `build_profile_ext.py`, `hourly_ext15.parquet` — hrv/pshare/sus fields, frames15 formulas unchanged
- `build_signals_ext.py`, `signals_ext.csv` — the mirror-short signal rows, frames15 `armB_intra.py` unchanged
- `nbbo_ext.py`, `nbbo_ext.csv` — measured NBBO for the entry/exit legs of the mirror population
- `pull_log.csv` — every request considered, priced (Databento) or noted as Alpaca (not priced), cost
