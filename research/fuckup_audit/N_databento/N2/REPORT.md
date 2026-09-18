# Stage N2 — 2024H2 daily history, and K2 (52-week-high breakout on volume) with a FULL lookback

Pre-registered in `N_databento/PREREG.md` §N2 (2026-09-18, before the pull). Executed 2026-09-18.
Nothing was tuned: the rule, the universe, the fill, the costs, the stop, the book, the gates and
the capacity definition are Stage-K code (`K/build_k.py`, `K/report_k.py`) executing **unmodified** —
`N2/run_n2.py` only repoints `PANEL`, the output dir and `SPLITS`, restricts `FAMILIES/HOLDS` to K2,
and ANDs the universe with `bar_date >= 2025-07-01` (the first day the 250-session `high52` window
is full).

## 1. One page

**Does K2 clear G1/G2 once its 52-week lookback is real? No. 0 of the 4 pre-registered cells pass
G1 on TRAIN, G2 is therefore unreachable, and TEST (2026-06-01..2026-09-04) was NOT read.**

| cell (hold × slots) | TRAIN n | tr/wk | **TRAIN net bps** | t | VAL n | **VAL net bps** | t | wk green% | p_adj (VAL) | G1 | G2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `K2_h10_n10` | 159 | 5.9 | **−11.6** | −0.11 | 127 | **+19.2** | +0.12 | 40.9 | 0.844 | fail | fail |
| `K2_h10_n20` | 308 | 11.0 | **−18.4** | −0.27 | 250 | **+154.3** | +1.17 | 56.5 | 0.413 | fail | fail |
| `K2_h5_n10` | 281 | 10.4 | **−91.5** | −1.56 | 227 | **−22.9** | −0.33 | 45.5 | 0.933 | fail | fail |
| `K2_h5_n20` | 509 | 18.9 | **−70.4** | −1.76 | 421 | **−10.5** | −0.22 | 45.5 | 0.922 | fail | fail |

The one positive-looking cell is the same one Stage K already refused to carry forward, and it
behaves the same way with the real lookback: `K2_h10_n20` is **+154.3 bps on VAL, −18.4 on TRAIN,
and −164.7 bps with the top 5% of trades removed** (+3.0 bps with winners capped at their 95th
percentile). Its whole VAL mean is in the right tail, its search-adjusted permutation p is 0.41, and
its TRAIN sign is wrong. Tails for all four cells: `phaseA.md` §Tails.

**What the pull bought.** Stage K ran K2 on a panel that began 2025-01-02, so its `high52` was a
truncated `min_periods=60` window — coverage **79.08% of TRAIN** liquid symbol-days. With 128
sessions of 2024H2 in front, coverage is **99.07% TRAIN / 99.13% VAL / 98.92% TEST**
(`availability.md`); the residual ~1% is young listings with < 60 prior bars, which cannot fire K2 by
construction. The signal set is genuinely different (1,076 TRAIN signals over 26 weeks, ~41/week
pre-book), and the verdict is unchanged. The lookback was not what was wrong with K2.

**Power (the smallest per-trade mean that would have produced t = 2).** Per-trade SD is 900–2,100 bps
at these holds, which is the real story:

| split | book weeks | trades booked | MDE / trade | MDE / week of book return |
|---|---:|---:|---|---|
| TRAIN 2025-07-01..12-31 | 27–28 | 159–509 | **80 bps** (`h5_n20`) … **205 bps** (`h10_n10`) | 84 – 119 bps |
| VAL 2026-01-01..05-31 | 22–23 | 127–421 | **96 bps** … **314 bps** | 100 – 203 bps |
| TEST 2026-06-01..09-04 | ~14 | not read (526 signals, ~45% of VAL's) | ≈ **1.5× the VAL numbers** (145 – 470 bps) | ≈ 150 – 300 bps |

So the supported statement is: **no edge was detectable for a 52-week-high-on-volume breakout in the
≥$10M/day US common-stock universe, at 5- and 10-day holds, in a 10–20 position equal-$ long-only
book, over 2025-07..2026-05, at the PREREG cost model — and the test could not have seen a per-trade
effect smaller than ~0.8% (TRAIN) / ~1.0% (VAL) even in its densest cell.** A half-year TRAIN split
at 6–19 trades/week cannot resolve anything a multi-day book plausibly earns; this is a null with
very little power, not a refutation.

**Capacity** (1% of the name's 20-day median dollar volume, one size for the book): **$0.41M–$0.56M
per position → a $4.2M–$11.1M book**, unchanged from Stage K. The best cell's VAL months are
+$287,817/month mean on an $8.95M book (+3.21%) with a worst month of −$528,791; its TRAIN months are
−$33,192/month. Full table in `phaseA.md` §Capacity.

**Cells looked at: 4** (K2 × holds {10, 5} × books {10, 20}), plus one non-gated secondary-universe
control (`K2_h10_n10_sec`: TRAIN +26.2 bps t 0.25, VAL +131.1 bps t 0.68 — same shape, so the class
map is not what is hiding an edge). Cumulative for this family across the program: **24** (Stage K's
20 + these 4). Permutation p is computed across the 4 cells (`perm_p.csv`).

**Data cost: $2.2241** (`metadata.get_cost` before the first byte, budget $5).

## 2. The pull (step 1)

`fetch_daily_2024h2.py` + `convert_daily_2024h2.py`. EQUS.SUMMARY `ohlcv-1d`, `ALL_SYMBOLS`,
2024-07-01 → 2025-01-01 (exclusive): **1,421,466 records, 11,970 symbols, 128 sessions
2024-07-01..2024-12-31**, priced at **$2.2241** against the $5 stop. Raw DBN kept at
`data/research/databento/equs_summary_ohlcv1d_ALL_20240701_20241231.dbn.zst` (41 MB).

`ALL_SYMBOLS` DBN carries no symbol mappings, so `instrument_id → raw_symbol` was resolved
separately (`symbology_2024h2.json`, free; 11,970 intervals, all 1,421,466 rows inside their
interval, 0 unmapped) and `bar_date` is the **UTC date of `ts_event`** — the convention of the
existing 2025-2026 file (an ET conversion would shift every bar one day earlier).

Output `data/research/databento/equs_daily_2024H2.parquet`: **layout identical** to
`equs_daily_2025_2026.parquet` — same columns in the same order, same dtypes
(`bar_date` str, `symbol` str, `instrument_id` uint32, OHLC float64, `volume` uint64). The existing
parquet was opened read-only and never rewritten.

## 3. Price-scale check (step 1, PLAN §1 / CLAUDE.md check 3)

`pricescale_2024h2.md`, `pricescale_2024h2.csv`. 200 random (symbol, date) keys of the NEW file that
also exist in `data/cache.db::daily_bars` (Alpaca, read-only; the cache holds 27,079 rows in 2024H2 —
the bull-flag universe only, so 11,331 panel rows had to be sampled to find 200 matches):

| field | within 0.01% | **off by > 0.5%** | median abs |
|---|---:|---:|---:|
| open / high / low / close | 99.0% | **0.5% (1 of 200)** | 0.0000% |
| volume | 99.0% | **0.0%** | 0.0000% |

**The 2024H2 file is RAW and agrees with Alpaca to float precision** — no split/dividend adjustment,
so the CLAUDE.md price-scale hazard does not apply. The single disagreement is the zero-price row
pathology Stage K already documented (`ALDF 2024-12-31`, Databento close exactly 0.0000 vs Alpaca
9.93). `build_k.build_features` converts every zero/NaN price row to NaN *everywhere* before any
rolling statistic, so those rows cannot enter a `high52` window or fabricate a breakout.

## 4. Freeze (TEST was not read)

`build_k.main('A')` computes statistics for `('TRAIN', 'VAL')` only; the TEST column of
`signal_counts.csv` is a **count of signals (526), not a return**. No TEST price entered any number
here. With 0 of 4 cells passing G1, PLAN §1's "TEST is read once, for selections that earned it"
leaves nothing to confirm, so phase B was not run.

Per the N2 decision rule, **the XNAS.ITCH 2018-05 → 2024-06 daily pull (≈ $23) was NOT priced and NOT
pulled**: it is conditional on K2 clearing G1 *and* G2, and K2 cleared neither.

## 5. Files

`fetch_daily_2024h2.py`, `convert_daily_2024h2.py`, `pricescale_2024h2.{py,md,csv}`,
`build_panel_n2.py`, `run_n2.py`, `phaseA.md`, `cells_trainval.csv`, `perm_p.csv`,
`signal_counts.csv`, `availability.md`, `trades/*.csv`,
`equs_instrument_symbol_map_2024h2.csv`, `symbology_2024h2.json`.
The two parquets (`data/research/databento/equs_daily_2024H2.parquet`, 32 MB; the derived
`daily_panel_2024H2_2026.parquet`, 231 MB — 6,464,692 rows, 16,263 symbols, 548 sessions
2024-07-01..2026-09-04) and the raw DBN are gitignored.
