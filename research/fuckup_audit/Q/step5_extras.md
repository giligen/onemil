# Step 5 — split/price-scale, breakeven cost, tails, power, report-only variants

## a) Price-scale check (all bars come from ONE source: Alpaca SIP 1-min, `etf_1min.db`)

| symbol | prev_close -> open jumps outside 0.75x..1.33x | dates |
|---|---|---|
| SPY | 0 | — |
| QQQ | 0 | — |
| TQQQ | 1 | 2020-03-16 7.13->5.06 (0.709x) |

This table is the RESIDUAL after `zsim.load_symbol` back-adjusts the five known TQQQ splits
(2017-01-12 2:1, 2018-05-24 3:1, 2021-01-21 2:1, 2022-01-13 2:1, 2025-11-20 2:1 -- factors derived as
`(TQQQ open/prev_close) / (1 + 3 x QQQ gap)` = 0.4996..0.5000 / 0.3333). The one remaining row,
2020-03-16 at 1.0225 implied, is the COVID crash gap and is correctly NOT treated as a split.

No daily-file / intraday-file price comparison exists in this rule (bands, VWAP, signal and fill all
come from the same 1-min table), so the split risk is confined to the 14-day sigma and prev-close
anchor crossing a split date.

## b) Gross reference and the breakeven cost per leg (QQQ, live fill, MOC flat)

| period | cost/leg (bp) | bps/traded day 1x | t | ann % 1x | SR 1x | MDD % 1x |
|---|---|---|---|---|---|---|
| IS 2016-2023 | 0.0 | 7.97 | 4.10 | 12.34 | 1.45 | 9.4 |
| OOS 2024-2026 | 0.0 | 6.08 | 1.69 | 8.74 | 1.03 | 9.0 |
| IS 2016-2023 | 0.25 | 7.22 | 3.71 | 11.09 | 1.31 | 9.7 |
| OOS 2024-2026 | 0.25 | 5.33 | 1.48 | 7.56 | 0.90 | 9.8 |
| IS 2016-2023 | 0.5 | 6.48 | 3.32 | 9.85 | 1.18 | 9.9 |
| OOS 2024-2026 | 0.5 | 4.57 | 1.27 | 6.40 | 0.77 | 10.6 |
| IS 2016-2023 | 0.75 | 5.73 | 2.94 | 8.63 | 1.04 | 10.2 |
| OOS 2024-2026 | 0.75 | 3.81 | 1.06 | 5.25 | 0.64 | 11.4 |
| IS 2016-2023 | 1.0 | 4.99 | 2.55 | 7.42 | 0.91 | 11.0 |
| OOS 2024-2026 | 1.0 | 3.06 | 0.85 | 4.11 | 0.52 | 12.2 |
| IS 2016-2023 | 1.5 | 3.50 | 1.79 | 5.04 | 0.63 | 13.7 |
| OOS 2024-2026 | 1.5 | 1.54 | 0.43 | 1.87 | 0.26 | 13.7 |
| IS 2016-2023 | 2.0 | 2.01 | 1.02 | 2.71 | 0.36 | 16.4 |
| OOS 2024-2026 | 2.0 | 0.03 | 0.01 | -0.32 | 0.00 | 15.2 |
| IS 2016-2023 | 3.0 | -0.97 | -0.49 | -1.79 | -0.17 | 23.7 |
| OOS 2024-2026 | 3.0 | -3.00 | -0.82 | -4.56 | -0.50 | 18.7 |

Breakeven cost per leg, IS 2016-2023: **2.67 bp** (gross 7.97 bps/traded day).

Breakeven cost per leg, OOS 2024-2026: **2.01 bp** (gross 6.08 bps/traded day).

## c) Tail dependence (QQQ, live fill + 0.5 bp/leg, 1x)

**IS 2016-2023** — total return 77.3%; top 5 days = 23% of it; top 1% of days = 42%; top 5% = 144%.
**OOS 2024-2026** — total return 17.6%; top 5 days = 102% of it; top 1% of days = 75%; top 5% = 206%.

| period | variant | days | bps/calendar day | t | $/month at $60K |
|---|---|---|---|---|---|
| IS 2016-2023 | full | 2000 | 3.86 | 3.32 | 487 |
| IS 2016-2023 | top 1% days removed | 1980 | 1.29 | 1.27 | 162 |
| IS 2016-2023 | top 5% days removed | 1900 | -4.20 | -4.99 | -529 |
| IS 2016-2023 | bottom 5% days removed | 1900 | 9.41 | 8.97 | 1185 |
| IS 2016-2023 | both 5% tails removed | 1800 | 1.21 | 1.93 | 152 |
| IS 2016-2023 | daily return capped at +1% | 2000 | 1.01 | 1.07 | 128 |
| IS 2016-2023 | daily return capped at +0.5% | 2000 | -2.84 | -3.61 | -357 |
| OOS 2024-2026 | full | 678 | 2.60 | 1.27 | 328 |
| OOS 2024-2026 | top 1% days removed | 671 | -0.53 | -0.35 | -66 |
| OOS 2024-2026 | top 5% days removed | 644 | -5.13 | -4.08 | -647 |
| OOS 2024-2026 | bottom 5% days removed | 644 | 7.40 | 3.74 | 933 |
| OOS 2024-2026 | both 5% tails removed | 610 | -0.50 | -0.49 | -63 |
| OOS 2024-2026 | daily return capped at +1% | 678 | 0.03 | 0.02 | 3 |
| OOS 2024-2026 | daily return capped at +0.5% | 678 | -3.11 | -2.53 | -391 |

## d) Power — the smallest daily effect this OOS window could have seen

- IS 2016-2023: n = 2000 days, daily sd = 52.1 bps; MDE at 80% power / 5% two-sided = **3.26 bps/day** = 411 $/month at $60K 1x. Observed = 3.86 bps/day.
- OOS 2024-2026: n = 678 days, daily sd = 53.5 bps; MDE at 80% power / 5% two-sided = **5.75 bps/day** = 725 $/month at $60K 1x. Observed = 2.60 bps/day.

## e) Report-only variants (NOT adopted — each is an extra cell)

| variant | period | bps/traded day 1x | t | ann % 1x | SR 1x | MDD % 1x | trades/day |
|---|---|---|---|---|---|---|---|
| flat at 15:30 (skip the last 30 min) | IS 2016-2023 | 5.56 | 3.09 | 8.40 | 1.09 | 9.4 | 0.89 |
| flat at 15:30 (skip the last 30 min) | OOS 2024-2026 | 4.49 | 1.34 | 6.33 | 0.82 | 10.5 | 0.86 |
| paper cadence (flat 15:59) | IS 2016-2023 | 6.48 | 3.32 | 9.85 | 1.18 | 9.9 | 0.89 |
| paper cadence (flat 15:59) | OOS 2024-2026 | 4.57 | 1.27 | 6.40 | 0.77 | 10.6 | 0.86 |

## f) Gross $/share P&L by ENTRY minute index (k = ET minute − 570; 31 = 10:01 fill)

| period | k | ET | trades | sum $/sh | mean $/sh |
|---|---|---|---|---|---|
| IS | 31 | 10:01 | 522 | 73.51 | 0.1408 |
| IS | 61 | 10:31 | 215 | -7.97 | -0.0371 |
| IS | 91 | 11:01 | 141 | 16.82 | 0.1193 |
| IS | 121 | 11:31 | 149 | 20.56 | 0.1380 |
| IS | 151 | 12:01 | 133 | 5.45 | 0.0409 |
| IS | 181 | 12:31 | 114 | 4.30 | 0.0377 |
| IS | 211 | 13:01 | 94 | -3.15 | -0.0336 |
| IS | 241 | 13:31 | 72 | 17.04 | 0.2367 |
| IS | 271 | 14:01 | 87 | 21.07 | 0.2422 |
| IS | 301 | 14:31 | 84 | 5.43 | 0.0646 |
| IS | 331 | 15:01 | 79 | 28.16 | 0.3565 |
| IS | 361 | 15:31 | 87 | 19.66 | 0.2260 |
| OOS | 31 | 10:01 | 177 | -5.48 | -0.0310 |
| OOS | 61 | 10:31 | 60 | 35.86 | 0.5976 |
| OOS | 91 | 11:01 | 52 | 23.79 | 0.4575 |
| OOS | 121 | 11:31 | 51 | -2.22 | -0.0435 |
| OOS | 151 | 12:01 | 26 | -9.71 | -0.3735 |
| OOS | 181 | 12:31 | 35 | 55.78 | 1.5936 |
| OOS | 211 | 13:01 | 29 | 1.92 | 0.0661 |
| OOS | 241 | 13:31 | 21 | 6.21 | 0.2957 |
| OOS | 271 | 14:01 | 33 | -15.42 | -0.4672 |
| OOS | 301 | 14:31 | 38 | 7.28 | 0.1915 |
| OOS | 331 | 15:01 | 32 | -9.63 | -0.3010 |
| OOS | 361 | 15:31 | 31 | -5.18 | -0.1669 |

Hold time: median 60 min, mean 115 min, p90 298 min. Round trips/day OOS 0.86.

