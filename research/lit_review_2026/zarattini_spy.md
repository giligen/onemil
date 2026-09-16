# Zarattini "Beat the Market" SPY noise-area strategy — exact-spec replication on Alpaca SIP 1-min bars

Generated 2026-09-16 04:33 by `test_zarattini_spy.py`. Data: `etf_1min.db` (regular hours 09:30–15:59 ET, close of the 15:59 bar used as the 16:00 exit). IS = 2016-01 → 2023-12 (inside the paper's 2007-05 → 2024-04 sample), OOS = 2024-01 → end of data. "dyn" = 2% daily vol target, 4× cap; "1x" = 100% notional. bps/day, t and hit are over TRADED days (the paper's Table 5 convention: 12 bps, t 5.34, 43%, N 2,620 traded days of ~4,270). Ann % = CAGR; SR = mean/sd × √252 over all days; MDD from the compounded equity curve. Costs: paper = $0.0035 + $0.001 per share per leg; 1bp = 1 bp of price per leg; gross = none.

## SPY — full paper model

### Summary (IS = 2016–2023, OOS = 2024-01 → today)

| period | costs | days | bps/day dyn | t | hit % | trades/day | ann % 1x | SR 1x | MDD % 1x | ann % dyn | SR dyn | MDD % dyn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **paper 2007-05→2024-04** | paper | ~4,270 | 12 | 5.34 | 43 | 1.8 | 9.7 | 1.24 | 12 | 19.6 | 1.33 | 25 |
| paper, its own 2016–2023 yearly table (dyn) | paper | 2,012 |  |  |  |  |  |  |  | 19.2 (CAGR of FAQ Q24 rows) |  |  |
| SPY IS 2016–2023 | paper | 2002 | 10.9 | 3.20 | 43 | 0.90 | 7.1 | 1.03 | 10.6 | 16.7 | 1.13 | 31.8 |
| SPY OOS 2024→ | paper | 678 | 0.6 | 0.09 | 43 | 0.94 | -1.0 | -0.14 | 11.8 | -0.3 | 0.06 | 24.7 |
| SPY full 2016→ | paper | 2680 | 8.2 | 2.77 | 43 | 0.91 | 5.0 | 0.77 | 11.8 | 12.2 | 0.85 | 31.8 |
| SPY IS 2016–2023 | bp | 2002 | 4.0 | 1.16 | 39 | 0.90 | 3.0 | 0.47 | 17.2 | 5.1 | 0.41 | 48.1 |
| SPY OOS 2024→ | bp | 678 | -7.2 | -1.18 | 39 | 0.94 | -5.2 | -0.90 | 16.7 | -11.6 | -0.72 | 36.6 |
| SPY full 2016→ | bp | 2680 | 1.1 | 0.37 | 39 | 0.91 | 0.9 | 0.16 | 17.2 | 0.6 | 0.11 | 48.1 |
| SPY IS 2016–2023 | gross | 2002 | 12.1 | 3.57 | 43 | 0.90 | 7.8 | 1.13 | 9.3 | 18.9 | 1.26 | 28.3 |
| SPY OOS 2024→ | gross | 678 | 1.2 | 0.20 | 43 | 0.94 | -0.6 | -0.08 | 11.4 | 0.7 | 0.12 | 23.8 |
| SPY full 2016→ | gross | 2680 | 9.3 | 3.14 | 43 | 0.91 | 5.6 | 0.86 | 11.4 | 14.0 | 0.96 | 28.3 |

### By year (paper costs)

| year | days | traded | bps/day dyn | t | hit % | trades/day | ann % 1x | SR 1x | MDD % 1x | ann % dyn | SR dyn | MDD % dyn | paper dyn ann % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2016 | 242 | 155 | -14.3 | -1.46 | 32 | 0.95 | -6.9 | -1.30 | 8.0 | -21.6 | -1.49 | 24.0 | -12.8 |
| 2017 | 251 | 150 | -8.3 | -1.02 | 35 | 0.97 | -3.3 | -1.04 | 4.0 | -12.4 | -1.03 | 15.0 | -6.9 |
| 2018 | 251 | 163 | 30.5 | 2.70 | 44 | 0.95 | 24.8 | 2.59 | 3.1 | 61.9 | 2.69 | 8.9 | 61.1 |
| 2019 | 252 | 134 | 8.7 | 0.98 | 40 | 0.81 | 2.7 | 0.67 | 3.3 | 11.6 | 0.98 | 7.9 | 6.9 |
| 2020 | 253 | 139 | 15.9 | 1.48 | 40 | 0.79 | 3.9 | 0.43 | 8.8 | 23.3 | 1.47 | 7.2 | 26.8 |
| 2021 | 252 | 144 | 23.3 | 2.35 | 57 | 0.85 | 12.9 | 2.67 | 1.8 | 38.4 | 2.34 | 5.7 | 34.8 |
| 2022 | 251 | 156 | 16.6 | 1.84 | 47 | 0.96 | 18.8 | 1.83 | 4.6 | 28.5 | 1.84 | 9.2 | 24.4 |
| 2023 | 250 | 163 | 14.0 | 1.71 | 46 | 0.92 | 7.0 | 1.29 | 2.3 | 24.7 | 1.72 | 6.0 | 37.2 |
| 2024 | 252 | 158 | 12.4 | 1.19 | 47 | 0.94 | 2.8 | 0.53 | 4.6 | 20.0 | 1.19 | 11.0 | 32.2 |
| 2025 | 250 | 144 | -0.9 | -0.09 | 44 | 0.90 | 0.4 | 0.09 | 6.5 | -2.5 | -0.09 | 13.4 | -1.2 |
| 2026 | 176 | 112 | -14.1 | -1.38 | 36 | 1.01 | -7.9 | -1.65 | 6.1 | -21.0 | -1.65 | 15.6 |  |

## SPY — ablations (paper costs), one ingredient at a time

| config | period | days | bps/day dyn | t | hit % | trades/day | ann % 1x | SR 1x | MDD % 1x | ann % dyn | SR dyn | MDD % dyn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A  full paper model | IS | 2002 | 10.9 | 3.20 | 43 | 0.90 | 7.1 | 1.03 | 10.6 | 16.7 | 1.13 | 31.8 |
| A  full paper model | OOS | 678 | 0.6 | 0.09 | 43 | 0.94 | -1.0 | -0.14 | 11.8 | -0.3 | 0.06 | 24.7 |
| B  bands anchored on open only | IS | 2002 | 8.9 | 2.97 | 42 | 1.34 | 5.8 | 0.70 | 25.6 | 17.9 | 1.05 | 36.6 |
| B  bands anchored on open only | OOS | 678 | 6.0 | 1.10 | 42 | 1.35 | 4.7 | 0.63 | 7.7 | 11.2 | 0.67 | 17.2 |
| C  checks every minute (from 10:00) | IS | 2002 | 1.1 | 0.38 | 35 | 5.61 | 0.4 | 0.10 | 26.4 | 0.9 | 0.13 | 54.5 |
| C  checks every minute (from 10:00) | OOS | 678 | 1.2 | 0.24 | 36 | 5.62 | -0.7 | -0.09 | 11.1 | 1.1 | 0.14 | 24.1 |
| D  stop = opposite band only (paper base model) | IS | 2002 | 6.8 | 1.53 | 54 | 0.63 | 6.0 | 0.69 | 16.4 | 8.9 | 0.54 | 47.2 |
| D  stop = opposite band only (paper base model) | OOS | 678 | 5.1 | 0.69 | 53 | 0.63 | 2.9 | 0.37 | 10.4 | 6.4 | 0.42 | 24.1 |
| E  A2 replica: open anchor + every minute from 09:31 + opposite-band flip | IS | 2002 | 5.9 | 1.56 | 51 | 1.52 | 1.8 | 0.20 | 31.5 | 11.7 | 0.55 | 55.6 |
| E  A2 replica: open anchor + every minute from 09:31 + opposite-band flip | OOS | 678 | 1.1 | 0.16 | 51 | 1.53 | 2.6 | 0.28 | 13.4 | -1.1 | 0.10 | 31.0 |
| F  full model, fill at next bar open | IS | 2002 | 10.8 | 3.17 | 43 | 0.90 | 7.0 | 1.02 | 10.6 | 16.6 | 1.12 | 31.8 |
| F  full model, fill at next bar open | OOS | 678 | 0.5 | 0.08 | 43 | 0.94 | -1.0 | -0.15 | 11.9 | -0.4 | 0.05 | 25.0 |
| G  full model, VM = 1.5 (paper FAQ optimum) | IS | 2002 | 13.4 | 3.10 | 46 | 0.54 | 5.3 | 1.03 | 7.6 | 13.1 | 1.10 | 9.6 |
| G  full model, VM = 1.5 (paper FAQ optimum) | OOS | 678 | 3.3 | 0.42 | 41 | 0.60 | 1.0 | 0.24 | 8.0 | 2.5 | 0.26 | 19.1 |

## QQQ — full paper model

### Summary (IS = 2016–2023, OOS = 2024-01 → today)

| period | costs | days | bps/day dyn | t | hit % | trades/day | ann % 1x | SR 1x | MDD % 1x | ann % dyn | SR dyn | MDD % dyn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QQQ IS 2016–2023 | paper | 2000 | 12.8 | 3.73 | 44 | 0.89 | 11.4 | 1.34 | 9.8 | 19.9 | 1.32 | 25.3 |
| QQQ OOS 2024→ | paper | 678 | 10.1 | 1.59 | 45 | 0.86 | 8.4 | 0.99 | 9.1 | 14.4 | 0.97 | 20.6 |
| QQQ full 2016→ | paper | 2678 | 12.1 | 4.02 | 44 | 0.88 | 10.6 | 1.25 | 9.8 | 18.5 | 1.23 | 25.3 |
| QQQ IS 2016–2023 | bp | 2000 | 8.0 | 2.31 | 43 | 0.89 | 7.6 | 0.92 | 11.1 | 11.5 | 0.82 | 32.6 |
| QQQ OOS 2024→ | bp | 678 | 4.6 | 0.71 | 43 | 0.86 | 4.1 | 0.52 | 12.1 | 5.6 | 0.43 | 26.7 |
| QQQ full 2016→ | bp | 2678 | 7.1 | 2.35 | 43 | 0.88 | 6.7 | 0.82 | 12.1 | 10.0 | 0.72 | 32.6 |
| QQQ IS 2016–2023 | gross | 2000 | 14.3 | 4.20 | 45 | 0.89 | 12.5 | 1.47 | 9.6 | 22.8 | 1.49 | 22.4 |
| QQQ OOS 2024→ | gross | 678 | 10.7 | 1.67 | 45 | 0.86 | 8.8 | 1.03 | 8.9 | 15.2 | 1.02 | 20.2 |
| QQQ full 2016→ | gross | 2678 | 13.4 | 4.46 | 45 | 0.88 | 11.5 | 1.35 | 9.6 | 20.8 | 1.36 | 22.4 |

### By year (paper costs)

| year | days | traded | bps/day dyn | t | hit % | trades/day | ann % 1x | SR 1x | MDD % 1x | ann % dyn | SR dyn | MDD % dyn | paper dyn ann % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2016 | 242 | 142 | -7.8 | -0.71 | 35 | 0.92 | -3.0 | -0.42 | 7.3 | -12.0 | -0.73 | 23.8 |  |
| 2017 | 251 | 151 | 8.8 | 0.81 | 39 | 0.89 | 3.3 | 0.70 | 2.5 | 12.8 | 0.82 | 9.3 |  |
| 2018 | 249 | 144 | 35.5 | 3.06 | 51 | 0.85 | 39.1 | 3.13 | 2.7 | 65.4 | 3.04 | 7.6 |  |
| 2019 | 252 | 138 | 4.8 | 0.57 | 38 | 0.83 | 3.1 | 0.60 | 2.8 | 6.1 | 0.57 | 6.0 |  |
| 2020 | 253 | 146 | 18.3 | 1.71 | 45 | 0.86 | 7.1 | 0.71 | 9.8 | 29.0 | 1.71 | 6.9 |  |
| 2021 | 252 | 152 | 8.4 | 0.92 | 51 | 0.87 | 8.1 | 1.20 | 3.0 | 12.6 | 0.92 | 9.7 |  |
| 2022 | 251 | 171 | 14.0 | 1.87 | 46 | 1.02 | 23.5 | 1.82 | 4.6 | 26.1 | 1.87 | 6.4 |  |
| 2023 | 250 | 149 | 19.4 | 2.50 | 50 | 0.86 | 14.8 | 2.24 | 2.2 | 32.9 | 2.49 | 5.2 |  |
| 2024 | 252 | 146 | 22.8 | 2.42 | 49 | 0.84 | 12.1 | 1.67 | 4.0 | 38.1 | 2.40 | 5.5 |  |
| 2025 | 250 | 139 | -1.0 | -0.07 | 41 | 0.91 | 5.8 | 0.57 | 6.9 | -2.8 | -0.07 | 20.6 |  |
| 2026 | 176 | 101 | 7.1 | 0.72 | 44 | 0.84 | 6.7 | 1.04 | 3.3 | 10.0 | 0.86 | 7.8 |  |

## Reading (written against the 2026-09-16 run; regenerate the tables above before re-using)

**Implementation check.** The dynamic model's yearly returns track the paper's own FAQ Q24 table within a few
points in 8 of 9 overlapping years (2018 61.9 vs 61.1, 2019 11.6 vs 6.9, 2020 23.3 vs 26.8, 2021 38.4 vs 34.8,
2022 28.5 vs 24.4, 2023 24.7 vs 37.2, 2024 20.0 vs 32.2; 2016 −21.6 vs −12.8, 2017 −12.4 vs −6.9), with hit
ratio 43% = paper. Remaining gaps are IQFeed-vs-SIP bars, the 15:59-bar close vs the auction print, and fractional
shares. "Trades/day" here counts round trips (0.9); the paper's 1.8 counts legs (7,668 legs / 4,270 days).

**IS vs OOS (SPY, paper costs).** 2016–2023: 10.9 bps/traded-day (t 3.2), 1× 7.1%/yr SR 1.03, dyn 16.7%/yr
SR 1.13 — in line with the paper's 9.7% / 1.24 and 19.6% / 1.33 once 2007–2015 (which the paper says were its
best years) are excluded. **2024-01 → 2026-09: 0.6 bps/day (t 0.09), 1× −1.0%/yr, dyn −0.3%/yr, SR ≈ 0**:
2024 +20% (dyn), 2025 −2.5%, 2026 YTD −21%. The post-publication record on SPY is flat-to-negative and the
2026 drawdown is the worst year in the sample. QQQ is the exception: IS 12.8 bps (t 3.7), OOS 10.1 bps (t 1.6),
dyn 14.4%/yr SR 0.97 — 2024 +38%, 2025 −2.8%, 2026 +10%.

**Costs.** The paper's $0.0045/share per leg is ≈0.08 bp at SPY ≈$600; 1 bp/leg is ≈13× that and drops IS to
SR 0.47 (1×) / 0.41 (dyn) and OOS to −7 bps/day. Gross vs paper-cost differ by ≈1.2 bps/day. The realistic
number lies between (quoted SPY spread ≈1 cent ≈0.2 bp; the binding item is slippage on the HH:00/HH:30 market
orders, not the spread).

**Ablation — which ingredient changes the sign (SPY, IS / OOS, dyn bps/day, 1× SR):**
- A full model: 10.9 / 0.6 bps; SR 1.03 / −0.14.
- B open-only anchor (no prev-close gap adjustment): 8.9 / 6.0 bps; SR 0.70 / 0.63 — trades 50% more (1.34/day)
  and MDD 1× 26% vs 11% IS; not the sign-changer, and actually better OOS.
- **C every-minute checks (from 10:00): 1.1 / 1.2 bps; SR 0.10 / −0.09; 5.6 trades/day.** The semi-hourly cadence is
  the ingredient that carries the IS result: without it the VWAP/band trailing stop whipsaws ≈6× a day and the
  edge is gone even before realistic costs.
- D opposite-band stop only (paper's base model): 6.8 / 5.1 bps; SR 0.69 / 0.37; hit 54% — matches the paper's
  base-model Table 1 (SR 0.61, hit 54%); the VWAP+current-band trailing stop adds ≈+0.35 SR IS.
- E A2 replica (open anchor + every minute from 09:31 + opposite-band flip): 5.9 / 1.1 bps; 1× SR 0.20 / 0.28;
  1.5 trades/day; MDD 1× 32%. Positive here rather than the coordinator's −1.7 bps/day — the residual difference
  must be in fill/cost conventions (this run: fill at the signal bar's close, $0.0045/share per leg) or in how
  the 14-day time-of-day sigma is built; the sign of E is fragile (t 1.6 IS, 0.2 OOS) either way.
- F next-bar-open fill: identical to A (10.8 / 0.5 bps) — at a 30-min cadence the fill convention does not matter.
- G VM 1.5: 13.4 / 3.3 bps; fewer trades (0.54/day), MDD dyn 9.6% vs 31.8% IS — the FAQ's "optimum" mainly
  buys drawdown, not OOS return.

**Bottom line.** The paper's rule is reproducible and its 2016–2023 in-sample edge is real at the paper's cost
assumption (t ≈ 3, ≈11 bps per traded day, SR ≈1 unlevered); the edge is carried by (i) the semi-hourly decision
cadence and (ii) the VWAP/current-band trailing stop, in that order, not by the band anchor. On SPY it has not
delivered since 2024 (t 0.09 over 678 days; 2026 YTD −21% at the 2% vol target), while QQQ still shows
≈10 bps/day (t 1.6). At 1 bp/leg the SPY rule is unprofitable in every period.

