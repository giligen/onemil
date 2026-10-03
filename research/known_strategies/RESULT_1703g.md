# Cell 1,703g v2 (OHLC stops, years/365.25)

## 1. BITX tracking check (2x daily reset, sim vs real BITX)
- Source: yfinance download (auto_adjust=False close); BTC priced at 16:00 ET from Alpaca hourly bars; n=819 sessions 2023-06-28..2026-10-01
- Correlation 0.9972 (bar >= 0.99: MET); tracking diff (sim - real) 25.6 %/yr with ER+financing, 30.3 %/yr before costs; real BITX mean 54.7 %/yr vs sim 80.3 %/yr; compounded CAGR real 8.4 % vs sim 42.1 % (BTC 1x 36.6 %); beta real-on-sim 1.015

## 2. Anchor (1x / no stop, 2018-2026)
- This run: 37.8 % / -58.1 % (open-execution, 25 bp ER); 1703e formula on the new OHLC closes: 37.7 % / -58.1 %; target 37.7 / -58.1 (+-0.5 pt)
- Anchor REPRODUCED

## 3. Cells (CAGR % / maxDD % / CAGR/DD / stops hit / h1 / h2 returns)
| L | stop | 2018-26 | 2022-26 | stops | h1 | h2 | pass |
|---|---|---|---|---|---|---|---|
| 1 | none | 37.8 / -58.1 / 0.65 | 9.1 / -51.0 / 0.18 | 0 | 9.95 | 0.51 | n |
| 1 | 2.0 | 27.9 / -68.7 / 0.41 | 6.0 / -47.5 / 0.13 | 16 | 5.53 | 0.32 | n |
| 1 | 1.0 | 22.2 / -60.6 / 0.37 | -2.8 / -38.7 / -0.07 | 61 | 5.63 | -0.13 | n |
| 1 | 0.5 | 12.8 / -69.4 / 0.19 | -11.2 / -54.8 / -0.20 | 109 | 4.05 | -0.43 | n |
| 2 | none | 60.0 / -84.5 / 0.71 | 7.4 / -78.2 / 0.10 | 0 | 42.30 | 0.41 | n |
| 2 | 2.0 | 32.1 / -94.2 / 0.34 | 0.9 / -74.6 / 0.01 | 16 | 9.91 | 0.04 | n |
| 2 | 1.0 | 25.8 / -89.7 / 0.29 | -13.2 / -68.2 / -0.19 | 61 | 13.52 | -0.49 | n |
| 2 | 0.5 | 14.6 / -93.2 / 0.16 | -23.6 / -82.0 / -0.29 | 109 | 10.81 | -0.72 | n |

References (2018-26 / 2022-26 CAGR % / maxDD %): btc_1x_hold 22.5/-81.5 , 13.3/-67.0; btc_2x_hold -9.9/-98.6 , -7.7/-93.3; btc_1x_trend_1703e 37.7/-58.1 , 8.5/-51.0

## 4. Verdict vs PREREG bar (CAGR/DD >= 0.65 AND maxDD better than -45 % AND halves same-signed)
- Cells passing (2018-26): 0 of 8. Best by CAGR/DD: L=2 stop=none 60.0 % / -84.5 % / 0.71
- 2x cells reportable only if the BITX check is met: met

## 5. Adversary caveats
1. BTC UTC-day bars, 24/7: the ETF session (09:30-16:00 ET, weekend gaps) is ignored; real fills are worse at weekend/overnight gaps. Stop gap-through uses the UTC open.
2. Stop level fixed at entry from a 20d mean true range at t-1; flat after a stop until the signal turns negative then positive. 10 bp per switch also on stops; cash earns 0.
3a. BITX tracking: correlation passes the PREREG bar but real BITX trails the sim by ~25 pts/yr (futures roll/basis, swap costs not modelled), so every 2x CAGR here is OVERSTATED by that drag. Pre-2021 OHLC = yfinance scaled to Alpaca at the splice day; sim 2x for pre-2023 is a model (ER 185 bp + T-bill+50 bp financing).
4. 8 cells x 2 windows; tail-dependence shown by extop5_cagr in 1703g_cells.csv; 2018-21 vs 2022-26 regimes differ strongly (h1 vs h2).
5. Independent re-read still required (PREREG) before the owner sees any number.

## 6. Independent re-read (REREAD_1703g.py, written from the PREREG prose, reviewer's own code)
- 1x/none 37.5 % / -58.2 % (agent 37.8 / -58.1); 2x/none 59.1 % / -84.5 % 2018-26 and 6.0 % / -78.2 % 2022-26 (agent 60.0 / -84.5, 7.4 / -78.2): AGREE.
- Executable on the real ETFs (session closes, signal = BTC 20-day at the prior UTC close, 10 bp/switch): BITX hold 18.5 % / -83.4 % and
  BITX-trend 76.9 % / -47.3 % since 2023-06-27 (BTC 36.6 % / -53.1 %); IBIT hold 24.2 % / -53.3 % and IBIT-trend 32.2 % / -26.3 % since
  2024-01-11. One regime (post-ETF bull); the 2022 bear is outside the real-product window — the simulated 2x/none carries that year at -78 %.
- BITX is a FUTURES product: 25.6 %/yr tracking shortfall vs the daily-reset simulation (roll + 185 bp) while held; 54 % time in market => ~14 %/yr.
