# RESULT 1,700j -- drawdown frontier of the reconciled momentum sleeve (V2 top-N weekly)

PREREG_1700j.md + Amendment 1. Daily sim 2017-02-06..2026-09-28 (9.64 y), $50K start, open-to-open, Monday equal-weight delta-trade costs (5 bps + half H-L proxy, cap 20 bps), stops on closes executed next open. Band-based cost, NOT measured NBBO; panel survivorship/adjusted-price caveats of RECON_1700_sleeve apply. SPY: CAGR 15.1%, max DD -32.0%. REF should reproduce ~26.6% / -42% (weekly basis).

## Step 1 -- five deepest REF drawdowns

| # | peak | trough | recovered | depth | SPY dd | loss by peak names / rotation | holdings at peak (top 8 by weight) |
|---|---|---|---|---|---|---|---|
| 1 | 2021-02-16 | 2022-07-14 | 2024-03-06 | -44.5% | -9.1% | 62% / 38% | PACB BLNK FCEL MARA BILI FUTU APPS PDD |
| 2 | 2020-02-14 | 2020-03-19 | 2020-06-16 | -37.1% | -32.0% | 60% / 40% | ROKU SHOP AMD KLAC EQIX TSM ZTS ASML |
| 3 | 2025-02-13 | 2025-04-07 | 2025-07-08 | -33.6% | -18.8% | 70% / 30% | APP HOOD SE PLTR SMR WMT CVNA RKLB |
| 4 | 2025-10-15 | 2025-11-21 | 2026-01-29 | -29.4% | -1.8% | 65% / 35% | QBTS RGTI OKLO QUBT BE RBLX SOFI RKLB |
| 5 | 2026-06-03 | 2026-07-29 | none | -29.1% | -3.9% | 91% / 9% | AAOI AEHR LITE AXTI VIAV CIEN TSEM BE |

Loss split = P&L peak->trough of names held at the peak vs names entered after (rotation). Themes: read from the holdings column.

## Step 2 -- frontier sorted by max DD (pass = DD >=10 pts better, CAGR <=5 pts lower, rolling-5y >=90%, >=3 of 5 episodes cut by >=25% relative)

| cell | CAGR | max DD | end $ | worst yr | yrs>SPY | roll5y | Sharpe | turn/yr | cost/yr | wks red. | ep1..5 depth | cut | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| J1_T2S1 | 18.4% | -30.2% | 255,278 | -9.6% (2018) | 5/10 | 82% (56) | 0.79 | 11.6x | 3.26% | 228 | -30% -27% -24% -21% -15% | 5/5 | no |
| J3_T2D1 | 16.3% | -30.8% | 214,702 | -7.3% (2018) | 3/10 | 86% (56) | 0.72 | 10.0x | 2.81% | 280 | -31% -18% -28% -28% -24% | 2/5 | no |
| J4_T2S1N1 | 16.5% | -30.9% | 217,602 | -7.9% (2022) | 5/10 | 66% (56) | 0.74 | 10.9x | 3.03% | 178 | -31% -29% -24% -23% -17% | 3/5 | no |
| D1 | 16.6% | -31.0% | 218,866 | -4.5% (2018) | 6/10 | 91% (56) | 0.71 | 9.2x | 2.57% | 257 | -31% -23% -29% -28% -24% | 2/5 | no |
| S1_30 | 18.9% | -34.4% | 265,133 | -6.6% (2018) | 6/10 | 68% (56) | 0.80 | 10.1x | 2.79% | 252 | -30% -34% -23% -19% -15% | 4/5 | no |
| T1_15 | 22.7% | -35.8% | 358,840 | -8.0% (2018) | 4/10 | 100% (56) | 0.78 | 14.7x | 4.18% | 0 | -36% -27% -32% -31% -27% | 1/5 | no |
| S2_35 | 21.0% | -36.0% | 314,469 | -7.0% (2018) | 6/10 | 84% (56) | 0.82 | 10.5x | 2.93% | 178 | -34% -36% -26% -22% -17% | 2/5 | no |
| N2_40 | 22.5% | -40.3% | 353,899 | -8.8% (2022) | 6/10 | 98% (56) | 0.83 | 9.8x | 2.69% | 0 | -40% -36% -28% -21% -30% | 1/5 | no |
| J2_T2N1 | 19.8% | -41.7% | 285,917 | -6.8% (2022) | 5/10 | 91% (56) | 0.75 | 12.2x | 3.42% | 0 | -42% -30% -28% -28% -29% | 0/5 | no |
| N1_30 | 23.3% | -42.3% | 376,460 | -8.3% (2022) | 6/10 | 98% (56) | 0.82 | 10.5x | 2.92% | 0 | -42% -36% -28% -26% -32% | 0/5 | no |
| T2_20 | 23.7% | -44.0% | 387,442 | -10.2% (2018) | 5/10 | 98% (56) | 0.80 | 13.5x | 3.81% | 0 | -44% -28% -32% -30% -28% | 0/5 | no |
| REF | 27.2% | -44.5% | 507,823 | -7.0% (2018) | 6/10 | 100% (56) | 0.85 | 11.7x | 3.27% | 0 | -45% -37% -34% -29% -29% | 0/5 | no |
| C1_drift | 23.9% | -51.2% | 395,032 | -5.8% (2018) | 5/10 | 95% (56) | 0.77 | 10.1x | 2.81% | 0 | -51% -37% -34% -35% -32% | 0/5 | no |
| T3_25 | 23.2% | -53.2% | 373,324 | -9.4% (2018) | 5/10 | 95% (56) | 0.78 | 12.8x | 3.61% | 0 | -53% -32% -34% -30% -28% | 0/5 | no |

Pass list: NONE. Best CAGR per |max DD|: T1_15 (22.7% / -35.8%); REF = 27.2% / -44.5%.
Cells: 14 + REF, frozen grid, nothing added after numbers. Stop re-entry rule (blocked at the rebalance where the sale executes) and the 25%-relative "cut" definition were fixed in the script docstring before any read.
