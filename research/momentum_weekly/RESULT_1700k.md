# RESULT 1,700k -- diversification caps on the momentum sleeve (PREREG_1700k, 8 cells + REF)

Engine = 1700j reconciled book (daily sim 2017-02-06..2026-09-28, $50K, open-to-open, Monday equal weight /20, delta costs, band cost not NBBO). Corr 63d / beta 252d daily close-to-close through the prior trading day; names with <40 valid days = unknown, placed after all known names. Clustering: scipy average linkage on 1-corr (unknown names = singleton). Caps that cannot reach 20 names hold fewer (rest cash). SPY 15.1% / -32.0%. Ref episodes (peak->trough): 2021-02-16..2022-07-14; 2020-02-14..2020-03-19; 2025-02-13..2025-04-07; 2025-10-15..2025-11-21; 2026-06-03..2026-07-29.

| cell | CAGR | max DD | end $ | ep1..5 depth | worst yr | yrs>SPY | roll5y | pair corr | turn/yr | cost/yr | cut eps | <20 wks | unk | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REF | 27.2% | -44.5% | 507,823 | -45% -37% -34% -29% -29% | -7.0% (2018) | 6/10 | 100% | 0.31 | 11.7x | 3.27% | - | 0 | 0/0 | no |
| K1_corr70 | 24.9% | -40.8% | 426,103 | -41% -38% -33% -29% -30% | -6.2% (2018) | 6/10 | 98% | 0.27 | 13.1x | 3.67% | - | 0 | 0/0 | no |
| K2_corr60 | 19.3% | -39.7% | 274,502 | -40% -30% -32% -27% -27% | -12.7% (2022) | 6/10 | 95% | 0.23 | 14.8x | 4.12% | - | 7 | 0/0 | no |
| K3_clu4 | 22.8% | -42.6% | 361,498 | -43% -31% -33% -29% -27% | -11.5% (2022) | 5/10 | 95% | 0.26 | 12.8x | 3.56% | - | 5 | 0/0 | no |
| K4_clu3 | 20.8% | -40.8% | 307,961 | -41% -31% -33% -28% -26% | -10.2% (2022) | 5/10 | 95% | 0.25 | 13.3x | 3.71% | - | 6 | 0/0 | no |
| K5_beta2 | 22.0% | -42.8% | 339,056 | -43% -37% -21% -19% -9% | -3.2% (2018) | 6/10 | 98% | 0.29 | 12.2x | 3.38% | 345 | 0 | 0/0 | no |
| K6_K1K5 | 19.8% | -43.5% | 285,545 | -44% -37% -22% -20% -7% | -3.8% (2018) | 6/10 | 91% | 0.26 | 13.2x | 3.63% | 345 | 0 | 0/0 | no |
| K7_noADR | 25.7% | -42.6% | 451,904 | -43% -37% -32% -31% -30% | -5.8% (2018) | 5/10 | 100% | 0.31 | 11.5x | 3.21% | - | 0 | 0/0 | no |
| K8_K3T1 | 19.1% | -34.5% | 269,444 | -35% -22% -31% -30% -23% | -10.1% (2018) | 4/10 | 95% | 0.26 | 15.5x | 4.39% | 2 | 5 | 0/0 | no |

Pass list: NONE. Best CAGR/|DD|: K1_corr70 (24.9% / -40.8%); REF 27.2% / -44.5%, CAGR/|DD| 0.61.
K7 2020 return 68.1% vs REF 85.3% (cost of the ADR exclusion in the 2020 China-ADR year). ADR-named symbols in assets: 4712.

Which episodes each cap cuts: no cell passes. K1/K2 (correlation caps) shave ep1 by 4-5 points and ep2 only at 0.60 (-37 to -30), leaving eps 3-5 unchanged: the cap lowers pair correlation (0.31 to 0.27/0.23) but equal-weight top-ranked names still fall together. K3/K4 (clusters) cut ep2 (-37 to -31) and shave ep1 by 2-4 points; they also hold fewer than 20 names in 5-7 weeks and lose 4-6 CAGR points. K5/K6 (beta cap) is the only lever that cuts the later episodes (ep3-5 depths -21/-19/-9 vs -34/-29/-29, 3 of 5 cut by 25 percent) but leaves ep1-2 (-43/-37) untouched, so max DD barely moves (-42.8) for 5 CAGR points. K7 (ADR exclusion) costs 1.5 CAGR points, cuts ep1 by 2 points and 2020 by 17 points of return (68 vs 85 percent). K8 (K3 + 15 percent stop) has the best max DD (-34.5, 10 points better, ep2 -22) but loses 8.1 CAGR points. Frontier remains: every 1 point of DD bought costs 0.7-1.0 CAGR points. Caveats: band costs not NBBO; 'cut eps' column lists episode numbers (K5/K6 '345' = episodes 3,4,5); unknown-name count 0 because the 252-day signal already guarantees >= 40 return days.
