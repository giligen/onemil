# REBUILD 1,703a/b (independent, from PREREG Amendment 1 prose; REBUILD_1703ab.py, log REBUILD_1703ab.log)
Window 2017-01-03 .. 2026-09-28 (509 Mondays, 117 month-ends), $50K, marks at the open. U2 + hygiene (gtype==0) as 1700s_lowvix.py.

| cell | CAGR | max DD (daily) | end $ | Sharpe | worst yr | 2-sided turnover/yr |
|---|---|---|---|---|---|---|
| a1 weekly top-20, 1/N reset | -4.8 % | -58.7 % | 30,975 | -0.21 | -28.8 % (2022) | 66.6x |
| a2 G&H monthly top-30, 6-mo overlap | +10.95 % | -29.5 % | 137,511 | 0.74 | -10.1 % (2022) | 3.8x |
| b dual momentum (SPY/EFA vs SHY, else IEF) | +6.02 % | -38.7 % | 88,314 | 0.45 | -25.0 % (2022) | 4.7x |
| SPY buy-and-hold (open marks) | +15.2 % | -32.1 % | 197,855 | 0.90 | -18.7 % | 0 |
By year: REBUILD_1703ab_by_year.csv. a2 ramps in over the first 6 months (1/6 of cash per month; ~2.5 % CAGR-drag possible early).

## Ties (a)
Names tied at ratio = 1.0 per Monday: median 0, mean 0.24, max 7, never >= 20. The tie-break (126-session return) is
therefore INERT; the suspected defect is NOT the cause. a1 reproduces the first run (-4.8 % / -58.7 %) to the digit.
a1 loses because the top-20-by-ratio set churns ~1.3x of the book per week (66x/yr two-sided; band cost 5-20 bp/side)
and the near-high names are low-vol defensives/mean-reverting at a weekly horizon; a2 (published form) works: +10.95 %.
Top-20 (signal-date ratio): 2021-02-01 ACIA .997 ABT .991 CMG .980 RP .971 BRK.B .965 ANET .960 CGC .960 AVGO DHR NOW TMO
MSFT WORK PSA LLY TXN GOOG CMCSA TTWO GOOGL; 2024-06-03 HCA RTX SO PM DECK MCK TMUS BAC WMT T BBY WELL GAP VRTX TSCO VRSK NEE GD
LMT KDP (all .993-.999); 2026-09-21 TWST 1.000 NTRA HALO SMTC MPC VLO DGX ABBV PSX DINO FFIV TMO RBRK GH WAT AAPL TD VRSN IQV GILD.

## b hand-checks (panel adjusted closes: close_t / close_t-252 - 1; hold, then next-month return net of 2 bp)
Holdings overall: SPY 71, EFA 29, IEF 16 months. 2020-03-31: SPY 235.37/256.09 = -8.09 %, EFA 44.00/52.33 = -15.92 %, SHY +5.30 % -> IEF (Apr -0.2 %).
2020-04-30: SPY +1.63 %, EFA -11.95 %, SHY +5.37 % -> IEF (May -0.0 %); SPY back in from 2020-05-29 (SPY +11.4 % > SHY +4.7 %).
2022 month-end signal (SPY / EFA / SHY 12m) -> hold, return of the following month:
01-31 +21.3/+7.0/-1.4 -> SPY (-3.5 %); 02-28 +13.6/+0.4/-1.8 -> SPY (+4.5 %); 03-31 +14.2/-1.0/-3.1 -> SPY (-9.1 %);
04-29 +0.0/-9.2/-3.7 -> SPY (+0.8 %); 05-31 -0.3/-10.8/-3.1 -> SPY (-8.9 %); 06-30 -10.6/-17.5/-3.5 -> IEF (+2.2 %);
07-29 -5.2/-14.3/-3.3 -> IEF (-4.8 %); 08-31 -11.2/-20.3/-4.1 -> IEF (-3.2 %); 09-30 -15.5/-25.2/-5.1 -> IEF (-1.6 %);
10-31 -14.6/-23.2/-4.9 -> IEF (+3.0 %); 11-30 -9.2/-9.0/-4.2 -> IEF (-0.5 %); 12-30 -18.4/-14.5/-3.9 -> IEF (+2.6 %).
All 14 rows agree with the closes printed in the log (HAND lines). Why b is only 6 %: SHY's 12m return was 3-6 % in 2019-20 and
2023-25, so the rule sits in IEF when SPY < ~4 % (2019, 2020-21 H1, 2022 whipsaw: IEF -> Oct-22/Mar-23 lagged the rebound);
2022 IEF itself lost ~-17 % (SPY->IEF switch came after the Jan-May SPY drop). Literature DD ~-20 % not reproduced: 2022 -25 %.

## Causality (shift test: ALL rows after the signal date deleted, features recomputed)
a1 2021-01-29 signal: identical 20 holdings. a2 2022-06-30: identical 30. b 2022-06-30: identical decision and returns (IEF).
## Suspicious / caveats
- a1 ties inert, so Amendment 1's tie hypothesis is refuted; a1's failure is the weekly 1/N reset on a churning set + cost, not a bug I can find.
- Open-mark DD on all cells; a2 first 5 months partially in cash (ramp); a2 sells/buys whole tranches (no cross-tranche netting).
- U2 incl. delisted names held at the last open (ffill), as the sleeve. Prints of 2021-02-01 names (ACIA, CGC, WORK) are plausible U2 members.
