# RESULT 1,746 — FINRA short-volume ratio, liquid tier (E1) — Stage 1 — KILL

Source: Vortex Capital (Oct 2026) https://www.vortexcapitalgroup.com/insights/the-50-percent-short-volume-myth-what-the-daily-short-sale-file-actually-predicts ; DLW RFS 2009 https://diether.org/papers/short_strategies.pdf (both via CANDIDATES E1).

## Coverage (VOID rule first)
CNMS-only: 2,083 / 2,731 trading days (76.3 %) -> **VOID by the literal PREREG** (CNMS files exist only from 2018-08-01).
Amendment 1 (post-hoc, logged): pre-2018-08 days consolidated by summing the per-facility files (FNSQ+FNYX+FORF+FNQC+FNRA);
validated on 5 overlap days (2018-08..2019-03): 100 % exact equality on every matched symbol. Coverage then 2,731/2,731, LOST 0.
Alpaca bars: 3,997 candidate symbols (median FINRA volume >= 100k), 2,998 returned (LOST 999 = delisted, survivorship-biased; reject stands, a pass would need PIT re-read).

## Universe (liquid: 20d median dollar volume > $100M at t-1, raw close >= $5; names/day)
2016 353, 2017 373, 2018 444, 2019 431, 2020 490, 2021 587, 2022 572, 2023 519 (deciles of ~35-60 names).

## Numbers (L = bottom-SVR decile, S = top; gross, entry open t+1, exit close t+5; NW lag 5)
| split | LS bp | NW t | iid t | ex-top-5 % bp | green months | MDE bp |
|---|---|---|---|---|---|---|
| TRAIN | +2.8 | 0.63 | 0.89 | -107 | 53 % | 12.6 |
| 2016-18 | +2.6 | 0.55 | | | | |
| 2019-20 | +3.2 | 0.37 | | | | |
| VAL | +10.1 | 1.20 | 1.90 | -112 | 53 % | 23.6 |

Year LS bp (liquid): 2016 -12.1, 2017 +16.3, 2018 +2.9, 2019 -3.4, 2020 +9.8, 2021 -1.6, 2022 +31.1 (t 1.7), 2023 +0.6.
Next-day exit (r1): TRAIN +2.7 (t 2.1), VAL -1.1 (t -0.5). SVR 5-day average: VAL +5.5 (t 0.5). $10-100M tier: VAL +6.4 (t 1.1), r1 VAL -2.9.
Sign matches the claim (low SVR > high SVR) but the size is 3-10 bp, not 27 bp; the published "never positive" year pattern is not reproduced (2016, 2019, 2021 negative).

## Cost
Measured NBBO half-spread 09:30:05-09:35 on entry day, 400 VAL name-days: mean 13.8, median 10.5, p90 27.7 bp.
Net VAL = 10.1 - 2x(13.8+2) = -21.5 bp (entry half-spread only); -49 bp with round-trip spread. Net < 0 in every split.

## KILL list (literal)
VAL |NW t| 1.2 < 2: KILL. ex-top-5 % VAL <= 0: KILL. Net VAL <= 0: KILL. TRAIN halves same sign and 2022 positive: pass. **KILL.** TEST not read (sealed).
Caveat: ex-top-5 % is the one-sided repo definition (drops the best 5 % of leg name-days), harsh by construction, but the kill stands on t and cost alone.
Cells: 1,746 (24 reported rows = 1 cell, 3 variants). Files: 1746_fetch.py, 1746_score.py, 1746_cost.py, 1746_summary.csv, 1746_yearly.csv.
