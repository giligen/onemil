# RESULT 1,700q — do fear gauges predict the next week? (PREREG_1700q.md, frozen 10/2)
Run: `1700q_fear.py` (one process, 76 s, 2.5 GB cage), `1700q_trade.py` (tradable test), log `1700q.log`. Outputs `1700q_cells.csv` (36 rows), `1700q_series.parquet`, `1700q_model.csv`, `1700q_trade.csv`.
**REF reproduced before any predictor was read: CAGR 27.18 %, max DD -44.50 %, end $507,823.**

## Base rate first (the "always up" forecast)
Share of up weeks (Monday open to next Monday open): SPY 2017-26 **61.8 %** (n 503); sleeve 57.7 %; sleeve minus SPY 54.1 %; SPY 1993-2026 58.4 % (n 1,756).

## Predictor x target (2017-01..2026-09, 503 wks; top-minus-bottom quintile spread in %-points / t)
| pred | T1 SPY | T2 sleeve | T3 sleeve-SPY | T1 halves 17-21 / 22-26 | T1 ex-5 % tails | T1 mono | T1 MDE |
|---|---|---|---|---|---|---|---|
| P1 | +0.26 / +0.6 | +1.60 / +2.2 | +1.34 / +2.7 | +0.18 / +0.35 | -0.02 | +0.3 | 1.17 |
| P2 | +0.48 / +1.2 | +0.83 / +1.1 | +0.36 / +0.6 | +0.81 / +0.08 | +0.01 | +0.7 | 1.11 |
| P3 | +0.57 / +1.4 | +1.97 / +2.7 | +1.40 / +2.8 | +0.54 / +0.77 | +0.41 | +0.7 | 1.17 |
| P4 | +0.31 / +0.8 | +1.40 / +1.9 | +1.09 / +2.2 | -0.02 / +0.50 | -0.12 | +0.6 | 1.15 |
| P5 | +0.36 / +0.9 | +0.94 / +1.3 | +0.59 / +1.1 | +0.13 / +0.68 | +0.03 | +0.2 | 1.13 |
| P6 | -0.33 / -0.8 | +0.25 / +0.3 | +0.57 / +1.0 | -0.15 / -0.55 | -0.39 | -0.5 | 1.19 |
| P7 | -0.44 / -1.0 | -0.27 / -0.4 | +0.17 / +0.4 | -0.75 / -0.27 | -0.00 | -0.9 | 1.19 |
| P8 | -0.75 / -1.9 | -1.22 / -1.8 | -0.47 / -1.0 | -0.67 / -0.91 | -0.33 | -1.0 | 1.13 |
| P9 | -0.86 / -2.1 | -1.05 / -1.5 | -0.19 / -0.3 | -1.49 / -0.43 | +0.29 | -0.3 | 1.14 |
| P10 | -0.58 / -1.4 | -0.33 / -0.4 | +0.24 / +0.4 | -0.59 / -0.56 | -0.41 | -0.9 | 1.14 |
| P11 | -0.84 / -1.9 | -0.69 / -1.0 | +0.16 / +0.4 | -0.86 / -0.82 | -0.22 | -0.6 | 1.22 |

| long window | n | spread %-pt | t | halves 93-09 / 10-26 | ex-5 % | mono | MDE | pass |
|---|---|---|---|---|---|---|---|---|
| P1 T1 1993-2026 | n 1730 | +0.47 | +2.36 | +0.47 / +0.54 | +0.33 | +0.7 | 0.56 | no |
| P2 T1 1993-2026 | n 1755 | +0.43 | +2.23 | +0.42 / +0.44 | +0.10 | +0.9 | 0.54 | no |
| P9 T1 1993-2026 | n 1755 | -0.70 | -3.34 | -0.58 / -0.89 | -0.06 | -0.9 | 0.59 | PASS |
(P2 1993-2026: spread +0.43, t +2.2, ex-5 % +0.10, no pass.) Other cells (P12 CNN index: NOT RUN, endpoint returns 418 bot-block; no free history, coverage 0 %).
MDE is 1.1-1.2 pp/week on T1 and ~2 pp on T2/T3 in the 503-week window: only a quintile spread above that could be seen; none was.

## Pass rule (|t| >= 3.2, halves agree, sign kept ex-5 % tails, |mono| >= 0.8): **1 of 36 passes**
* The one: **P9 (SPY prior 5-day return), T1, 1993-2026**: spread -0.70 pp (high last week -> lower next week), t -3.34, halves -0.58 / -0.89, quintile means 0.64/0.26/0.13/0.16/-0.07 % (monotone -0.9). **Ex-5 % tails the spread falls to -0.06 pp (sign kept, size gone: ~90 % of it sits in the top/bottom 5 % of weeks).** In the 2017-26 window the same cell is t -2.1 and flips sign ex-tails. It is a tail-carried short-term-reversal effect on SPY, not a tradable signal.
* Best |t| among the 33 sleeve-window cells: P3 (VIX/VIX3M) on T3 t +2.8, P3 on T2 +2.7, P1 on T3 +2.7; none reaches 3.2; halves agree in sign but ex-5 % tails the spreads shrink 35-55 %.
* Multiplicity: 36 tests, Bonferroni t 3.2; at that bar a single t -3.3 on a tail-carried cell is inside what chance gives.

## Joint model (logistic, standardised P1-P11, walk-forward fit through Y-1, predict Y; OOS 2020-2026, n 351)
| target | OOS hit | always-up | z | pred-up / pred-down wks | mean up / down | t | pass |
|---|---|---|---|---|---|---|---|
| T1 SPY sign | 57.8 % | 61.3 % | -1.3 | 317 / 34 | +0.26 % / +0.75 % | -1.0 | no |
| T2 sleeve sign | 53.8 % | 57.5 % | -1.4 | 304 / 47 | +0.54 % / +2.13 % | -1.6 | no |
The model does WORSE than always-up and its predicted-down weeks are the better weeks. Decile calibration is non-monotone (T1 decile 4 predicted 63 % up, actual 31 %; top decile predicted 81 %, actual 51 %).

## Tradable test (run because P9 passed; approximation on the saved weekly sleeve returns, 10 bp per unit exposure switch)
Flag = P9 >= expanding-window quantile (threshold from prior weeks only); sleeve to cash / half size the next week. REF on the same weekly basis: 27.26 % / -41.9 % / $508K.
Cash: q70 15.2 % / -36.4 %; q75 13.8 % / -39.9 %; q80 17.8 % / -33.5 %; q85 20.8 % / -38.4 %; q90 21.0 % / -40.5 %. Half: 21.5-24.2 % / -36.8 to -40.6 %.
**0 of 10 cells improve both CAGR and DD; 0 of 8 neighbours.** Cutting after strong SPY weeks removes the sleeve's best weeks (its T2 is -1.05 pp after high P9, t -1.5, not significant). NOT recommended.

## Verdict
No fear / volatility / credit / breadth / F&G-proxy gauge predicts the next week's SPY or sleeve direction at the frozen bar on 2017-26; the joint model does not beat always-up (it loses 3.5-3.8 pts, z -1.3/-1.4). The one letter-of-the-rule pass (SPY 5-day reversal, 1993-2026) is tail-carried and fails the tradable test. Statement is limited to: these 11 daily gauges, weekly horizon, SPY and this sleeve, 2017-26 (1993-26 for P1/P2/P9), linear-logistic joint form, no put/call leg, CNN index unavailable.

## Checks and adversary caveats
* Shift test: 12 random Fridays, all 11 predictors recomputed on data deleted after that Friday = 0 mismatches; breadth rebuilt from truncated rows on 4 Fridays = 0 mismatches. Percentiles trailing 252-day (min 126 obs); P7/P8 denominators count only names with a full trailing window (an earlier build counted missing windows as zero and was fixed before any read).
* Coverage: every series 100 % of 503 weeks except P11 95.8 % (21 weeks lost to trailing-percentile warm-up in 2017); fetch LOST count = 0 (CBOE VIX/VIX3M/VIX9D/VVIX all fetched via redirect; VIX3M from 2009, VIX9D from 2011, VVIX from 2006: fine for 2017+).
* Price-scale: yfinance SPY vs panel SPY daily-return corr **0.99603 < 0.999 required**; 0.99969 excluding the 10 worst days, all in March 2020 (panel closes differ from adjusted closes up to 2.7 pp on single days). Panel SPY/TLT/HYG/IEF are raw closes (no dividends), so P6/P9/P10 carry small ex-dividend noise. The long-window cells use yfinance only.
* Quintile cut points use the full window (descriptive); the model is the causal read. Entry week = first trading day of the week, so holiday weeks are 4 days; weeks with gaps > 9 days dropped. Breadth universe = point-in-time eligible U2 names but only symbols that ever qualified (survivor-free within the panel, not delisted-complete beyond it).
* The tradable test uses weekly returns and 10 bp switching, not the daily engine (cost inside T2 already); its DD is weekly-compounded (REF -41.9 % vs daily -44.5 %). A rejection this clear does not need the exact engine; a pass would have.
* Not tested (outside the list): intraday VIX, options skew, put/call, nonlinear models, horizons other than one week, conditional-on-regime interactions.
