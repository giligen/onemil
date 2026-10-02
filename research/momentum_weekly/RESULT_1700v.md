# RESULT 1,700v -- information-discreteness (continuous-path) selection on the guarded sleeve (PREREG_1700v.md, FROZEN)

GREF reproduced BEFORE any cell was read: CAGR 29.34% / max DD -38.3% / end $596,394 (target 29.34 / -38.3 / 596,394); ratio 0.77, halves 0.67/0.99. 2017-01..2026-09, $50K, 1700s engine, guard ON.
ID = sign(window ret) x (share down days - share up days) of close-to-close returns, window rows t-252..t-21 (231 returns) or t-126..t-21 (105); zero-return days in neither share, kept in the denominator; t = Friday close before the Monday rebalance.
A50/A67: eligible guarded universe, drop ID above the cross-sectional median / 2/3 quantile, rank the rest by sleeve score. B40/B60: K best by score, hold the N lowest ID.
causality: all data after 2021-06-04 deleted, signals+selections rebuilt for that date: 8/8 cells identical holdings (mismatch: none); eligible n=409
Sanity ID prints (AMC, NVDA on 2021-06-04, 2024-06-07): see 1700v.out.
| cell | CAGR | DD | ratio | end $K | h1/h2 ratio | beta | ov | corr | pairedEx5 | I | R | P |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A50|w252|N20 | 24.8% | -46.3% | 0.53 | 422 | 0.48/1.02 | 1.24 | 0.86 | 0.96 | -0.248% | . | . | . |
| A50|w252|N30 | 22.6% | -43.7% | 0.52 | 356 | 0.49/0.88 | 1.18 | 0.86 | 0.94 | -0.340% | . | . | . |
| A50|w126|N20 | 22.3% | -41.2% | 0.54 | 346 | 0.61/0.72 | 1.22 | 0.78 | 0.95 | -0.292% | . | . | . |
| A50|w126|N30 | 20.5% | -38.8% | 0.53 | 302 | 0.54/0.67 | 1.19 | 0.77 | 0.94 | -0.367% | . | . | . |
| A67|w252|N20 | 26.6% | -44.3% | 0.60 | 485 | 0.58/1.03 | 1.27 | 0.93 | 0.98 | -0.178% | . | . | . |
| A67|w252|N30 | 23.6% | -43.4% | 0.54 | 385 | 0.55/0.90 | 1.19 | 0.93 | 0.96 | -0.289% | . | . | . |
| A67|w126|N20 | 21.9% | -46.2% | 0.47 | 336 | 0.47/0.75 | 1.26 | 0.88 | 0.98 | -0.228% | . | . | . |
| A67|w126|N30 | 22.3% | -38.4% | 0.58 | 347 | 0.56/0.77 | 1.20 | 0.88 | 0.96 | -0.294% | . | . | . |
| B40|w252|N20 | 19.0% | -46.3% | 0.41 | 268 | 0.37/0.86 | 1.15 | 0.56 | 0.90 | -0.462% | . | . | . |
| B40|w252|N30 | 21.7% | -41.2% | 0.53 | 331 | 0.50/0.86 | 1.17 | 0.77 | 0.93 | -0.387% | . | . | . |
| B40|w126|N20 | 18.4% | -43.7% | 0.42 | 254 | 0.39/0.71 | 1.20 | 0.56 | 0.92 | -0.435% | . | . | . |
| B40|w126|N30 | 21.8% | -37.7% | 0.58 | 334 | 0.56/0.76 | 1.20 | 0.77 | 0.95 | -0.351% | . | . | . |
| B60|w252|N20 | 16.4% | -46.3% | 0.35 | 216 | 0.40/0.65 | 1.14 | 0.45 | 0.88 | -0.536% | . | . | . |
| B60|w252|N30 | 19.3% | -39.8% | 0.49 | 274 | 0.43/0.95 | 1.13 | 0.58 | 0.90 | -0.479% | . | . | . |
| B60|w126|N20 | 17.5% | -44.1% | 0.40 | 236 | 0.40/0.69 | 1.15 | 0.42 | 0.89 | -0.485% | . | . | . |
| B60|w126|N30 | 17.9% | -44.5% | 0.40 | 244 | 0.40/0.75 | 1.16 | 0.57 | 0.91 | -0.471% | . | . | . |

COUNTS: improve (ratio >= GREF 0.77+0.10 AND CAGR >= 25%) 0/16 (need 12); ratio beats GREF in both halves 0/16 (need 12); paired weekly diff >= 0 ex-top-5% 0/16 (need 8); median cell shallower by >= 3 pts in 1/3 GREF episodes (need 2).
VERDICT: FAIL. Improve by selection {'A50': 0, 'A67': 0, 'B40': 0, 'B60': 0}, window {126: 0, 252: 0}, N {20: 0, 30: 0}.
Median cell (9th of 16 by ratio): A50|w252|N30 CAGR 22.59% / DD -43.7% / end $355,842 / ratio 0.52 / Sharpe 0.85 / beta 1.18 (alpha t 0.9) / overlap with GREF 86% / weekly corr 0.94 / turnover 21.1x, cost 2.91%/yr / paired -0.143%/wk (t -1.8, ex-top5% -0.340%) / worst yr -8.3% / yrs>SPY 6/10.
GREF three deepest episodes (GREF depth -> median cell depth): 2021-02-16..2021-05-11 -38.3% -> -43.3%; 2020-02-14..2020-03-19 -37.1% -> -35.8%; 2025-02-13..2025-04-07 -33.6% -> -24.8%
Median cell by year (GREF | cell | SPY): 2017: 20%|14%|20% 2018: -5%|-3%|-6% 2019: 15%|24%|31% 2020: 85%|73%|18% 2021: 25%|-1%|30% 2022: -0%|-8%|-19% 2023: 17%|15%|27% 2024: 41%|33%|25% 2025: 38%|37%|18% 2026: 73%|56%|13%
Adversary caveats: (1) 16 neighbours of one published idea, not independent; family count +16; all share GREF engine, band-model costs (not NBBO) and the guard, whose own selection (273-bar lookback) was whole-sample chosen. (2) 2020 crash is the shared floor; ID cannot see it. (3) A-cells use a cross-sectional ID quantile over the whole eligible universe (~ price>=10, ADV>=200M); B-cells a rank-within-top-K re-order, so the N lowest-ID rule is also a K-N cut. (4) Two half-splits only; paired ex-top-5% trims the best weeks only. (5) Daily bars: zero-return days include illiquid unchanged closes; ID denominators keep them. (6) N=30 cells are compared with GREF N=20 (paired/overlap use GREF at the cell N for overlap only).
