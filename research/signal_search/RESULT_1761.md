# RESULT 1,761 — daily first close above 52-week / 20-day high: KILL

Daily caches 1748/1746 (7,292 symbols), TEST cut at load. Universe at t−1: rank ≤ 3,000, raw px ≥ $3. Event: adjusted close above the prior 252-d (20-d) close-high after 20 days below; volume ≥ 1.5× prior-20-d mean. Entry open t+1, exit close t+5/t+20, excess vs the t−1 universe. Cost 2 bp/leg; sensitivity adds measured exit half-spread (median 9.6, mean 15.8 bp; extreme-loser sample). Scale-check drops 0.35 %. Data start 2016-01, so the 252-d high exists only from late 2016.

Primary (252-d, rank 1,001–3,000, 5 d, with volume), excess bp (day-t):
- TRAIN 16–18: +24 (1.3), 19.1 events/wk
- TRAIN 19–20: +58 (2.1), 17.0/wk
- TRAIN: +37 (2.4), ex-top-5 % −84
- VAL 21–23: +15 (0.64), 15.7/wk, ex-top-5 % −120

VAL also: capped +20 % −30 bp; green months 53 %; MDE 67 bp; years 21 +10, 22 +95, 23 −23; net +11 at 0 spread, +2 with median exit half-spread, −4 with mean. Kill rules hit: VAL excess ≤ +30 bp, |t| < 2, ex-top-5 % ≤ 0 (the mean sits in the right tail). Halves same sign, 2022 positive, frequency passes.

Other reads (VAL, 5 d, rank 1,001–3,000): 20-d high with volume +6 bp (t 0.4; TRAIN −11); 20-d without volume +6 (t 0.8; TRAIN −12). 252-d without volume +33 bp (t 2.3, ex-top-5 % −70, net +29; TRAIN +13, t 1.4, ex-top-5 % −79): more events, same tail dependence. Price $3–10 +71, $10–30 +30, >$30 −10 (small n).

Caveats: close-based highs; survivor-leaning cache; overlapping 5-d windows (week-clustered t 0.61); about 300 cells counted.

CHILDREN:
1. Family anchoring, parent 1,761: 252-d breakout without the volume filter, rank 1,001–3,000, $3–10 only, 5 d long. Kill: VAL excess ≤ +50 bp, or ex-top-5 % ≤ 0, or TRAIN halves opposite.
2. Family new timing, parent 1,761: mirror, first close below the 252-d low with volume, short 5 d. Kill: VAL gross after 30 bp borrow plus 4 bp ≤ +30, or ex-top-5 % ≤ 0, or 2022 negative.
