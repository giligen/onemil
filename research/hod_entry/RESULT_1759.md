# RESULT 1,759 — F3b leveraged-ETF close rebalance, SPY/QQQ minutes (PREREG_1759)

Verdict: **KILL** (SPY, r', k 1 %, 15:00 to 15:55, cost 2 bp). Scorer `1759_score.py`, grid `1759_grid.csv`.

Data: 2016-01-06 to 2023-12-29; TEST rows dropped 235,358 (SPY); days dropped for missing bars 21 SPY / 25 QQQ; days kept 1,991 / 1,987.

VAL (2021-23) SPY primary: n 200, mean -0.30 bp, t -0.10, hit 50.5 %, ex-top-5 % -4.8, worst -186, MDE 8.1 bp. Up n 108 +1.7 bp (t 0.6); down n 92 -2.6 bp (t -0.5). By k: 0.5 % n 396 +1.5; 1.5 % n 100 +0.6; 2.0 % n 42 +4.0 (t 0.6). None clears +5 bp / t 2.

TRAIN primary: 2016-18 n 110 -4.2 bp (t -0.9); 2019-20 n 135 +6.0 (t 0.9); halves opposite. Years: 2016 -11.2 (t -2.6), 2017 +9.6 (n 6), 2018 0.0, 2019 -1.5, 2020 +8.8, 2021 -11.3 (t -2.7), 2022 +2.9, 2023 +3.3.

Comparator |r'| < 0.5 % (VAL n 353): -0.06 bp, not below the trade, so no gradient with move size.
15:30 to 15:55 variant (VAL n 200): -5.2 bp, t -2.6, hit 42 %: late flow mean-reverts.
Open-based r (VAL n 135): +1.2 bp, t 0.3.
QQQ (VAL n 301): +6.8 bp, t 2.5, hit 57 %, ex-top-5 % +1.2 bp. A secondary read among about 100 grid rows with no TRAIN support: a lead only.

Caveats: days treated as iid; event days (FOMC) not controlled; 2017 k 1 % n 6. MDE 8 bp at n 200, so effects under 8 bp are not excluded.

CHILDREN:
1. QQQ-only flow: PREREG QQQ r' k 1 % with TRAIN halves, scaling by TQQQ/SQQQ AUM, as a fresh cell.
2. Reversal: 15:30 to 15:55 faded against |r'| ≥ 1 % (VAL implied about +1 bp net), under its own PREREG with TRAIN halves and tail check.
