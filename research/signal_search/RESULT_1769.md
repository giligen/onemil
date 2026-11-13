# RESULT 1,769 (J8: short the >= 10 % gapper at 09:35, stop = first-5-min high, cover 10:30) -- KILL

PREREG_1769 rule, frozen. TEST 2024+ never fetched.

**Population.** Gap >= 10 %, raw open >= $5. Events per year 2016-23: 381, 389, 513, 527, 1,924, 1,181, 747, 614. Fetch coverage 99.7 % TRAIN / 99.5 % VAL. No bars at 09:30-09:35 (own bucket, untraded): 2.2 % / 1.6 %. First bar late (traded): 3.5 % / 1.5 %.

**VAL 2021-23 (n 2,498, 16.6 names/wk):** net -56.6 bp, day-t -4.2, hit 35 %, stop-hit 48 %, worst-5 % day mean -767 bp, max single loss -2,223 bp, ex-top-5 % -138 bp, weekly P10 -268 bp, green weeks 34 % vs null 47 %. Years 2021 / 2022 / 2023: -22 / -86 / -87 bp. Net at borrow 100 bp: -107.

**TRAIN (n 3,652):** -76.6 bp, day-t -4.6. Halves 2016-18 -69.2, 2019-20 -80.5. Every year 2016-23 negative.

**Gross vs cost.** Gross +61 bp VAL, +40 TRAIN. Charged: 4 bp fees, 63 bp half-spread (band medians of the 1748 sample, quoted at the next-day open, likely overstated for 09:35), 50 bp borrow. Even at zero spread VAL net is about +7 bp, below the +50 bar; TRAIN about -14.

**Sensitivities (not searched).** 1768 stop convention (trigger at H): VAL -85.7. Cover 10:00: VAL -79.3. Gap 5-10 %, $1-5 and the 15:55 cover were not fetched.

**Kill checks failed:** VAL net <= 50, day-t < 2, ex-top-5 <= 0, 2022 negative, borrow-100 <= 0. Passed: halves same sign, worst-5 % > -8 %, >= 3 names/wk.

**Caveats.** Minute bars resolve stop and cover in one bar to the stop. The stop trigger is PREREG-literal (H + 50 bp). The 2026 +142 bp (1768) is a one-year read that 2016-23 does not reproduce.

CHILDREN:
1. Gross-only anatomy by year, 2016-26, before any cost model.
2. Per-trade NBBO at 09:35 on 2025-26 names replacing the open-quote band.
3. Tick tape on the 48 % stop-hit trades.
