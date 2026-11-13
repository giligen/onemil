# RESULT 1,757 — three flow reads (1757_score.py). TEST dropped at load: 3,521 signals, 946,097 path rows.
Data are 2025-01..2026-05 only (TRAIN 2025, VAL 2026); no 2022 sign exists. VAL read once, after the TRAIN CSV.

**F1 late-day momentum: KILL.** No volume column, so state = within 1 % of the path high at 15:30. Paths end at the 15:55 bar; the sell is that close less 2 bp. VAL near-high: 40.7/wk, net -12.4 bp, t -2.5, ex-top-5 % -24.8; gross +12.6 bp, so cost eats it. TRAIN -28.3 (halves -28.4/-28.2). Not-near VAL -14.7; near minus not-near +2.3 bp: no state information.

**F2 overnight drift: KILL, and VOID by rail.** RAW daily bars; 2,920 corporate-action rows dropped. Coverage 65.7 % (4,165 arms are 2x wrapper ETFs absent from the daily cache) fails the 80 % rail. Covered VAL top third: 94/wk, excess net -3.5 bp, t -0.26, ex-top-5 % -35.6; top minus bottom -12.6 bp. TRAIN -7.0 (halves -8.3/-5.5). Universe = close >= $1, $vol >= $1M (my definition).

**F4 breadth to SPY: KILL.** SPY from paths.spy_close (no SPY daily on disk), 10:30 to 15:55 bar; 324 of 344 days. TRAIN Q5-Q1 +18.8 bp, t 0.8, halves +75/+13, not monotone (Spearman 0.30). VAL Q5-Q1 +40.2 bp, t 2.4, but Q1 holds 4 days (VAL breadth runs above the TRAIN cuts) and Q5 is +6.8 bp (t 0.8); long-top-quintile book +5.8 bp, 28 days, t 0.7.

Programme count 1,756 to 1,757.

CHILDREN:
1. F1b: near-high at 15:30 on arms with measured half-spread <= 3 bp (gross +12.6 bp needs cost < 10 bp); frequency under 3/wk is the risk.
2. F4b: breadth as an absolute-count regime state, TRAIN-frozen threshold, longer arm history (needs >= 10 cycles).
3. F2b: rerun once wrapper-ETF daily bars exist (restores the 80 % rail).
