# RESULT 1,767 (J2): where the gapper fade lives

PREREG frozen 21:05 UTC. Population 1,428 (215 gaps > 200 % dropped). Prices = last bar ending at the instant; bp of the 09:30 open; day-clustered t; symbol-median half-spread. TEST sealed.

## Read 1: clock (gap >= 10 %; TRAIN n 586, VAL n 498)
Cumulative bp at 09:35 / 10:00 / 10:30 / 15:55: TRAIN -108 / -294 / -396 / -431 (t -6.4); VAL -82 / -271 / -359 / -437 (t -7.7). Share of the fade done by 10:00: 68 % / 62 %; by 10:30: 92 % / 82 %.

## Read 2: the 10:00 rule
10:00 to 15:55 after a declared fade: -148 TRAIN, -202 VAL; holders -113, -95. Difference (hold minus fade): +35 bp t 0.3 TRAIN, +107 t 1.1 VAL; halves -250 / +298.

## Read 3: high clock
Day high set by 09:35 / 09:45 / 10:00: 44 / 56 / 59 % TRAIN, 54 / 63 / 68 % VAL; later 41 % / 32 %. Drawdown from the first-15-minute high to close: -850 bp.

## Read 4: short at 10:00 if faded and >= $5, stop at the morning high + 50 bp, cover 15:55
VAL: n/wk 10.5, net +110 bp, day-t 1.8, ex-top-5 % -35, worst-5 % mean -1,818 bp, stop-hit 19 %, MDE80 175 bp.
TRAIN: net +24 bp, t 0.3, ex-top-5 % -149, stop-hit 34 %. Halves: -108 (t -1.3) / +135 (t 0.9).

## Verdict: KILL
Fails day-t < 2, halves opposite, ex-top-5 % <= 0, tail worse than -8 %. Names/wk and VAL net > 50 bp pass. The names fall on average, but the short's mean comes from a few full fades and its worst 5 % lose 18 % through the stop.

CHILDREN:
1. Short at 09:35-09:45 with the opening-range high as stop (44 % of the fade is still ahead).
2. Defined-risk expression (puts or a stop-limit) to cap the squeeze tail, same names.
3. Short only when the day's high is already set by 09:45 (causal), score the remainder.
