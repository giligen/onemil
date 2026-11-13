# RESULT 1,753 — long the post-break dip / range-low retest / trail: KILL ×3

TEST dropped at load: 3,521 signals, 946,097 path rows. No half_spread: 75 TRAIN / 50 VAL arms dropped. TRAIN = 2025 (52 wk), VAL = 2026 (22 wk). A first VAL run crashed before printing any number; VAL was then read once.

**D2 caveat:** paths start at the entry bar, so the pre-break 5-bar low is unavailable. `signals.stop` (= level − 1 R0) stands in for it. That makes D2 = D1(a 1, b 0) with a session-long trigger (to 15:25).

## TRAIN picks (by H2, `1753_grid.csv`)
- D1 a 1 / b 0
- D2
- D4 N 10

All 8 cells are negative in both halves. Gross is +0.04..+0.11.

## VAL (`1753_val.csv`)
| read | fills/wk | net own-R | net R0 | day-t | ex-top-5 % | placebo | B0 same arms | |
|---|---|---|---|---|---|---|---|---|
| D1 | 65.7 | −0.84 | −0.30 | −5.6 | −1.22 | −1.90 | −1.31 | KILL |
| D2 | 124 | −1.20 | −0.31 | −3.7 | −1.52 | −1.41 | −1.16 | KILL |
| D4 | 24.5 | −0.47 | −0.47 | −10.5 | −0.60 | −0.43 ≥ rule | +0.45 | KILL |

The D1/D2 lift over B0 is mechanical. On D4 the trail gives back 0.92 R0 against B0's fixed target (t −15.7).

## Winners vs losers (TRAIN AUC, `1753_anatomy.csv`)
- **Cost share hs/R0:** D4 0.38, D2 0.45. Winners are cheap names.
- **R0 bps:** D4 0.62.
- **Dip-bar close depth:** D1 0.60. Winners' trigger bar closed higher.
- **Later in the day:** 0.56.

CHILDREN:
1. **Cost-killers, parent 1,753 D4.** Held-at-N10 arms with low hs/R0 (H1 tercile) and R0 ≥ median keep B0's fixed +2 R target. Data: exit-lab files. Kill: VAL net ≤ 0, day-t < 2, ex-top-5 % ≤ 0, < 3 fills/wk, or complement ≥ rule.
2. **Mean-reversion harvest, parent 1,753 D1.** Hammer dip: the trigger bar touches level − 1 R0 and closes in the top half of its range. Buy the next open, with a passive limit exit at the level. Same data. Kill: the same five, plus TRAIN gross ≤ 2× round-trip cost.
