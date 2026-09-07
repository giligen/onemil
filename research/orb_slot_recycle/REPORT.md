# ORB no-fill slot recycling — REPORT (2026-09-07 22:40 UTC) — REJECT, hypothesis closed

Replay validated: the simulator reproduces the pipeline's honest book to the dollar (114 picks / 67 fills / $6,531 / MDD −$551). Breakout minutes from a full bar walk (`breakout_times.csv`, 12,851 candidates; trigger = range_high × 1.003 at/after 09:35).

| T (min) | max/day | picks | fills | fills/mo | total | MDD | red | 25H1 | 25H2 | 2026 | recycled fills | recycle vetoed | released | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | — | 114 | 67 | 3.19 | 6,531 | −551 | 6 | 2,442 | 1,446 | 2,643 | 0 | 0 | 0 | |
| 15 | 1 | 58 | 56 | 2.67 | 5,527 | −683 | 8 | 2,129 | 1,461 | 1,937 | 1 | 28 | 59 | REJECT |
| 15 | 3 | 60 | 56 | 2.67 | 5,527 | −683 | 8 | 2,129 | 1,461 | 1,937 | 1 | 32 | 59 | REJECT |
| 30 | 1 | 66 | 62 | 2.95 | 6,401 | −683 | 7 | 2,649 | 1,653 | 2,100 | 1 | 22 | 53 | REJECT |
| 30 | 3 | 68 | 62 | 2.95 | 6,401 | −683 | 7 | 2,649 | 1,653 | 2,100 | 1 | 25 | 53 | REJECT |
| 45 | 1 | 68 | 64 | 3.05 | 6,760 | −571 | 6 | 2,484 | 1,653 | 2,623 | 1 | 22 | 51 | REJECT |
| 45 | 3 | 70 | 64 | 3.05 | 6,760 | −571 | 6 | 2,484 | 1,653 | 2,623 | 1 | 20 | 51 | REJECT |

## Why it fails — two mechanics, both visible in the numbers
1. **Releasing a slot at T cancels picks that trigger LATER.** Baseline fills 67; at T=15 only 56 of them had triggered by then, 62 by T=30, 64 by T=45. The "no-fill" picks are partly late fills, and a late fill under the static lock is still a fill. Every release cutoff costs real fills before it can add any.
2. **There is almost nothing to recycle INTO.** Across 51–59 released slots per variant, the next-ranked eligible candidate produced ONE fill. 20–32 replacements were vetoed by the same PDR/G1/range-size/catalyst stack (below the top-3 the ranked list is mostly what the vetoes exist to remove); the rest never triggered either. The fill-rate lever the 44% no-fill number suggested is empty: the no-fills are not "slots wasted on the wrong pick", they are mornings where the breakout did not come.

## Consequence for the ORB ROI question
Fills per month cannot be raised by slot management. What remains: budget (the ramp, adopted 9/7), the selection refit (`research/orb_refit_walkforward/REPORT.md`, +11–36% total / MDD −60% on the honest book, proposal pending), and the wrapper in/out decision (2026-first says out). Nothing else on the fill side.
