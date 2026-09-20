# orb_inplay — Cell D: range-stop + static-lock exit

**Verdict: FAIL (all three cells).** Combined VAL net R/trade **-0.057** (need >=+0.10, t_cl
2.0) with t_cl **-0.63** — fails the point estimate before the significance bar is even reached.
Combined TRAIN net **-0.126**, t_cl **-2.82** — significantly NEGATIVE, not merely non-significant.
Long and short fail individually on VAL and TRAIN. Neither side nor the combined book is close to
the bar in either split.

## Numbers (1x book, $66,000 equity, net of measured cost + $0.0035/share/leg)

| split | side | n | fills/wk | gross R (t) | net R/trade | SE(cl) | t_cl | MDE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | combined | 955 | 18.0 | +0.020 (0.46) | **-0.126** | 0.045 | -2.82 | 0.123 |
| TRAIN | long | 554 | 10.5 | +0.076 (1.29) | -0.057 | 0.065 | -0.88 | 0.167 |
| TRAIN | short | 401 | 7.6 | -0.059 (-0.94) | **-0.222** | 0.062 | -3.56 | 0.182 |
| TRAIN_H1 | combined | 463 | 17.1 | +0.085 (1.31) | -0.075 | 0.069 | -1.08 | 0.188 |
| TRAIN_H2 | combined | 492 | 18.2 | -0.042 (-0.73) | -0.175 | 0.058 | -3.04 | 0.162 |
| VAL | combined | 397 | 18.0 | +0.111 (1.40) | -0.057 | 0.091 | -0.63 | 0.233 |
| VAL | long | 231 | 10.5 | +0.105 (1.27) | -0.039 | 0.102 | -0.38 | 0.258 |
| VAL | short | 166 | 7.5 | +0.118 (0.79) | -0.083 | 0.162 | -0.51 | 0.427 |

TRAIN halves are same-signed (both negative: H1 -0.075, H2 -0.175) — item 4 of the pass bar is the
only item cleared. Fills/week clears the >=3/wk floor by 4-6x on every cell. Full detail (WR,
avg win/loss, ex-top-1%/5%, top-5 share): `score_d.out` / `results_D.json`.

**Skip share**: 633/6,309 eligible (ok + skipped) picks, **10.0%**, dropped for range < 0.5% of
price. **Admitted positions/day (1x cap)**: mean 3.84, max 8 — the cap binds on 4,324/5,676
(76%) of `ok` trades because this exit's R (mean **3.1%** of price) is ~6x the base book's R
(0.5%, ATR-derived), so fixed 1%-of-equity risk sizing buys much smaller notional per name and the
1x cap admits only ~4 of the day's 20 candidates before it's exhausted.

## Cadence bar (`scripts/cadence_bar.py`, trades = `trades_D_cadence.csv`)

```
--split VAL
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median 5.0 wk  P90 6.6 wk        [fail]   gaps: [2, 5, 7]
C2 bleed     P90 -9.64 R     cycles net>0 33% [fail]
C3 reds      P10 -6.92 R  min -13.46 R  MDD 37.39 R   under-water max 18 wk   [fail]
C4 green     33%  null 50%                   [fail]
C5 fills/wk  18.05                             [pass]
C6 tail      C6 not audited
C7 power     cycles 3   bootstrap P90-gap 75% UB 8.4 wk    [fail]
```

```
--split TRAIN
CADENCE BAR  (unknown, TRAIN, live config: N/A slots, N/A, R = $N/A)
C1 gap       median 2.5 wk  P90 10.0 wk        [fail]   gaps: [1, 1, 11, 1, 9, 4]
C2 bleed     P90 0.00 R     cycles net>0 67% [fail]
C3 reds      P10 -10.99 R  min -13.99 R  MDD 122.54 R   under-water max 52 wk   [fail]
C4 green     40%  null 50%                   [fail]
C5 fills/wk  18.02                             [pass]
C6 tail      C6 not audited
C7 power     cycles 6   bootstrap P90-gap 75% UB 16.0 wk    [fail]
```

Only C5 (frequency) passes on both splits; C1/C2/C3/C4/C7 all fail — the book has no reliable
renewal cadence, bleeds between the rare green weeks, and its green-week share (33-40%) sits at or
below the count-matched coin-flip null (50%).

## The one caveat

Gross R/trade is **never statistically distinguishable from zero** in either split (TRAIN t=0.46,
VAL t=1.40) — this is a pick+direction mechanism whose raw edge is already indistinguishable from
noise under the range-stop, same as it was under the ATR-stop in the base book (Cell A/B). The
range-stop/lock exit is materially cheaper (avg cost 0.13-0.20R/trade here vs ~1.1R implied for
the ATR-stop book) and still cannot turn a net profit, because there may be no real gross edge to
harvest in the first place. A cost-model change cannot rescue this cell; the failure is upstream of
costs. MDE on VAL (0.233R) means a true edge smaller than that would not have been detectable here
— but the point estimate is already negative, so this is a negative result, not merely an
underpowered null.

## Files
`score_d.py` (sim), `trades_raw_D.csv`, `book_D_1x.csv`, `trades_D_cadence.csv`, `score_d.out`, `results_D.json`.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PKSsd3LjBkXYzPECLNYR5W
