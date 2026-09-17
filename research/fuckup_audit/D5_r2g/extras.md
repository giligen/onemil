---

## 6. Two follow-ups the tables forced

### 6a. Is "first entry of the day" anything but the 09:3x bar?

The book is FIRST-COME (12/day, 4 concurrent), so `seq` is just entry order in time. Cross-tab of net R per trade, seq ordinal x entry hour, ALL splits pooled and then per split for the 09:xx column only.

| ordinal | 09:xx n | 09:xx R/tr | 10:xx n | 10:xx R/tr | 11:xx+ n | 11:xx+ R/tr |
|---|---|---|---|---|---|---|
| 1 | 386 | +0.284 | 22 | +0.237 | 9 | +0.592 |
| 2 | 329 | +0.093 | 57 | +0.199 | 24 | -0.076 |
| 3 | 261 | -0.019 | 103 | -0.040 | 31 | +0.025 |
| 4+ | 273 | +0.003 | 259 | +0.014 | 255 | +0.028 |

Within the 09:xx hour only, per split:

| ordinal | TRAIN n | TRAIN R/tr | VAL n | VAL R/tr | TEST n | TEST R/tr |
|---|---|---|---|---|---|---|
| 1 | 222 | +0.142 | 96 | +0.706 | 68 | +0.148 |
| 2 | 175 | +0.022 | 90 | +0.278 | 64 | +0.025 |
| 3 | 121 | -0.029 | 81 | +0.164 | 59 | -0.250 |
| 4+ | 105 | +0.002 | 94 | +0.065 | 74 | -0.073 |

### 6b. The worse-than-1R losses

A stop exit books `min(stop, bar open) x 0.999`. When the next bar opens BELOW the stop, the loss is bigger than the 1R the stop nominally risked. The rule floor is `R >= 1% of entry`, so a 1.1%-wide stop on a thin tape turns a 4% gap-down bar into a -3.6R print.

- trades with net R <= -1.5: **16** of 2009 (0.8%), **-34.5 R** — 4.0% of ALL loser R, against a book total of +163.4 R.
- their median stop distance **1.65%** of entry vs **6.13%** for the book; median 5-min $ volume **$34,769** vs **$336,948**.
- 15 of 16 had a stop closer than 3% of the entry price; 15 were thin.

| stop distance band | n | R/trade | total R | mean net R of its losers |
|---|---|---|---|---|
| < 2% | 65 | +0.052 | +3.3 | -1.461 |
| 2-3% | 87 | +0.518 | +45.1 | -1.169 |
| 3-4% | 102 | +0.497 | +50.7 | -0.988 |
| 4-6% | 698 | +0.069 | +48.2 | -0.761 |
| 6-9% | 733 | +0.012 | +9.1 | -0.713 |
| >= 9% | 324 | +0.021 | +6.9 | -0.525 |

**Declared cell: a minimum stop distance** (known at the fill — the stop is the running low, the entry is the fill). Subset filter, not a re-book.

| rule | TRAIN n | TRAIN R/tr | TRAIN $/mo | VAL n | VAL R/tr | VAL $/mo | TEST n | TEST R/tr | TEST $/mo |
|---|---|---|---|---|---|---|---|---|---|
| as booked | 1112 | +0.0620 | $1,725 | 531 | +0.2066 | $6,584 | 366 | -0.0420 | $-1,473 |
| stop distance >= 2% of entry | 1074 | +0.0719 | $1,929 | 518 | +0.1735 | $5,392 | 352 | -0.0200 | $-674 |
| stop distance >= 3% of entry | 1029 | +0.0783 | $2,015 | 493 | +0.1013 | $2,997 | 335 | -0.0467 | $-1,499 |
| stop distance >= 4% of entry | 990 | +0.0676 | $1,673 | 457 | +0.0402 | $1,102 | 308 | -0.0683 | $-2,015 |

**Declared cell: stop floor combined with the sequence rule.**

| rule | TRAIN n | TRAIN R/tr | TRAIN $/mo | VAL n | VAL R/tr | VAL $/mo | TEST n | TEST R/tr | TEST $/mo |
|---|---|---|---|---|---|---|---|---|---|
| seq <= 2 | 487 | +0.1076 | $1,310 | 204 | +0.4632 | $5,669 | 136 | +0.0966 | $1,259 |
| stop >= 3% | 1029 | +0.0783 | $2,015 | 493 | +0.1013 | $2,997 | 335 | -0.0467 | $-1,499 |
| seq <= 2 AND stop >= 3% | 426 | +0.0984 | $1,048 | 179 | +0.2343 | $2,516 | 116 | +0.0792 | $880 |
| seq <= 2 AND stop >= 3% AND 09:xx | 339 | +0.1056 | $895 | 161 | +0.2501 | $2,416 | 112 | +0.0688 | $739 |

Tail check on the surviving rule (`seq <= 2 AND stop >= 3%`): mean net R with the top 1% and top 5% of trades removed, and with winners capped at +3R.

| split | n | R/tr | ex-top-1% | ex-top-5% | winners capped +3R |
|---|---|---|---|---|---|
| TRAIN | 426 | +0.0984 | +0.0154 | -0.1582 | -0.0038 |
| VAL | 179 | +0.2343 | +0.1025 | -0.0570 | +0.0962 |
| TEST | 116 | +0.0792 | +0.0287 | -0.1326 | +0.0293 |

Monthly net R of `seq <= 2 AND stop >= 3%`:

| 2025-01 | 2025-02 | 2025-03 | 2025-04 | 2025-05 | 2025-06 | 2025-07 | 2025-08 | 2025-09 | 2025-10 | 2025-11 | 2025-12 | 2026-01 | 2026-02 | 2026-03 | 2026-04 | 2026-05 | 2026-06 | 2026-07 | 2026-08 | 2026-09 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -7.6 | +9.5 | +17.2 | +5.3 | +6.7 | -5.1 | +9.2 | -10.8 | +15.1 | +3.1 | +5.2 | -5.9 | +19.8 | +10.0 | -0.1 | +16.0 | -3.7 | -2.7 | +8.1 | +1.7 | +2.1 |

Months green **14/21**; worst month **-10.8 R** (= $-3,229 at $300 risk); mean month **+4.43 R** ($1,329).
