# Cell 1,725 EXPLORATORY: afternoon runner vs fader at 14:00 (n = 67, NOT a ship signal)

PREREG_1725 frozen rules R1 / R2, live HOD paper book, 231 fills, free Alpaca 1-min bars (cell 1,724 cache).

- Positions open at 14:00 (live replay): **67** (A 32 / B 35)
- Runners (live-replay R >= +1.0): **20**, overall rate **30%**
- Hold side = live_R; squeeze side = V2 (take the profit at the 14:00 bar open if open P&L >= 0, else live).

## Primary (features from bars m < 840, i.e. through the 13:59 bar; squeeze at the 14:00 open as in V2)

### R1 near the high (open P&L >= +0.5R and within 0.5R of the high so far)

| side | n | runners | rate |
|---|---|---|---|
| hold | 21 | 13 | 62% |
| squeeze | 46 | 7 | 15% |

Fisher exact p (two-sided) = 0.0003

| split | n open | n hold | hold runners/rate | squeeze runners/rate | Fisher p | paired mean dR | t | ex-top-5 % | touched |
|---|---|---|---|---|---|---|---|---|---|
| ALL | 67 | 21 | 13/21 62% | 7/46 15% | 0.0003 | -0.0014 | -0.03 | -0.0614 | 31 |
| A | 32 | 9 | 6/9 67% | 5/23 22% | 0.0350 | -0.0688 | -0.98 | -0.1190 | 17 |
| B | 35 | 12 | 7/12 58% | 2/23 9% | 0.0030 | +0.0601 | 0.75 | +0.0007 | 14 |

### R2 momentum intact (MFE >= +1R and price >= 50 % of MFE)

| side | n | runners | rate |
|---|---|---|---|
| hold | 23 | 13 | 57% |
| squeeze | 44 | 7 | 16% |

Fisher exact p (two-sided) = 0.0015

| split | n open | n hold | hold runners/rate | squeeze runners/rate | Fisher p | paired mean dR | t | ex-top-5 % | touched |
|---|---|---|---|---|---|---|---|---|---|
| ALL | 67 | 23 | 13/23 57% | 7/44 16% | 0.0015 | -0.0375 | -0.77 | -0.0929 | 29 |
| A | 32 | 6 | 5/6 83% | 6/26 23% | 0.0112 | -0.0902 | -1.12 | -0.1514 | 20 |
| B | 35 | 17 | 8/17 47% | 1/18 6% | 0.0072 | +0.0106 | 0.18 | -0.0397 | 9 |

## Sensitivity (features incl. the 14:00 bar; squeeze at the 14:01 open)

### R1 sensitivity

| side | n | runners | rate |
|---|---|---|---|
| hold | 22 | 13 | 59% |
| squeeze | 45 | 7 | 16% |

Fisher exact p (two-sided) = 0.0005

| split | n open | n hold | hold runners/rate | squeeze runners/rate | Fisher p | paired mean dR | t | ex-top-5 % | touched |
|---|---|---|---|---|---|---|---|---|---|
| ALL | 67 | 22 | 13/22 59% | 7/45 16% | 0.0005 | -0.0281 | -0.51 | -0.0873 | 31 |
| A | 32 | 8 | 6/8 75% | 5/24 21% | 0.0099 | -0.0963 | -1.34 | -0.1449 | 19 |
| B | 35 | 14 | 7/14 50% | 2/21 10% | 0.0153 | +0.0343 | 0.42 | -0.0241 | 12 |

### R2 sensitivity

| side | n | runners | rate |
|---|---|---|---|
| hold | 25 | 13 | 52% |
| squeeze | 42 | 7 | 17% |

Fisher exact p (two-sided) = 0.0049

| split | n open | n hold | hold runners/rate | squeeze runners/rate | Fisher p | paired mean dR | t | ex-top-5 % | touched |
|---|---|---|---|---|---|---|---|---|---|
| ALL | 67 | 25 | 13/25 52% | 7/42 17% | 0.0049 | -0.0594 | -1.19 | -0.1145 | 28 |
| A | 32 | 7 | 5/7 71% | 6/25 24% | 0.0318 | -0.1151 | -1.43 | -0.1763 | 20 |
| B | 35 | 18 | 8/18 44% | 1/17 6% | 0.0178 | -0.0085 | -0.14 | -0.0582 | 8 |

## Pass bar (for the 500-fill run, not today)
Hold-side runner rate >= 2x squeeze side with Fisher p <= 0.01 in BOTH halves AND paired dR >= +0.10 with ex-top-5 % > 0. At n = 67 none of this can be met with confidence; nothing ships.

## Implementation caveats
- Timing: PREREG says features at the '14:00 bar close'; V2 sells at the 14:00 bar OPEN, so a bar-close feature would be look-ahead. Primary uses the last bar closed before 14:00 (as cell 1,724 open_r_1400); the literal reading is the sensitivity with a 14:01 squeeze exit.
- 'Day's high so far' and MFE are the high since the FILL minute (cache bars start at the fill), not the true session high; R1's 'within 0.5R of the day high' is therefore 'of the post-entry high'. For positions entered near the HOD these coincide; others understate the distance.
- Runner uses the REPLAYED live R (not the ledger R); replay matches the ledger exit reason on 95.7 %.
- Volume ratio (13:30-14:00 / 30 min after fill) is computed and stored but used by no rule.
- Fisher p is two-sided; halves are alternating trade dates (column `half`); rules were fixed before any number was seen; 2 rules x 2 timings = 4 cells counted in the multiplicity ledger.
- Squeeze side executes V2 (exit only if open P&L >= 0), so negative-open positions on the squeeze side still ride live.

## Afternoon drift from 14:00 to exit

EXPLORATORY. drift = live-replay exit R - open P&L R at the 13:59 bar close (primary timing); bps = exit price vs 13:59 close. Squeeze rows would, if drift ~ 0, have nothing to earn from an early exit.

| group | n | mean dR | sd | t | mean bps | median bps |
|---|---|---|---|---|---|---|
| R1 hold ALL | 21 | -0.043 | 1.002 | -0.20 | -9 | +73 |
| R1 hold A | 9 | +0.231 | 0.940 | 0.74 | +49 | +108 |
| R1 hold B | 12 | -0.248 | 1.037 | -0.83 | -53 | +41 |
| R1 squeeze ALL | 46 | -0.023 | 0.654 | -0.24 | -10 | -17 |
| R1 squeeze A | 23 | +0.125 | 0.629 | 0.95 | +23 | +15 |
| R1 squeeze B | 23 | -0.172 | 0.658 | -1.26 | -43 | -44 |
| R1 squeeze open>=0 | 31 | -0.040 | 0.650 | -0.35 | -26 | -18 |
| R1 squeeze open>=0 A | 17 | +0.067 | 0.548 | 0.51 | +3 | -3 |
| R1 squeeze open>=0 B | 14 | -0.171 | 0.757 | -0.85 | -61 | -59 |
| R1 squeeze open<0 | 15 | +0.012 | 0.683 | 0.07 | +23 | +45 |
| R1 squeeze open<0 A | 6 | +0.289 | 0.858 | 0.83 | +80 | +113 |
| R1 squeeze open<0 B | 9 | -0.173 | 0.510 | -1.02 | -15 | -39 |
| R2 hold ALL | 23 | -0.144 | 1.012 | -0.68 | -32 | +14 |
| R2 hold A | 6 | +0.291 | 1.069 | 0.67 | +55 | +118 |
| R2 hold B | 17 | -0.298 | 0.977 | -1.26 | -63 | -110 |
| R2 squeeze ALL | 44 | +0.030 | 0.616 | 0.33 | +2 | +1 |
| R2 squeeze A | 26 | +0.123 | 0.632 | 1.00 | +25 | +50 |
| R2 squeeze B | 18 | -0.104 | 0.584 | -0.76 | -30 | -29 |
| R2 squeeze open>=0 | 29 | +0.040 | 0.592 | 0.36 | -9 | -5 |
| R2 squeeze open>=0 A | 20 | +0.074 | 0.566 | 0.58 | +8 | +27 |
| R2 squeeze open>=0 B | 9 | -0.035 | 0.674 | -0.16 | -46 | -18 |
| R2 squeeze open<0 | 15 | +0.012 | 0.683 | 0.07 | +23 | +45 |
| R2 squeeze open<0 A | 6 | +0.289 | 0.858 | 0.83 | +80 | +113 |
| R2 squeeze open<0 B | 9 | -0.173 | 0.510 | -1.02 | -15 | -39 |
