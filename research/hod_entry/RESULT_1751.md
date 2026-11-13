# RESULT 1,751 — confirmation entry after the HOD break held N minutes: KILL (TEST sealed) — 2026-11-13

Programme count 1,751. Files `1751_score.py`, `1751_anatomy.csv`, `1751_grid.csv` (246 cells), `1751_rule.json`, `1751_val.csv`.
TEST dropped at load: 3,521 signals / 946,097 path rows. Arms without measured half-spread dropped (TRAIN 75, VAL 50).
B0 replay parity on TRAIN: exit price 99.75 %, exit reason 99.93 %. Volume is not in paths.parquet, so vol_ratio was skipped.

## Winners vs losers (TRAIN, forward sign from c_N)
- **dist_N and maxdd_N separate, and the separation grows with N**: AUC 0.62 → 0.74 (dist) and 0.61 → 0.67 (maxdd) from N 3 → 30, day-t 5–17.
  Their AUC against the **B0** sign is higher at every N (dist 0.79 at N 20). The forward sign is mostly geometric: a higher close sits farther from
  the fixed stop and nearer the target.
- held: AUC 0.54–0.55. above_vwap: 0.50–0.52. ret_SPY: 0.52–0.62 (t < 2). bars_since_open: 0.60 (t 1.5). None of these is usable.
- **The never-retest premium is almost all already earned.** At N 20, held vs not held: forward gross +0.16 vs +0.08 R0 and to the close
  +0.21 vs +0.19, while B0 is +0.86 vs −0.18. That is about 92 % of the 1.04 R gap earned before N. N 30 shows the same (+0.14/+0.05 forward, +0.97/−0.15 B0).
- Across all arms the forward gross is +0.06 to +0.11 R0, but costs take about 0.30 R0 (spread on both legs, stops 44–54 %). **0 of 246 cells are net > 0 on TRAIN.**

## Rule (frozen on TRAIN before VAL)
The best TRAIN-H2 net cell with ≥ 3 fills/wk: **N 20, held == 1**. TRAIN results: n 472, 9.1/wk, gross +0.16, net −0.15 R0 (t −1.5),
H1 −0.24 / H2 −0.07, ex-top-5 % −0.24.

## VAL (read once, 2026-01..05, 22 wk)
| | rule | placebo (not held) | all arms |
|---|---|---|---|
| n / wk | 368 / 16.7 | 3,575 / 162 | 3,943 |
| gross R0 | −0.07 | +0.02 | +0.01 |
| net R0 (day-t) | **−0.39 (−4.9)** | −0.33 (−8.7) | −0.34 |
| net, new-risk units | −0.23 | −0.33 | −0.32 |
| ex-top-5 % / capped +3 | −0.49 / −0.39 | −0.46 | −0.47 |
| B0 on the same arms → lift | +0.59 → **−0.97 (t −24)** | | B0 all −0.30 |

The rule's other VAL numbers:
- weekly P10: −14.8 R
- green weeks: 18 % vs a count-matched null of 16 % (p 0.50)
- all 5 months negative
- cost ×1.5: −0.54
- next-bar-open entry: −0.39
- MDE (80 % power): 0.22 R

**KILL**: every kill criterion fails except fills/wk (net ≤ 0, t < 2, ex-top-5 ≤ 0, the placebo is not below the rule, lift ≤ +0.10). Waiting for the hold only buys the move that has already happened.
