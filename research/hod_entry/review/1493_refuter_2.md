# Refuter 2: cell 1,493 (retest-bounce exit surface), through statistics, multiplicity and artifacts

**Verdict: the FAIL holds, and it is stronger than the builder reports. Two of the builder's secondary claims are wrong and must not be relayed to the owner.**
Everything below was recomputed from `cell_1493_fills.csv`, `bars_fills_1478.db`, `features_1478_A.csv` and `cell_1480_fills.csv`.
Scripts: `review/refuter2_1493_a1.py` (surface, tails, cache-only), `a2.py` (placebo decomposition) and `a3.py` (real-SIP surface, mirror).

## 1. Surface, selection and multiplicity: confirmed
- 493,515 rows, 55 cells, 3,957 TRAIN / 5,016 VAL fills. **0 of 55 cells have positive net return on either holdout.** Only 1 cell on TRAIN and 2 on VAL have positive gross return: 3.0%|NONE, gross +0.009 % (TRAIN) and +0.028 % (VAL).
- Maximum day-clustered t is −0.77 on TRAIN and −0.53 on VAL. The best TRAIN cell (excluding the 0.5 % stop row and M) is also the best VAL cell: 3.0%|NONE at −0.116 % / −0.078 %. The gap between the TRAIN-selected cell and the VAL-best cell is 0, so there is no over-selection on TRAIN. The Spearman rank correlation of the grid between TRAIN and VAL is 0.79. All neighbours are negative.
- Tails for 3.0%|NONE (TRAIN / VAL):

  | Measure | TRAIN | VAL |
  |---|---|---|
  | ex-top-1 % | −0.31 % | −0.25 % |
  | ex-top-5 % | −0.71 % | −0.64 % |
  | winner-capped at +3 % | −0.70 % | −0.63 % |
  | drop the best 2 days | −0.24 % | −0.20 % |
  | median | −1.05 % | −0.95 % |
  | largest single day's contribution | ≤ 0.06 pp | ≤ 0.06 pp |

  This is a lottery-shaped loser: the maximum single fill is +28 % / +30 %.
- **Builder error (minor):** the RESULT says capping winners "barely moves it" and quotes −0.078 %. The capped mean is actually −0.63 %. The error runs in the conservative direction.

## 2. Cache-only share: the check passes on paper, but it hides the 1,427 artifact
The cache-only share is 18.2 %, against 19.5 % for the population. That is within 5 pp, so the check passes. **But the outcome on cache-only rows is completely different:**

| Rows (3.0%\|NONE) | TRAIN | VAL |
|---|---|---|
| Cache-only (store_served_1438 = 1) | **+0.89 %** | **+0.66 %** |
| Real-SIP | **−0.36 %** | **−0.23 %** |

- On cache-only rows, 43 of 55 cells are positive on TRAIN and 29 on VAL.
- On real-SIP rows, 0 cells are positive. The best real-SIP cell is 3.0%|tgt0.5 at −0.18 % (TRAIN) and −0.15 % (VAL).

The published surface is therefore **inflated** by the sparse-cache level artifact. The honest book is 0.1–0.3 pp worse on every cell. This makes the FAIL stronger rather than weaker. Any future read must be split by store.

## 3. Placebo: the builder's "adverse timing, −0.6 pp, t −5.8 to −12.6" is an ARTIFACT
The placebo draws its minute from anywhere in [09:45, 15:00] outside the retest window. That includes minutes before the break. This population is defined by a later new high-of-day break at the level, so a long entered before the break is guaranteed to see price rise to the level. That is look-ahead baked into the population.

I re-drew the placebo separately for pre-break and post-window minutes (seed 1493), on the 3.0%|NONE cell:

| Split | Placebo window | Real | Placebo | Margin | Day-clustered t |
|---|---|---|---|---|---|
| TRAIN | pre-break | −0.10 | **+2.44** | −2.54 | −25.9 |
| TRAIN | post-window | −0.03 | −0.18 | **+0.16** | **+1.3** |
| VAL | pre-break | −0.06 | **+2.56** | −2.62 | −38.8 |
| VAL | post-window | −0.02 | −0.05 | **+0.03** | **+0.26** |

**The whole negative margin comes from pre-break placebo minutes.** Against a valid (post-window) placebo, the retest timing is NEUTRAL: it neither beats the placebo nor loses to it. The RESULT sentence "the retest instant is a worse-than-average entry time / adverse selection" is false. The placebo leg of the pass bar is uninformative as built.

Two further defects in the placebo:
- It uses the salted Python `hash((seed, day, symbol))`, so the placebo minutes are **not reproducible** across processes (PYTHONHASHSEED).
- Its walk starts at pm + 1 and never checks the stop inside the entry bar.

The verdict does not change, because the cell fails its own mean, t and tail tests before the placebo matters.

## 4. Mirror M vs the 1,480 short: the "bounce" was the short's cost plus its lower entry
On the 1,480 short (the shortable subset, n = 999 TRAIN / 991 VAL):

| Measure | TRAIN | VAL |
|---|---|---|
| Net return | −0.525 R | −0.670 R |
| Raw return | −0.116 R | −0.230 R |
| Median R | 1.00 % of price | 1.00 % of price |
| Net return, % of price | −0.61 % | −0.77 % |
| Raw return, % of price | −0.15 % | −0.28 % |
| **Cost, % of price** | **0.47 %** | **0.49 %** |

About 75 % of the short's loss is cost. The short sells at the bid, 13–14 bps below the print, and pays the stop-limit tail on a 67–71 % stop rate.

Paired on the same (day, symbol):

| Measure | TRAIN | VAL |
|---|---|---|
| M raw | −0.28 % | −0.21 % |
| Short raw | −0.14 % | −0.28 % |
| **Sum of the two raw returns** | **−0.43 %** | **−0.49 %** |
| Long entry minus short entry | **+0.48 %** | **+0.44 %** |

The two legs reconcile almost exactly: the price path between the two entries is directionless, and the long simply buys about 0.45 % higher than the short sells. **There is no bounce regularity. The deduction in PREREG_1493 ("the one directional regularity measured is the bounce after the dip") is refuted**, not just "smaller than the mirror suggests". Also, M is not the geometric mirror of the short's loss. The short loses when price goes up about 1 % before it goes down about 2 %, which corresponds to cell 2.0%|tgt1.0: P(target first) is 0.62, but the cell is still −0.167 % on VAL.

## 5. Count-matched null
The null is drawn from the base break-entry outcome, which is itself about −0.2 %. The "100th percentile" result only says that the retest entry is less bad than the break entry. That is already known from 1,481 (the immediacy cost recovered). It is not evidence of edge, as the builder already notes.

## Bottom line
- **The FAIL stands, and on real-SIP rows it is worse than reported.** The TRAIN selection is stable (TRAIN-best = VAL-best), the tails are negative and no day dominates.
- Do NOT relay "the retest is adverse timing (placebo t −5.8 to −12.6)". That result comes from pre-break look-ahead in the placebo, and the valid placebo margin is 0.
- Do NOT relay "a bounce exists but is smaller". The short's loss is cost plus entry price. Gross, the path after the dip has no direction on this population.
- Future placebos on this HOD population must draw minutes AFTER the break only, and future surfaces must be split by store.
