# Past quarter (TEST 2026-06-01..09-18, owner-unsealed 9/21) — production vs union of pools, week by week

Walked on the June–Sep wide features (`out/orb_features_20260921_1842.csv`), each pool alone through the
production pipeline (`run_quarter_pools.sh`), union = production first + add-ons to 8 slots (`quarter_weekly.py`).
R = $375 (stage), 4× = $1,500 shown as arithmetic only (cap and slippage at 4× unmodeled).

| book | fills | $ at $375 | $ at 4× | mean R | green wk | worst wk |
|---|---|---|---|---|---|---|
| production (gap ≥ 5 %, $3–30) | 42 | +1,548 | +6,191 | +0.098 | 7/14 | −382 |
| union | 109 | +1,469 | +5,878 | +0.036 | 8/16 | −835 |
| add-on gap 4–5 %, $3–30 | 56 | **−1,732** | | −0.082 | | |
| add-on gap 3–5 %, $30–50 (S3) | 11 | **+1,654** | | +0.401 | | |

Weekly (production / union, $ at $375): Jun-1 260/302 · Jun-8 −65/612 · Jun-15 39/99 · Jun-22 −73/298 ·
Jun-29 162/−22 · Jul-6 −27/−451 · Jul-13 −315/−217 · Jul-20 111/351 · Jul-27 1,325/1,330 · Aug-3 −71/−532 ·
Aug-10 −169/−835 · Aug-17 0/−17 · Aug-24 734/554 · Aug-31 22/495 · Sep-7 0/−26 · Sep-14 −382/−471.

## Read
* The union added nothing over the quarter (−$78 on 67 add-on fills) at 2.6× the fills and a deeper worst week.
  The gap 4–5 % slice, which was the marginal one in-sample (+0.07/+0.08 R), is −0.08 R out of sample: rejected.
* S3 (gap 3–5 %, $30–50) paid again: +0.40 R on 11 fills, after TRAIN +0.19 and VAL +0.53 — positive in all three
  windows, 25 fills total across 21 months (~1.2/wk). It was the lead BEFORE TEST was opened (frozen VAL +0.35,
  t 2.7), so TEST confirms rather than selects it.
* Production in the quarter: +0.10 R/fill, a quarter of its in-sample rate (n 42, SE ≈ 0.07 R).
* TEST is now consumed for these pools; the next holdout for ORB is live fills only.

## Decision (amends PREREG_LIVE_UNION.md)
Dry day 9/22 runs BOTH pools as measurement (zero orders). Only `addon_p30` is eligible for real orders after the
dry day and the owner's word; `addon_gap4` stays dry-only (instrumentation) and is never enabled without a new
pre-registration. Kill rules unchanged, applied to the p30 cohort.
