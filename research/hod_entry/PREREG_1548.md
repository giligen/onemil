# PREREG — cells 1,548–1,549: the EXTENSION ANATOMY and the sweep entry on predicted extenders

FROZEN 2026-09-26 18:50 UTC before any number. Programme count on the HOD line: 1,547 → 1,549. Owner 9/26: "both
sound interesting, maybe we should dive more into both" (the corrected extension model).

## What was seen (disclosed) and the deduction
Cell 1,478-L3 v2 (arm bar = the last bar closed before the fill minute, PREREG_1478 amendment 3): the label "the day's
high after bar j reaches level × 1.05" is predictable at the arm bar — VAL AUC 0.715 (placebo 0.49; the metadata decoy
0.58 > 0.55, so the store-identity cohort carries part of the label: every number below is reported on real-SIP rows
beside the whole). Kept top tercile (TRAIN-H2 threshold 0.3070 on hgb_prob_L3): VAL n 2,048, precision 44.6 % vs a
27.6 % base rate; the kept extenders earn +0.79 R under the base trade, the kept non-extenders −0.81 R, mean −0.10 R.
Breakeven precision for this trade structure is 50.6 %. TRAIN-H2's +0.32 R is in-sample (fit and threshold on TRAIN)
and is not evidence. Deduction: the model is a real 1.6× lift; the trade that wraps it loses a near-full R on every
miss and a 2 R target caps every hit. The open question is the PATH of the extenders: how deep they dip below the
level before the +5 % touch and how long they take — if 75 % of extenders never dip more than s below the level, a stop
at s with a +5 % limit target changes the arithmetic; if they sweep the consolidation first, a resting bid in the sweep
is the entry. Nothing on this population has yet used the extension as the TARGET.

## Population
The kept top tercile of L3 v2 (`model_1478_L3_v2_predictions.csv`, hgb_kept_L3 == True): TRAIN-H2 1,466 / VAL 2,048
base fills; TEST sealed and absent. Bars: `bars_fills_1478.db`; the level, consolidation low, fill and fill minute
from `causal_arming_causal.csv` (status == fill). Real-SIP flag: `features_1478_A.csv` store_served_1438 == 0.

## Part A — anatomy (report-only, both holdouts, all rows and real-SIP rows)
For extenders (L3 == 1) and non-extenders separately: the maximum drawdown below the LEVEL (in % of the level) from the
fill bar until the +5 % touch (extenders) or until 15:55 (non-extenders); the minutes from the fill bar to the +5 %
touch; the share of extenders whose drawdown breached the consolidation low before the touch; the share whose base
trade was stopped before the touch; the base exit mix per group; the distribution quantiles (10/25/50/75/90) of the
drawdown and of the minutes-to-touch. This table is the deliverable even if Part B fails.

## Part B — two pre-registered cells, parameters fixed from TRAIN-H2 quantiles, VAL read once each
Let d = the TRAIN-H2 median drawdown of the EXTENDERS below the level (floored at 0.20 %), s = the TRAIN-H2 75th
percentile of the extenders' drawdown (so three extenders in four survive the stop), W = the TRAIN-H2 75th percentile
of the extenders' minutes-to-touch (capped at 120 minutes). All three are computed once on TRAIN-H2 and written to the
result before VAL is read.
* 1,548 SWEEP: from the fill bar a BUY LIMIT rests at level × (1 − d) for W minutes (through-print rule: a bar low
  strictly below the limit fills at the limit; the fill bar itself may fill); stop = level × (1 − s) (stop-limit
  standard); target = level × 1.05 as a resting limit (fills at the target when a bar's high exceeds it); 15:55 exit at
  the bid. Unfilled → no trade (reported with their base outcome).
* 1,549 WIDE: the base entry (the actual fill at the ask, `fill`), stop level × (1 − s), target level × 1.05 limit,
  15:55 at the bid — isolates the stop/target change from the entry change.
Path semantics as `sip_rebuild.walk_path` (stop first on a bar touching both; gap-through at the open). Costs: entry
half-spread once for 1,549 (the base fill's `half_entry` from features_1478_A), none for the passive 1,548 entry;
stop-limit standard on stops; target limit no cost; EOD at the bid (RESULT_1443 means). Units: net R with R = entry −
stop, and net % of price. Report: n, fill share (1,548), mean, day-clustered t, ex-top-5 %, winner-capped at +3 R,
fills/week under the live cap (12/day, 4 concurrent), real-SIP-only mean and t, count-matched null (random kept fills'
base outcome on the same days, 1,000 draws, seed 1548), the exit mix, and the same two cells on the NON-kept fills
(the dropped two terciles) as the calibration line — the lift of the model must show as kept > dropped.

## Pass bar (frozen; VAL, per cell)
Mean net R ≥ +0.15 and mean net ≥ +0.15 % of price, day-clustered t ≥ 2.5, ex-top-5 % > 0, winner-capped positive,
≥ 3 fills/week, null percentile ≥ 99, real-SIP-only mean ≥ +0.10 R with t ≥ 2, TRAIN-H2 same sign with t ≥ 1, kept >
dropped on both holdouts, median R ≥ 0.5 % of price (the rail). TEST once for the better cell if it passes.

## Independent check and consequences
Rebuild from this prose (never reading the builder's code): Part A quantiles within 5 % relative, Part B fills Jaccard
≥ 0.99 and ≥ 99 % of rows within 0.01 R, the same d, s, W. Refuters: look-ahead (d, s, W from TRAIN-H2 only; the kept
flag is the arm-bar probability; the level and consolidation low are known at the arm bar; nothing after the fill bar
decides the entry), obtainability (a resting bid filled only through a print below it; the +5 % sell limit through-print;
halts), statistics (tails, day concentration, drop the best 2 days, the decoy-cohort split). PASS → the live engine gets
`entry_mode: sweep_limit` with the model's probability gate — dry 5 sessions on the parity ledger, then $50 real orders.
FAIL → the extension predictor is closed as a money signal on this population: predictable, not tradable, with the
anatomy on record.

## Not allowed
Choosing d, s or W on VAL; more than the two cells; reading TEST more than once; reusing the leaky features_1478_A.
