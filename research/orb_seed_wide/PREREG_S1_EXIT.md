# PREREG — S1 stratum exit pass. Cells 1,319–1,321

Committed BEFORE any variant is walked. Follows `REPORT_S1_FILTERS.md` (selection cuts do not transfer).
Population: `out/runS1_features.csv` (gap 3–5 %, open $3–30), production selection frozen (orb.yaml
literals), 8 shared slots, R = $375, cost setting M — identical to `out/runS1_true.csv` except the ONE
exit knob per cell. Baseline exit anatomy on TRAIN: stop 46 % at −0.32 R, tag_bb 20 % at −0.11 R,
scale_eod 8 % at +0.96 R, lock 10 % at +0.16 R.

## Cells (three, fixed here)
| cell | variant | env | mechanism |
|---|---|---|---|
| 1,319 E1 | touchgo OFF | `ORB_TOUCHGO_ENABLED=0` | Rule M/D were fitted on the gap ≥ 5 % / $3–30 book; on 3–5 % gappers the first-bar failure signal may cut winners (41 TRAIN tag_bb trades, −0.11 R — shallow, not the −0.32 R of a stop) |
| 1,320 E2 | lock arm 1.0 R / stop 0.5 R | `ORB_BT_LOCK_ARM_R=1.0` | earlier ratchet: fewer full stop-outs after a run; the owner's shallow-reds preference |
| 1,321 E3 | scale 50 % at +2 R | `ORB_BT_SCALE_FRAC=0.5 ORB_BT_SCALE_LEVEL_R=2.0` | earlier partial — the BF P1 mechanism that raised green share and cut reds |

Nothing else changes; the three are NOT stacked in this pass.

## Pass bar (per cell, TRAIN and VAL)
1. VAL mean R ≥ +0.15 and TRAIN mean R ≥ +0.10;  2. VAL day-clustered t ≥ 2.0;  3. TRAIN halves both ≥ 0;
4. ex-top-5 % ≥ 0 both splits;  5. weekly MDD (R) ≤ baseline on both splits;  6. VAL fills/wk ≥ 3 (C5);
7. cadence bar block printed (diagnostic). Delta vs baseline reported trade-by-trade on (date, symbol).

## Decision rule
Pass → independent re-walk (agent that has not read the pipeline) before any claim, then a live
PREREG for the widened seed with this exit. No pass → S1 is reported as "frequency without edge under
production selection and three exits", S3 stays the only live-widening lead, and the S1 line moves to the
regime-conditional pass (trade S1 only in HMM calm weeks) under its own PREREG.

## Not allowed
Re-tuning the knob values after seeing VAL; stacking; touching orb.yaml or trading/.
Scorer: `score_s1_exit.py` (baseline vs variant, same statistics as `score_s1_filters.py`).
