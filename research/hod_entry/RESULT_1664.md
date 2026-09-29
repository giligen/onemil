# RESULT — cell 1,664: synthesis of the HOD re-read at the measured cost (cells 1,660–1,668), 2026-09-29 19:50 UTC

Judge: main session. Every cell below ran under its own frozen PREREG; every number is per half (TRAIN-H2 / VAL) at the
measured cost (entry 7 bps, stop 6, target 0, EOD 11) on the 1,438 fill book (9,911 fills; floored ≥ 1.5 %: 5,506).
Bar: net ≥ +0.05 R and t ≥ 2.5 in BOTH halves, ex-top-5 % > 0 in both, ≥ 3 fills/week. MDE of the floored book: 0.077 /
0.066 R. Programme count on the HOD line: > 2,000 cells.

| cell | question | verdict |
|---|---|---|
| 1,660 | exit lab (35 variants) re-read | base −0.014/−0.061 R; floor 1.5 % +0.007/−0.028; no variant clears (best M2 tail-carried) |
| 1,661 | entry limit width on the dry cross records | +0.10 % = +0.047 R vs base on n 31 — watch-only, too thin |
| 1,662 | re-score of 16 cell families | nothing clears; 1,488 flagged (reconstructed join); 1,619 a pseudo-replication artifact |
| 1,663 | seven cost-axis cuts | 0 of 24 causal reads; the 4 numeric "passes" condition on exit type (excluded) |
| 1,665 | relative volume to the arm minute (owner) | VOID under the rail on the old store (own-day gap 19 %, missingness 7 pp); definition A re-run pending the prior-20-session backfill; direction on the covered part negative |
| 1,666 | independent rebuild of 1,488 from prose | REFUTED: paired −0.043/−0.044 R, day-t −4.9/−5.1, MDE 0.02 — the +0.34 R was the added cohort's own R |
| 1,667 | every causal arm-time feature (owner) | daily features flat; intraday features flat after the backfill (10 reads |t| ≥ 2.5, all negative direction); n_cross "pass" = full-day count (leak, VOID) |
| 1,668 | post-entry failure detection (owner) | 0/36 rules; classifier AUC 0.63–0.67 out of sample but cutting on it is −0.05..+0.02 R (3/4 negative); patterns ≈ 0 importance; placebo clean |

## What this says
1. The cost model was 3× too high and fixing it moved every cell up 0.02–0.05 R; none crossed into a positive on both
   halves. With the 1.5 % floor the book sits at ≈ 0 on TRAIN and −0.03 R on VAL (t −2). Filters cannot prove a lift
   under ~0.10 R on a third of the book here; every causal cut tested is inside that band.
2. Failure is predictable but not tradable: the post-entry classifier separates later stop-outs from targets (AUC
   0.65), yet cutting the "likely stop" trades loses — the cut pays the round trip on trades whose expected value,
   conditional on the signal, is still above the cut. Same first-passage logic as the shallow-stop cells.
3. Volume, shape and patterns add nothing once distance-to-level and return-in-R are known.

## Decisions (pre-committed)
* Nothing ships beyond the 1.5 % stop floor (paper since 9/29 17:24 UTC). Forward read at 100 paper fills, the two
  buckets (1.5–3 %, ≥ 3 %) reported separately (PREREG_1659 amendment 2).
* Second paper mechanism = the +0.10 % entry limit (1,661), ONLY after one clean floored session with ≥ 5 fills, one
  mechanism per session (rule 9/29): earliest 2026-10-01.
* 1,665 definition A is re-run on the completed store when the prior-20-session backfill finishes; if VOID again the
  cell closes as unresolvable on this data.
* No further filter cells on this population. The next HOD frame is a NEW signal definition with gross edge first
  (`research/ideas_web/RANKED_20260928.md` queue), under its own PREREG.
