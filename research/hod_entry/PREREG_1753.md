# PREREG 1,753 — harvest the measured post-break MEAN REVERSION (book D1/D2/D4): long the dip, never short

FROZEN 2026-11-13 19:25 UTC before any number. Programme cell count: 1,752 → 1,753. Rules: write only
`research/hod_entry/1753_*`, `RESULT_1753.md`; no git/service/orders/network; `ulimit -v 2500000; nice -n 19`,
peak RSS < 1 GB (paths.parquet is 98 MB — pyarrow column selection); budget ≤ 50 calls. Data: exit-lab
`paths.parquet`, `signals.parquet`, `b0_trades.csv` (as 1,751). TEST rows dropped at load (count them).

## Claim (parent: cell 1,743 — VR(10) 0.87/0.83, VR(30) 0.74/0.77, CIs < 1 in every tercile)
The post-break path mean-reverts. A LONG that buys the dip after the break and targets a partial return to the
level (never the extension, never short) converts the reversion into R. Three reads, all long-only, all from the
break bar's close onward (causal):
D1 DIP: price first trades ≤ level − a × R0 (a ∈ {0.5, 1.0}) within 30 min of the break → buy at that bar's close +
   half_spread + 2 bp; target level − b × R0 (b ∈ {0.0, 0.25, 0.5}, b < a); stop level − (a + 0.5) × R0 (slip as
   PREREG_1740); time-out 30 min after entry → exit at close. R unit = a − b? No: R unit = entry − stop (0.5 R0).
D2 RANGE-LOW RETEST: price back inside the range and touching the pre-break 5-bar low (from signals/paths) → long
   there, target the level, stop 0.5 R0 below the low, 30-min time-out.
D4 GEOMETRY: no retest of the level by N ∈ {10, 20} → trailing stop at max(level, high − 1 R0) to the close
   (what 1,751 measured as "earned before N" — can a trailing stop keep it?). Compared with B0 on the same arms.
Grid: a × b for D1, N for D4 (≤ 12 cells); rank on TRAIN-H2 net; pick one per read with ≥ 3 fills/wk.

## Numbers (TRAIN halves, TRAIN, VAL once)
n/wk, net mean R, day-clustered t, ex-top-5 %, winner-capped +3 R, weekly P10, green-week share vs null, year
signs, placebo = the same dip rule on a random non-break minute of the same symbol-day (dip from any local high),
cost ×1.5, MDE at 80 % power.

## Kill (any one, per read)
VAL net ≤ 0 or day-t < 2; TRAIN halves opposite sign; VAL ex-top-5 % ≤ 0; < 3 fills/wk; placebo ≥ rule on VAL.
Survive → TEST once → independent rebuild → dry-run spec.

## Output
`1753_score.py`, `1753_grid.csv`, `1753_val.csv`, `RESULT_1753.md` ≤ 350 words ending with "CHILDREN:" (≥ 2 new
hypotheses from the winners-vs-losers table, family + parent + data + kill). Return ≤ 120 words.
