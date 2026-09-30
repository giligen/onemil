# PREREG — cell 1,674: the bar's power error — pooled reads of the same-signed cuts, the joint book on VAL, and capped-day selection (FROZEN 2026-09-30 05:22 UTC)

Owner 05:10 UTC: "find the errors and oversights in your research, there's money there." Oversight #2 (a statistical
error in MY bar): every synthesis cell required net ≥ +0.05 R with t ≥ 2.5 in BOTH halves. The MDE per half is
0.066–0.077 R, so a TRUE +0.05 R lift clears one half about half the time and both halves about a quarter of the time.
The bar was calibrated to throw away real +0.03–0.05 R lifts. Several cuts came back SAME-SIGNED on both halves with
similar magnitudes and were called "fail" on t alone:
* 1,665 RVOL_A(20): low tercile +0.033 / +0.035 R, high tercile −0.078 / −0.067 R (TRAIN / VAL)
* 1,667: intraday activity features (dollar volume / ADV20, cumulative volume) — 10 reads |t| ≥ 2.5, all "high = worse"
* 1,663: ≥ 3 % stop × 11:00–12:30: +0.011 / +0.012
* 1,658: ≥ 3 % stop bucket: −0.08 / +0.14 (disagrees on sign, n small — read pooled)
* 1,660: MOC end-of-day exit +0.010 / +0.008 vs the bid exit
These may add up. The correct read is the pooled estimate with a day-clustered SE, a sign-agreement check across
halves, and then a JOINT book defined on TRAIN and read ONCE on VAL.

## Population and cost
The 1,663 join (fills_1658 ⋈ causal_arming_causal), floored r_pct ≥ 1.5 % (n 5,506); features already on disk:
`1665_features.csv` (rvol_a20, rvol_a5, rvol_b), `1667_features.csv` (F1–F15), `1663_features.csv` (ATR %, stop
bucket, time of day, target bucket), `1660_per_fill.csv` (MOC vs bid EOD per fill). Cost: net_R as before; the MOC
line uses 1,660's MOC column.

## Part A — pooled reads (both halves together, day-clustered SE, MDE at the pooled n)
For each candidate cut: pooled mean of the kept book, pooled ΔR vs the floored base, day-clustered t, ex-top-5 % ΔR,
fills/week, and the sign-agreement line (TRAIN sign, VAL sign). Candidates (fixed list): drop the top RVOL_A(20)
tercile; drop the top RVOL_A(5) tercile; drop the top RVOL_B tercile; drop the top tercile of F15 (dollar volume to the
level bar / ADV20); drop the top tercile of F11 (level vs VWAP); keep ≥ 3 % stops only; keep 11:00–12:30 only; the MOC
exit. Tercile edges are computed on TRAIN only and applied to VAL (no look-ahead in the cut itself).

## Part B — the joint book (defined on TRAIN, read once on VAL)
Selection rule (fixed here, before any number): from the Part A list, a cut enters the joint book only if its TRAIN
ΔR is positive with TRAIN day-clustered t ≥ 1.0 AND its VAL sign agrees (the VAL magnitude is NOT used for selection).
The joint book = floor 1.5 % + every entering cut applied together (intersection; the MOC exit applies to EOD exits).
Report on VAL only: mean net R, day-clustered t, ex-top-5 %, MDE at the joint n, fills/week at the live config, and
the TRAIN in-sample mean as a caveat line. Also the "leave-one-out" table: the joint book without each cut.

## Part C — capped-day selection (the live cap binds: 12 fills/day vs ≈ 30/day in the book)
Simulate the live cap on the floored book per day: take the first 12 fills in time order (the engine's behaviour) vs
priority orders (fixed list): largest r_pct first; lowest RVOL_A(20) first; lowest F15 first; the joint-book members
first then the rest. For each: the capped book's mean net R per half and pooled, day-clustered t, fills/week (= 12 ×
days with ≥ 12 fills + …), ex-top-5 %. This is the read that turns a within-day feature into money without dropping
frequency.

## Pass bar (for the joint book and for each capped-selection rule)
Pooled mean ≥ +0.05 R with day-clustered t ≥ 2.5, ex-top-5 % > 0, both-halves SIGN agreement, ≥ 3 fills/week; the
joint book's VAL read is the one that counts (TRAIN is in-sample by construction). A pass → independent rebuild from
prose before the owner sees a number; then paper as ONE mechanism (the whole joint rule), forward read at 100 fills.

## Multiplicity
Part A: 8 cuts × 3 reads; Part B: 1 joint + ≤ 8 leave-one-out; Part C: 5 orders × 3. ≈ 50 reads. Programme count: > 2,650.

## Not allowed
Adding candidates not in the list above; using VAL magnitudes to choose the joint; tercile edges from the full sample;
pooled-only numbers WITHOUT the sign-agreement line; changing the cap or the priority list after seeing numbers.

## Output
`research/hod_entry/RESULT_1674.md` (≤ 140 lines), `1674_reads.csv`, `1674_joint_per_fill.csv`, `1674_pooled.py`,
`1674_pooled.log`. The agent returns ≤ 150 words.
