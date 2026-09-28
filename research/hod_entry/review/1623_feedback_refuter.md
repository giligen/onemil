# Cell 1,623 (the day's own feedback gate): adversarial refuter

Verdict under review: **FAIL** (builder `RESULT_1623.md` and rebuild `REBUILD_1623.md` agree, kept-set Jaccard 1.0).
Refuter verdict: **the FAIL stands (refuted = false).** I found three defects. None of them changes the verdict, and
each one, once corrected, makes the gate look the same or worse.

Check script: `review/1623_feedback_refuter_chk.py`, output in `review/1623_feedback_refuter_chk.json`. It reproduces
the builder's n_res, F and G+ on 9,911/9,911 rows with 0 mismatches (F max diff 9e-16). It also reproduces both
seed-1623 placebo draws exactly: builder n 208 / −0.1058, rebuild n 75 / −0.5725. TEST was not read.

## 1. Causality of F

| check | result |
|---|---|
| Outcome/exit pairing (`why` in causal_arming vs model_1478) | 9,911/9,911 agree (stop 5,699 · target 2,757 · eod 1,321 · stop_bar 76 · eod_fallback 58) |
| outcome_R vs causal_arming net_R | Same exit event; the two differ by cost only (median 0.05 R, 652 rows > 0.25 R, 69 sign flips) |
| Open positions at 15:55 | Clean. EOD exits sit at minute 955 and the latest fill is at 841 (14:01), so an EOD exit never counts as resolved |
| exit_m ≥ fill_min | 0 violations |
| Day-level label | None. F uses only same-day outcomes that have already resolved. No day aggregate enters F |

**Defect A (same-minute look-ahead, shared by both builds).** The two time columns are on different scales:

- `exit_m` is the integer start of the minute bar in which the stop or target was touched (99.2 % of rows are integers; the rest are stop_bar/no_path rows, where exit_m equals that fill's own fill_min).
- `fill_min` is the fractional time of the print.

The rule `exit_m < fill_min` therefore admits `exit_m == floor(fill_min)`. That counts an exit anywhere in f's own fill
minute as resolved, including one up to 59 s after the fill. The PREREG says "EXIT minute strictly before f's fill
minute", so both builds read this line the same wrong way. An independent rebuild cannot catch a spec reading that
both sides share.

Scale of the defect:
- 1,732 of 9,911 fills have at least one same-minute resolution in their F.
- 293 of the 687 kept fills do.

Re-run with the exit bar required to close before the fill minute, and again with one more minute of latency:

| rule | VAL kept n | VAL mean | VAL t | VAL ex-top-5 % | TRAIN-H2 n | TRAIN-H2 mean | TRAIN-H2 t |
|---|---|---|---|---|---|---|---|
| builder: exit_m < fill_min | 356 | +0.049 | 0.34 | −0.053 | 331 | −0.390 | −3.51 |
| strict minute: exit_m < floor(fill_min) (81 out, 19 in) | 326 | +0.039 | 0.29 | −0.061 | 299 | −0.393 | −3.23 |
| +1-min latency (122 out, 32 in) | 320 | +0.029 | 0.22 | −0.073 | 277 | −0.428 | −3.97 |

The defect flatters VAL by about 0.01 R and does not change the verdict. Cell 1,624 counts arms with
`m_hi < fill_min`, the same pattern, so its refuter should run the same same-minute test.

**Defect C (the exit-minute caveat is mis-stated, and harmless).** The "80.5 % exit_m agreement with
rebuild_1481_fills.csv" is not a check of the base exit. In that file, `exit_m` belongs to cell 1,481's retest trade:
its own entry, stop_used and target, which give net_R_prime. The base trade's fields there are base_why and
base_net_R. The exit minute actually used, causal_arming's own column, is the right one: its `why` matches the
outcome file on every row.

## 2. Is G+ a time-of-day gate in disguise? No, it is a day gate

| holdout | all fills | after-11:00 gate | n_res ≥ 2 population | G+ kept | kept median fill (ET) | kept share after 11:00 (all) | TOD-matched lift of kept (30-min buckets) |
|---|---|---|---|---|---|---|---|
| VAL | −0.171 (n 5,513) | −0.236, t −5.4 (n 1,757) | −0.192 | +0.049 | 10:18 | 9 % (32 %) | +0.253, t 1.75 |
| TRAIN-H2 | −0.167 (n 4,398) | −0.152, t −3.0 (n 1,583) | −0.176 | −0.390 | 10:34 | 35 % (36 %) | −0.264, t −2.24 |

- n_res ≥ 2 does not wait for the afternoon. It is reached early: the median first kept fill of a day is at 10:03 ET in VAL and 10:23 in TRAIN-H2.
- 91 % of VAL kept fills are before 11:00.
- The plain after-11:00 gate is worse than the base in VAL.
- The time-adjusted lift is positive in VAL and negative in TRAIN-H2.

So time of day neither creates nor hides an effect. The kept set is a set of days on which the morning went well.

## 3. Tails and day concentration of the kept set

**VAL** (356 kept fills on 19 days):
- 2026-04-02 alone contributes 111 of the kept fills (31 %) and +37.7 R, which is 216 % of the +17.4 R kept sum.
- That day was a day on which everything worked: its 139 fills average +0.61 R.
- Excluding that day: n 245, mean −0.083. Excluding the top 3 days: −0.315.
- 47 % of kept days are positive, and the top 5 days hold 68 % of the kept fills.

**TRAIN-H2** (34 days): 21 % of days positive; mean −0.475 excluding the top day.

The effective sample is about 20 days, and the VAL mean comes from one day.

## 4. Cache-only share

The share criterion passes: VAL 17.4 % against a base of 18.7 %, TRAIN-H2 22.1 % against 20.4 %, bar 19.5 % ± 5 pp.

That criterion hides a population split. Cache-only fills average +0.155 R (VAL) and +0.300 R (TRAIN-H2) in the base
book, against −0.245 and −0.287 for the other fills. This is the sparse-store population behind the 1,427 artifact.

Inside VAL's kept set:
- The 62 cache-only fills average +0.297 and sum to 18.4 R, more than the 17.4 R kept total.
- The other 294 kept fills average −0.003 (t −0.03).
- The 20 cache-only kept fills on 2026-04-02 average about +1.10 R.

So VAL's positive number comes from the cache-only fills of one day.

## 5. The shuffle placebo

**Defect B (the placebo t is mis-defined in both builds).**
- The builder's t of −0.61 is the one-sample t of the placebo set's own mean, not a t on the margin.
- The rebuild's t of 3.70 is an iid Welch t, not day-clustered.
- The day-clustered two-sample margin t is 0.69 under the builder's draw (block reading), which fails. Under the rebuild's draw (row reading) it is 2.88 with real-day clusters or 3.37 with pseudo-day clusters, which passes, but only on that one draw.

**500 seeds per reading** (VAL test, TRAIN-H2 shuffled):

| reading | margin p5 / p50 / p95 | share with margin ≥ 0.10 | share with margin ≥ 0.10 and clustered t ≥ 2 |
|---|---|---|---|
| block (builder) | −0.017 / +0.228 / +0.472 | 80 % | **15.6 %** |
| row (rebuild) | −0.020 / +0.231 / +0.505 | 80 % | **9.6 %** |

- The rebuild's seed-1623 placebo mean of −0.57 is a tail draw; the median over seeds is −0.18. Its PASS is luck, and the criterion fails under either reading.
- **Design weakness:** the cross-holdout margin is roughly the kept mean minus the other holdout's base rate (about −0.17). It measures "kept vs base", not "own day vs a random day", which is why 80 % of seeds clear the 0.10 margin.

**The informative null.** Keep each fill in its own holdout and at its own clock time, but compute F from a randomly
chosen other day of the same holdout (1,000 permutations). This controls for time of day and for the holdout's base rate.

| holdout | real kept mean | null p5 / p50 / p95 | share of null ≥ real | real lift vs dropped | share of null lift ≥ real |
|---|---|---|---|---|---|
| VAL | +0.049 | −0.430 / −0.195 / +0.051 | 0.052 | +0.235 | 0.049 |
| TRAIN-H2 | −0.390 | −0.414 / −0.186 / +0.056 | 0.927 | −0.241 | 0.939 |

The two holdouts land in opposite tails of the same-day null. There is no evidence that a day's own resolved outcomes
grade its later fills. The flat 5-bin autocorrelation table in RESULT_1623 says the same.

## Verdict

**FAIL stands; not refuted.** None of the three defects moves any failing criterion:
- A: the same-minute look-ahead gives −0.01 R on VAL once corrected.
- B: the placebo t is mis-defined; the corrected statistic fails on the builder's draw and in 84–90 % of seeds.
- C: the exit-minute caveat is mis-stated.

**What the frame does not carry:** on the 9,911-fill causal HOD-break book (TRAIN-H2 and VAL, cost as in 1,478), the
mean of a day's resolved fills does not predict the day's later fills at a +0.5 R threshold. The VAL +0.049 comes from
one day (2026-04-02) and from its cache-only rows. TRAIN-H2 is −0.39 R (t −3.5).

**Corrections for the record:**
1. RESULT_1623 should report the placebo margin t as 0.69 (day-clustered, two-sample).
2. Its exit-minute caveat should drop the 1,481 comparison.
3. Cell 1,624 needs the same-minute test on `m_hi < fill_min`.
