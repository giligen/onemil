# HOD-OFI REPORT
coverage_usable=0.994 avail=OK gap_pp=0.44 fallback_share=0.264 n_signal=12135 n_usable=12066

## Corrections vs run 1
## Defects → fixes (each with a unit test in `research/hod_ofi/test_pipeline.py`, synthetic data, pytest)
1. **CKS ask side has the wrong sign** (`ofi_updates`). Cont-Kukanov-Stoikov:
   e = [b>b' ? q_b : b==b' ? q_b − q_b' : −q_b'] + [a>a' ? +q_a' : a==a' ? q_a' − q_a : −q_a]
   (primes = previous record). Now: ask moved up → −q_a' (should be +q_a'); ask unchanged → q_a − q_a' (sign flipped);
   ask moved down → +q_a' (should be −q_a). Test: a hand-computed 5-record book (ask lifted, ask size added, ask
   improved, bid improved, bid size cut) with the exact expected e per record.
2. **Coverage counts seconds with an UPDATE, not seconds with a quote** (PREREG: "≥ 60 % of its window seconds have a
   quote"). New `coverage_frac` = share of the 300 window seconds s ∈ [end−300, end) at whose end (instant s+1, capped
   at `end`) the prevailing mbp-1 record — the last record at or before that instant, INCLUDING records fetched before
   the window start — is two-sided (bid > 0, ask > 0, ask ≥ bid). No prior record → unquoted. The main session measured
   this on 187 windows: 99 % usable vs 22 % under the old rule. Test: a book with one quote at t=−10 s and no updates
   → coverage 1.0; a book whose first record arrives at window second 150 → 0.5.
3. **Lee-Ready must use the quote strictly before the trade**: `merge_asof(..., allow_exact_matches=False)`, and all
   sorts by `sec` must be stable (`kind='mergesort'`) so same-timestamp records keep Databento's order (`sec` drops
   nanoseconds). Test: a same-timestamp quote change must not classify its own trade.
4. **Fallback end is not causal**: when no XNAS print ≥ the level is found in minute `entry_m`, end = S (the minute's
   START), not S + 60 (the old value used up to 60 s after the break). Keep the `locate_fallback` flag; report its share.
5. **Placebo ↔ signal join is ambiguous** when a name-day has several signals. Add `sig_entry_m` (the signal's
   `entry_m`) to BOTH rows in `build_windows` and to `window_features.csv`; join each placebo to its own signal on
   (day, symbol, sig_entry_m). Assert the seeded placebo minutes are unchanged: every placebo window of the 2 test days
   must find its raw rows (count and report any that do not).
6. **Score (`cmd_score`)**:
   * drop NaN feature rows before any cut / rank / Spearman (TSI and placebo Spearman were NaN);
   * `t_lift` = t of the keep dummy in OLS `net_R ~ 1 + keep` with day-clustered SE (statsmodels,
     `cov_type='cluster'`, groups = day); print the iid t beside it;
   * the pass-bar "deciles monotone (Spearman ≥ 0.6)" = Spearman between decile index 1–10 (qcut of the feature
     within that holdout, `duplicates='drop'`) and the decile's mean net R — NOT the trade-level rho (report that as
     info). Write both decile tables (TRAIN-H2, VAL: n, mean net R per decile) into REPORT.md;
   * placebo: cut the placebo feature at ITS OWN TRAIN-H1 median; placebo lift = kept − dropped mean net R of the
     matched signal's trade on TRAIN-H2 and VAL. PASS additionally requires placebo VAL lift < +0.10;
   * REPORT.md header: coverage, fallback share, winner/loser missingness gap, and a "Corrections vs run 1" section
     listing defects 1–6 verbatim from this file.
   Unchanged: the three cells, the TRAIN-H1 median cut, lift ≥ +0.10 on TRAIN-H2 and VAL, VAL t ≥ 2, ≥ 3 fills/wk,
   ex-top-5 % lift ≥ 0, the 60 % / 80 % / 5 pp availability rail.
7. **(Main-session amendment, 2026-09-24 ~19:55 UTC, before any corrected number.) The window end was a spec error.**
   `SPEC.md` clarification 1 assumed an intra-bar stop at the level. The HOD-break book enters at the OPEN of bar
   `entry_m` after the break bar (`entry_m − 1`) CLOSES (`research/bf_zero/spread_study.py:48`), so the decision
   instant is S = the start of minute `entry_m` — exactly the PREREG's "before the break bar closes". Every signal
   window now ends at S; `locate_fallback` is a diagnostic only (no XNAS print ≥ entry inside the entry minute). Found
   by the window-causality lens: 58 % of signal windows have ROUND-LOT XNAS prints ≥ entry in the minute before
   `entry_m` (natural for a next-open entry, impossible for a stop at the prior high), and the old end sat a median
   ~6 s (up to 60 s) after the decision.

## F1_OFI5 (OFI_5)
cut(TRAIN-H1 median)=35.1335  placebo_cut(own TRAIN-H1 median)=0.5001
TRAIN-H2: n=3440 lift=0.217 decile_spearman=0.842 trade_spearman(info)=nan
VAL: n=4661 lift=0.258 t_cluster=5.81 t_iid=6.43 decile_spearman=0.939 trade_spearman(info)=nan
fills/week=37.41 placebo_trade_spearman=nan placebo_lift TRAIN-H2=0.068 VAL=0.172 (PASS additionally needs placebo VAL lift < 0.10)
ex_top5_lift=0.147
VERDICT=FAIL

### F1_OFI5 decile table -- TRAIN-H2 (n, mean net R)
 decile   n  mean_net_R
      1 344   -0.286637
      2 344   -0.495900
      3 344   -0.325717
      4 344   -0.393973
      5 344   -0.301335
      6 344   -0.178904
      7 344   -0.191265
      8 344   -0.167931
      9 344   -0.066707
     10 344   -0.092618

### F1_OFI5 decile table -- VAL (n, mean net R)
 decile   n  mean_net_R
      1 467   -0.381275
      2 466   -0.574286
      3 466   -0.499151
      4 466   -0.358240
      5 466   -0.295301
      6 466   -0.314715
      7 466   -0.225175
      8 466   -0.110105
      9 466   -0.135449
     10 466   -0.067969

## F2_OFI1 (OFI_1)
cut(TRAIN-H1 median)=6.7285  placebo_cut(own TRAIN-H1 median)=0.0000
TRAIN-H2: n=3424 lift=0.111 decile_spearman=0.467 trade_spearman(info)=nan
VAL: n=4624 lift=0.150 t_cluster=3.70 t_iid=3.73 decile_spearman=0.503 trade_spearman(info)=nan
fills/week=37.78 placebo_trade_spearman=nan placebo_lift TRAIN-H2=0.070 VAL=0.078 (PASS additionally needs placebo VAL lift < 0.10)
ex_top5_lift=0.038
VERDICT=FAIL

### F2_OFI1 decile table -- TRAIN-H2 (n, mean net R)
 decile   n  mean_net_R
      1 343   -0.257971
      2 342   -0.188986
      3 342   -0.340897
      4 343   -0.423035
      5 342   -0.350523
      6 342   -0.299777
      7 343   -0.263234
      8 342   -0.249363
      9 342   -0.115626
     10 343    0.004851

### F2_OFI1 decile table -- VAL (n, mean net R)
 decile   n  mean_net_R
      1 463   -0.163959
      2 462   -0.354204
      3 462   -0.392851
      4 463   -0.631035
      5 462   -0.268245
      6 462   -0.311558
      7 463   -0.238144
      8 462   -0.235027
      9 462   -0.179754
     10 463   -0.117468

## F3_TSI5 (TSI_5)
cut(TRAIN-H1 median)=0.2234  placebo_cut(own TRAIN-H1 median)=0.0000
TRAIN-H2: n=3385 lift=-0.093 decile_spearman=-0.321 trade_spearman(info)=nan
VAL: n=4575 lift=-0.035 t_cluster=-0.82 t_iid=-0.86 decile_spearman=-0.224 trade_spearman(info)=nan
fills/week=39.00 placebo_trade_spearman=nan placebo_lift TRAIN-H2=0.086 VAL=0.081 (PASS additionally needs placebo VAL lift < 0.10)
ex_top5_lift=-0.154
VERDICT=FAIL

### F3_TSI5 decile table -- TRAIN-H2 (n, mean net R)
 decile   n  mean_net_R
      1 339   -0.345438
      2 338   -0.223366
      3 339   -0.032327
      4 338   -0.199452
      5 339   -0.256254
      6 338   -0.184977
      7 338   -0.338294
      8 339   -0.239790
      9 338   -0.253415
     10 339   -0.401272

### F3_TSI5 decile table -- VAL (n, mean net R)
 decile   n  mean_net_R
      1 458   -0.496808
      2 457   -0.263307
      3 458   -0.192140
      4 457   -0.192110
      5 458   -0.214073
      6 457   -0.196561
      7 457   -0.196965
      8 458   -0.302936
      9 457   -0.362642
     10 458   -0.431459

## Main-session adversarial review (2026-09-24 ~21:45 UTC) — post-hoc lenses, disclosed as such
**Frozen verdicts: F1 FAIL (placebo leg), F2 FAIL (deciles, placebo-adjacent), F3 FAIL (negative).** Two lenses change
what F1 means; neither changes a verdict.
1. **The placebo leg failed on a design flaw (mine).** SPEC clarification 2 let the random placebo minute fall AFTER
   the break; a post-break window carries the outcome (buyers keep lifting after a break that works). Split by timing:
   pre-break placebo lift −0.022 R (TRAIN-H2, n 564) / −0.043 (VAL, n 670); post-break +0.090 (n 2,848) / +0.208
   (n 3,976). OFI is not a name-level trait — but see 2.
2. **The F1 lift is entirely COST, not information.** Gross (`raw_rr`) kept vs dropped: TRAIN-H2 +0.019 vs +0.054,
   VAL +0.016 vs +0.013; cost_R kept 0.159 / 0.171 vs dropped 0.410 / 0.427 — the whole +0.22 / +0.26 net lift.
   Rank corr(OFI_5, cost_R) = −0.50 / −0.54: depth-normalised OFI is large where the spread is small relative to R.
   Inside cost quintiles the gross lift is noise of both signs (VAL −0.12, +0.07, −0.08, −0.07, −0.13). The
   monotone deciles are a liquidity gradient; the best decile still nets −0.093 / −0.070 R.
**Conclusion.** On 12,066 usable HOD-break signals (2025-01..2026-05) the break's GROSS expectancy is ≈ 0 in every
order-flow bucket (VAL gross lift +0.003, SE ≈ 0.04 → a gross lift > ~0.08 R is excluded); net sign is set by cost.
No filter on this population can create an edge the population does not have — together with the 35-cell exit lab
this closes filtering of THIS population; a new HOD frame needs a different signal definition with gross edge first.
