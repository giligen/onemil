# RESULT 1,706 — marketable-at-submit cohort (built 10/23 per PREREG.md; script build_1702_1706.py, log run_1702_1706.log, per-fill fills_1702.csv col cohort5)
Data: same 137 ORB fills (LIVE 123 + PAPER 14), broker_ts.csv, 0 lost. chase = broker created->filled <= T s; resting = > T. R as in 1,702. CI = day-clustered bootstrap 2000.
drift = (fill - entry_quote_ask)/ask bps. Descriptive per PREREG; no rule change follows.
## Frozen threshold 5 s
chase   n 11 (11 days, ALL live)  meanR -0.31 [-0.76,0.21]  ΣR -3.4  Σ$ -1,220  drift median -3 bps  exits: stop 5, tag_bb 4, force_close 2
resting n 126 (60 days)           meanR -0.15 [-0.36,0.09]  ΣR -18.9 Σ$ -4,987  drift median +55 bps (mean +78)  exits: stop 48, stop_mkt_fallback 4, tag_bb 34, force_close 27, lock 13
diff chase - resting = -0.16 R, CI [-0.62,0.40]
## Sensitivity
 2 s: chase n 8  meanR -0.13 [-0.69,0.47] Σ$ -207  drift med -12 | resting n 129 -0.17 [-0.36,0.08] | diff +0.04 [-0.63,0.73]
10 s: chase n 24 meanR -0.33 [-0.67,0.04] Σ$ -2,967 drift med +5  | resting n 113 -0.13 [-0.36,0.13] | diff -0.20 [-0.61,0.26]
By account (5 s): live chase 11 (-0.31, -$1,220), live resting 112 (-0.09, -$4,060); paper resting 14 (-0.64, -$927); paper chase 0.
Tail (5 s): chase top R 1.49 (ex-top-1 -0.49); resting top R 7.43/4.51/3.57 (ex-top-6 mean -0.35) -> the resting book's total is tail-carried; its median fill is a loser.
## Read
Chase fills pay no drift (median -3 bps vs +55 for resting, i.e. they fill at/below the decision ask) yet their R is not better; point estimate is WORSE than resting
at 5 and 10 s, ~equal at 2 s. All differences have CIs spanning zero by a wide margin (half-width ~0.5 R vs MDE 0.36 R at n 123 iid): NOT RESOLVED, not "no effect".
With n = 11 (8 / 24) the cohort is a handful of names; frequency of chase fills is ~9 % of fills.
## Caveats (adversarial)
1. "Chase" is a selection, not a treatment: orders already above the level at submit are names that gapped through/ran in the first 2 s of the window; they differ in
   setup quality, not just in execution. A rule change (refuse marketable submits) would also delete the unfilled counterfactual (passive-entry adverse selection, memory).
2. created->filled ≤ 5 s also catches orders marketable only because the 50 bps chase guard let them cross; drift here uses entry_quote_ask whose timestamp is undocumented.
3. 3 thresholds x 3 outputs, plus 1,700/1,701: no multiplicity correction (programme count 1,700+). Symbols repeat across days (day clustering only).
4. Sign in 1,700 (longer wait = better R, rho +0.28 live) is the same statement from the other side: resting fills carry the +4-7R tail; the chase cohort is too small
   to say whether it lacks that tail or just lacks draws. 14 paper rows are all resting and all post-9/28 (paper OMS queue ~7 s makes L_brk > 2 s unreachable for paper marketable fills,
   so paper chase = 0 by construction at 2 s and likely at 5 s: the paper cohort split is VOID at small T).
5. Not independently rebuilt yet.
