# REPORT — cell 1,438: causal arming (PREREG_1438.md + amendment) — TEST NOT read

| split | fills (rate) | mean net R | day-clust t | ex-top-5 % | fills/wk | coverage / gap | verdict |
|---|---|---|---|---|---|---|---|
| TRAIN-H2 | 4398 (84.5 %) | -0.208 | -8.14 | -0.322 | 50.6 | 100.0 % / 0.0 pp | report-only |
| VAL | 5513 (82.1 %) | -0.224 | -8.65 | -0.339 | 50.2 | 100.0 % / 0.0 pp | FAIL: mean ≥ +0.10, t ≥ 2, ex-top-5 % > 0 |

* Overlap with 1,427 fills (day, symbol) — TRAIN-H2: both 1114 (+0.242 here / +0.319 in 1,427), 1,438-only 3284 (-0.361), 1,427-only 51 (-0.448); VAL: both 1388 (+0.276 here / +0.269 in 1,427), 1,438-only 4125 (-0.393), 1,427-only 55 (-0.525).
* 30 bps stop-slip (every stop exit 0.3 % worse): mean net R TRAIN-H2 -0.319, VAL -0.333.
* Report-only rv incl. bar j+1 tick volume to the fill: TRAIN-H2 4414 fills -0.207 (t -8.17); VAL 5531 fills -0.223 (t -8.67).
* Scope: superset symbol-days with >= 1 armed bar whose next bar crosses; minute bars cache.db + bars_sip.db (amendment); 127 superset symbol-days had < K+2 bars in both. Coverage / gap use a proxy winner (fill at the trigger on the first crossing armed bar).

* **1,427 level check vs bars_sip.db (all 1,427 signals, TRAIN-H2/VAL/TEST):** fills 3528/3580 match; NO-FILLS 2751/8115 match, 5364 have a 1,427 level BELOW the true HOD (cache.db sparse, e.g. FUN 2025-07-01: 7 cache.db bars, level 30.77 vs true 31.84). With the true level these signals' first cross mostly fills: 1,611 TRAIN-H2/VAL fills in 1,427's own break bar that 1,427 scored as ask-above-limit no-fills earn −0.687 R here. The 1,427 fill condition was largely selecting symbol-days where cache.db had the correct level; 1,427's TEST PASS is therefore compromised.
