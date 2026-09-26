# RESULT — cell 1,441: prior-day-high (PDH) break, same order (causal arming) — TEST NOT read

| split | fills (rate) | mean net R | day-clust t | ex-top-5 % | fills/wk | coverage / gap | verdict |
|---|---|---|---|---|---|---|---|
| TRAIN-H2 | 145 (61.2 %) | +0.016 | -0.44 | -0.098 | 5.3 | 100.0 % / 0.0 pp | report-only |
| VAL | 239 (70.9 %) | -0.168 | -3.37 | -0.281 | 10.1 | 100.0 % / 0.0 pp | FAIL: mean ≥ +0.10, t ≥ 2, ex-top-5 % > 0 |

* 30 bps stop-slip (every stop exit 0.3 % worse): mean net R TRAIN-H2 -0.072, VAL -0.270.
* Scope: superset symbol-days (day high >= PDH+1c, PDH >= open x (1+min_dist), PDH >= floor+1c); 259238/350694 base symbol-days had no prior-session bars_sip.db data (no_pdh, excluded); 1 symbol-days had < K+2 minute bars. Coverage / gap use the proxy winner (fill at the trigger on the first crossing armed bar; B0 path physics).

## Judge's note (2026-09-26 00:30 UTC)
74 % of the base symbol-days had no prior-session bars in `bars_sip.db` (it holds only the HOD superset's own days), so the
population actually tested is "prior-day-high break on the day AFTER a mover day" — narrower than the spec, but a condition
known at the signal bar (not outcome-based). The verdict is FAIL on that population; the untested remainder is disclosed,
not assumed. Levels were never taken from cache.db or daily bars (data rule).
