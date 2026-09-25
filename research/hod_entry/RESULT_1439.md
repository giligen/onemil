# RESULT — 1,439: low-of-day mirror short (corrected population; first run VOID, see the note at the end)

Scored 2026-09-25 21:31 UTC by `score_1439.py` on the full corrected run (33,171 symbol-days, 95 no-bars, ticks for every
crossed bar). Primary book = fills on non-SSR days on shortable names (the pass bar applies); secondary = all fills.

| holdout / book | n | fill rate | mean net R | 30 bps stop-slip | day-clustered t | ex-top-5 % | fills/wk (12/4) | coverage | gap | SSR share | shortable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN-H2 primary | 2,398 | 0.43 | −0.150 | −0.150 | −2.25 | −0.262 | 36 | 100 % | 0 pp | 0 % | 100 % |
| TRAIN-H2 all fills | 4,696 | 0.84 | −0.129 | −0.129 | −2.10 | −0.239 | 48 | 100 % | 0 pp | 23 % | 59 % |
| VAL primary | 3,344 | 0.48 | −0.289 | −0.289 | −4.80 | −0.407 | 43 | 100 % | 0 pp | 0 % | 100 % |
| VAL all fills | 5,877 | 0.84 | −0.279 | −0.279 | −5.82 | −0.396 | 50 | 100 % | 0 pp | 25 % | 67 % |

**FAIL, decisively.** The short mirror of the HOD-break resting order loses on both holdouts (bar was ≥ +0.10 R, t ≥ 2),
worse on VAL, and worse ex-top-5 %. SSR and borrow remove 49–57 % of raw fills without changing the sign. Caveat: the
stop-slip column charges only stops hit inside the fill bar (a scorer simplification); charging every stop exit lowers
the book further. Unknown prior close on 77 fills was flagged SSR (conservative). Cell closed; the frame "short the new
low" carries the same non-information as the long on this population.

## VOID note — first full run (20:22 UTC)
`load_population_low` kept only symbol-days whose daily LOW fell to ≤ $19.99 while every trigger sits at ≥ $20.01; the
running low at arm time is above the floor, so the sub-floor print always came AFTER the entry (the population knew the
stock kept falling): +0.61 / +0.75 R, t 3–5 — a look-ahead, not a finding. Filter fixed to the necessary condition
(day high ≥ floor + 1c, HOD's exact mirror), regression test in `test_cell_1439.py`, rerun above. Programme count
unchanged (a coding-error fix). Void fills kept in the session scratchpad only.
