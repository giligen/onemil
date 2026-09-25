# VOID 2026-09-25 20:40 UTC — judge's review: the population filter was a look-ahead
`load_population_low` kept only symbol-days whose daily LOW fell to ≤ $19.99 (floor − 1c) while every trigger sits at
≥ $20.01; the running low at arm time is above the floor, so the sub-floor print always came AFTER the entry — the
population knew the stock kept falling. The numbers below (+0.61 / +0.75 R primary) are NOT a finding. Filter fixed to
the necessary condition (day high ≥ floor + 1c, the exact mirror of HOD's), regression test added, full rerun started
20:40 UTC; this file is overwritten by the rerun's score. Programme count unchanged (a coding-error fix, not a cell).

# Cell 1,439 — low-of-day mirror SHORT — FINAL

Full run: `cell_1439_fills.csv` (75,689 symbol-days, TRAIN-H2 39,539 / VAL 36,150; 0 `no_tape` rows,
coverage 100% both holdouts, gap 0 pp). Scorer: `score_1439.py` (run once, post-amendment).
PRIMARY book = fills with SSR false AND shortable true; SECONDARY = all fills (report-only).

| holdout | book | n | fill rate | mean net R | stop-slip R | t (day-clust) | ex-top-5% | fills/wk | coverage | gap | SSR share | shortable share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN-H2 | PRIMARY | 48 | 0.270 | +0.747 | +0.747 | 4.64 | +0.693 | 2.67 | 100% | 0.0pp | 0% | 100% |
| TRAIN-H2 | secondary | 147 | 0.826 | +0.491 | +0.491 | 4.66 | +0.417 | 5.37 | 100% | 0.0pp | 39.5% | 40.8% |
| VAL | PRIMARY | 67 | 0.289 | +0.606 | +0.606 | 3.11 | +0.542 | 3.35 | 100% | 0.0pp | 0% | 100% |
| VAL | secondary | 182 | 0.784 | +0.561 | +0.561 | 4.52 | +0.488 | 8.09 | 100% | 0.0pp | 36.8% | 46.7% |

**Verdict: FAIL** (not VOID — coverage/gap clear on both holdouts). Every other bar clears on the
PRIMARY book (mean net R ≥ +0.10, VAL t ≥ 2, ex-top-5% > 0, ≥ 60% shortable — 100% by construction
of the primary filter) but PRIMARY fills/week on TRAIN-H2 = 2.67, under the ≥ 3/week bar (VAL clears
at 3.35). SSR/shortable filtering removes ~73% of raw fills, which is why the secondary book's raw
+0.5R/fill collapses to a primary frequency too thin to pass. Caveat: 7/147 (TRAIN) and 2/182 (VAL)
fills had unknown prior close and were conservatively flagged SSR (excluded from primary) — this can
only understate, not inflate, the primary frequency and P&L.
