# PREREG — cell 3, vol line: tracking gate on 5-day returns (2026-10-09)

Cell 2 (`REPORT2.md`) VOID at the pre-registered DAILY tracking gate (VXX R² 0.855, SVXY 0.489): the VX futures
settle at 16:15 ET, the ETPs close at 16:00 — a 15-minute timing offset that adds daily noise but does not accumulate.
This cell pre-declares the fairer gate BEFORE any strategy number is read: R² of overlapping 5-day log returns,
synthetic leg vs real ETP, ≥ 0.95 on both legs (VXX 2018-01-18 → 2026-10, SVXY 2016-01 → 2026-10, era-correct
leverage); also report the 20-day R² and the annualised return difference (synthetic − real) per leg in %/yr — if
|difference| > 3 %/yr on either leg the rebuild is biased: VOID. If the gate passes, run the PREREG_1 rule and
variants on the synthetic legs 2011-01 → 2026-10 exactly as PREREG_1/2 specify (same splits, cost, placebo, cadence
bar with R = $50, ex-top-5 %, both 2018-02-05/06 day losses on $5K) and ALSO on the real ETP legs for the overlapping
years as a side-by-side column. Reuse `run2.py` / `index_rebuild.csv`. Outputs: `run3.py`, `tracking3.csv`,
`trades3.csv`, `weekly3.csv`, `REPORT3.md` ≤ 70 lines. Rules as PREREG_1 (write only under `research/vix_term/`, no
orders, no config/.env/crontab/cache edits, no git, ≤ 40 calls). Return ≤ 120 words: 5-day R² and return difference
per leg, the 1.05/0.95 line per half at measured cost incl. ex-top-5 %, verdict. This task IS the owner's request.
