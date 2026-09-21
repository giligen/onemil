# S1 exit pass — cells 1,319–1,321. VERDICT: no cell passes; the S1 stratum is flat under four exits

PREREG `PREREG_S1_EXIT.md` (commit 19f00bd) before any walk. Population `out/runS1_features.csv`, production
selection frozen, 8 slots, R = $375, cost M; one knob per cell via the new `ORB_BT_*` research overrides (warned in
the config banner). Books `out/runS1_{E1,E2,E3}.csv`, logs beside them; scorer `score_s1_exit.py`.

| cell | split | n | mean R | t_clu | ex-top-5 % | wk MDD (base) | trades changed | halves |
|---|---|---|---|---|---|---|---|---|
| baseline | TRAIN | 210 | −0.026 | −0.87 | −0.087 | −9.24 | | +0.02 / −0.06 |
| baseline | VAL | 141 | +0.040 | +0.95 | −0.023 | −2.85 | | |
| E1 touchgo off | TRAIN | 210 | −0.042 | −1.29 | −0.104 | **−12.19** | 41 (sum −3.3 R) | +0.01 / −0.08 |
| E1 touchgo off | VAL | 141 | +0.081 | +1.14 | −0.025 | **−3.70** | 30 (sum +5.7 R) | |
| E2 lock arm 1 R | TRAIN | 210 | −0.033 | −1.40 | −0.082 | −9.22 | 46 (sum −1.5 R) | −0.02 / −0.04 |
| E2 lock arm 1 R | VAL | 141 | +0.041 | +1.17 | −0.010 | −1.88 | | |
| E3 scale 50 % @ 2 R | TRAIN | 210 | −0.026 | −0.91 | −0.079 | −8.84 | 38 (sum −0.05 R) | +0.01 / −0.05 |
| E3 scale 50 % @ 2 R | VAL | 141 | +0.037 | +0.92 | −0.019 | −2.85 | | |

Pass bar: every cell fails 1 (TRAIN ≥ +0.10, VAL ≥ +0.15) and 2 (t ≥ 2). E1 also fails 5 (MDD deeper on
both splits): touchgo is protective on 2025 and its VAL gain is 30 trades, sd-sized. E2 halves both negative.
E3 is a no-op at this R scale (the 2 R partial rarely triggers before the stop).

## Adequacy
n = 141 VAL, sd ≈ 0.45 R → MDE at t = 2 ≈ +0.08 R per trade for a full-book change; the largest VAL gain seen
(E1, +0.04 R) is under it, and it comes with a 2025 loss. Compounding sizing means `_sized_pnl` drifts on every
trade after a changed one; the "trades changed" column is on `pnl_pct`, the R columns on `_sized_pnl`.

## What this pass establishes
Under production selection the gap 3–5 % / $3–30 stratum is flat at every exit tried (static lock, no touchgo,
early lock, early partial) and at every selection cut tried (`REPORT_S1_FILTERS.md`). Its 6.6 fills/wk are
frequency without edge; adding it to the live seed would dilute the production book's +0.41 R. Per the PREREG
decision rule the S1 line moves to the regime-conditional pass (HMM calm weeks) under its own PREREG, as a
report-only check on the walked baseline; S3 (gap 3–5 %, open $30–50) stays the only live-widening lead.
Cell count: **1,321**.
