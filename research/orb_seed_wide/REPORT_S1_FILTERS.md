# S1 loser filters — cells 1,316–1,318. VERDICT: no cell passes; F-B reverses on VAL

PREREG `PREREG_S1_FILTERS.md` (commit d9c34de) before any VAL number. Book `out/runS1_true.csv`,
entered rows, R = $375 (`_sized_pnl`), TRAIN 2025 (50 fill-weeks), VAL 2026-01..05 (22 fill-weeks),
TEST sealed. Scorer `score_s1_filters.py`; per-cell trade CSVs `out/s1_f*_{TRAIN,VAL}.csv`.

| cell | split | n | mean R | t_clu | ex-top-5 % | fills/wk | dropped cohort R | halves |
|---|---|---|---|---|---|---|---|---|
| baseline S1 | TRAIN | 210 | −0.026 | −0.87 | −0.087 | 4.20 | | |
| baseline S1 | VAL | 141 | +0.040 | +0.95 | −0.023 | 6.41 | | |
| F-A gap ≥ 4 | TRAIN | 85 | +0.100 | +1.70 | +0.036 | 1.70 | −0.112 | +0.19 / +0.05 |
| F-A gap ≥ 4 | VAL | 69 | +0.063 | +1.09 | +0.013 | 3.14 | **+0.018** | |
| F-B rr ≤ 2 | TRAIN | 135 | +0.029 | +0.78 | −0.037 | 2.70 | −0.125 | +0.07 / −0.00 |
| F-B rr ≤ 2 | VAL | 82 | **−0.040** | −1.02 | −0.092 | 3.73 | **+0.151** | |
| F-AB | TRAIN | 57 | +0.184 | +2.58 | +0.109 | 1.14 | −0.104 | +0.26 / +0.14 |
| F-AB | VAL | 38 | +0.007 | +0.11 | −0.049 | 1.73 | +0.052 | |

Pass bar: F-A fails 1 (VAL +0.063 < +0.15), 2 (t 1.09), 5 (VAL dropped cohort +0.018, not < 0);
F-B fails 1, 2, 4, 5 and its sign flips on VAL; F-AB fails 1, 2, 4, 5, 6.

## Adequacy (a null is a claim about my test first)
* F-A MDE at t = 2 on VAL (n 69, sd ≈ 0.45 R) is ≈ +0.11 R; the point estimate +0.06 is inside it. What
  is excluded: a gap-≥4 edge of the size the production seed shows (+0.41 R). Direction is consistent
  everywhere (both splits, both halves, ex-top-5 % ≥ 0 both splits) — a small real effect is not ruled
  out, but it is below the bar and the dropped VAL cohort was not a loser cohort.
* F-B was the second-strongest of 23 scanned features (t −2.5 on TRAIN) and reverses on VAL: the
  expected one false positive of the scan, now identified. Range extension is NOT a usable veto in S1.
* Cadence bar: no week in any cell reaches +5 R at this sizing; the walked book's realized |R| per trade
  is compressed (mean stop −0.32 R) because the $10K-stage notional cap binds on low-priced names, so
  C1/C2/C7 are structurally unreachable on this book at this stage; reported, not scored.

## What this establishes
Selection cuts learned from S1's own TRAIN losers do not transfer to VAL. The one durable structure is
the gap monotone (gap ≥ 4 keeps TRAIN and VAL positive) — already known from the production seed, and
too weak alone. Per the PREREG decision rule the next pass is the EXIT pass (stops = 46 % of TRAIN
trades at −0.32 R mean, scale_eod = +0.96 R mean on 8 %), under its own PREREG; and the S3 cell
(gap 3–5 %, open $30–50) remains the live-widening lead. Cell count: **1,318**.
