# Recalibration: BF/ORB honest books at realized entry cost (2026-09-20)

Diagnostic (1,295 BF / 1,296 ORB). Method: `PREREG_RECAL.md`. Script: `recal.py` (validated:
BF formula reproduces `f45_bf_book.csv`'s own `pnl_meas` exactly; ORB-P reproduces the known
honest B+ total $14,061.55 to the dollar; ORB R-proxy = `pnl_pct/range_size_pct` matches
`trading/ramp_bt_band.py::load_orb_bt_r` verbatim; BF R = `pnl/R_dollar` matches its `BF_R_BASIS_RISK`).
BF exit leg halved (`sp_x x 0.5`) in ALL three settings per instructions — this alone recovers
most of frames14's VAL loss before any entry recalibration.

## Bull flag (regen-7 P1, 56 trades; frames14's own pnl_meas total $67,201, vs REPORT.md's
$69,255 — small file/arm mismatch, not investigated further)

| setting | entry bps | TRAIN 2025 $ | VAL Jan-May26 $ | TRAIN MDD | VAL MDD | pooled mean R |
|---|---|---|---|---|---|---|
| P (current measured) | 93.9 | $92,548 | **+$11,736** | -$17,500 | -$20,375 | +0.462 |
| M (midpoint 33.9/12.6) | 23.2 | $115,554 | **+$25,668** | -$14,335 | -$16,280 | +0.691 |
| O (realized 12.6) | 12.6 | $119,032 | **+$27,774** | -$13,971 | -$15,661 | +0.726 |

**BF VAL does NOT flip sign under M or O — it is already positive at P** because halving the
exit charge (per instructions) recovers most of what frames14's full-measured-cost VAL loss
(-$8,045) was driven by. Entry recalibration (M, O) then adds further upside on top: VAL
+$11.7K -> +$25.7K -> +$27.8K, MDD improves ~25%. Day-clustered VAL t: P 0.56, M 1.25, O 1.35
(n=12 weeks, n=15 trades — underpowered either way).

**Ramp-band number under M**: pooled mean R/trade = **+0.691** (vs +0.462 at P, the level
`trading/ramp_bt_band.py::bf_reference()`'s P1.csv reference currently implies). A P1-under-M
reference CSV would shift the bootstrap band's center up by ~+0.23R — makes the live-vs-BT
band easier to clear, not harder; the band was never rebuilt for this diagnostic.

## ORB (B+ book, 165 filled, $10K stage; exit unchanged, entry flat-bps-on-notional)

| setting | entry bps | TRAIN 2025 $ | VAL Jan-May26 $ | TRAIN MDD | VAL MDD | pooled mean R |
|---|---|---|---|---|---|---|
| P (current) | 13.5 | $6,662 | $6,386 | -$642 | -$489 | +0.607 |
| M (midpoint 13.5/3.1) | 8.3 | $6,808 | $6,456 | -$635 | -$479 | +0.620 |
| O (realized 3.1) | 3.1 | $6,953 | $6,525 | -$629 | -$468 | +0.632 |

ORB does not move materially under any setting (~+$150-290 TRAIN, +$70-140 VAL, out of
$6-7K) — its entry-leg cost is already a small share of stage-size $/trade, and current-model
(P) already sits close to the realized number, so there is little room to move.

## Caveats
- BF's `sp_e`/`sp_x` semantics were reverse-engineered by exact reproduction of the shipped
  `pnl_meas`, not derived from a documented bps convention — M/O apply a level-shift via
  rescale-to-target-median, preserving shape only.
- ORB has no per-trade spread column; entry cost is a flat bps on notional, ignoring the
  bimodal wide/narrow-spread mix the live number (3.1 vs 13.5) is itself an aggregate of.
- TEST (>=2026-06-01) sealed, not opened, per instructions.
- n=15 BF VAL trades is not enough to resolve sign confidently even before this exercise;
  read the VAL numbers as directional, not decisive.
