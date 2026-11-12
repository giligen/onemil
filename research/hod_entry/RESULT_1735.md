# RESULT 1,735 — the stop is not the cost (no-stop / soft-stop / dead-trade variants)

Scorer `1735_score.py`; per-signal `1735_per_signal.csv`; metrics `1735_summary.csv`. TRAIN + VAL only (12,135 signals; 12,010 with a spread; TEST never loaded). Independent rebuild (V1, V4) NOT yet done: nothing here reaches the owner as a finding until `REBUILD_1735.md` agrees.

**B0 reproduction.** 12,010 rows vs `b0_trades.csv`: TRAIN mean -0.2212 vs -0.2212 (delta 0.0000), VAL -0.3043 vs -0.3043; max row diff 9e-16. Gate passed.

**Disclosure.** The exit lab already scored time stops X7 (60 min below entry, dR -0.004) and X7b (30 min, entry+0.25 R, dR -0.011), both ~0. No-stop and soft-stop were not scored there; V4/V5 agree with X7/X7b.

| Variant | cost R | VAL dR | day-t | ex-top5 % dR | VAL own mean R | own day-t | placebo pct | Verdict |
|---|---|---|---|---|---|---|---|---|
| V1 no-stop, half | 0.157 | +0.217 | 6.7 | +0.061 (TRAIN -0.014) | -0.087 | -1.8 | 100 | FAIL |
| V2 no-stop+target, half | 0.157 | +0.202 | 9.0 | +0.087 | -0.103 | -2.9 | 100 | FAIL |
| V3 soft stop 2R close | 0.315 | +0.041 | 1.5 | -0.117 | -0.264 | -4.5 | 92 | FAIL |
| V4 dead-trade 20 m | 0.315 | -0.006 | -0.3 | -0.071 | -0.310 | -10.1 | 38 | FAIL |
| V5 dead-trade 60 m | 0.315 | +0.010 | 1.1 | -0.046 | -0.294 | -8.0 | 87 | FAIL |

(B0 VAL mean -0.304.) TRAIN dR: V1 +0.176 (t 6.4), V2 +0.133 (4.5), V3 +0.002, V4 -0.034, V5 -0.015. TRAIN halves same sign for V1, V2, V4, V5; V3 flips.

**Reading.** V1/V2 pass every paired clause bar V1's TRAIN ex-top-5 % (-0.014) but the pre-committed own-book clause fails: the half-size no-stop book is still negative on VAL (-0.09/-0.10 R baseline-R, t -1.8/-2.9), weekly P10 -60 to -75 R, green weeks 23-32 %, max drawdown 550-575 R. Removing the stop makes the book less negative, not a book. The soft stop and time stops do nothing (cost doubles to 0.315 R because exits are at the open with full size).

**Caveats.** Cost = 2 x half-spread + 2 bp/side from the exit-lab model, scaled by s (not measured NBBO per trade). s = 0.5 halves the dollar book, so "dR" is in baseline-R and the dollar stake falls by half. A no-stop long carries uncapped gap/halt tail risk the path data cannot show (R is ~1.3 % of price). V3 same-bar close-vs-target tie resolved as soft stop; soft-stop next-open fills are obtainable only if the close is observed live before the open. Placebo = day-level sign-flip of paired dR (300 draws), my reading of "shuffle the day label". Disclosed: V1 dR is large and significant, so the reduction in loss is real vs B0, but both books are negative.
