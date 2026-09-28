# Refuter — cell 1,617 (Frame A, PREREG_1617: held break overnight, MOC -> MOO)

Verdict under review: FAIL (builder cell_1617.py, rebuild rebuild_1617.py agree). **Refuted: NO — the FAIL stands under every lens.**
Scripts: `review/refute1617_tails.py`, `review/refute1617_pull_bars.py` (reads research/bf_zero/bars_sip.db read-only;
output rows kept in the session scratchpad), `review/refute1617_scale_causal.py`. Net = raw close->next open bps - 10.

## 1. Raw price scale / splits
- Panel close vs the 15:59 ET SIP minute close (7,980 of 9,911 fills with bars): median |diff| 5.0 bps, p90 26 bps, 0.6 % > 100 bps, 0 > 20 %.
- Panel next_open vs the next day's 09:30 SIP bar open (5,550 matched): median |diff| 0.0 bps, p90 9 bps, 0 > 20 %.
- close/level ratio 0.545..1.322 (1 row outside 0.7..1.5) — daily and intraday are on the same raw scale; held/failed labels recompute exactly (0 mismatches).
- next_open is always the NEXT trading day (gap = 1 session for all 43,730 matched rows); no stale multi-day gaps.
- Zero-return mass is 2.3 % (the median of exactly -10 bps net is a coincidence of the median landing on ret=0, not a stale-open defect).
- A bar-priced return (15:59 close -> 09:30 open) on the held population: VAL -11.8 bps (n 1,334), TRAIN -6.6 — same sign as the panel.

## 2. Earnings / halts
- No earnings calendar in the repo (confirmed; 0 % coverage — a disclosed gap, not fixable here). Symmetric substitute:
  VAL winsorised 5/95 = -9.6 bps, clipped at +-10 % = -9.3 bps -> removing the big-gap nights (where earnings live) does not reveal an edge.
- Halts at the open: the first RTH bar is 09:30 on 5,532/5,548 matched next days (16 at 09:31-09:32). Not material.

## 3. Placebo
- Held minus in-play universe, same nights: VAL -11.2 bps (t -0.74). The universe itself is flat on VAL (-1.2 bps). Note the
  universe contains the held rows themselves (mild contamination toward zero) — it cannot flip a negative margin.

## 4. Tails (VAL, held-break, n 2,865, mean -7.00)
| lens | VAL | TRAIN-H2 |
|---|---|---|
| ex-top-5 % | -63.6 | -50.1 |
| ex-bottom-5 % | +45.8 | +55.7 |
| winner-capped +10 % | -16.2 | -3.5 |
| winsorised 5/95 | -9.6 | +2.0 |
| best 2 nights dropped | -18.5 | +1.5 |
| worst 2 nights dropped | -0.7 | +16.7 |
| equal-weight day mean | -14.6 | +13.5 |
Symmetric fat tails around ~0; no tail treatment moves VAL above 0, let alone the +8 bps bar.

## 5. Month concentration
VAL by month: Jan +6.9, Feb -47.9, Mar -69.8, Apr +61.0, May +7.3 (3/5). TRAIN: 2/6 positive, carried by Oct (+88.6).
VAL spans 5 months, not 6 (disclosed by the builder; stricter than written but irrelevant — every other line fails).

## 6. Held-break condition read from the daily close — a CAUSALITY defect, not verdict-changing
The PREREG population (close >= level) uses the closing auction price itself, which is not known at the MOC cutoff (15:50).
Causal version (15:49 SIP close >= level): 3,563 of 5,073 held rows keep the label, 1,510 flip, 266 failed rows join.
Causal held VAL: n 2,170, -10.6 bps, t -0.39, ex-top-5 % -64.8; TRAIN-H2 +13.7, t 0.56. Still FAIL. (15:49 bar coverage 80 %;
rows without bars fall into 'not held' — stated.) A PASS under the close-defined rule would still have needed this repair before live.

## 7. Other checks
- Cost: at ZERO auction cost VAL gross = +3.0 bps, t ~0.1 -> still fails the +8 bps / t 2.5 bar; the 5 bps/leg assumption is not what kills it.
- TRAIN-H2: the base book's TRAIN is entirely H2 (2025-07..12), so the builder's TRAIN line IS TRAIN-H2 — correct.
- Leveraged wrappers are ~1/3 of the population (legacy-list fallback, imprecise). Ex-wrappers VAL -11.8 bps, wrappers +2.5 — no hidden edge in either slice (post-hoc, report only).
- No duplicate (symbol, day) keys in the held population.

## Conclusion
No defect found that changes the verdict. The one real defect (close-defined held condition is not causal at the MOC cutoff)
makes the cell look, if anything, slightly better than the causal version. Cell 1,617 FAILS; frame A closes with its numbers.
Adequacy note: n 2,865 VAL nights, day-clustered SE ~28 bps -> MDE at 80 % power / 2.5 t ~ 95 bps/night; the test can only
exclude large overnight drifts, not a few bps. The universe placebo (n 18,418) is equally flat, so no overnight drift exists at
the population level to be selected by the held-break label.
