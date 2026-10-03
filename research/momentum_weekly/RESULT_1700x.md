# RESULT 1,700x -- short overlay on the guarded sleeve (PREREG_1700x.md, FROZEN)

Long-leg reproduction: engine end $596,394 = simulate(ranked_g) $596,394; stored 1700u guard curve end $596,394, DD -38.25% vs -38.25%. PASS (<0.5 %).
GREF: CAGR 29.34% / DD -38.3% / ratio 0.77; halves 0.67 / 0.99; worst week -18.52%; episodes 2021-02-16..2021-05-11 -38.3%; 2020-02-14..2020-03-19 -37.1%; 2025-02-13..2025-04-07 -33.6%.

| cell | CAGR | DD | ratio | h1 | h2 | worst wk | P10 | green | ep1/2/3 | ovl in GREF worst10 | ovl worst wk | cost+borrow /yr | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| X-beta|h25 | 27.8% | -41.9% | 0.66 | 0.52 | 1.01 | -16.4% | -5.2% | 57% | -42%/-33%/-34% | +0.95% | -3.0% | 0.15% | N |
| X-beta|h50 | 26.1% | -46.0% | 0.57 | 0.39 | 0.99 | -18.1% | -5.2% | 55% | -46%/-28%/-34% | +1.90% | -6.0% | 0.34% | N |
| X-loser|h25 | 28.9% | -45.5% | 0.63 | 0.49 | 1.07 | -17.9% | -5.3% | 56% | -46%/-31%/-33% | +0.34% | -8.9% | 1.75% | N |
| X-loser|h50 | 28.4% | -54.7% | 0.52 | 0.35 | 1.11 | -24.6% | -5.7% | 55% | -55%/-30%/-33% | +0.69% | -17.8% | 3.76% | N |

Monthly returns % (GREF | loser h25 | loser h50):
  2020-03: -8.7 | -3.6 | +2.5
  2020-04: +12.4 | +5.7 | -1.5
  2020-05: +6.1 | +5.4 | +4.5
  2020-06: +24.0 | +22.2 | +19.9
  2020-07: +13.9 | +18.1 | +23.6
  2026-03: -13.6 | -13.4 | -13.1
  2026-04: +39.6 | +43.8 | +48.3
  2026-05: +22.9 | +22.9 | +22.9

Rule: DD better >= 8 pt AND ratio >= GREF+0.10 AND both halves >= GREF half AND worst week no worse than 2 pt. NO cell passes: short overlay CLOSED as a DD repair at these ratios.
Adversary caveats: (1) delisted losers carry a stale ffilled open (no delisting squeeze/halt gap); short squeeze, recall and hard-to-borrow costs beyond 3 %/yr are not modelled. (2) Marks at daily opens, borrow on 252-day year. (3) Costs are the band model, not NBBO; the book is re-sized (delta) each Monday. (4) 4 cells, one window, Oct-2026 knowledge of the crash months. (5) Proceeds earn 0, no margin interest/limits modelled.
