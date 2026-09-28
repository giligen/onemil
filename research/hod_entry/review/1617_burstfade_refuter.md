# Refuter: PREREG_1617 Frame B (burst fade, cells 1,619 / 1,620)

Verdict under test: builder `cell_1619.py` says 1,619 FAILS (VAL −0.075 R_f, t −4.24). Rebuild says FAIL (−0.60 R_f).
**Refuted: NO. The FAIL stands, and every lens makes it worse, not better.** I found defects in the builder, but none
of them changes the verdict. Probe: `review/refute_1619_probe.py` re-walks the builder's own 3,750 fills under
switchable rules. Summaries: `review/refute_1619_analyze.py`. Rows: `review/refute_1619_probe.csv`.

## Why no fix can flip it: the payoff geometry
- Entry is at level ×1.0015, the cover at level − $0.01 and the stop at level ×1.0075. So R_f ≈ 0.6 % of price and
  **the most a trade can make is +0.333 R_f**.
- Measured means: a cover is +0.29 R, a stop is −1.34 R.
- Breakeven needs 82.4 % covers. The +0.15 bar needs **91.7 %**. VAL shows 77.2–77.8 %; TRAIN shows 74.4–74.9 %.
- With the stop tail added back the book is still negative: about −0.03 R on VAL and −0.07 R on TRAIN.
- The 1,840 excluded gap-unconfirmed entries and the 629 no-tape entries cannot rescue it. For the pool to reach
  +0.15, they would need to average above +0.6 R, which is more than the +0.333 R ceiling.

## Lenses
| Lens | Finding | Direction |
|---|---|---|
| Cover rule | The PREREG says the cover fills on a print STRICTLY below. The code fills on a touch (`<=`). Using the strict rule: 1,619 VAL −0.084 (t −4.77), TRAIN −0.131. | The builder is optimistic. This is a spec deviation. |
| Obtainability of the resting offer | 70 % of the first prints above the limit are odd lots (median size 24). In 9 % of fills no round-lot print trades above the limit at all. Fills with ≥ 500 round-lot shares above the limit average **−0.106 R (VAL)** and −0.178 R (TRAIN). The thin, doubtful fills average +0.04 / +0.07 R. | Adverse selection. The positive part of the book is the least obtainable. Requiring ≥ 100 round-lot shares: 1,619 VAL −0.100 (t −5.15); 1,620 VAL −0.014 (t −0.59). |
| Short stop in a fast market | On tape, the builder fills the stop AT the stop price. The print that triggers the stop sits a mean 4.8 bps above it (p90 13.8, p99 59.5). Filling at that print instead: 1,619 VAL −0.099 (t −5.46). The builder's charged tail of 11.9–13.8 bps already covers the mean excess, so this lens is roughly neutral. The worst single trade is −7.9 R (a gap through the stop). | Neutral to optimistic |
| Borrow / SSR | Every filled symbol is also `easy_to_borrow` in the snapshot. Alpaca shorts ETB names only, so the population is fine. SSR matters little: a resting offer ABOVE the bid is allowed under Rule 201. The real defect: the flags are a static 2026 snapshot applied to 2025–26 days. Names that were later delisted or went hard-to-borrow are excluded (3,680 names, including those missing from the file), and the squeeze-prone names are the ones most likely to be dropped. | Survivorship makes the builder optimistic |
| Runner tail | The never-retest cohort (the stops) costs −0.79 % of price, which is 1.3 R. The left tail is heavy (min −7.9 R). The right tail is capped at +0.33 R. | Structural. This is a short-gamma book. |
| Tails and days | VAL ex-top-1 % −0.079, ex-top-5 % −0.096. Only 7.4 % of the positive sum comes from the top 5 %, so the tail is not what drives the result. Green days: 39 % (VAL), 32 % (TRAIN). **Every month is negative on both splits** (11/11). | The FAIL is broad, not tail-driven |
| Mirror pairing units | `base_net_pct = outcome_R × R_pct` is in % of price, and so is the short's net_pct, so the units match. But the base R is 1.5 % of price and the short R is 0.6 %, and the two exit on different horizons. That makes it a pairing on the same fills, not a true mirror. The builder's row puts the correlation (−0.103) in the `placebo_or_paired` slot, which is not a paired difference. Short + long on VAL = −0.013 %, on TRAIN = −0.226 %. | Reporting defect. No effect on the verdict. |
| Spread quartile | Tight-spread Q1 on VAL is +0.021 R (t 0.73, n 537) and TRAIN Q1 is −0.027 R. Choosing Q1 would be selecting after seeing VAL, which the PREREG forbids, and it is not significant anyway. | — |

## Rebuild disagreement
The Jaccard (0.97) and the 0.19 % agreement within tolerance both fail the bar. The cause is in the rebuild: it
charges half the spread on BOTH passive legs. A resting limit fills AT its limit under the PREREG, so the rebuild's
−0.60 R double-counts cost. Its price walk agrees with the builder's (91 % of exit reasons match; 79 % of fills
agree within tolerance on gross). The independent-check bar is therefore formally unmet. The verdict does not
depend on it, because the geometric ceiling above holds under either cost model.

## Consequence
Frame B closes as FAIL with its numbers. The 1,620 report-only VAL result (+0.004 R, t 0.17) falls to −0.014 R under
the strict cover rule plus round-lot obtainability. It must not be read as a lead.
