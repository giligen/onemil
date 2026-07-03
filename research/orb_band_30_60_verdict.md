# ORB $30-60 band expansion — NO-GO (2026-07-05)

Owner-approved study, run against a pre-declared ship bar (both OOS eras
positive, >= +$15K combined OOS, MDD <= $25K, corr <= 0.5). Full log:
/tmp/orb_band_study_report.txt; candidates CSV:
/tmp/orb_band_30_60_candidates.csv; script:
research/scripts/orb_band_study.py (rerunnable, bars now cached).

## B1 — widen universe to $3-60 through the frozen pipeline (the simple ship)
FAILED. B1-BASE = **-$6,704** total; era-split +$16,715 (25H1 = TRAIN,
in-sample flattery) / **-$31,617 (25H2)** / +$8,198 (2026). MDD worsens
-$20.1K -> -$25.1K. Classic overfit signature: the frozen small-cap
z-fit mis-scores mid-cap feature distributions OOS — mid-caps rank high
enough to displace small-caps but don't out-earn what they displace.

Nuance worth keeping: the 79 band picks themselves made +$25.1K with
**0.00 daily correlation** to base — genuinely independent money — but
cost ~$31.8K in displaced small-cap picks. Positive, yet worse than
what they push out of the 4 shared slots.

## B2 — band-own fit (the would-be separate strategy)
Also FAILED, decisively: +$8,868 TOTAL over 18 months on the $100K
model (negligible), 25H1 negative, top-5 = 275% of P&L (no stable
edge, pure lottery-ticket noise). No-veto variant: -$21,155.

## Why (mechanism)
As hypothesized pre-study: $30-60 gappers with 500K+ volume are
earnings/M&A events, not low-float squeezes. The +50%-day monster tail
that pays for ORB's 60% losing trades does not exist above $30. 54% of
discovered candidates broke out (same as small-caps) — the ENTRY
dynamics transfer; the PAYOFF distribution does not.

## Standing conclusions
- Universe stays $3-30. One config knob NOT to turn.
- Do not revisit as "separate strategy with tuned exits" without new
  evidence — B2's concentration profile says there's no edge to tune.
- The $1-3 band (bigger ranges, worse spreads) remains untested but is
  now LOWER priority: this result plus 50bps-per-tick friction makes
  the prior unfavorable.
- 2,132 mid-cap symbol-day 1-min bar sets are now cached (additive) —
  future studies on this band are ~2 min, not 20.
