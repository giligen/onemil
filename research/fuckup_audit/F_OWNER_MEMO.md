# Owner memo — the state after stages A–E (2026-09-17 01:15 UTC)

**No book passed the gates.** This is the honest deliverable (b) of PLAN §5: the closest miss per hypothesis and the
smallest effect each test could see. Everything below is net of the CORRECTED cost contract, honest next-open fills,
the 12/day 4-concurrent book, TRAIN 2025 / VAL Jan–May 26 / TEST Jun–Sep 26 with TEST read once.

## What was wrong in my research (measured, fixed in code)
| defect | size | status |
|---|---|---|
| cost model: band spreads 3.8× too wide vs our fills + entry double-count | **+0.41R/trade** | fixed (contract c, `A/acore.py`) |
| the ≥5%-range universe's causal floor, fill convention, gate at +10R/wk | design | replaced; the honest gate is t ≥ 2 / VAL sign |
| resting stop-limit fill never scored | +30% trades but a WORSE price on shared signals (t −61) | rejected |
| stop at the obvious level (owner) | real: 34% pure wicks; stop −1% = +0.03R | too small; not adopted |
| day direction (owner) | real non-causally (t 4.6); no 09:30-knowable carrier | intraday market state kept as a feature only |
| premarket-$ feature availability leak in D1 | fake +0.415R at t 5.1 on VAL | caught before TEST; standing availability audit |

## What the data says, consistently, across ~4,000 cells
- Raw 1-minute long breakout/pullback families on ≥5%-range small caps: ≈ 0 gross, −0.02..−0.10 net. F5 (HOD-break)
  gross-negative at zero cost (t −7.9). F12 retest / F13 sweep-and-reclaim destructive (−0.2..−0.35R).
- Hold-to-close beats every target/lock exit; wider stops are cheaper; the resting order is not a free lunch.
- Selection (walk-forward gradient boosting on 47 clean causal features): **+0.05R/trade** over first-come, dies without
  the top 5% of trades, fails the reversed-tape gate on VAL where it looked good.
- The ORB news × premarket-$ rule is negative as a SELECTION rule everywhere, including its own 09:35 window on a causal
  gap universe. Within newsy names, more premarket dollars is WORSE (crowding). Live ORB uses it as a sizing
  multiplier on an already-selected book — a different object; this does not overturn the ORB rulebook, it bounds it.
- The causal universe (gap ≥3% / prev-day range ≥8%) is worse than the ≥5%-range set: the rows the old floor discarded
  are the losing rows. "The move already happened" is a feature, not a bias.
- Smallest per-trade effect visible at 80% power: 0.06–0.15R (1.2–6.8 R/week at 4 slots). The $10K/month target needs
  ≈ +0.3R at $400 risk. Effects below ~0.06R are NOT excluded; they are also not worth the owner's capital.

## What is OOS-positive anywhere in the program
1. QQQ noise-band sleeve (Zarattini-type): ≈ 10 bps/day OOS 2024→, t 1.6, Sharpe ≈ 1 unlevered ≈ $5K/yr on $60K.
2. F6 red-to-green with the range floor: +0.09/+0.21/+0.10R but August 2026 is the whole TEST book, dies ex-top-5%.
3. Live ORB honest book: positive, 28% one month, 49% three trades — paused by the owner's rule.

## Stages G and Q (2026-09-17 03:00 UTC) — both closed
- **Short side (G)**: 0 of 24 cells clear G1 (perm p 1.000); gross ≈ flat (+0.11..−0.01R by family), costs 0.05–0.12R.
  Green-to-red and HOD-rejection shorts are destructive (−0.10..−0.16R): a consolidation break is where the stops are
  in EITHER direction. SSR blocks 7–15% of signals, borrow ~0.02R — neither changes a verdict. Borrow availability is
  not modelled. The M18 open fade does not survive into borrowable names at a 09:35 entry.
- **QQQ noise sleeve (Q)**: fills are obtainable and unbiased, breakeven 2 bp/leg vs a 0.09 bp spread — but the OOS
  return is 102% five days (2025-04-09 alone 45%), MDE 5.75 bps/day vs 2.6 observed. ≈ $320/month at $60K 1×,
  worst month −$2,200. Dry-run spec written; not proposed for capital.
- **The one direction-agnostic, live-computable filter this program found**: `range_so_far_pct ≥ 5` at the signal bar
  beats its absence in 12/12 short pairs and 8/10 long pairs. It does not make a book; it makes every book less bad.

## Conclusion
The intraday 1-minute breakout/pullback line on US small caps — long and short, two universes, 19 families, two fills,
seven stop/exit shapes, feature selection, day context, the ORB rule book — has no edge detectable at 4 slots over
2025-01→2026-09 at the corrected cost, with the smallest visible effect 0.06–0.15R per trade. Effects smaller than
that cannot fund $10K/month at this account size. Every number is on disk with its cell count; every positive number
the program produced was a defect, and each defect is now a standing rule.

## What would need the owner's money to test next (not started)
1. Order-flow confirmation at the breakout minute (the CKS OFI plan the owner asked to remember): Databento MBP-1
   windows for the F6/F8 signal minutes (~90K symbol-minutes; tens of dollars) + the NBBO proxy side by side. The only
   feature family in the literature with a real short-horizon signal that this program has not touched.
2. An earnings calendar (post-earnings drift on small caps): no free reliable source; a paid feed or a Databento
   corporate-events add-on.
3. Nothing else in this design space is worth another cell.

- Stage G: the SHORT side of the same populations, mirrored families, borrowable names only (price ≥ $10, ADV20 ≥ $10M,
  no wrappers), same honest machinery, pre-registered — the data's sign points there in every first-hour cell.
- QQQ noise-band sleeve: obtainability audit on the 1-min ETF store and a dry-run spec.
