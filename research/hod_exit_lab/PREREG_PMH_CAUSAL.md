# PREREG — PMH break on a CAUSAL universe. Cells 1,396–1,399 (replaces the void 1,389–1,392)

Frozen before any bar is walked. Everything not stated here is as `PREREG_PMH.md` (signal, entry, stop floor,
exits, splits, statistics, placebos, verification), with these corrections:

## Universe (every condition knowable at 09:30:00 ET)
The ORB wide seed symbol-days (`research/orb_seed_wide/cache/universe_gap3.0_p50.0.pkl`, 23,767 pairs, Jan-2025
.. Sep-2026): open-price gap ≥ 3 % vs the prior close, open $3–50, prior-day volume ≥ 500K, all from daily bars
and the 09:30 open. NO condition may reference anything after 09:30 (no HOD signal, no later volume, no outcome).
The universe-causality trace is the FIRST verification gate, on Opus, before any cell is refuted on other lenses.
Pre-market bars: `scripts/orb_premarket_backfill.py --candidates research/hod_pmh_causal/pm_candidates.csv`.

## Rails added
* Availability: ≥ 80 % of the universe symbol-days must have ≥ 20 pre-market bars with prints, else the pass is
  VOID at the exhibit; report the winner/loser missingness gap.
* Bar density in the hold: a signal is walked only if ≥ 80 % of the 1-minute bars between entry and 15:55 exist;
  report the share dropped; the stop check on a missing minute is treated as a touch (conservative arm reported).
* Cost: the entry-minute NBBO is not in `nbbo.csv` for these names; base cost = the ORB study's measured
  minute-of-day half-spread table is NOT allowed (band). Use the Alpaca quote at the entry minute where the
  `orb_seed_wide` opens cache has it, else mark cost as "proxy: 15 bps half-spread" and report gross beside net.

## Cells
1,396 P1 (+2 R target) · 1,397 P2 (no target) · 1,398 P3 (breakeven lock) · 1,399 P4 (P1 in 09:31–10:00).
First deliverable: the drift exhibit with the minute-since-entry table, and the overlap line with the ORB
production book (share of PMH breaks that are also ORB picks that day — a different question from HOD overlap:
if > 50 %, the population is "ORB by another entry" and is compared to the ORB book instead).
Pass bar and placebos as `PREREG_PMH.md`. Programme count after this pass: 1,399.
