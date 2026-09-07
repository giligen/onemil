# ORB no-fill slot recycling — study design (pre-registered 2026-09-07; NOT run yet)

**Problem.** 44% of the honest book's picks never trigger (57 of 130; 47 of 114 after the 9/8 vetoes). Each holds one of 3 slots for up to 60 minutes and books $0. Fills per month, not risk per trade, set ORB's return on budget (9/6: sizing knobs are inert at the stage — the notional cap binds).

**Hypothesis.** Releasing a slot that has not triggered by T minutes after the range end to the NEXT-ranked candidate that has ALSO not yet broken out raises fills per month without lowering the quality of what fills.

**Why this is not the refuted refill.** The PDR/catalyst refill was toxic because it replaced a VETOED pick at 9:35 with a below-cutline name at the same instant — pure selection dilution. Slot recycling replaces a pick that the market itself declined to trigger, after waiting T minutes, with a candidate whose breakout has not happened yet either (a candidate that already broke out before the release time is NOT eligible — its entry would be a chase). The distinction is testable: if recycled fills have the same R distribution as first-pick fills, the hypothesis holds; if they look like the refill study's fills, it dies.

**Mechanics to add to the pipeline (BT) and engine (live) as ONE spec.** `filter.slot_recycle.{enabled, release_after_min: T, max_recycles_per_day}`. BT: for each day, after top-K selection, for each pick with `entered=0` (no-fill) OR whose breakout time > release time: at `range_end + T` release the slot to the next candidate in rank order whose breakout time is ≥ release time (or who never breaks out — then it books $0 like any no-fill). Live: the stop-limit's existing 60-min auto-cancel becomes T-min; on cancel, `check_entries` may place ONE replacement order for the next-ranked candidate that has no breakout yet (its stop-limit above range_high), with the same vetoes applied.

**Grid (fixed in advance).** T ∈ {15, 30, 45} minutes; `max_recycles_per_day` ∈ {1, 3}. Six runs plus baseline on the static-lock dump (`research/orb_veto_study/candidates_static_lock_dump.csv` — it has every candidate's breakout time and exit).

**Pass rule (pre-committed).** KEEP a (T, max) cell only if: fills/month up ≥ 20% AND total P&L ≥ baseline AND MDD not worse AND the recycled fills' mean R ≥ 0.5 × first-pick fills' mean R AND every era not worse than baseline − $100. Otherwise REJECT and close the hypothesis (no T search beyond the grid).

**Cost.** ~half a day: pipeline recycling logic + engine replacement order path + parity tests. Runs in seconds on the dump.
