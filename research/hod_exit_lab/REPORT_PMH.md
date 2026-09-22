# PMH (pre-market-high break) — cells 1,389–1,392: VOID. A fabricated book, caught at the last gate. 2026-09-22

Workflow `hod-pmh-population`: 19 agents, 1.36 M tokens. All four cells "passed" the pre-registered bar with
+0.62 to +0.99 R net per trade, t 9–12, D1/D3 placebos beaten by 0.3–0.9 R, all three Sonnet refuter lenses
cleared, and the Opus independent rebuild reproduced every number to 98.5 % trade agreement. **None of it is
reportable.** The rebuild's own reading, confirmed here:

1. **Universe look-ahead — my spec error.** `PREREG_PMH.md` set the universe to "the HOD-break universe's
   symbol-days (gappers with a signal that day)". A symbol-day enters that universe only because it produced an
   HOD-break signal, and for **92.4 % of the PMH trades that HOD signal fires AFTER the PMH entry**. The universe
   therefore pre-selects days that went on to make a new high after the entry — a book that cannot lose much (VAL
   exit mix 43 % target / 37 % close / 20 % stop on a capped +2 R rule is impossible without selection). CLAUDE.md
   rail 2 ("membership must be knowable at the signal bar") fails by construction. The PREREG even wrote the
   condition down and did not recognise it as look-ahead; the same class as the ignition day-cohort artefact.
2. **Availability rail.** Usable coverage 36 % of the candidate symbol-days (8,393 of 15,656 have ≥ 20 pre-market
   bars; 2,850 more fail the PM-volume gate), far under 80 %.
3. **Sparse-bar fill realism.** Thin names have missing 1-minute bars inside the hold, so stop touches can be
   skipped; shared by both implementations, biases the book up.
4. The PREREG's own overlap kill-switch passed (12.5 % of PMH entries within 5 min of the HOD signal) — because
   the HOD signal comes later, which is precisely the problem.

**What the rails did and did not do.** Three Sonnet refuters with sequencing / fills / stability lenses cleared
a fabricated book; the universe-causality question was not one of their lenses, and it is the one that matters.
The Opus rebuild asked it. From now on the FIRST verification gate on any new population is a universe-causality
trace (is every membership condition known at the signal bar?), run on Opus, before any other lens is paid for.

**Salvage.** The mechanism (a level the crowd watches, broken once) is untested, not refuted. It is re-registered
as `PREREG_PMH_CAUSAL.md` on a universe knowable at 09:30 (the ORB wide seed: gap ≥ 3 % at the open, $3–50, prior
volume ≥ 500K, from daily bars), with pre-market bars fetched for that universe and a bar-density rail on the hold
window. Cells 1,389–1,392 are consumed; programme count 1,392.
