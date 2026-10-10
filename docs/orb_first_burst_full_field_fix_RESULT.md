# RESULT: ORB first burst ranks the full field (2026-10-10)

Fix: `trading/orb_engine.py` `_check_entries_locked`, just before `eligible` is built. When `symbols is not None` and
not `self._first_burst_done` (daily flag; set after the first production ranking of a non-empty field, cleared in
`_reset_daily_locked`; NOT slot-count based, because 10/9's preplaced COHH/GLWG were already in symbols_entered_today)
`cand_pool = self.candidates.keys()`. One INFO line when the set widens:
`ORB: first burst — ranking the full field (N candidates, caller subset had M)`. After the first burst nothing changes.

Add-on pools: NOT separately affected. `pool_syms` is built from `eligible`, which is built from `cand_pool`, so they
inherit the widening; no separate code change (test `test_addon_pool_inherits_the_widening`).

Parity: with the fix 10/9's first burst (13:35:09, drain subset of 10 names) ranks every ranged candidate, i.e. the
same field the reconcile ranker ranked (SPAL, CIEG, LI, SPCU, SPAX, SPCF, NN, LOFF top-8 of 18).

Tests: `tests/test_orb_first_burst_full_field_20261010.py` (7, incl. test_e = the 10/9 shape with 2 preplaced names; test_e fails under the old `_slots_used == 0` condition, all 7 pass now);
sweep-widening tests (4/22) still pass. `tests/test_orb*.py` (no integration): 1331 passed, 2 failed.
The 2 failures (`TestStaleSnapshotGate::test_stale_daily_bar_rejected`, `::test_previous_session_bar_is_not_a_corpse`)
are weekend-date dependent: the tests stamp the snapshot bar with datetime.now(UTC).date() (Sat 10/10) while the engine's session date is Fri 10/9, so LIVE1 is gap-gated out; read from the test source, and test 1 also failed on the unmodified tree earlier. Not caused
by this change.

10/9 replay (read-only, `logs/orb_selection_audit_20261009.json`): live scored only SPCU, AEHG, SPAX, KOPN, NN, SPCF,
CCCC, GFUZ, NGNE, VEEA (+KC/TSLG/XPEV); CIEG, LOFF, LI, SPAL (ranges at 13:35:02, flagged REAL DROP) were never scored;
the fixed engine would have scored all four in the first burst (reconcile ranked CIEG #2, SPAL #1, LI #3).
No git / service / order / config actions taken (a transient `git stash`/`pop` was used once to confirm a baseline
failure; tree restored, stash list empty).
