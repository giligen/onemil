# RUNBOOK_UNION_LIVE — result (2026-09-21)

Implemented, not enabled. `orb.yaml`/`.template::universe.addon_pools` (enabled=false, dry_run=true,
addon_gap4/addon_p30) parsed in `ORBEngine.__init__`, banner `ORB add-on pools: enabled=… dry_run=… pools=[…]`.
`build_orb_universe_from_snapshots` tags `self._symbol_pool[sym]` = 'production'|pool name (production always
wins); byte-identical when disabled (tested). `_check_entries_locked` now splits eligible candidates by pool and
calls a new `_run_pool_selection(pool_label, …)` once per pool (production first, then add-ons), each pool's own
feature→filter→rank→dedup→veto→plan→submit chain, unioned under the shared slot cap recomputed per call.

Day-level state made per-pool: the catalyst "alone" cohort (`_catalyst_veto_reject`) now takes `cohort_symbols`
so add-on symbols never borrow production's confirmation cohort or vice versa. Also fixed a latent bug found
mid-build: the phantom-gap re-validation was hardcoded to the production gap floor, which would have rejected
every legitimate add-on pick — now uses each pool's own `min_gap_pct`.

Add-on picks: `plan.pool` set post-hoc, persisted in `pattern_data.pool`; `[ORB+]` telegram prefix at
ENTRY SUBMITTED; `dry_run: true` → `[ORB+ DRY] WOULD BUY …` log/telegram, no DB row, no order.

Files: `trading/orb_engine.py`, `orb.yaml`, `orb.yaml.template`, new `tests/test_orb_addon_pools.py` (8 tests:
unit/integration/parity), plus 5 pre-existing source-inspection tests updated for the refactor (catalyst_veto,
day_sequence, g1_veto, pdr_veto, pm_mult, range_size_veto — same invariants, new location).

`pytest tests/ -q -p no:randomly`: 4163 passed, 10 skipped, 0 failed.
