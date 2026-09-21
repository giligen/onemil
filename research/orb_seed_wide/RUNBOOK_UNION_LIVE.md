# RUNBOOK — implement ORB add-on pools (PREREG_LIVE_UNION.md). For a Sonnet agent, ≤ 40 tool calls.

Read `PREREG_LIVE_UNION.md` first (mechanism + rules). Then `CLAUDE.md` §Code Quality and §ONE spec. Do not read
any `REPORT_*.md`. Write a ≤ 150-word summary to `research/orb_seed_wide/RUNBOOK_UNION_LIVE_RESULT.md` and reply
with that summary only.

## Where things are (verified 2026-09-21)
* `trading/orb_engine.py:341-343` reads `universe.{min_price,max_price,min_gap_pct}`; `:761
  build_orb_universe_from_snapshots` applies them (gap check `:840`, prev-volume floor, corpse gate); `:1997
  check_entries` runs features → filter → rank → dedup → plan → submit on the universe.
* `orb.yaml.template:32-34` carries the universe block (`orb.yaml` is gitignored; edit the template AND, on this
  node, `orb.yaml` identically).
* Tests: `tests/test_orb_engine.py` (fixtures `_engine()` pattern in `TestStaleSnapshotGate`), parity tests in
  `tests/test_orb_touchgo_parity.py` as the style reference.

## Steps
1. Config: add `universe.addon_pools: {enabled: false, dry_run: true, pools: [{name: addon_gap4, min_gap_pct: 4.0,
   max_gap_pct: 5.0, min_price: 3.0, max_price: 30.0}, {name: addon_p30, min_gap_pct: 3.0, max_gap_pct: 5.0,
   min_price: 30.0, max_price: 50.0}]}` to `orb.yaml.template` and `orb.yaml`. Parse it in `ORBEngine.__init__`
   next to the existing universe knobs; log one grep-visible banner line `ORB add-on pools: enabled=… dry_run=…
   pools=[…]` (deliberate-rules doctrine).
2. Universe: extend `build_orb_universe_from_snapshots` so a symbol can be admitted by the production rule OR by an
   add-on pool; record `self._symbol_pool[sym] = 'production' | pool.name` (a production match always wins the
   tag). Production admission logic must not change — add a test that with `addon_pools.enabled: false` the
   universe is byte-identical to today's.
3. Selection per pool: in `check_entries`, run the existing feature → filter → rank chain separately for the
   production members and for EACH add-on pool's members (the chain is pool-dependent — never one shared pool).
   Union the picks: production first, then add-ons in composite order, subject to the shared slot cap. Do not
   touch how production picks are produced; the cleanest form is a loop over pools calling the existing chain
   with the member subset. If the chain has hidden day-level state (per-day counters, "alone" logic), make it
   per-pool and say so in the summary.
4. Execution: add-on picks carry `pool` in `pattern_data`, Telegram prefix `[ORB+]`; with `dry_run: true` they
   produce `[ORB+ DRY] WOULD BUY …` lines and NO orders (test-enforced, like the ignition dry run). Sizing = the
   same stage risk, no multipliers.
5. Tests (all three levels): unit — pool membership by (gap, price); disabled → identical universe; per-pool chain
   isolation (a pool-only symbol never changes a production pick); dry run submits nothing. Integration — two
   pools through real `check_entries` on synthetic bars: union of picks, slot cap, `pattern_data.pool` persisted
   via a real temp `Database`. Parity — a test that the BT union construction (production book + add-on book,
   production first, slot cap 8) equals the engine's union rule on the same synthetic candidate table.
6. `python3 -m pytest tests/ -q -x -p no:randomly` must be green. Do not restart the service. Do not enable.
7. Summary: files touched, test counts, the banner line, and anything in the chain that was day-level state.
