# PARITY_1691 — live `addon_gap35_range5` admission vs the idea1/P1 backtest

Script: `research/orb_freq/1691_parity_check.py` (read-only; imports the real
`trading/orb_addon_gates.{PoolGateInputs,evaluate_pool_gates}`; never touches
orb.yaml/config/live service). Rows: `research/orb_freq/1691_parity_rows.csv`.

## Part 2 — fail-closed (synthetic input): PASS
`evaluate_pool_gates` on an all-`None` `PoolGateInputs()` returns
`admitted=False` both for a synthetic cfg activating all 7 `GATE_KEYS` and
for P1's real cfg (`min_move_to_range_high_pct=5.0` only); `gate_values`
carries `None` for every active key, so the input was checked, not silently
skipped. No gap.

## Part 1 — admission parity, 2026-06-01..2026-09-26
Harness ground truth = `fastpath/idea1_features.csv` (the pre-ranking
admitted list `1684_fastpath.py` writes — "before selection"), which only
covers **2025-01-02..2026-09-18**; 2026-09-19..09-26 (186 re-derived
candidates, 0 engine-admits) has **no harness artifact at all**, reported
separately.
Coverage window 2026-06-01..2026-09-18: 3,446 daily-bar `idea1_pre`
candidates, re-derived independently from `data/cache.db::daily_bars` (not
imported from the harness). Raw agreement 68.8% is **not the real number** —
mostly a data-coverage artifact: `bars_sip.db` has zero minute bars for
2,617/3,446 (76%) candidates, and 83 more have only a partial (1-4 of 5)
range window. Only 742 rows resolve a value at all.

Of those 742: agree 441 (79 both-admit, 362 both-reject); disagree 301 (298
engine-only, 3 harness-only). Spot-checked a 5-row sample of the 298
engine-only rows directly against the wide-seed CSV
(`orb_seed_wide/out/orb_features_20260921_1842.csv`): **all 5 are completely
absent from it** — not a computed "move<5%", just never in the harness's
population (same failure mode as `bars_sip.db`, one layer upstream).
Excluding those 298 population-absent rows: **441/444 = 99.3% agreement, 3
mismatches** (not individually traced; low value at n=3). The
move-to-range-high mechanism itself is parity-by-construction: the harness's
`entry_price` *is* `trade.range_high` (`study_orb_features.py:435`), and both
sides use the bar whose ET wall clock is exactly 09:30 plus the next 4
one-minute bars (`study_orb.py:141`; `orb_engine.py`'s `range_open` comment
says "9:30 bar open — BT-parity"). Caveat: the absence check was a 5-row
spot sample, not exhaustive over all 298 engine-only rows.

**Decision instant**: gate values are bar-anchored, not anchored to when the
code runs, so engine processing latency (the entry-drain thread, elsewhere
documented at 20-45s+ under load) does not change `range_high` for this
pool. Not verified this session: whether `preplace_at_close`'s pre-close
provisional rank touches add-on-pool gating or only production's submit.

## Part 3 — engine paths the harness never modeled
1. **Pool-order precedence (fix/confirm first).** `build_orb_universe_from_
   snapshots` assigns each symbol to the *first* matching pool in
   `addon_pools.pools[]`, then `break`s. In `orb.yaml.template`, `addon_gap4`
   (gap 4-5%, $3-30) is listed *before* `addon_gap35_range5` (gap 3-5%,
   $3-30) — so live, `addon_gap35_range5` can only ever receive gap-in-[3,4)
   names whenever both pools are enabled together; [4,5) always goes to
   `addon_gap4` first. The backtest used the full [3,5) band with no
   competing pool. **If `addon_gap4` is enabled concurrently, expect roughly
   half the backtested frequency.** This review used only
   `orb.yaml.template` (per the task) and never read live `orb.yaml` —
   confirm the live pool list/order before trusting the idea1 frequency.
2. **Gap ceiling not re-validated at 09:35.** `_run_pool_selection`'s
   phantom-gap guard only re-checks the pool's *min* gap against the settled
   bar-1 open, never the *max*. A candidate whose real gap drifts above 5.0%
   after the 09:30 snapshot check still reaches `evaluate_pool_gates` and can
   be admitted above the ceiling the harness's population strictly enforces.
3. **Gate pass ≠ order.** Passing `evaluate_pool_gates` still funnels through
   `composite_score`/`filter_threshold`/quintile/Q1-skip/ranking/dedup and the
   shared slot cap (production consumes slots first) — none of that is in
   `idea1_features.csv` by design; the harness's ranked/filled book is
   `1684_pool_books.csv`, not diffed here.
4. **Exit mechanics** (stops/trailing/time-stop) are out of scope here;
   add-on pools share production's `StopMonitor` per CLAUDE.md, not
   independently verified in this review.

## Bottom line
Parity holds **by construction** for P1's one active gate and for
fail-closed behavior. The large raw disagreement is a minute-bar/population
coverage artifact, not a code defect. What should change before the pool
runs today: confirm live `orb.yaml`'s `addon_pools.pools[]` order/membership
— if `addon_gap4` is enabled alongside `addon_gap35_range5`, the live pool
structurally sees about half the backtested idea1 gap band.
