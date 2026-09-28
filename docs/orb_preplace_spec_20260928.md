# ORB preplace-at-close — spec (2026-09-28)

Owner ask: "1 sec latency is what we need." Today's measured 09:35 path
(`docs/orb_latency_day1_20260928.md`, journal 2026-09-28 13:35 UTC, after the
day's two prewarm fixes): `post_open_range_sweep` 6.4s (vendor bar lag on
~33 candidates), `bar_arrival` 1.5s, `rank_and_submit` 0.7s,
`blocked_outside_orb` 3.9s, plus ~0.66s serial per order. Preplace-at-close
removes all of that from the critical path by ranking and building order
plans *before* the range closes, then firing the submit at 09:35:00.0 from a
dedicated scheduled thread instead of the scanner tick.

Flag: `entry.preplace_at_close` (orb.yaml, default `false`) +
`entry.preplace_rank_lead_s` (default `3.0`). Both read in `ORBEngine.__init__`
(`trading/orb_engine.py`). Flag off is byte-identical to today's path —
`_maybe_arm_preplace_scheduler` and `_reconcile_preplaced` are internally
gated and never construct a Timer / never touch order state when off
(`tests/test_orb_preplace_at_close.py::TestFlagOffByteIdentical`).

## Design

**(1) Provisional ranking, ~09:34:57 ET** — `_preplace_provisional_rank`.
Armed by `_maybe_arm_preplace_scheduler` (called every tick from
`_check_entries_locked`, idempotent via `_preplace_armed_today`) via a
`threading.Timer`, not the scanner tick. Builds the provisional range per
production-pool candidate from `_provisional_range_for`: the Alpaca
snapshot's `dailyBar` high/low (at 09:34:57 only ~5 minutes of the session
have printed, so the daily bar *is* the opening range so far), widened by
any 1-min bars already ingested via the WS stream this morning
(`self._bar_windows`) — the "minuteBar for completed bars" role, served from
data already in memory rather than a second fetch. Same client as
`build_universe` (`self.alpaca.get_snapshots` / `self._get_snapshots_warm`
under `execution.prewarm_seed`). Scores + ranks with the exact production
functions (`composite_score`, `assign_quintile`, `skip_q1`, `ranking_order`,
`dedup_candidates`, the four post-ranking vetoes, `self.planner.build`) —
never a re-implementation. **Never mutates `CandidateState.range_data`**:
each candidate is scored on a `copy.copy` with `.range_data` swapped to the
provisional value, so the real 09:35 ingestion path and reconciliation both
still see a clean, untouched final range. Logs one INFO line:
`[ORB PREPLACE] provisional top-N @ HH:MM:SS.mmm ET: SYM(rh=$..,Qn), ...`.

**(2) Submission, exactly 09:35:00.0 ET** — `_preplace_submit_at_close`,
fired by its own `threading.Timer` (armed alongside the rank timer; delay
computed from `09:35:00.000 - now`, clamped `>= 0`, so a late-armed cycle
fires immediately — never *before* the target, since a stop placed before
the range closes could trigger inside the range). Submits every provisional
plan **concurrently** via the same bounded `ThreadPoolExecutor` (≤ 8
workers) pattern `execution.fast_submit` already uses, calling the
unchanged `_submit_entry` per plan — identical order params, chase guard
and sizing as the normal path. Per order: `[ORB] SUBMIT LATENCY sym=.. 
seconds=.. preplaced=1`, plus the existing `_check_first_submit_latency`
tripwire (unchanged function — now fires ~0s after 09:35:00 instead of
several seconds).

**(3) Reconciliation, normal 09:35 tick** — `_reconcile_preplaced`, called
once from `_check_entries_locked` right after the post-open sweep, gated on
`_preplace_submitted_today and not _preplace_reconciled_today` (so it can
never run before the submit timer fires, and never twice). Re-scores the
preplaced symbols on their now-final `cand.range_data`, re-ranks + dedups
the same way, then per symbol:
- **already filled**, same trigger (± 0.5¢) → kept.
- **already filled**, trigger differs → WARNING with both levels (parity
  deviation, counted); fill stands, no unwind.
- **still in the final top-K**, trigger differs (± 0.5¢) → cancel + rebuild
  the plan on the final range/composite/quintile + resubmit (reuses
  `self.planner.build` / `_submit_entry` exactly as the normal path).
- **dropped from the final top-K** (final range never completed, or scored
  below threshold, or edged out by dedup) → cancel, **no refill** — the
  symbol is added to `self._pdr_vetoed_today`, the *same* no-refill slot
  accounting a post-ranking veto uses, so the slot stays empty everywhere
  that set is consulted (`_check_entries_locked`, `_run_pool_selection`).
- **new entrants**: not handled by this method at all — preplaced symbols
  are excluded from the normal ranking pass below (via `plan_submitted` /
  `open_positions` / `_pdr_vetoed_today`), so whatever
  `_run_pool_selection('production', ...)` submits on this same tick
  already *is* exactly the added set (`n_added = len(submitted)`).

One line: `[ORB PREPLACE] PARITY n_preplaced=.. n_kept=.. n_replaced=.. 
n_cancelled=.. n_added=.. n_filled_before_reconcile=..`, logged right after
the production `_run_pool_selection` call so `n_added` is available.

**(4) Dry-run / paper.** `_preplace_submit_at_close` checks
`self.strategy_dry_run` (same flag the late path reads fresh every tick,
including a tripwire-forced flip) and logs
`[ORB DRY] WOULD PREPLACE SYM stop $.. limit $.. shares .. risk $.. at
HH:MM:SS.mmm ET` instead of submitting, writes the same
`_append_dry_ledger_row` CSV row with a new **`preplaced`** column
(0 = normal path, 1 = this path — added as a trailing column so existing
readers keyed by name are unaffected) and the same `dry_trades` DB row via
`_record_orb_dry_entry`. This lets the dry week measure this path's latency
with zero live orders.

## Known, deliberate limitations (documented per CLAUDE.md — no silent scope)

- Preplace covers the **production pool only**; add-on pools stay on the
  existing late (09:35+) path.
- The four post-ranking vetoes (PDR/G1/range-size/catalyst) are evaluated
  **once**, at provisional-rank time, and are not re-run at reconciliation.
  PDR/G1 are prior-day-only so this is exact; range-size/catalyst are
  evaluated on the provisional range/cohort and are not re-checked against
  the final range — a known conservative simplification (can occasionally
  veto a symbol that would have passed on final data). Worth revisiting
  only if the dry-week data shows it matters.
- The PARITY summary line is logged from inside the same
  `_check_entries_locked` tick as the production ranking pass; if a
  same-tick gate (daily loss limit / kill rails / PDT / time cutoff) trips
  *between* reconciliation and that point, the counters were still applied
  correctly (cancels/replaces went through) but the summary line itself is
  lost for that day since `_preplace_reconciled_today` latches. Accepted
  given how early in the session those gates would have to fire.
- Epsilon for "trigger differs": `>= 0.5¢` (`abs(final_rh - prov_rh) >=
  0.005`), to avoid float-noise replacements on an unchanged range.

## Rollback

Set `entry.preplace_at_close: false` (or leave unset — that's the shipped
default). No restart-order dependency: the flag is read once per engine
construction, and `_maybe_arm_preplace_scheduler` / `_reconcile_preplaced`
are no-ops with it off — nothing else in the entry path changes.

## Tests

`tests/test_orb_preplace_at_close.py` (22 tests, all passing standalone and
inside the full ORB suite run):
`TestProvisionalRange` (snapshot-only, widened-by-WS-bars, degenerate→None),
`TestSchedulerArming` (exact 09:34:57/09:35:00.0 targets, never-negative /
never-early when armed late, arms once/day, flag-off never arms),
`TestConcurrentSubmitAtClose` (per-order SUBMIT LATENCY with `preplaced=1`,
failed submit not refilled, empty state, flag-off no-op),
`TestDryModePreplace` (`[ORB DRY] WOULD PREPLACE` + ledger row
`preplaced=1` + `dry_trades` insert), `TestReconciliation` (same-trigger
kept, higher-final-high replaced, dropped-from-final-topk cancelled +
no-refill, filled-before-reconcile warns + counted, filled-same-trigger
kept-not-warned, empty state), `TestTripwireReadsPreplaceTimestamps`
(near-zero first-submit delay reported), `TestFlagOffByteIdentical`
(reconciliation never runs off, arming reached-but-inert off).

Full-suite result (2026-09-28, this change):
`python3 -m pytest tests -q --no-header -p no:randomly` —
ORB-scoped subset (`-k orb`): **1087 passed**, 0 failed
(includes all 22 new tests). Full-suite run: see session log — expected
0 failed (unchanged pre-existing tests untouched; only additive code paths,
all internally flag-gated).
