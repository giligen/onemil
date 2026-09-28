# ORB dry-week day 1 latency — 2026-09-28

Measured (journalctl -u onemil-trader, 09:35 ET window, ET = UTC-4):

```
[ORB] LATENCY TRIPWIRE: first order submit at 09:35:26 ET = 26.2s after 09:35:00
  measured: universe_seed 13.7s, post_open_range_sweep 6.4s, bar_arrival 1.5s,
            rank_and_submit 0.7s, scoring 0.0s, ranking 0.0s, blocked_outside_orb 3.9s
```

`_latency_phases` (trading/orb_engine.py:741, `_record_latency_phase` at :1429) accumulates
from `ORBEngine.__init__` (08:30 ET boot) to the day's first submit — the breakdown is a
whole-morning cost ledger, not a slice of the final 26.2s window. `orb_engine.reset_daily()`
is never wired into the scanner's daily boundary, so this ledger only starts fresh on a
process restart (out of scope here — today's boot was at 12:30 UTC).

## Cause 1 (fixed): first-rank grace counted add-on-pool rangeless names

`_should_defer_first_rank` / `_first_rank_grace_elapsed` treated ANY rangeless candidate in
`self.candidates` as reason to defer the day's first placement burst, including add-on-pool
(dry-only) names. Evidence (09:35:00-26 ET):

```
09:35:00.397  post-open range sweep — backfilling 33 candidates ... : SRZN,ZSQR,WTTR,STAA,WHLR,CIFG,PLU,ANNX,CLRO,MSGY...
09:35:01.154  post-open sweep — 21 candidates still rangeless ...; retrying once in 4s
09:35:06.140  post-open range sweep — filled 25/33 ranges          <- PRODUCTION field complete here
09:35:06.144  first-rank GRACE — 8 pool candidate(s) still rangeless (SRZN,ZSQR,WTTR,STAA,WHLR,CIFG...);
              deferring ranking until field completes or 09:35:25 ET
09:35:17.419  first-rank GRACE — 8 pool candidate(s) still rangeless ...; deferring ... until 09:35:25 ET
09:35:25.177  post-open range sweep — backfilling 8 candidates ...; filled 0/8
09:35:25.548  ORB SCORED: ... (ranking finally proceeds)
09:35:26      LATENCY TRIPWIRE
```

The 8 names holding the gate (SRZN, ZSQR, WTTR, STAA, WHLR, CIFG, PLU, YDES) are all
`addon_gap4`/`addon_p30` pool tags, not production. By 09:35:06 the production field was
already complete (25/33 filled by the sweep retry); the gate should have cleared then. It
instead held ranking hostage for the full `entry.first_rank_grace_s` (25s) window, costing
~19s of the 26.2s delay — the single largest contributor.

**Fix**: `trading/orb_engine.py` — added `_production_rangeless()` (new helper, ~line 3411)
and scoped both `_should_defer_first_rank` (rangeless list, ~line 3435) and
`_first_rank_grace_elapsed` (~line 1421) to it. Add-on pool candidates (`_symbol_pool[sym] !=
'production'`) no longer count toward the rangeless field or extend the grace; an untagged
symbol still defaults to `'production'` (matches `_run_pool_selection`'s default), so
existing/unit-test candidates seeded outside snapshot admission are unaffected.

Tests added: `tests/test_orb_selection_race.py::TestGraceScopeProductionOnly`
- `test_pool_only_rangeless_does_not_defer` — 8 rangeless pool names + complete production
  field → no deferral.
- `test_production_rangeless_still_defers` — 1 rangeless production name → grace fires, same
  as pre-fix behaviour.
- `test_elapsed_ignores_pool_rangeless` — `_first_rank_grace_elapsed` clears once the
  production field completes even while pool names stay rangeless.

## Cause 2 (diagnosed, NOT fixed — needs a design decision): universe_seed 13.7s

`universe_seed` wraps `source_loader()` in `ORBEngine.build_universe` (orb_engine.py:815-822),
i.e. the whole `_orb_universe_source()` call including the sqlite prefilter / WARM lookup and
the snapshot fetch. Six ORB ticks ran between boot and the first submit; only two had an
individually-logged sub-timer, and they don't add up to 13.7s:

| tick (ET)  | logged sub-timer                                   | elapsed |
|------------|-----------------------------------------------------|---------|
| 09:30:22.3 → 09:30:30.7 | (none — SQL query itself) | **~8.4s unaccounted** |
| 09:30:30.7 → 09:30:32.7 | `prewarm cache MISS: 2555/2555 ... fetched fresh` | 1.99s |
| 09:31:23 → 09:31:25     | `prewarm cache MISS: 2555/2555 ... fetched fresh` (ALL flip stale) | 1.84s |
| 09:32:16, 09:33:14, 09:34:14 | `prewarm cache STALE: ~250 ... fetched fresh` | 0.34s each |
| 09:35:16 → 09:35:17     | `prewarm cache STALE: 243 ... fetched fresh` | 0.41s |

Sum of logged sub-timers ≈ 5.3s; the remaining ≈8.4s falls entirely in the gap between the
prior main-loop cycle's `CYCLE TIMING` log (09:30:22.318) and the first `ORB universe seed`
DEBUG log (09:30:30.694, `scanner/realtime_scanner.py:853`) — before any sub-timer starts.
That gap is the `daily_bars` window-function query in `_orb_universe_source`
(`scanner/realtime_scanner.py:838-849`, `SELECT symbol FROM (... ROW_NUMBER() OVER (PARTITION
BY symbol ORDER BY bar_date DESC) ...) WHERE rn=1 AND close BETWEEN 1.0 AND 50.0 AND
volume>=500000`), which only runs once per day (result cached in `_orb_sqlite_seed_cache` for
the rest of the day, logged as `ORB prewarm cache MISS (sqlite_prefilter): recomputed 2270
symbols`). Matches the build_universe comment's own prior measurement ("15.3s daily_bars
window-function scan", "18.7s-measured full-table scan").

A second, smaller contributor: at 09:31:23 ALL 2555 cached snapshots flip STALE at once
(`_is_complete` requires `daily_bar_date == today`; Alpaca's snapshot feed evidently still
tags the daily bar with yesterday's date for a beat right after the open), forcing a second
full re-fetch (1.84s) one tick after the first. By 09:32 only ~250 remain incorrect.

**Verdict on today's incident**: the sqlite prefilter recompute happened at 09:30:22-30,
comfortably inside the 5-minute premarket-to-range-close window, and did NOT itself block the
09:35:00-09:35:26 critical path — by 09:35 every tick's sub-timer was back to ≤0.41s. It is
real cost (13.7s charged to `universe_seed` in the whole-morning ledger) but not today's
proximate delay; Cause 1 (first-rank grace) was. It IS a latent risk: on a day where the
first RTH tick lands later (slow main-loop cycle, restart near the open, etc.) this ~8-17s
one-time cost would land inside the 09:35 window directly.

**Proposed fix (not implemented — design decision)**: warm `_orb_sqlite_seed_cache` /
`_orb_sqlite_seed_cache_date` during the scanner's pre-market startup path
(`scanner/realtime_scanner.py` `run()`, ~line 205-220, where `orb_engine.build_universe` is
already called once at startup) instead of lazily on the first RTH tick — `daily_bars`
prev-day data doesn't change intraday, so there's no correctness reason to wait for market
open. Open questions before implementing: (a) should this block scanner startup (~8-17s,
harmless pre-market) or run async; (b) does it need its own `prewarm_seed_enabled` gate or
piggyback the existing one; (c) how to test it without a real DB fixture reproducing the
2270-row window-function query cost. Left as a diagnosis per the task's "needs a design
decision" branch — no behaviour changed for Cause 2.

## Expected day-2 latency

With Cause 1 fixed: sweep (6.4s, structural — initial pass + `sweep_retry_delay_s=4.0`) +
ranking/scoring (~0.1s) + whatever main-loop-cycle overhead lands in `blocked_outside_orb` —
expect first submit around 09:35:07-10 ET (≈7-10s after range close), under the
`latency_warn_secs` (10s) tripwire threshold most days, vs. today's 26.2s. Cause 2's ~13.7s
whole-morning cost is unchanged and will still show in the ledger, but as diagnosed above it
should keep landing before 09:35 rather than inside it — worth re-checking on day 2's tripwire
log (if it fires) to confirm the sqlite prefilter's first-tick timing again.
