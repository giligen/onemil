# Bull Flag Discipline Program — owner order 2026-08-22 ("make BF disciplined as the others are")
**Goal: BF gets everything ORB/ignition have — bounded downside, live parity
measurement, a validated expectation band, and a dated verdict — while it KEEPS
TRADING. Pause only ever happens the disciplined way: a gate fires.**

## The gap inventory (why BF bleeds opinion instead of producing evidence)
1. No per-book kill rails (only the account-level $5K daily limit).
2. Parity harness exists (scripts/bf_decision_parity.py, nightly since 8/15)
   but is STARVED — reports BT_STALE because the Stage-1 cache ends 8/07;
   nobody refreshes it nightly.
3. Six documented BT/LIVE structural drift sources (threshold mismatch, scan
   timing, entry latency, exit divergence, ...) — so Stage-2 is a relative
   tool, not a band.
4. No pre-declared expectation band or verdict date; live record (Mar +$5,684
   then 5 months −$3,885) has no contract to be judged against.

## Phase 1 — RAILS + CONTRACT (build now, live at Monday boot)
- **BF kill rails**, mirroring trading/orb_engine's DB-derived fail-closed
  pattern (tests/test_orb_kill_rails.py is the template):
  - daily −$800: no NEW entries rest of day (entry gate)
  - weekly −$1,200: flatten BF + no entries rest of ISO week
  - monthly −$2,500 (≈1.4× worst live month): BF AUTO-PAUSES (engine enabled
    latch off) + [BF] ABANDON-GATE telegram → owner keep/kill decision WITH
    evidence attached. This is the disciplined form of "pause".
  - Config: trading.bull_flag.kill_rails.{enabled,daily_usd,weekly_usd,
    month_pause_usd}; env kill BF_KILL_RAILS=0. Same notify-once latches,
    ET-dated, fail-closed (-1e9 on DB error) as ORB.
- **Pre-declared verdict contract** (this section IS the contract):
  - Window: 2026-08-24 → 2026-10-15 (retirement-plan Month-2 boundary).
  - PASS = (a) monthly pause-gate never fires; (b) parity harness live with
    real (non-STALE) comparisons ≥ 80% of trading days from 9/1; (c) live
    cumulative over the window ≥ −$1,500 OR ≥1 live monster (≥ +2R) captured;
    (d) drift sources quantified with the band published (Phase 3).
  - FAIL of (a) → auto-pause happened, owner decides with data.
    FAIL of (b/c/d) at 10/15 → BF pauses pending the missing evidence.
  - No sizing change: current size stands; the rails are the ceiling.

## Phase 2 — FEED THE PARITY HARNESS (next week)
- **Nightly Stage-1 cache increment**: append TODAY's movers to
  data/bull_flag_cache_e50_x30.csv via the existing --build-cache path scoped
  to the single day (APPEND-ONLY + dated backup first — the never-overwrite
  rule; verify by row-count delta + md5 of the pre-section). OS-cron after
  close. bf_decision_parity then produces real BOTH/BT_ONLY/LIVE_ONLY daily.
- Green check gains a BF parity section (hard-flag on BT_STALE > 3 days,
  divergence counts trending).
- EOD dive + weekly report consume it (weekly already has the BF line).

## Phase 3 — THE BAND (weeks 2-3)
- Quantify the six drift sources on the repaired cache (each a bounded study;
  entry-latency + exit-divergence first — they dominate).
- Re-run the deferred cache-level A/Bs on the CLEAN cache (V-rev bonus,
  vol-trail, two-tier — the 8/15 repair re-validated regime sizing already).
- Publish analysis_results/bf_expected_book.csv (monthly distribution at live
  sizing, drift-haircut documented) → weekly report tracks live-vs-band
  exactly like ORB. From then on BF is judged, not felt.

## Rollout
- Phase 1 tonight: shared-style rails module + tests (~90% cov), full suite,
  Sunday rehearsal adds a BF-rails drill (mock breach → entry gate → flatten
  → pause latch + telegram), Monday boot ships rails enabled.
- Rollback: BF_KILL_RAILS=0 or config flip (rails only gate/flatten — they
  never place entries — so rollback risk is nil).
- Monitor: journalctl | grep "BF RAIL"; EoD dive line; weekly contract table.

## Appendix — BF-rails Sunday drill (added with the Phase-1 build, 2026-08-22)
Mock breach → entry gate → flatten → pause latch + telegram, on the rehearsal
box against a THROWAWAY copy of trades.db — never the live DB.
1. Redirect the pause flag: `export BF_MONTH_PAUSE_FLAG=/tmp/bf_pause_drill.flag`
   (and point the engine at the throwaway DB).
2. Insert mock closed bull_flag losses (row shape:
   tests/test_bf_kill_rails.py::_insert) and watch the rails fire in order:
   - −$900 today → log `BF RAIL: DAILY KILL` + `[BF] DAILY KILL` telegram;
     entry gate: pattern checks skipped rest of day.
   - two −$700 this ISO week → `BF RAIL: WEEKLY KILL` + flatten via the real
     `_force_close_all` path + `[BF] WEEKLY KILL` telegram.
   - −$2,600 this month → `BF RAIL: MONTH PAUSE` + `[BF] ABANDON-GATE`
     telegram with the month's trade list; flag file written. Restart the
     service → boot log shows `MONTH-PAUSE flag present ... at boot`.
3. Clear: owner deletes the flag file → next boot/reset_daily logs
   `owner cleared the pause`.
4. EoD visibility: `python3 -c "import sys; sys.path.insert(0,'.');
   import scripts.report_common as rc;
   print(rc.bf_rails_line(rc.bf_rails_status('YYYY-MM-DD')))"`.
Grep contract: every rail log line contains `BF RAIL`
(`journalctl -u onemil-trader | grep "BF RAIL"`), the BF analogue of ORB's
`[ORB] ... KILL` lines. Automated equivalent of this drill:
`pytest tests/test_bf_kill_rails.py` (includes the mock-month integration
sequence daily gate → weekly flatten → monthly pause).
