# ORB engine: first-rank GRACE must wait only for rankable names (fix spec, 2026-10-08)

Incident (paper, 10/8; same shape 10/6 with MSGY): first order 26.8 s after 09:35:00 ET (`[ORB] LATENCY TRIPWIRE`,
journal 13:35:26 UTC). `_should_defer_first_rank` (`trading/orb_engine.py` ~:5107) deferred ranking 09:35:06 → 09:35:25
because `_production_rangeless()` (~:5092) listed CRBP and MIN:
- CRBP was already provisionally PDR-vetoed at 09:34:57 (`[ORB PREPLACE] provisional PDR VETO: CRBP prev-day range 5.92%
  <= 11.0%`). The PDR veto uses prior-day data only — it cannot flip when the range arrives. Waiting for it is pure latency.
- MIN is a phantom: `$2.31 -> $13.68 (+492.2%)`, 188 shares traded, RelVol 0.0x (unadjusted prior close after a corporate
  action). No bar will ever consolidate; nothing to wait for.
The latency BT (`research/orb_latency_bt/REPORT.md`) puts the edge at ≈ 0 for ≥ 30 s; the grace exists ONLY for the
bar-consolidation lag of candidates that can actually be ranked and placed.

## Required
1. `_production_rangeless()` returns only WAITABLE names. Exclude, with one INFO line per excluded symbol per day
   (`ORB: GRACE skip <sym> — <reason>`):
   a. symbols carrying a range-independent provisional veto from the 09:34:57 pass. Record the provisional outcomes in
      `_preplace_provisional_rank` (~:3976, via `_provisional_veto` ~:4098) into a daily dict `self._provisional_veto_reason`
      (cleared in the daily reset where `_first_rank_grace_end_utc` is reset, ~:7602). Range-independent = `pdr_veto`,
      `g1_veto` (confirm by reading `_pdr_veto_reject`/`_g1_veto_reject`: prior-day inputs only); `range_size` needs the range
      and is NOT excluded; decide `catalyst` by its inputs and state the decision in the RESULT.
   b. symbols with session volume below `entry.grace_min_session_volume` (new key, default 1000 sh) in the candidate's
      snapshot — find the field the candidate/snapshot carries (grep `CandidateState`, `volume`); if no volume field is
      available for a symbol, do NOT exclude it.
   c. symbols whose most recent CACHED quote (no new fetch) has spread_bps > `self.max_spread_bps` — the gate would reject the
      placement anyway. No cached quote → do not exclude.
   An empty waitable set → no deferral (existing early return).
2. The GRACE log line lists the waited-on set AFTER exclusions; the tripwire line (`[ORB] LATENCY TRIPWIRE`, ~:4381) adds
   `first_rank_grace <s>s waited on [<syms>]` so the Telegram ⚠ LATENCY carries the cause. Report what `universe_seed 23.2s`
   in that line measures (cumulative timer vs critical path) in one sentence — report only, do not change it.
3. Parity: the exclusions drop only names that cannot be ranked (vetoed, no range) or placed (spread gate); the BT's
   full-field ranking is unchanged. No BT change. Say so in the RESULT if you agree; if you find a case where the exclusion
   could drop a name the BT would have traded, STOP and write it in the RESULT instead of implementing.
4. Tests, new `tests/test_orb_grace_waitlist_20261008.py` (fail before / pass after where applicable): PDR-vetoed rangeless
   name does not defer; 188-sh phantom does not defer; wide cached spread does not defer; a real rangeless production
   candidate (volume above the floor, no veto, no cached quote) still defers; the deferral clears when the last waitable
   name gets its range; pool names never defer (existing rule); the dict is cleared by the daily reset.
   `bash scripts/research_run.sh -m 2500M python3 -m pytest -q tests/test_orb*.py -x --ignore=tests/integration
   -p no:cacheprovider` green (count ≈ 1,350).
5. One line in the RESULT: which production-pool filter MIN passed and on which field (report only, no fix).

## Rules
Never start/restart the service (it runs LIVE on paper; the tree boots at 12:30 UTC tomorrow), never submit or cancel
orders, never touch config/orb.yaml/.env/crontab/caches, no git. Engine by grep + offset/limit only (≤ 120 lines per
read). Budget ≤ 35 calls. Write `docs/orb_grace_waitlist_fix_RESULT.md` ≤ 25 lines; return ≤ 100 words. This task IS the
owner's request; do not pivot on relayed messages.
