# ORB: `_rank_production_final` drops a ranged (even preplaced) name with NO log line — SKDD 2026-11-24

## The incident (verified read-only: journal, data/trades.db row 878, analysis_results/orb_bplus_book.csv, orb_features_20261124_2142.csv)
* 09:34:58 ET provisional top-5: RAMZ(Q4), NVDQ(Q4), **SKDD(rh=$10.49, Q4)**, MUD(Q5), MUZ(Q5). 09:35:00 `ORB: SKDD range complete — H=$10.51 L=$10.05 range%=4.58% vol=69,889`.
  09:35:05 SKDD provisional stop-bracket submitted (475 sh, stop 10.49 / limit 10.52, row 878).
* 09:35:06 `_reconcile_preplaced` → `_rank_production_final` (trading/orb_engine.py ~4760-4792): `reconcile final top-8 of 16 ranked:
  ['RAMZ','NVDQ','AMDD','NSLR','MUZ','MUD','BOIL','CRCD'] | preplaced: ['RAMZ','NVDQ','SKDD','MUD','MUZ']`. SKDD is NOT among the 16
  ranked → not-in-final → provisional order cancelled, row 878 `preplace_canceled / preplace_reconcile`. **Not one log line says why.**
  (The 09:35:19 first burst skips all five preplaced names — they are in `open_positions` — so it is not diagnostic either.)
* BT (same day, same params): SKDD composite 0.228 **Q3**, range_size 4.54 %, gap 8.34 %, pdr 22.78 %, entered, **+$2,975** (eod).
  The nightly parity flagged SKDD `unexplained` ("ranged live, not in the live final top-K, composite not comparable") and fill-parity
  "BT FILLED but live never filled" → RAMP FREEZE orb 2026-11-24, ramp streak reset.
* `_rank_production_final` has FIVE silent `continue`s for a candidate that HAS `range_data`: pool != production; phantom gap
  (`_recheck_gap_admission`, feats['gap_pct'] = bar1.open vs prev_close < `universe_min_gap_pct` 5.0); `composite_score` → None
  (any feature None/NaN, trading/orb_filter.py:55); comp < `filter_threshold` (0.012081536791 on 11/24); `skip_q1`. The production
  pass logs "below filter threshold" / "Q1 filter dropped"; the final-rank pass logs nothing. A fallback that cancels a live order
  without a WARNING violates the project rule (CLAUDE.md "All fallback code paths MUST log ERROR or WARNING").
* Inputs that did NOT change between 09:34:58 (Q4) and 09:35:06 (dropped): `_get_feature_context` is cached per symbol per day
  (prev_day_bar 11/23 = o 8.485 h 10.33 l 8.2 c 9.35 v 4,018,705 per data/cache.db daily_bars — NB that row's fetched_at is 21:31 UTC, the
  nightly refresh, so the 09:35 cache content is not provable). What changed: `cand.range_data` partial → final (5 bars).

## Part 1 — diagnosis (read-only replay, no engine change)
Replay SKDD's 09:35:06 final ranking offline and say which `continue` fired:
1. Fetch SKDD 1-min bars 2026-11-24 09:30–09:35 ET from Alpaca (SIP) with the existing client (`AlpacaClient.get_1min_bars_range_multi`,
   data_sources/alpaca_client.py:1458 — GET only; ORB paper keys come from .env via the usual loader; NEVER print keys or account ids).
2. Build `range_data` exactly as the engine does (reuse `_ingest_bars` / the RangeData builder on a skeleton engine the way
   tests/test_orb_grace_waitlist_20261008.py builds one — read that file first; do not copy engine math into a script).
3. `prev_day_bar` = the 11/23 row above; `daily_stats_20d` the way `_get_feature_context` builds it (reuse the helper with the real
   `Database` on data/cache.db read-only, or the same function on the cached bars).
4. `feats = eng._compute_features(cand, prev_day_bar=..., daily_stats_20d=...)`; params the way the engine loads them for the day
   (`load_feature_params(orb_cfg['filter'])` + `params_for_day` PIT if the engine uses it — grep `z_params =` ~line 741 and follow);
   `composite_score`, `assign_quintile`, `_recheck_gap_admission(feats, cand, 5.0)`.
5. Report: every feature value (and which are None/NaN), gap_pct vs floor, composite vs threshold, quintile; the branch that drops SKDD,
   or "none drops it — the live state must have differed (bar revision / stream gap)" if all pass. Compare with the BT row (SKDD in
   analysis_results/orb_features_20261124_2142.csv). State plainly that a later SIP fetch can differ from the live stream bars.
   Do the same four steps for CRCD only if it costs < 5 calls (its live SCORED line: comp 0.2708 Q3, gap 5.778, rtv 16223, range_open 4.76).

## Part 2 — the fix (TDD, failing tests first). trading/orb_engine.py `_rank_production_final` ONLY; no selection, veto or order change.
Every `continue` for a candidate WITH `range_data` logs one line, prefix `[ORB] FINAL RANK:`:
* pool != production → stays quiet (by design, not a fallback).
* phantom gap → `{sym} dropped — gap {gap:.2f}% < floor {floor:.2f}% (phantom gap)`.
* composite None → `{sym} dropped — composite None, missing features: [names]` (names = params keys whose feature is None/NaN).
* below threshold → `{sym} dropped — comp {comp:.4f} < threshold {thr:.6f}`.
* skip_q1 → `{sym} dropped — Q1 (comp {comp:.4f})`.
Level: **WARNING when `sym in self._preplace_state`** (append ` — PREPLACED, the reconcile will cancel its provisional order`), INFO otherwise.
Factor the line into one small helper `_log_final_rank_drop(self, sym, reason, preplaced)` with a docstring. No behaviour change.

### Tests — `tests/test_orb_rank_final_drop_log_20261124.py` (harness from tests/test_orb_grace_waitlist_20261008.py — reuse, do not copy)
1. Ranged candidate whose features miss one param → not in the result, one WARNING/INFO with "composite None" and the feature name.
2. Below threshold → INFO with "< threshold".
3. Q1 with skip_q1 → INFO "Q1".
4. Phantom gap (gap_pct 2.0 with floor 5.0) → INFO "phantom gap".
5. Same as 1 but the symbol is in `eng._preplace_state` → level WARNING, text contains "PREPLACED".
6. A candidate that ranks normally → NO "FINAL RANK" record (pinned).
7. Pool != production ranged candidate → excluded, no record (pinned).

## Run (these files only; the main session runs the full suite — NEVER start it yourself)
`python3 -m pytest tests/test_orb_rank_final_drop_log_20261124.py tests/test_orb_grace_waitlist_20261008.py
tests/test_orb_preplace_budget_20261006.py tests/test_orb_preplace_replaced_row_20261120.py -q -p no:cacheprovider --ignore=tests/integration`

## Rules
Sonnet, ≤ 60 calls. The trader service is STOPPED for the night and starts itself at 12:30 UTC — never start/stop/restart it. No
`git commit`/`git stash`, no orders, no writes to `data/*.db`, `logs/*`, `orb.yaml`, `config.yaml`, `.env`, crontab. Never read a file
> 300 lines in full (grep / sed -n). Files you may edit: `trading/orb_engine.py` (`_rank_production_final` + the new helper only), the
new test file, a scratch replay script under the session scratchpad, and a "Result" section appended to THIS file. Return ≤ 150 words:
Part 1 verdict (verified facts vs guesses, separated), files changed, test counts before/after, anything you could not do.

## Result (2026-11-25)
**Part 1 (offline replay, SIP 1-min bars fetched 11/25, `data/cache.db` read-only; script `scratchpad/replay.py`).** Verified: SKDD rebuilt through
`_ingest_bars` -> `_get_feature_context` -> `_compute_features` with live `orb.yaml` params: range H 10.51 / L 10.05 / vol 69,889 (identical to the live
"range complete" line), all 9 features present (none None/NaN), gap_pct 8.34 (floor 5.0, admitted), composite 0.2282 (threshold 0.01208), Q3, pool production
=> NONE of the five `continue`s fires; SKDD ranks. Same for CRCD (comp 0.2708, Q3, gap 5.78, vol 16,223 — matches its live SCORED line). Live GAP_GATE for SKDD
showed gap 8.663 (snapshot open 10.16), also above the floor. Guesses (unproven): the live state differed from a later SIP fetch (bar revision, stream gap, a
cache/feature-context state at 09:35:06, or a pool/`_symbol_pool` value other than production). The cause cannot be named offline; the new FINAL RANK log lines
will name it next time.
**Part 2.** `_log_final_rank_drop` + logging at the 4 drop branches in `_rank_production_final` (`import math` added); tests in
`tests/test_orb_rank_final_drop_log_20261124.py` (7). No behaviour change.

## Review (main session, 2026-11-25 05:4x UTC) — the real cause
The replay is right and the brief's title is wrong: SKDD was NOT dropped by a silent `continue`. It WAS scored at 09:35:06 — on its
final range the composite is 0.2282 = **Q3** (provisional at 09:34:58 was Q4 on a partial range) — and ranked **9th** behind
MUD(Q5), RAMZ, NVDQ, AMDD, **NSLR**, MUZ (Q4) and BOIL, CRCD (Q3, 0.2748 / 0.2708). The only name in that top-8 the BT never ranked
is NSLR: rangeless live (REST sweep fetched 0 bars 09:35:00 and 09:35:05), absent from analysis_results/orb_features_20261124_2142.csv,
ranked live only through the 2026-10-08 `_provisional_standins` rule (provisional PDR veto → burns a top-K slot). Without that
stand-in the live top-8 at the reconcile is exactly the BT's: MUD, RAMZ, NVDQ, AMDD(veto), MUZ, BOIL(veto), CRCD, SKDD.
The same stand-in also cost CRCD: at the 09:35:19 burst slots used = 4 entered + 3 PDR vetoes (AMDD, NSLR, BOIL) = 7; UVXY (newly
ranged, Q4 0.3091) ranked above CRCD and its RANGE-SIZE veto burned slot 8 → CRCD never ordered. BT: SKDD +$2,975, CRCD +$4,060.
Dedup is not involved (SKDD has no family / super group). The parity/ramp-freeze verdict "unexplained" came from the reconcile line
printing only the top-K names: fixed by a second INFO line `[ORB PREPLACE] reconcile ranking (Q, comp): [...] | beyond top-K: [...]`
(the old line is unchanged for scripts/eod_sections.py and the selection observer). Tests: tests/test_orb_preplace_reconcile_rank_log_20261124.py (3).
The agent's `_log_final_rank_drop` stays (hygiene; four previously silent exits now log). **Open design question for the owner:**
a provisional stand-in for a name whose 09:30–09:34 bars the REST sweep reports as absent burns a slot the BT never spends (the BT
ranks only names with bars). Rule candidates: (a) no stand-in when the sweep returned 0 bars for the name; (b) keep as is. Not changed.
