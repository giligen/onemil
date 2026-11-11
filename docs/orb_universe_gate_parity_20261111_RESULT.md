# ORB universe gate parity — RESULT 2026-11-11

| gate | live 09:30 | live re-check | BT SQL universe | BT builder features |
|---|---|---|---|---|
| min gap 5 % | resolve_gap_input open (bar, else snapshot) | bar1.open (`range_open`) | daily_bars.open | bar1.open in feats, NEVER gated (fixed: `builder_admission` re-check) |
| prev vol >= 500K | yes | no | yes | no |
| price $3-$30 | on that open | no | on daily open | no |
| test ticker | seed only | no | none | none (admit has it) |
| 9:35 RTH vol >= 15K | none | none | BT-ONLY (`MIN_935_VOL_FLOOR`) | - |

Changes: new `trading/orb_universe_gate.py` (`admit`, `Admission`); engine 09:30 filter `orb_engine.py` `is_production` and both re-checks call it (helpers `_production_universe_cfg_admit`, `_recheck_gap_admission`); re-check with no `gap_pct` now WARNs (was a silent fail-open); `_log_final_ranges` logs `[ORB] final ranges:` once/day; builder `study_orb_features.py` (`builder_admission`, `diagnose_extract_failure`, `--trace-drops`, `--end-date`, per-reason COUNT line); `study_orb_broad.load_broad_universe(drop_sink=)`. No threshold changed.

ASTY 10/12 and SK 10/23: dropped by the BT-only 9:35 volume floor (RTH volume to 9:35 = 12,866 and 7,645 vs 15,000). Not a universe gate in live. The rulebook does not name it, so reported, not changed.

ASTX 9/30 (hypothesis): no `ORB SCORED` line, entry submitted 13:35:07 with preplaced=1, so it likely never hit the scoring re-check. Journal rotated, so this is not proven.

Range (V): the inclusion rule is identical (>= open, < open+5 min, 5 bars); only the bar feed differs (live stream/REST vs cache). No shared function needed.

Tests: 19 new, 146 pass in the listed set, 0 failures.

Not done: the builder run for 10/12 and 10/23 did not finish (universe load > 15 min on this node); the reasons come from direct cache queries. FEATURES_CODE_VERSION not bumped (would break the nightly incremental run). The nightly rows will now differ from older CSV rows.
