# Spec — HOD arm-gate parity: the research twin's min_r_pct / min_price gate goes live (Fable, 2026-10-30 evening)

## Defect
`research/hod_entry/causal_arming.py` l.81 rejects an arm when `trigger < min_price` or
`(trigger − stop) / trigger × 100 < p.min_r_pct` (1.5 %). The live arm path in `trading/hod_break.py` /
`trading/hod_break_engine.py` applies neither (the engine only uses `params.min_r_pct` as a FALLBACK stop distance,
~l.1493/1566). Result: 82 of 231 live paper fills since 9/29 had R < 1.5 % of price (down to 0.24 %). Cell 1,729
(`research/hod_live_anatomy/RESULT_1729.md`) shows that cohort is −0.088 / −0.097 R net on 4,761 BT fills (t −3.3);
the live +0.06 R on 82 was noise (CI −0.33..+0.45). BT and live must run ONE rule: the twin's gate goes live.

## Change (Sonnet; one agent; ≤ 40 calls; no commit/stash, no restart, no orders, no writes to data/*.db, config.yaml,
## orb.yaml, .env, crontab, logs state files; one pytest process at a time; never read a file > 300 lines in full)
1. Find the live arm decision (grep `def .*arm` in `trading/hod_break.py` and `trading/hod_break_engine.py`; the point
   where level/trigger and consol_low/stop are known and the resting order is about to be placed). Add ONE gate there,
   evaluated exactly as the twin: `trigger < min_price` or `(trigger − stop) / trigger * 100 < min_r_pct` → do not arm,
   log INFO `[HOD] ARM SKIP {sym}: r_pct={…:.2f}% < {min_r_pct}%` (or `trigger {…} < min_price {…}`), count it in the
   engine's daily arm stats if such a counter exists. Read `min_r_pct` and `min_price` from the SAME config keys the
   twin reads (`research/hod_entry/causal_arming.py` l.~55–65 shows which keys; the engine already has
   `self.min_price` and `self.params.min_r_pct`). If the twin's config value differs from the engine's, the engine
   adopts the twin's and the RESULT says so. Put the gate predicate in a small shared helper used by BOTH the twin and
   the engine if the import graph allows (parity by construction); otherwise write a parity test that calls both.
2. Tests `tests/test_hod_arm_gate_parity_20261030.py` (fixtures from `tests/conftest.py`, `MagicMock(spec=…)`):
   unit — r_pct 1.2 % → skipped with the log line; 1.6 % → armed; trigger below min_price → skipped; boundary exactly
   1.5 % → armed (twin uses `<`). Parity — the same 200 synthetic (trigger, stop) pairs through the twin's predicate and
   the engine's give identical decisions. Integration — an engine built as the nearest existing HOD engine test does,
   fed a bar sequence that would arm a tight base, places no order and writes no ledger row.
3. Suite once: `timeout 900 python3 -m pytest tests -x -q --ignore=tests/integration -p no:randomly`. Zero failures.
4. `docs/hod_arm_gate_parity_spec_20261030_RESULT.md`: diff summary, the config keys and values used, expected drop in
   arms/fills (≈ 35 % of fills by the live ledger), test counts, deviations. No rehearsal (Fable runs it Sunday).

## Not in scope
Any other HOD rule; exits; the ORB engine; the dry ledger format.
