# RESULT — HOD arm-gate parity (spec `hod_arm_gate_parity_spec_20261030.md`, 2026-10-30)

## Diff summary
* `trading/hod_break.py`: new `arm_gate_reject(trigger, stop, min_price, min_r_pct) -> None | 'price' | 'r_pct'` —
  the ONE predicate (`trigger < min_price` or `(trigger - stop) / trigger * 100 < min_r_pct`, both strict `<`).
* `research/hod_entry/causal_arming.py`: `arm_state` now calls `arm_gate_reject` (was an inline copy; behaviour identical).
* `trading/hod_break_engine.py`: `HodBreakEngine._apply_arm_gate` (called in `_evaluate_resting` right after
  `arm_state`, before the dry ledger / `_arm_live_order` / cancel-on-`arm_lost` logic, so dry and live are gated alike).
  Gated arm -> `new_arm = None`; one INFO per distinct (sym, trigger, stop): `[HOD] ARM SKIP {sym}: r_pct=1.20% < 1.5%`
  or `trigger 19.50 < min_price 20.00`. Counter `engine.arm_gate_skips = {'price', 'r_pct'}` (+ `_arm_skip_logged`
  dedupe set), both reset in `_roll_session`.
* Pure `trading.hod_break.arm_state` is unchanged (no gate inside it; the existing arm_state-vs-twin parity test still holds).

## Config keys and values (engine == twin, no adoption needed)
* `config.yaml hod_break.min_price` = 20.0 (engine `self.min_price`; twin `live_params()` floor)
* `config.yaml hod_break.params.min_r_pct` = 1.5 (engine `self.params.min_r_pct`; twin `p.min_r_pct`)
* Test `test_live_config_keys_equal_the_twins` pins both against the real config.

## Expected effect
82 of 231 live paper fills since 9/29 had R < 1.5 % of price, i.e. about 35 % of fills / arms with a cross are removed
(the whole cohort is -0.088 / -0.097 R net, cell 1,729). The dry ledger also loses these rows from Monday on.
Note: the existing entry-time r_min check (`_try_enter`, next_open mode) is unchanged.

## Tests (`tests/test_hod_arm_gate_parity_20261030.py`, 15)
Unit 5 predicate (1.2 % skip, 1.6 % arm, price skip, price boundary, exact 1.5 % arms) + 5 engine (log line/counter/dedupe,
wide passes, price log, boundary, session reset); parity 2 (200 synthetic (trigger, stop) pairs through
`causal_arming.arm_state` vs `arm_state + _apply_arm_gate`, identical; live config keys); integration 3 (tight base through
`_ingest_bars`: no ARMED, no ledger row, no live order; control with gate relaxed writes the row; wide base still arms).
Suite: see the line appended below.

## Deviations
None from the spec. The gate lives in the engine (not inside `arm_state`) so the skip can be logged with its reason;
parity is by the shared predicate plus the 200-pair test. No rehearsal (Fable, Sunday).

**Suite result:** `pytest tests --ignore=tests/integration`: 5408 passed, 6 skipped, 0 failed (744 s; takes > 900 s limit under load, run with a 2700 s timeout).
