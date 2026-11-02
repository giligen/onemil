# HOD config whitelist silently dropped the squeeze keys (Fable, 2026-11-02) — THIRD occurrence of the 9/25 defect

## What happened
`config.py::Config.hod_break_cfg` (l. 842–911) returns an explicit dict of keys; a `hod_break` YAML key absent from it
is silently dropped (the 9/25 lesson is written in the docstring; it bit again on 10/1 `telegram_per_trade`). On 10/31 the
passive squeeze shipped with `squeeze_enabled / squeeze_minute / squeeze_max_r` in config.yaml and
`cfg.get('squeeze_*')` in `HodBreakEngine.__init__` (l. 288–290) — never added to the whitelist. Same for
`fc_leg_poll_s` (l. 277). Result on 11/2, the first live day: `squeeze_enabled=False`, the 14:00 squeeze never ran
(FTAI −0.63 R and WING +0.12 R open at 14:00, no SQUEEZE / WARNING line, `squeeze_done` 0 on every row). The Saturday
boot rehearsal cannot catch it (no session) and the `[HOD] engine gates` line does not print the squeeze fields.

## Fix (root cause, not a patch)
1. `config.py`: add the four keys to `hod_break_cfg` (and to the R2G block at l. ~893 if it mirrors the same engine
   keys — check) with the engine's defaults: `squeeze_enabled` bool False, `squeeze_minute` int 840,
   `squeeze_max_r` float 0.5, `fc_leg_poll_s` float (import `HodBreakEngine.FC_LEG_POLL_S` or repeat its literal
   with a comment naming the source).
2. **Make the drop loud, permanently.** A shared helper `_whitelist_check(block_name, raw: dict, returned: dict)` in
   `config.py`, called by `hod_break_cfg` (and its R2G twin): every raw YAML key not in the returned dict →
   `logger.error(f"config.yaml {block_name}: key '{k}' is not read by the engine (not in Config.{block_name}_cfg) — "
   f"ignored; add it to the whitelist or delete it")`. ERROR = Telegram at boot, as intended (a silently ignored
   trading parameter is never harmless). Nested blocks that the engine reads whole (`failure_short`, anything
   returned as a dict) count as one key.
3. `[HOD] engine gates` line (`hod_break_engine.py` l. ~387): append `squeeze={on|off}@{minute}/<{max_r}R` and
   `entry_mode`, so the boot journal proves the flags.
4. Tests (`tests/test_hod_config_whitelist_20261102.py`):
   a. `hod_break_cfg` on a YAML with the four keys returns them with the right types/values.
   b. Unknown key `zzz_unknown: 1` in the block → one ERROR record naming the block and key; no ERROR with a clean
      block.
   c. **Parity test:** every `cfg.get('<key>'` literal in `trading/hod_break_engine.py.__init__` (regex over the
      source) is a key of `Config.hod_break_cfg` for a YAML that sets them all → a future `cfg.get` without a
      whitelist entry fails CI. Likewise every key in the live `config.yaml` `hod_break` block (read-only) is
      whitelisted.
   d. The gates line contains `squeeze=on@840/<0.5R` when enabled.
5. A squeeze unit test already exists (tests/test_hod_exec_instruments_20261031.py) — confirm it builds the engine with
   a raw dict and therefore never went through `Config` (that is how this slipped); add one test that builds the
   engine from `Config.hod_break_cfg` on a tmp YAML with `squeeze_enabled: true` and asserts
   `engine.squeeze_enabled is True`.

## Deliverables
`docs/hod_config_whitelist_drop_spec_20261102_RESULT.md`: the keys added, the ERROR text, test counts, suite count.
Return ≤ 150 words.

## Rules
No commit/stash, no restart (onemil-trader is LIVE; the fix is picked up by the 11/3 12:30 UTC boot after Fable's boot
rehearsal tonight), no orders, no writes to data/*.db, config.yaml (read-only in tests), orb.yaml, .env, crontab,
logs/*; one pytest process at a time, the suite once at the end in the background with a scratchpad log, never piped,
always `--ignore=tests/integration`; never Read > 300 lines at once. This task IS the owner's request.
