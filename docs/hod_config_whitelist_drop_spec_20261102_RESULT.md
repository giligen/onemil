# RESULT: HOD config whitelist drop fix (2026-11-02)

## Keys added to `Config.hod_break_cfg` (and, via the shared `_engine_mechanics_cfg`, `red_to_green_cfg`)
The spec named four; the new source-parity test found every `cfg.get(...)` the engine reads, so all missing ones were added
(defaults = the engine's own): `squeeze_enabled` (False), `squeeze_minute` (840), `squeeze_max_r` (0.5), `fc_leg_poll_s`
(10.0 = `HodBreakEngine.FC_LEG_POLL_S`, literal repeated in `Config._FC_LEG_POLL_S_DEFAULT`, parity-tested),
`oco_settle_timeout_s` (5.0), `oco_settle_poll_s` (0.25), `oco_retry_delay_s` (2.0), `eod_exit_mode` ('market'),
`eod_limit_timeout_s` (20.0), `stream_list_dir` ('logs'); HOD only: `live_parity_ledger_path`, `live_orders_state_path`.
`book` is deliberately not a hod_break key (R2G dispatch key). Live config.yaml hod_break/red_to_green keys are all whitelisted.

## Loud drop
`Config._whitelist_check(block_name, raw, returned)`, called by both properties. Keys read inside `params` and nested blocks
(`failure_short`) count as read. Text (logger.error, so Telegram at boot):
`config.yaml {block}: key '{k}' is not read by the engine (not in Config.{block}_cfg) — ignored; add it to the whitelist or delete it`

## Gates line (`hod_break_engine.py`)
Appended: `entry_mode={mode} squeeze={on|off}@{minute}/<{max_r}R` e.g. `squeeze=on@840/<0.5R`.

## Tests: `tests/test_hod_config_whitelist_20261102.py` (13)
Keys/types/defaults flow, R2G twin, unknown key -> exactly one ERROR, clean block -> none, R2G ERROR, source-regex parity
(engine `cfg.get` literals vs whitelist), FC poll default parity, live config.yaml read-only whitelist check, gates line
on/off, and an engine built FROM `Config.hod_break_cfg` honours `squeeze_enabled` (the existing 10/31 squeeze tests use a raw
dict via `tests/test_hod_break_engine.cfg`, never `Config` — that is how it slipped).

## Suite
`pytest tests --ignore=tests/integration`: 5523 passed, 6 skipped, 0 failed (284.8 s). Log: scratchpad `suite_whitelist.log`.
No commit, no restart; takes effect at the next boot (11/3 12:30 UTC) after the boot rehearsal.
