# Cron code fixes 2026-11-12 — result

1. **Universe build**: `data_sources/alpaca_client.py` `invalid_symbols_in_error`, `AlpacaClient._fetch_daily_chunk` (new, called from `get_daily_bars`): a 400 `invalid symbol: X` drops X, WARNING, retries the chunk (bounded at 5, one name per response), LOST kept in `lost_daily_bar_symbols` / `daily_bars_requested`; other errors raise. `main.py` `run_batch` (crash -> ERROR + traceback + exit 1) / `_run_batch`, `lost_symbols_verdict` (`Batch complete: … lost=N [names]`, LOST > 2 % = ERROR).
2. **Cron guard** (`run_guarded` in each `__main__`): `scripts/{trader_watchdog (new main()), bf_decision_parity, hourly_summary, guardrail, hod_break_deadman_flat, eod_report, orb_pick_parity, daily_green_check}.py`; batch via §1.
3. **BF paused**: `scripts/bf_decision_parity.py` `STATUS_PAUSED`, `build_report` (BT only, no live comparison), `clear_false_paused_freeze` (only marker-reason freezes, logged reason "paused book, parity not applicable"), main skips breaches; `scripts/eod_report.py` `parity_section` prints the status.
4. `docs/active_crons.md` regenerated.

Tests: `tests/test_cron_fixes_20261112.py` 21 pass; related set (BF parity, EOD, green check, alpaca client, universe builder, cron guard) 443 pass, 0 failures; guardrail/hourly/main/pick-parity files run separately.
Not done: full suite and `get_daily_bars_range` (same invalid-symbol exposure) not changed; no real-API probe of the 400 message format (regex built from the logged text).
