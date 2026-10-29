# RESULT — assets SDK enum fallback (spec 2026-10-29)

## Files changed
* `data_sources/alpaca_client.py`: `RawAsset`, `_asset_from_json`, `_is_sdk_validation_error`; `AlpacaClient.get_assets_tolerant`,
  `get_asset_tolerant`, `_assets_raw_rest`, `_assets_raw_warn` (one WARNING/hour via `_assets_raw_warned_at`);
  `get_all_tradeable_assets`, `is_marginable`, `get_shortability`, `get_asset_name` routed through them; `exchange` via `hasattr(.., 'value')`.
* `scripts/momentum_sleeve.py` (`fetch_assets`, `apply_fractionable`), `trading/macd_wave_engine.py`, `macd_wave_backtest.py`: routed through the tolerant reads.
* Tests: new `tests/test_alpaca_assets_raw_fallback_20261029.py` (9 tests); `tests/test_momentum_sleeve.py` and `tests/test_momentum_sleeve_live.py`
  doubles now wire `get_asset_tolerant` (the `spec=AlpacaClient` mock no longer reaches `trading_client.get_asset` directly).

## Probe (paper MOM key, read-only, `scratchpad/assets_probe.py`)
assets 14,349; TXSE rows 20; `get_all_tradeable_assets` 11,221; `get_shortability(ICOM)` (TXSE) = shortable False, easy_to_borrow False.
Single WARNING emitted: "20 validation errors for list[Asset]; unknown exchange values {'TXSE'} — returned 14349 assets".

## Deviations
* Request params use `request.to_request_fields()` (exactly what the SDK sends) instead of hand-picking four keys.
* `get_assets_tolerant` takes an extra `operation` keyword (label for the timeout wrapper) before `**call_kwargs`.
* `apply_fractionable` is the ~l.666 `get_asset(sym)` site; existing momentum-sleeve test doubles were adjusted (see above).

## Follow-up (same day): MOM sleeve crash never reached Telegram
The sleeve attached no Telegram error handler at all, and an uncaught exception only printed a traceback. Added
`scripts/momentum_sleeve.py::main_guarded` (entry now `sys.exit(main_guarded())`): any `Exception` from `main()` is logged
`logger.error("momentum_sleeve crashed (%s): %s")` and re-raised unchanged (non-zero exit); `SystemExit`/`KeyboardInterrupt`
pass through. `_attach_telegram_error_handler` adds a root `TelegramErrorHandler` at crash time (idempotent, after
`main()`'s `basicConfig`, which an earlier root handler would have turned into a no-op). Limitation: other ERRORs logged
inside a non-crashing run are still not forwarded. Tests: 3 new in `tests/test_momentum_sleeve.py`.

### Follow-up 2: crash message must survive process exit
`TelegramErrorHandler._send_async` starts a daemon thread, so the re-raise/exit killed the send. Now the handler keeps
`_send_threads` (pruned) and `wait_for_sends(timeout=10.0)` joins them within a total timeout; `main_guarded` calls it on every root
`TelegramErrorHandler` (wrapped, never raises) after logging the crash and before `raise`. Tests: 2 in
`tests/test_telegram_error_handler.py`, 1 in `tests/test_momentum_sleeve.py`.

### Follow-up 3: one shared crash guard for every cron entry point
New `monitoring/cron_guard.py`: `attach_telegram_error_handler() -> bool` (idempotent, Config-based, never raises) and
`run_guarded(main, name)` (logs `"<name> crashed (Type): msg"` at ERROR, waits for the Telegram send, re-raises unchanged;
SystemExit/KeyboardInterrupt pass). The per-script copies in `scripts/momentum_sleeve.py` were deleted. Wired: `momentum_sleeve`,
`tom_sleeve` (both also call `attach_telegram_error_handler()` right after `basicConfig`, so their `logger.error` lines now reach
Telegram), `build_hod_volume_profile`, `orb_weekly_refit`, `orb_selection_observer`. Tests: `tests/test_cron_guard_20261029.py`
(6); the `main_guarded` tests moved out of `tests/test_momentum_sleeve.py`; the two tests that call `main()` directly stub the
attach so no real Telegram handler is created. Targeted run: 153 passed. Not done: other cron scripts not named here.
