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
