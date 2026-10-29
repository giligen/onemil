# Spec — asset reads must survive SDK enum gaps (TXSE), 2026-10-29 evening (Fable). Sibling of `docs/positions_sdk_enum_fallback_spec_20261029.md`.

## Defect (live, two days running)
alpaca-py 0.43.2's `Exchange` enum has no `TXSE`; 0.44.0 (latest on PyPI, checked 10/29) does not either. Since 10/28
the `/assets` list contains TXSE-listed assets, so `trading_client.get_all_assets()` raises
`pydantic_core.ValidationError` for the WHOLE list after a successful HTTP call:
* `logs/universe_build.log` 10/28 and 10/29 10:30 UTC: `ERROR Failed to get tradeable assets: N validation errors for
  list[Asset]` → the daily universe build (`main.py --batch`, `batch/universe_builder.py:148/435`) has failed two days
  and the ERROR goes to Telegram each morning.
* `logs/momentum_sleeve.log`: `scripts/momentum_sleeve.py:121 fetch_assets` died with the same traceback — the MOM
  sleeve's Monday 11:50 UTC fetch will crash on 11/3 unless fixed (no rotation = stale live positions).
* `get_asset(symbol)` (`is_marginable`, `get_shortability`, `get_asset_name`) fails only for TXSE-listed symbols
  (fail-closed today, but with a WARNING per call).
Same root cause and the same remedy as the positions fix (`_open_positions_raw_rest`): raw REST on validation failure.

## Change (Sonnet; one agent; ≤ 40 calls; no commit, no restart, no config, no orders, no cache writes)
1. `data_sources/alpaca_client.py`
   * `RawAsset` (dataclass, module level): `id, symbol, name, exchange: str, asset_class: str, status: str, tradable,
     marginable, shortable, easy_to_borrow, fractionable` (bools default False; strings default ''). `_asset_from_json(d)`
     maps one `/assets` row (keys as Alpaca returns them) to it; unknown keys ignored.
   * `get_assets_tolerant(self, request=None, **call_kwargs) -> list`: `request` defaults to active us_equity; runs
     `self._call_with_timeout(lambda: self.trading_client.get_all_assets(request), "get_all_assets", **call_kwargs)`;
     on `PydanticValidationError` (import already at l.43) → `self._assets_raw_rest(request, e)` which calls
     `self.trading_client.get('/assets', params)` with `status`/`asset_class`/`exchange`/`attributes` taken from the request
     (only the ones set), maps rows with `_asset_from_json`, logs ONE WARNING per hour (`_assets_raw_warned_at`, pattern as
     the positions fix): `"get_all_assets: SDK rejected the list (N validation errors; unknown exchange values {…}) — raw
     REST fallback returned M assets"`, and returns the list. Every other exception propagates unchanged (no silent empty).
   * `get_asset_tolerant(self, symbol, **call_kwargs)`: same shape for `get_asset(symbol)` → `/assets/{symbol}`; WARNING
     once per hour (not per symbol).
   * `get_all_tradeable_assets` uses `get_assets_tolerant`; `'exchange'` becomes
     `asset.exchange.value if hasattr(asset.exchange, 'value') else str(asset.exchange or '')`.
   * `is_marginable`, `get_shortability`, `get_asset_name` use `get_asset_tolerant` with their existing timeout kwargs.
2. `scripts/momentum_sleeve.py`: `fetch_assets` → `client.get_assets_tolerant(req, timeout=90, timeout_retries=2,
   rate_limit_retries=5)`; the `get_asset(sym).fractionable` call (~l.666) → `client.get_asset_tolerant(sym).fractionable`.
3. `trading/macd_wave_engine.py` (~l.776) and `macd_wave_backtest.py` (~l.176): route through `get_assets_tolerant`
   (mechanical; the service is not installed, the backtest is offline — keep behaviour otherwise identical).
4. Tests — `tests/test_alpaca_assets_raw_fallback_20261029.py` (fixtures from `tests/conftest.py`, `MagicMock(spec=…)` for
   domain classes; SDK objects may omit spec):
   * SDK raises a REAL `PydanticValidationError` (build it with
     `TypeAdapter(List[Asset]).validate_python([{...,'exchange':'TXSE',...}])` inside `pytest.raises`, or construct via
     `ValidationError.from_exception_data`) → `get_assets_tolerant` calls `trading_client.get('/assets', …)` once, returns
     `RawAsset`s, exchange `'TXSE'` kept as str, bools correct; WARNING logged once, second call within the hour silent.
   * `get_all_tradeable_assets` on the fallback path returns dicts with `'exchange': 'TXSE'` and applies
     `_is_common_stock` as before (one wrapper excluded).
   * `get_asset_tolerant` fallback → `/assets/{symbol}`; `get_shortability` on a TXSE asset returns the real flags.
   * A non-validation exception (e.g. `AlpacaAPIError`, timeout) propagates / keeps the existing behaviour (no fallback).
   * Integration (real instances, mocked network only): `momentum_sleeve.fetch_assets(client)` on the fallback path returns
     the tradable-filtered DataFrame with `symbol, name`.
5. Real-API probe (read-only, paper MOM key `ALPACA_MOM_API_KEY/SECRET`, `load_dotenv('/home/ec2-user/onemil/.env')`):
   `scratchpad/assets_probe.py` → prints `len(get_assets_tolerant())`, number of TXSE rows, `len(get_all_tradeable_assets())`,
   and `get_shortability` for one TXSE symbol. Never print keys. Expect > 10,000 assets and ≥ 10 TXSE rows.
6. Full suite once: `python3 -m pytest tests -x -q --ignore=tests/integration` (never two suites at once). Zero failures.
7. Write `docs/assets_sdk_enum_fallback_spec_20261029_RESULT.md`: files changed, test counts, probe output, anything
   that deviated from this spec.

## Not in scope
Upgrading alpaca-py (0.44.0 has no TXSE either); any change to `_is_common_stock`; the universe-build gates.
