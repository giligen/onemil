# Positions SDK enum fallback — RESULT (2026-10-29)

## Files
* `data_sources/alpaca_client.py`: `get_open_positions` now catches `pydantic.ValidationError` (and a `ValueError`
  whose text contains "validation error") from `get_all_positions()` and serves the list from raw REST.
  New helpers `_position_dict_from_model`, `_position_dict_from_json`, `_open_positions_raw_rest`. `exchange`
  (string, as sent) added to the dict on BOTH paths. WARNING `positions: SDK model rejected the broker payload
  (...) — served from raw REST (n positions)` once per client per hour (monotonic, lazy attr
  `_positions_raw_warn_ts`). A raw-path failure falls into the existing ERROR + `AlpacaAPIError` path.
* `data_sources/news_provider.py`: billing/auth 4xx (401/402/403, or 400 with "credit balance"/"billing") is one
  WARNING per hour per analyzer (`LLM classifier unavailable: ...`), result still False and NOT cached; other
  non-retryable errors keep ERROR.
* Tests: `tests/test_alpaca_positions_raw_fallback_20261029.py` (6), `tests/test_news_provider_billing_level_20261029.py` (2).

## Raw REST method
`self.trading_client.get('/positions')` (alpaca-py 0.43.2 `RESTClient.get(path, data=None)` in
`alpaca/common/rest.py`, returns parsed JSON), through `_call_with_timeout` (op `get_open_positions_raw`).
`side` is converted to `PositionSide` when valid so consumers comparing to the enum or "long" both work.

## SDK pin
alpaca-py 0.43.2 unchanged. 0.44.0 exists; whether it knows TXSE is unverified; an SDK bump is its own rehearsal item.

## Tests
* Slice: new 8 + `tests/test_alpaca_client.py` + `tests/test_news_provider.py` = 158 passed.
* Full `tests/` (`--ignore=tests/integration`): 5349 passed, 6 skipped, 0 failed (623 s).

## Not done
* Single-symbol `/positions/{symbol}` variant: the client has NO single-symbol position read
  (`get_open_position(`/`get_position` appear nowhere in `data_sources/alpaca_client.py`), so there was nothing to
  wrap and the "404 -> None" test does not apply. Other modules calling `trading_client` directly were not touched.
