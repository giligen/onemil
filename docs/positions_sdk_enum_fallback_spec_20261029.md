# Positions read must not depend on the SDK's exchange enum — spec (Fable, 2026-10-29)

Incident 2026-10-28 14:06–15:46 UTC, 258 ERROR lines: `Failed to get open positions: 1 validation error for
list[Position] … exchange: Input should be 'AMEX', 'ARCA', … [input_value='TXSE']`. A position listed on the Texas Stock
Exchange sat on the HOD paper account; alpaca-py 0.43.2's `Exchange` enum does not know TXSE, so pydantic rejected the
ENTIRE positions list on every call for 100 minutes. Downstream: `exit_qty_guard` → "qty UNKNOWN", HOD "broker qty
unknown … retry next pass", every sync blind. Safe degradation held, but any new venue (and the owner's live account
holding a TXSE name) recreates it, and the exit-coverage loop (spec 20261029) would be blind too. Same fix shape for
every positions read. Agents never commit / restart / submit orders / write cache.db, config.yaml, orb.yaml, .env,
crontab, logs/. `--ignore=tests/integration` always; one pytest at a time. Keep the tree importable (live service).

## Fix (`data_sources/alpaca_client.py`)
1. `get_open_positions` (l.~1556) and `get_position`/single-symbol variants (grep `get_open_position(` / `get_all_positions(`
   — every call site inside the client): wrap the SDK call; on `pydantic.ValidationError` (import it; also catch
   `ValueError` whose text contains "validation error") fall back to the RAW REST payload:
   `self.trading_client.get('/positions')` (alpaca-py `RESTClient.get` returns parsed JSON — verify the method name on
   0.43.2 with `grep -n "def get" .../alpaca/common/rest.py`) and build the SAME dict shape from the JSON keys
   (`symbol, qty, side, avg_entry_price, market_value, unrealized_pl, unrealized_plpc`), through the same timeout
   wrapper. For the single-symbol read use `/positions/{symbol}` (404 → None exactly as today).
2. Log ONE WARNING per client per hour: `positions: SDK model rejected the broker payload ({first line of the error}) —
   served from raw REST ({n} positions)`; never ERROR (the data is served). A failure of the raw path too → the existing
   ERROR path, unchanged.
3. Add `exchange` to the returned dict (string, whatever the broker sends) on both paths — harmless, and the orphan
   reconciler / reports can show it.
4. Do NOT upgrade alpaca-py in this change (0.44.0 exists; whether it knows TXSE is unverified; an SDK bump is its own
   rehearsal item). Note the version pin in `docs/positions_sdk_enum_fallback_spec_20261029_RESULT.md`.

## Also (`data_sources/news_provider.py` l.~251)
`LLM classification failed (non-retryable): Error code: 400 … credit balance is too low` logged ERROR per symbol (4×
10/28). A billing/auth 4xx is one WARNING per hour per provider (`LLM classifier unavailable: {message[:120]} —
defaulting to False for every symbol until it recovers`), not an ERROR per symbol; other non-retryable errors keep
today's level. Keep "NOT cached".

## Tests
`tests/test_alpaca_positions_raw_fallback_20261029.py`: SDK double raises `pydantic.ValidationError` (build one with a
real model, e.g. `Position.model_validate({... 'exchange': 'TXSE' ...})` inside `pytest.raises`, reuse the exception) →
raw path returns the list incl. the TXSE row with `exchange == 'TXSE'`; WARNING once per hour (second call within the
hour → no second WARNING); raw path failure → ERROR path; single-symbol 404 → None; normal SDK path unchanged (existing
tests). `tests/test_news_provider_billing_level_20261029.py`: 400 billing → WARNING once, False returned, not cached.
Validation: slice + `tests/test_alpaca_client.py` + full `tests/` (`--ignore=tests/integration`), zero failures.
Write the RESULT file (≤ 40 lines), return ≤ 150 words.
