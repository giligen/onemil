# BF wrapper-guard — implementation report (2026-09-24)

Spec: `docs/bf_wrapper_guard_spec_20260924.md`. Fixes the 9/23 live defect
(BF bought RGTZ, a 2x short wrapper, lost 4R) by giving live the same
name-based exclusion BT Stage-2 already applies.

## Files changed
- `trading/bf_universe_filter.py` — `is_bf_eligible` now logs WARNING once
  per symbol per process when the name is unknown and it falls back to
  `AlpacaClient._LEVERAGED_ETF_SYMBOLS` (fallback path, was silent before).
- `trading/trading_engine.py` — new `TradingEngine._bf_wrapper_excluded`
  (looks up the asset name via `alpaca.get_asset_name`, cached per symbol in
  `self._bf_name_cache`; calls the shared `is_bf_eligible`; logs INFO once
  per symbol/day via `self._bf_excluded_logged` on exclusion). Called as the
  first statement in `on_stock_qualified` — the single choke point for both
  the batch pattern check and the RT instant bar callback (`_on_bar_close`),
  since both are keyed off `_qualified_symbols`, which a wrapper now never
  enters. Both caches cleared in `reset_daily`.
- `tests/conftest.py` — **new file**. `bf_db`/`bf_mock_alpaca`/`bf_engine`
  fixtures: a real `TradingEngine` with `MagicMock(spec=...)` collaborators.
- `tests/test_bf_universe_filter.py` — +4 unit tests: RGTZ excluded, JAGX
  kept, unknown name + legacy-list symbol excluded+WARNING, unknown name +
  other symbol kept+WARNING.
- `tests/test_wrapper_universe_rule.py` — +`TestLiveBFWrapperGuard` (real
  `TradingEngine`): wrapper never reaches `_qualified_symbols`/pattern
  check/`submit_buy_stop_bracket_order`; common stock does (positive
  control — proves this harness would have caught 9/23); log fires once/day
  and the name lookup is cached (not refetched); 5-case parametrized parity
  test driving both `bf_universe_filter.is_bf_eligible` (BT call site) and
  `TradingEngine._bf_wrapper_excluded` (live call site) on the same
  (symbol, name) pairs.

## Test results
- New tests: `tests/test_bf_universe_filter.py` + `tests/test_wrapper_universe_rule.py` → **38 passed**, 0 failed.
- Full suite: `python3 -m pytest tests/ -q -x -p no:randomly` → **4180 passed, 10 skipped, 0 failed** (467s).

## Real-API probe (read-only, `.env` live keys, `get_asset_name` only — no orders)
```
RGTZ: name='Tidal Trust II Defiance Daily Target 2x Short RGTI ETF' is_bf_eligible=False
JAGX: name='Jaguar Health, Inc. Common Stock' is_bf_eligible=True
```
Confirms the live lookup + shared predicate reproduce the 9/23 case exactly.

## Boot-log line for the rehearsal
On a qualified wrapper, grep the service journal for:
```
BF UNIVERSE: <SYM> excluded — "<name>" is a leveraged/inverse wrapper (BT Stage-2 parity, bf_universe_filter)
```
e.g. `journalctl -u onemil-trader -S today | grep "BF UNIVERSE:"`. Absence of
this line all day is not itself a failure (no wrapper may have qualified);
confirm instead that no wrapper symbol ever appears in a trade/order log.
