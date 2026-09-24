# SPEC — live bull flag must not trade leveraged/inverse wrappers (parity defect, 2026-09-24)

## Defect (money-losing, live)
2026-09-23 live BF bought RGTZ ("Tidal Trust II Defiance Daily Target 2x Short RGTI ETF") and lost $600 (−4 R). The
backtest excludes wrappers by name at Stage 2 (`batch_backtest.py` ~line 268 → `trading/bf_universe_filter.py`,
`is_bf_eligible('RGTZ') == False`), so the BF parity check froze the ramp ("LIVE_ONLY — Stage-1 saw it, live traded
it, Stage-2 did not"). Root cause (main-session trace, confirm it): since the 9/5 ORB wrapper rule, the shared
scanner/bar universe carries wrappers; symbols reach BF via `TradingEngine._qualified_symbols` (batch pattern check,
`trading/trading_engine.py` ~3368) and the RT instant check (~1455) with NO BF name guard. CLAUDE.md promises
"excluded for BF by name" — live never implemented it after 9/5.

## Fix (Sonnet; ≤ 40 tool calls; TDD; do not commit; do not restart any service)
1. ONE predicate for BT and live: `trading/bf_universe_filter.is_bf_eligible(symbol, names)`. Live supplies names
   from the live asset-name lookup `AlpacaClient` already has (~line 1621, "asset's registered name ... wrapper-vs-
   stock"), cached per symbol per session; the BT keeps its offline dumps. Unknown name (lookup failed) → the legacy
   `_LEVERAGED_ETF_SYMBOLS` fallback AND a WARNING log (fallbacks must log).
2. Guard at the single choke point where a symbol becomes BF-eligible (where `_qualified_symbols` is populated, so
   both the batch check and the RT instant check are covered) — BF only. ORB, HOD and the scanner itself keep
   wrappers (ORB's 9/5 rule). Log once per symbol per day at INFO:
   `BF UNIVERSE: {sym} excluded — "{name}" is a leveraged/inverse wrapper (BT Stage-2 parity, bf_universe_filter)`.
3. Tests (`tests/`, fixtures in `tests/conftest.py`, `MagicMock(spec=...)` for domain classes):
   * unit: RGTZ name → excluded; "Jaguar Health, Inc. Common Stock" → kept; unknown name + legacy-list symbol →
     excluded with WARNING; unknown name + other symbol → kept with WARNING;
   * integration: a real `TradingEngine` path — a qualified wrapper never reaches pattern check or order submission;
     a common stock does (regression: the 9/23 RGTZ case);
   * parity: for a fixed list of (symbol, name) pairs, BT `filter_trades` and the live guard agree on every symbol.
4. Run the new tests, `tests/test_bf_universe_filter.py`, `tests/test_wrapper_universe_rule.py`, then the full suite
   (`python3 -m pytest tests/ -q -x -p no:randomly`) — zero failures.
5. Real-API read-only probe: the live name lookup for RGTZ and JAGX with real keys (`.env`), print both names and the
   guard's verdict.
6. Write `docs/bf_wrapper_guard_REPORT.md`: files changed, tests added, full-suite result, probe output, and the exact
   boot-log line the main session should grep for in the rehearsal.

Return ≤ 150 words.
