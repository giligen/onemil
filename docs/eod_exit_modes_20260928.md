# EOD exit modes (`eod_exit_mode`) — 2026-09-28

Flag-gated, default OFF (absent from `orb.yaml` / hod_break cfg — neither file was touched by this
change, so both engines default to `market`, byte-identical to pre-flag behaviour, proven by
`TestOrbEodExitFlagOff` / `TestHodEodExitFlagOff` in `tests/test_eod_exit.py`).

Shared spec: `trading/eod_exit.py` (pure functions — mode resolution, MOC cutoff check, NBBO-mid
calc, pricing-method tagging, telemetry-dict builder). Both the ORB (`trading/orb_engine.py`,
`_close_position_with_held_qty_retry` / `_submit_eod_exit_priced`) and HOD
(`trading/hod_break_engine.py`, `force_close_all`) engines call the same module, so the three modes
mean the same thing in both engines by construction (CLAUDE.md "ONE spec ... no accidental
behaviour").

## Config key

`eod_exit_mode` (orb.yaml `exit.eod_exit_mode`, hod_break cfg `eod_exit_mode`), `eod_limit_timeout_s`
(default `20.0`). Neither key was added to `orb.yaml`/`orb.yaml.template`/`config.yaml` by this
change — an operator opts in by adding the key.

| Value | Behaviour |
|---|---|
| `market` (default) | Unchanged: ORB `close_position()`; HOD marketable limit at `ref × 0.99` (`× 0.97` on the 3rd+ FC attempt). |
| `limit_then_market` | Rest a limit sell at the NBBO mid captured at the exit instant. If unfilled after `eod_limit_timeout_s` (default 20s), cancel and re-submit as a plain market order (`submit_market_sell_order`, new method on `data_sources/alpaca_client.py`, `TimeInForce.DAY`). |
| `moc` | Submit a market-on-close order (`submit_moc_sell_order`, new method, `TimeInForce.CLS`). Past the cutoff (below), falls back to `limit_then_market` with a WARNING. |

## Cutoff fact verified

Checked locally against the **installed alpaca-py's own docstring**
(`alpaca.trading.enums.TimeInForce`, `python3 -c "import alpaca.trading.enums as e, inspect; print(inspect.getsource(e.TimeInForce))"`):

> `cls`: ... "CLS orders submitted after 3:50pm but before 7:00pm ET will be rejected."

So `eod_exit.MOC_CUTOFF_ET = 15:50 ET` (10 minutes before the 16:00 close). `resolve_mode()` checks
this before every MOC submission; at/after 15:50 ET it downgrades to `limit_then_market` and returns
a warning string the caller logs (`TestResolveMode.test_moc_at_cutoff_falls_back_with_warning`).
`TimeInForce.CLS` itself is confirmed present in the installed `alpaca-py` (`TimeInForce` members:
`DAY, GTC, OPG, CLS, IOC, FOK`).

A resting MOC order is never age-cancelled by HOD's normal 60s (`FC_RESUBMIT_S`) resubmit ladder
(`force_close_all`'s `never_resubmit` check) — it rides to the close by design, same as Alpaca's own
semantics for a CLS order.

## Telemetry

Existing `trades` columns (no migration needed): `exit_pricing_method`, `exit_quote_bid`,
`exit_quote_ask`, `exit_price`, `exit_fill_latency_ms`, `exit_slippage`. Pricing-method tags are
namespaced `eod_*` so they never collide with StopMonitor's own `stop_loss` / `quote_tight` family:
`eod_market`, `eod_limit_mid`, `eod_limit_mkt_fb`, `eod_moc`, `eod_moc_cut_limit`,
`eod_moc_cut_mkt_fb`. `exit_slippage` = reference price (NBBO mid for `limit_then_market`; the
pre-submit bid for `market`/`moc`, RESULT_1443's convention) − exit price (positive = cost).
`exit_fill_latency_ms` is written when the fill is observed within this code's own poll window (ORB's
bounded 0.5s-interval poll up to `eod_limit_timeout_s`; HOD's next `force_close_all` pass) — a fill
observed later is picked up by the ordinary FC reconciliation with no latency figure, and the
**vs-official-close** column is intentionally left for a separate EOD report pass (fetch the day's
official close after hours) — this module does not fetch it.

## Deliverable B — HOD exit telemetry (previously absent)

Before this change, HOD's exits recorded **none** of the above columns, in either exit path:

- Bracket target/stop legs (`_book_leg_fill` → `_record_exit`): no telemetry at all.
- StopMonitor-routed exits (`_drain_stop_monitor_exits`): the `StopExitEvent` StopMonitor emits
  already carries `pricing_method` / `exit_quote_bid` / `exit_quote_ask` / `exit_limit_price` /
  `submitted_at` (exactly what the ORB engine reads via `trading/trading_engine.py` ~L2741-2767) —
  HOD was discarding all of it, keeping only `exit_price`/`exit_reason`.

Now: `_record_exit` takes optional `pricing_method` / `quote_bid` / `quote_ask` / `fill_latency_ms` /
`trigger_price` kwargs and writes the same five columns, plus one INFO line per exit
(`EXIT {symbol} method={method} fill_vs_trigger={bps}`):

- `_drain_stop_monitor_exits` now threads the `StopExitEvent` fields through (fixes the discard).
- `_book_leg_fill` tags bracket legs `bracket_target` / `bracket_stop_limit` with `pos.target` /
  `pos.stop` as the trigger (slippage measured for free, no new state); the `eod` leg reads
  `pos.eod_pricing_method` / `pos.eod_quote_bid` / `pos.eod_quote_ask`, set by `force_close_all` at
  submission time (new `Position` fields).
- `exit_quote_bid`/`exit_quote_ask`/`exit_fill_latency_ms` are left unset for bracket target/stop legs
  (no submission-time quote capture exists for those today — that needs entry-time instrumentation,
  out of scope for an EXIT-path change; `exit_pricing_method` + `exit_slippage` are still populated).

## Parity caveat

`moc` fills at the 16:00 ET closing auction. The ORB backtest force-closes at 15:45 ET
(`orb.yaml exit.force_close_time_et`) and the HOD backtest flattens at 15:55 ET (`flat_minute`) —
**neither backtest models a 16:00 exit.** A `moc` fill is not comparable to either book's backtested
P&L until the backtest itself is re-verified with a close-price exit. `trading/eod_exit.py` is a
live/dry **measurement** option (cost vs the bid at the exit instant, cost vs the official close) —
never a backtest-parity rule. `limit_then_market` and `market` do not move the exit time and carry no
such caveat.

## Rollback

Do not edit `orb.yaml` / hod_break cfg to add `eod_exit_mode` (both default `market`). If already set
non-default and something looks wrong: remove the key (or set it to `market`) — no restart-time
migration, no DB schema change, no state to unwind; the next `force_close_all` / EOD pass reverts to
`close_position()` / the `ref × 0.99` ladder. `git revert` the two new methods on
`data_sources/alpaca_client.py` and the wiring in `trading/orb_engine.py` /
`trading/hod_break_engine.py` is safe at any time — `trading/eod_exit.py` has no side effects of its
own.

## Files touched

- `trading/eod_exit.py` (new) — shared spec.
- `data_sources/alpaca_client.py` — `submit_market_sell_order`, `submit_moc_sell_order` (new methods,
  appended after `submit_limit_sell_order`; no existing method changed).
- `trading/orb_engine.py` — `eod_exit_mode`/`eod_limit_timeout_s` config read (`exit_cfg`, next to
  `force_close_time_et`); `_close_position_with_held_qty_retry` branches to new
  `_submit_eod_exit_priced` for non-market modes.
- `trading/hod_break_engine.py` — `eod_exit_mode`/`eod_limit_timeout_s` config read in `__init__`;
  `Position` gains `eod_pricing_method`/`eod_quote_bid`/`eod_quote_ask`; `force_close_all` mode
  branch; `_record_exit` telemetry kwargs; `_drain_stop_monitor_exits` and `_book_leg_fill` pass
  telemetry through.
- `tests/test_eod_exit.py` (new) — 27 tests: pure `eod_exit.py` logic (mode resolution incl. cutoff +
  WARNING, NBBO mid, pricing-method tags, telemetry builder), Alpaca client TimeInForce per mode
  (`DAY` / `CLS`), ORB flag-off byte-identical + `moc` + `limit_then_market` timeout escalation, HOD
  flag-off byte-identical + `moc`, and HOD exit-telemetry population (StopMonitor-routed, bracket
  stop leg, EOD leg).

## Test evidence

`python3 -m pytest tests/test_orb_engine.py tests/test_hod_break_engine.py tests/test_eod_exit.py -q
--no-header -p no:randomly` → 169 passed. Full suite `python3 -m pytest tests -q --no-header -p
no:randomly` → **4451 passed, 5 skipped, 0 failed** (2026-09-28).
