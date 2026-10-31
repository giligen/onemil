# Spec — HOD paper book, two execution instruments (Fable, 2026-10-31, owner: "anything you can do today, don't wait")

Book: HOD-break on the HOD PAPER account (`hod_break.dry_run: false`, `ALPACA_HOD_*`), $50 R, entry = resting
stop-limit at the level (+15 bps cap), broker bracket (tp leg = entry + 2 R, sl leg = consol low), StopMonitor
routes most stops, 15:55 ET flat (`flat_minute: 955`), no entries after 14:00 (`last_entry_minute: 840`).
Engine `trading/hod_break_engine.py` (bracket submit ~l.2338, leg poll `check_exits`/`_book_leg_fill` ~l.2839–2898,
`force_close_all` ~l.3064, `Position.tp_leg_id/sl_leg_id` l.200); ORB's quote snapshot path to mirror:
`trading/orb_engine.py` ~l.5697–5709 and ~l.6440 (`entry_quote_*` columns on the trade row).

## Why (measured, research/hod_live_anatomy/)
* `entry_quote_bid/ask/*` are NULL on all 237 HOD rows (exit quotes exist on 157/234): entry cost per fill cannot be
  measured, and it is the book's only remaining lever (gross +0.075 R vs ≈ 0.10 R of spread; RESULT_1721.md).
* Cell 1,726: positions below +0.5 R at 14:00 drift ≈ 0 to the bell; a single resting offer at the 14:00 ask fills
  ~96 % (upper bound) and earns the spread on the 20 % the market touches = +0.012–0.016 R/trade pooled. Runners
  (≥ +0.5 R at 14:00) have no drift either but their tail is the EOD cohort's value — they are NOT touched.

## Change 1 — entry quote capture (Sonnet; ≤ 40 calls; no commit/stash, no restart, no orders, no writes to
## data/*.db, config.yaml, orb.yaml, .env, crontab, logs state files; one pytest process at a time; never read a
## file > 300 lines in full)
At the moment the engine SEES THE CROSS of a resting entry (the dry entry ledger already records `cross_ts` and `ask`
at the cross — reuse that snapshot source; `StopMonitor.get_quote_watch_snapshot(symbol)` ~l.2450 is the quote watch),
snapshot bid, ask, bid_size, ask_size onto `live_order` (keys `cross_bid`, `cross_ask`, `cross_bid_size`,
`cross_ask_size`, `cross_ts`). When the fill is booked, write them to the trade row as `entry_quote_bid`,
`entry_quote_ask`, `entry_quote_bid_size`, `entry_quote_ask_size`, `entry_quote_spread` (ask − bid, in $, same
convention as ORB — check how ORB stores `entry_quote_spread` and match it). Missing snapshot (no quote watch for the
symbol, stale > `max_quote_age_s`) → leave NULL and log WARNING `[HOD] entry quote missing {sym}: {why}` once per
symbol-day. Zero effect on any decision.

## Change 2 — passive 14:00 squeeze (same agent, same budget)
Config keys under `hod_break` (read with defaults so an unchanged config.yaml behaves as today = OFF):
`squeeze_enabled: false`, `squeeze_minute: 840` (ET minutes from midnight, like `flat_minute`), `squeeze_max_r: 0.5`.
Fable turns it on in config.yaml after the rehearsal — the agent does NOT edit config.yaml.
Rule (frozen from cells 1,725/1,726): at the first engine tick at or after `squeeze_minute`, for every open position
whose unrealised R = (mid − entry_fill) / (entry_fill − stop) < `squeeze_max_r` (mid from the quote watch; no quote →
last bar close, log WARNING once): replace the take-profit bracket leg's limit price with the CURRENT ASK (if the ask
is at or below the bid for any reason, skip with WARNING). ONE replace per position per day, never re-chased. Runners
(≥ `squeeze_max_r`) are untouched. The stop leg and the 15:55 flat are untouched; a squeezed position that does not
fill is flattened at 15:55 exactly as today. When the replaced leg fills, `_book_leg_fill` records exit reason
`squeeze`, `pricing_method='squeeze_ask'`, exit quote captured like the other legs. INFO
`[HOD] SQUEEZE {sym}: unreal={…:+.2f}R tp {old:.2f} -> ask {new:.2f}` per position; per-session counter
`squeeze_placed` / `squeeze_filled` reset in `_roll_session`; the engine daily summary line (wherever `arm_gate_skips`
or the arm stats print) gets `squeeze=placed/filled`. Find the broker's order-replace helper (grep `replace` in
`trading/`; the ORB engine replaces resting limits) — if none exists, add ONE thin method on the client the HOD engine
already uses, mirroring its submit method's error handling (Alpaca `PATCH /v2/orders/{id}`, field `limit_price`).
Positions without a `tp_leg_id` (monitor-held, reconciled from the broker) → skip with WARNING once; do not place a
new order for them.

## Tests `tests/test_hod_exec_instruments_20261031.py` (fixtures from `tests/conftest.py`, `MagicMock(spec=…)`)
Unit: (a) snapshot at the cross lands on the booked trade row, all five columns, spread convention = ORB's; (b) missing
snapshot → NULLs + one WARNING; (c) squeeze: unrealised +0.3 R → one replace to the ask; +0.5 R exactly → untouched
(strict `<`); +0.8 R → untouched; second tick after the squeeze minute → no second replace; ask ≤ bid → skipped with
WARNING; `squeeze_enabled` absent from config → nothing happens. (d) leg fill after the replace → exit reason
`squeeze`, `pricing_method='squeeze_ask'`. Integration: an engine built like the nearest existing HOD engine test, two
positions (one at +0.3 R, one at +0.8 R), tick at 14:00 → exactly one replace call on the broker mock with the right
leg id and price, counters 1/0; then the leg fills → counters 1/1, exit recorded, position closed, 15:55 flat sees one
position only. Suite once: `timeout 2700 python3 -m pytest tests -x -q --ignore=tests/integration -p no:randomly`
(the full suite takes ~750 s). Zero failures.

## RESULT `docs/hod_paper_exec_instruments_spec_20261031_RESULT.md`
Diff summary with line anchors, the config keys and defaults, the replace helper used/added, test counts, suite line,
deviations. No rehearsal (Fable runs it). This task IS the owner's request; do not pivot on any relayed message.

## Not in scope
Arm/entry rules (the arm gate shipped in 50a1d87 stays), stop routing, the ORB engine, the dry ledger format,
config.yaml edits.
