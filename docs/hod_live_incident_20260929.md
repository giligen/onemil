# HOD-break live incident — 2026-09-29 (first live session at $50) — PAUSED at 14:05 UTC

## What happened
* 9/25 (found today): the flat-at-15:55 sweep sold CDNA a second time after the StopMonitor had already stopped it out
  → the live account was SHORT 57 CDNA for two sessions; the orphan reconciler labelled our over-exit "owner's manual
  trade" at two boots. Covered 13:41 UTC @ 65.1152 (−$92 on the short; the real trade was −$54).
* 9/29 13:39–14:03 UTC: the resting stop-limit entries FILLED at the broker for PRIM (25 sh @ 77.74), WRBY (76 sh @
  26.19) and ASTN (83 sh) but the engine never registered the fills: it kept the orders as "armed", replaced them at new
  levels ("cancelled (replace)" on a filled order) and re-armed — three NAKED positions with no StopMonitor watch, no
  OCO legs, unknown to the registry; PRIM traded 3.5 % under its entry, below its intended stop, with nothing watching.
  TWST, MRNA, AXTX, NBIG, NBIL were detected and managed (TWST stopped −$51, −1.00 R, 2.6 bps slip).
* Every fill logged ERROR "OCO submit returned no leg ids" although both legs existed at the broker (parser expects a
  shape Alpaca does not send) — ERROR-level Telegram noise.
* Resting entries rejected/canceled by the broker "not our cancel" (WRBY, PRIM ×3, SVRN) — reason under investigation
  (likely a buy stop at or below the market when the level was already broken).

## Action (act-as-owner rule: fix money-losing defects immediately or pause the book)
* HOD real orders PAUSED: `config.yaml hod_break.dry_run: true`, restart after the owner flattens the six live
  positions (`scripts/ops_flatten_symbol.py SYM --yes`; the harness blocks the main session from live orders).
* Fixes in progress (tests first, boot rehearsal before any live day): fill registration from the order stream AND a
  5-s poll with GET-before-cancel/replace; broker-truth (min(registry, broker qty), refuse at zero) on every exit path;
  the OCO parser; hod- prefix on every order; reconciler adoption/alert for positions in symbols we traded; the
  missed-entry path (no chase, no re-arm loop) once the reject reason is known; arm-time feature telemetry.
* The live $50 test resumes only after ONE clean dry session on the fixed code with the parity ledger clean.

## Money
Live HOD 9/29: TWST −$51 realized; the six open positions at 14:05 UTC ≈ −$60 unrealized in total (PRIM −$68, NBIL −$33,
NBIG −$15, AXTX +$25, ASTN +$17, WRBY +$14). Plus the CDNA over-exit −$92 from 9/25.

## Part 2 — CONI duplicate rows and the R2G sync (17:25 UTC)
**Timeline**: 17:25:35 UTC `[R2G] sync: CONI open in DB (25 sh) but broker holds 0 — exit pending
verification` — bogus: R2G was reading hod_break's own row. Three trades rows exist for one CONI
broker position: 394 (87sh, pending_new), 396 (112sh, re-adopted), 395 (25sh, inserted by a later fill).

**Root causes**
1. Instance methods (`_open_position_count`, `_drain_stop_monitor_exits`, the phantom-sync watch drop,
   `sync_positions`' open-count query) read the MODULE constant `STRATEGY_NAME` ('hod_break') instead of
   `self.STRATEGY_NAME`. The red_to_green instance (main.py:845, LIVE client) therefore queried and could
   mutate hod_break's own open DB rows and could drain hod_break's own StopMonitor exit events.
2. `_adopt_unregistered_positions_on_boot` saved PRIM/WRBY/ASTN/CONI as order_status 'pending_new' — a
   status `sync_positions` never restores (open = filled/partially_filled/exit_pending_verification only)
   — so the same broker position was re-adopted whole on the next boot (row 396 = row 394 + the interim
   fill). It also never linked `trade_id` into `cand.live_order`, so the later fill of the remaining
   resting order inserted a THIRD row (395) via `_on_live_fill`'s dedup instead of updating the adopted one.
3. `reconcile_pending_exits` declared exit_pending_verification rows UNRECONCILED (ERROR+Telegram) without
   ever checking whether the broker still held the shares — CONI was flagged while the broker held the
   full position throughout.

**Fix** (`trading/hod_break_engine.py`): every instance-method bare `STRATEGY_NAME` → `self.STRATEGY_NAME`
(4 sites). Boot adoption now inserts `order_status='filled'` (fill_price/filled_at/filled_qty set), links
`trade_id` into `cand.live_order`, and dedups against the DB — an existing open row for (symbol, strategy)
is merged (shares += diff, fill_price = weighted average) instead of a second row; a non-positive
difference adopts nothing. `reconcile_pending_exits` now fetches broker positions once: broker qty >= the
row's open qty restores it (`order_status='filled'`, WARNING only); fewer shares with legs not covering
the rest stays the ERROR+Telegram UNRECONCILED path. `sync_positions`' "broker holds N" WARNING now names
the account compared against.

**Tests** (`tests/test_hod_break_engine_boot_defects.py`, 10 new, real Database + real engine instances +
`MagicMock(spec=AlpacaClient)`; full suite green): R2G `sync_positions` ignores/never mutates a hod_break
row; `drain_exit_events` called with each instance's own strategy; adopt-as-filled links `trade_id`;
second boot restores via `sync_positions` and does not re-adopt; broker-ahead-of-DB merges at the weighted
average price; non-positive difference adopts nothing; broker-still-holds restores with no ERROR/Telegram;
broker-holds-fewer still errors; the account-lookup fires on the warning path.
