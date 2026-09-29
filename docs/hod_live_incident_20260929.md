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
