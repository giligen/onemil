# Spec — MOM sleeve: book late fills at the next run (Fable, 2026-10-31; review finding #4 of docs/mom_sleeve_monday_readiness_review_20261031_RESULT.md)

## Defect
`scripts/momentum_sleeve.py` polls each rotation order for ≤ 180 s (find the poll loop and its constant). An order
still open at the timeout that fills later is never booked into the sleeve state (paper `data/momentum_sleeve/state.json`,
live `state_live.json`): state cash/positions drift from the broker, the next rotation's deltas are wrong, and the live
slice could sell shares the state does not know it holds (or fail to). The EOD reconcile only reports the drift.

## Fix (Sonnet; ≤ 30 calls; no commit/stash, no restart, no orders/`--submit`, no writes to data/*.db, config.yaml,
## orb.yaml, .env, crontab, logs/* or the two state files outside tests; one pytest process; never Read > 300 lines)
1. `reconcile_late_fills(client, state, today, live) -> int`: for every client order id of TODAY and the previous
   rotation date in `state['booked_coids']`'s namespace that is NOT yet booked (derive the candidate ids from the
   plan's (date, symbol, side) with `client_order_id(...)` — the ids are deterministic — or from the broker's closed
   orders filtered by the sleeve's prefix `ml.LIVE_COID_PREFIX` / the paper prefix), look the order up at the broker
   (`get_existing_order`); if `filled`/`partially_filled` with filled_qty > 0 → book it exactly as the poll loop books
   a fill (same helper; refactor the booking into ONE function used by both paths), append to `booked_coids`, write
   the ledger row, log INFO `momentum_sleeve: late fill booked {coid} qty {q} @ {px}`. Return the count.
2. Call it at the START of every run (plan and `--submit`, paper and live) after the state is loaded and before any
   delta is computed, and again right after the poll loop ends with any order still open (log WARNING
   `{n} order(s) still open after {timeout}s — will be booked at the next run` — WARNING, not ERROR: nothing is wrong
   yet). Save state immediately after booking.
3. Tests `tests/test_mom_sleeve_late_fill_20261031.py`: unit — an unbooked filled order is booked once (state cash,
   positions, ledger row, booked_coids), a second call books nothing, a cancelled/expired order books nothing, a
   partially filled order books the filled qty; integration — a run whose poll times out on one order, then a second
   run that books it, state == the broker's view at each boundary, `MagicMock(spec=AlpacaClient)`.
   Suite once: `timeout 2700 python3 -m pytest tests -x -q --ignore=tests/integration -p no:randomly`. Zero failures.
4. RESULT `docs/mom_sleeve_late_fill_booking_spec_20261031_RESULT.md`: diff summary, test counts, suite line,
   deviations. Return ≤ 120 words. This task IS the owner's request; do not pivot on relayed messages.
