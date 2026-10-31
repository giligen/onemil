# RESULT — MOM sleeve late-fill booking (spec mom_sleeve_late_fill_booking_spec_20261031.md)

## Diff summary (scripts/momentum_sleeve.py, tests/test_mom_sleeve_late_fill_20261031.py)
* `book_fills(state, fills, live, final=True)` — the ONE booking function (live: `ml.apply_fills`; paper: marker only,
  the broker sync owns cash/positions; always appends to `booked_coids` unless `final=False` for a still-working partial).
  Used by `execute_live`'s phase booking and by the reconcile.
* `write_ledger_rows(...)` — ledger writer extracted from the two duplicated blocks (paper path, `submit_live`) and reused
  for late fills.
* `record_pending` / `settle_open_orders` — orders the poll gave up on go to `state['pending_coids']`
  ({coid: symbol, side, date}); one immediate re-check right after the poll; WARNING
  `{n} order(s) still open after {timeout}s — will be booked at the next run`. `poll_fills`' per-order message is now WARNING.
* `reconcile_late_fills(client, state, today, live) -> int` — looks each pending id up (`get_existing_order`), books
  filled/partially-filled qty once (partials: the increment later), writes the ledger row, INFO
  `late fill booked {coid} qty {q} @ {px}`, saves state right after. Called at the start of every `run` (after state load;
  paper after the broker sync; before the pause check and any delta) and inside `execute`/`execute_live`.

## Tests
12 new (9 unit incl. paper, partial, sell, lookup failure; 2 integration: poll timeout then next run books it with state ==
broker view at each boundary; fill landing during the immediate re-check). Suite (once, `timeout 2700 python3 -m pytest tests -x -q --ignore=tests/integration -p no:randomly`): 5453 passed, 6 skipped, 0 failed (a first run caught that `write_ledger_rows` skipped `official_opens` on empty fills; restored).

## Deviations
1. Candidates come from `state['pending_coids']` (recorded when a poll times out), not derived from the plan or listed
   from the broker: no plan exists at the start of a run, and the broker listing is not part of the test doubles.
2. A cancelled/expired order that had a partial fill (filled_qty > 0) is booked for that qty (otherwise the partial is
   lost); one with filled_qty 0 books nothing.
3. Partial fill of a still-working order is booked at once and tracked (`booked_qty/booked_notional`), so a later call books
   only the increment (spec only said "books the filled qty").
4. Dry runs (no `--submit`) also reconcile (spec: every run).
