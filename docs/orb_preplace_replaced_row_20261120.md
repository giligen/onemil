# ORB preplace: the REPLACED branch leaves the provisional trades row at `pending_new` (2026-11-20)

## The incident (verified read-only: journal 14:35 UTC, data/trades.db rows 842-847, ORB paper account GET)
* 09:35:05 ET the preplace scheduler submitted the provisional top-4: SPCQ (row 842), SNK (843), SSPC (845), CAPR (844).
* 09:35:06 `_reconcile_preplaced` (trading/orb_engine.py ~4696-4860): final top-8 = SPCG, SNK, BGS, UNHG, SPCQ, GAP, SSPC, EDZ.
  `PARITY n_preplaced=4 n_kept=0 n_replaced=2 n_cancelled=2 n_added=1`.
  - CAPR: not in the final set → `_cancel_preplaced_order` (~4911) cancelled the order AND marked row 844
    `preplace_canceled / preplace_reconcile`. Correct.
  - SNK, SSPC: in the final set with a moved range high → the pre-pass (~4744-4760) cancelled the provisional orders,
    the REPLACED branch (~4802-4850) deleted `open_positions[sym]` and submitted new production orders (rows 846/847,
    both filled, both stopped out, DB correct). Rows 843 and 845 were NEVER marked: still `pending_new` tonight while
    the broker shows both provisional orders `canceled`, filled 0.
  - SPCQ: same branch; the replacement was skipped ("breakout extended past limit … Not chasing" → `_submit_entry`
    returned None → `n_cancelled += 1`). Row 842 also stays `pending_new`; broker `canceled`, filled 0.
* Root cause: only the not-in-final branch marks the provisional row. The REPLACED branch (replacement submitted,
  PlannerReject / no buying power, or `_submit_entry` → None) never writes the row's terminal status. Every reconcile
  that moves a range high therefore leaves one dead `pending_new` row per replaced name (see also
  docs/orb_preplace_cancel_db_status_20261020 tests and the 10/30 "preplace writes two ledger rows" incident).

## The fix (TDD: failing tests first). trading/orb_engine.py only; no selection, sizing or order-path change.
1. Add `_mark_preplace_row_canceled(self, pos, reason: str) -> None`: `self.db.update_trade(pos.trade_id,
   {'order_status': 'preplace_canceled', 'exit_reason': reason})` wrapped in try/except → `logger.error` on failure
   (never raises into the reconcile). Make `_cancel_preplaced_order` call it with `'preplace_reconcile'` (behaviour
   unchanged, one write site).
2. In the REPLACED branch, right after the `_status not in self._TERMINAL_ORDER_STATUSES` guard (i.e. once the
   provisional order is confirmed terminal with `_late_qty == 0`) and BEFORE `del self.open_positions[sym]`:
   `if pos is not None and pos.trade_id: self._mark_preplace_row_canceled(pos, 'preplace_replaced')`.
   This covers all three outcomes of that branch (replacement submitted → new row; PlannerReject / no buying power;
   `_submit_entry` returned None) because the provisional order is gone in every one of them.
   A provisional order with `pos.order_id == ''` (already filled, `is_filled`) never reaches this branch — unchanged.
3. The `exit_reason` values `preplace_reconcile` (dropped from the final set) and `preplace_replaced` (range moved) let
   scripts/eod_sections.py `preplace_dropped` keep working (it reads `preplace_failed` / cancelled rows — read it and
   state in the Result whether `preplace_replaced` rows need to be excluded anywhere; do not change the reports unless a
   test proves a count changes).

## Tests — `tests/test_orb_preplace_replaced_row_20261120.py` (fixtures and the reconcile harness from
`tests/test_orb_preplace_cancel_db_status_20261020.py` and `tests/test_orb_preplace_late_fill.py` — reuse, do not copy)
1. SNK replay: provisional preplaced at 15.87, final range high 16.03, cancel → terminal `canceled` filled 0,
   replacement `_submit_entry` → a new order id → `db.update_trade` called once for the OLD trade_id with
   `{'order_status': 'preplace_canceled', 'exit_reason': 'preplace_replaced'}`; `n_replaced == 1`.
2. SPCQ replay: same but `_submit_entry` → None → old row marked `preplace_replaced`; `n_cancelled == 1`.
3. PlannerReject → old row marked; `n_cancelled == 1`.
4. Late fill (`_late_qty > 0`) → NOT marked (the fill path owns it; existing behaviour pinned).
5. Not terminal after the wait → NOT marked, the existing ERROR, no replacement (pinned).
6. CAPR path (not in final) still marks `preplace_reconcile` through the shared helper (pinned).
7. `update_trade` raising → one ERROR, reconcile continues, replacement still submitted.
8. Integration with a real temp `Database` (as `tests/test_orb_preplace_cancel_db_status_20261020.py` does): after the
   reconcile, `get_trade_by_order_id(provisional id)` has `order_status == 'preplace_canceled'` and the new row exists.

## Run (these files only; the main session runs the full suite — NEVER start it yourself)
`python3 -m pytest tests/test_orb_preplace_replaced_row_20261120.py tests/test_orb_preplace_cancel_db_status_20261020.py
tests/test_orb_preplace_late_fill.py tests/test_orb_preplace_budget_20261006.py tests/test_orb_preplace_at_close.py
tests/test_eod_sections.py -q -p no:cacheprovider --ignore=tests/integration` → all green.

## Rules
Sonnet, ≤ 50 calls. No `git commit`/`git stash`, no service restart (it is stopped for the night — leave it), no orders,
no writes to `data/*.db`, `logs/*`, `orb.yaml`, `config.yaml`, `.env`. Files you may edit: `trading/orb_engine.py`
(the helper + the REPLACED branch + `_cancel_preplaced_order` only), the new test file, and a "Result" section appended
to THIS file. Docstrings on the new function. Return ≤ 120 words: files changed, test counts, the eod_sections answer,
anything you could not do.

## Result (2026-11-20)
* `trading/orb_engine.py`: new `_mark_preplace_row_canceled(pos, reason)` (try/except -> logger.error); `_cancel_preplaced_order` uses it (`preplace_reconcile`); the REPLACED branch marks `preplace_replaced` after the terminal/no-fill guard, before `del open_positions[sym]`.
* Tests: 8 new in `tests/test_orb_preplace_replaced_row_20261120.py` (5 failed before the fix, 3 pinned behaviour passed); spec Run set 106 -> 114 passed. The SNK replay uses DFDV (a name in the shared fixture's final set).
* `scripts/eod_sections.py` needs no change: `preplace_dropped` (line 604-608) is `(preplace_failed | preplace_canceled rows) - final_ranked`; replaced names are in `final_ranked`, so they are subtracted. A replaced row is also in `CANCELLED_STATUSES` (line 585), so it is not counted as traded. No test proves a count change.
