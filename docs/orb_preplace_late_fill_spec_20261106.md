# ORB preplace: a cancelled provisional order that fills late leaves naked, unledgered shares — spec 2026-11-06

## Incident (paper ORB key, 2026-10-27 CRDU)
09:35:05 provisional stop-limit buy 906 sh @5.51 (order 9debf1bf, trades row 618). 09:35:13 reconcile: CRDU's final
trigger moved 5.50→5.54 → `_cancel_symbol_open_orders` (fire-and-forget) + `del self.open_positions[sym]` + a NEW
entry 599 sh @5.56 (order 4918c7bf, row 621) — log `[ORB PREPLACE] CRDU REPLACED`. The broker queued the cancel ~9 s
(paper OMS, `docs/orb_live_endpoint_gap.md`): 9debf1bf FILLED 861 sh at 09:35:20–23 and was cancelled at 09:35:23
with filled_qty 861. The engine never polled the old order again, so: ledger 599 sh, broker 1,460 sh; the stop leg
covered 599; 861 shares were NAKED until the 15:45 force close sold the broker qty (exit_coverage treats shares above
the watch as "not ours"). It happened to be +$600 — accidental behaviour, unacceptable even when profitable.
`_cancel_preplaced_order` (the "fell outside top-N" path, orb_engine.py ~4586) reads `filled_qty` ONCE immediately
after the cancel — same race, same hole.

## Fix (root cause: act on the cancel's RESULT, not its request)
1. New helper `OrbEngine._await_terminal(order_id, max_wait_s=15.0, step_s=0.5) -> (status, filled_qty)`: poll
   `self.alpaca.get_order` until status ∈ {canceled, filled, expired, rejected, done_for_day, replaced} or the wait
   is exhausted; on exhaustion log WARNING and return the last seen status/qty. Dict-or-object safe like the existing
   read. One shared helper, used by BOTH paths below.
2. Reconcile REPLACED path (~4540–4580): after the cancel, call `_await_terminal` on the provisional order BEFORE
   `_submit_entry`. If `filled_qty > 0`: do NOT submit the replacement; keep the provisional position (do not delete it
   from `open_positions`), run it through the existing `_confirm_fill` path with the filled qty so row 618-style rows
   get fill_price/shares/filled_qty and the StopMonitor watch + stop legs are sized to the filled qty; log WARNING
   `[ORB PREPLACE] {sym}: provisional order filled {q} sh before the cancel landed — kept as the entry, no replacement`.
   If the broker is still non-terminal after `max_wait_s`: log ERROR (Telegram) and do NOT submit a replacement that
   day for that symbol (a second order against an unknown first is the hole). Wait for all cancelled symbols of one
   reconcile together (one loop over the set, 15 s bound total), so the burst is not serialised.
3. `_cancel_preplaced_order`: replace the one-shot read with `_await_terminal`; existing `filled_qty > 0` branch stays.
4. Safety net in `trading/exit_coverage.py`: when `excess_qty > 0` on a symbol that has an ORB trades row TODAY on
   this account, log ERROR `[ORB] {sym}: broker holds {excess} sh above the ledger on a symbol ORB traded today —
   unledgered fill` (once per symbol-day). Owner/third-party shares on OTHER symbols stay WARNING-free as now.

## Tests (unit + integration, both required; `--ignore=tests/integration`; `MagicMock(spec=...)`)
- unit `tests/test_orb_preplace_late_fill.py`: (a) replace path, get_order returns pending → canceled with
  filled_qty 861 on the 3rd poll → no `_submit_entry`, `_confirm_fill` called with the provisional pos, WARNING;
  (b) canceled with filled_qty 0 → replacement submitted as today; (c) never terminal → ERROR, no replacement;
  (d) `_cancel_preplaced_order` with a late partial → returns False, row not marked; (e) `time.sleep` patched, total
  polls bounded. (f) exit_coverage excess on a symbol with a today's ORB row → one ERROR; on a foreign symbol → none.
- integration: the existing preplace reconcile test with a real `Database` tempdir: after the late fill the trades
  row carries shares == filled_qty == 861 and one row per symbol-day is live.
- Run the new file + `tests/test_orb_preplace*.py` + `tests/test_exit_coverage*.py` + `tests/test_orb_engine*.py`;
  zero failures.

## Rules
No git commit, no service restart, no orders, no DB/config/log writes. Budget ≤ 40 calls. RESULT →
`docs/orb_preplace_late_fill_spec_20261106_RESULT.md` (≤ 150 words); return ≤ 100 words.
