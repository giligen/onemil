# Telegram ERROR noise fix — spec (Fable, 2026-10-23)

Owner rule: **no harmless ERRORs or alarms on Telegram.** Every `logger.error`/`critical` in the trader goes to Telegram
through `monitoring/telegram_error_handler.py`. Between 10/22 and 10/23 eight such lines went out; every one was either
a false alarm or a self-healing race. No money was lost (CELC +$17, PHM +$40, SNDQ −$19 as booked, TXG not ours).
This spec fixes the four causes. Scope is ONLY what is written here. Agents never commit, never restart the service,
never submit orders, never write cache.db / config.yaml / orb.yaml / .env / crontab. Never run two pytest suites at once.
`tests/integration` is a REAL paper account — always `--ignore=tests/integration`.

## 1. Orphan reconciler: cross-account false "SUSPECT OVER-EXIT" (`trading/orphan_reconciler.py`)
Evidence: 10/22 12:31:30 and 10/23 12:32:01 (every boot) — `[HOD] SUSPECT OVER-EXIT: TXG (+7 sh @ $69.6750) is unowned
now but hod_break traded it within 7d`. The scan at `trading/trading_engine.py:5073` runs on the LIVE account (shared
with the owner). `_recent_any_strategy_trade` (l.186) looks at ANY strategy in the trades table, but `hod_break` trades
on the HOD paper account (PA39) and `orb` on the ORB paper account — their rows can never explain a live-account
position. The owner's own TXG position was therefore named, by symbol, on Telegram at every boot (owner positions are
report-only and never listed by symbol). The HOD probe confirms PA39 holds no TXG and HOD's 10/21 TXG row (id 578,
19 sh bought 74.99, sold 19 at 75.72) is fully closed.
Fix (mechanism, not a special case):
* `reconcile_strategy_orphans(...)` gets a new keyword `peer_strategies: Optional[Set[str]] = None` — the strategies
  that trade on the SAME broker account as the one being scanned. `_recent_any_strategy_trade(db, symbol,
  lookback_start, peer_strategies)` restricts the lookup to those strategies when the set is given (SQL `strategy IN`
  on the legacy path; filter the rows on the public-API path). `None` keeps today's behaviour (tests that rely on it).
* Each caller passes its peers. Derive them from the engines' broker ACCOUNT NUMBERS, never from strategy names:
  find how each engine (TradingEngine, HODBreakEngine, ORBEngine) already learns its account number at boot (grep
  `account_number`); if an engine has none cached, add a cached read-only `account_number` property (one REST call,
  cached, exceptions → None and a WARNING). `peer_strategies` for a caller = {its own strategy} ∪ {every other engine's
  strategy whose account_number equals the caller's and is not None}. The wiring point is wherever the engines are all
  known (main.py `run_scan` or the TradingEngine that owns the sub-engines — find it; do not invent a registry if one
  exists). When the account number of a peer is unknown (None) it is NOT a peer (fails toward the OLD "foreign" read,
  never toward an alarm — same direction as the existing docstring).
* The alert text for a true suspect must keep naming the symbol (it is OUR over-exit); nothing else changes.
Tests: (a) unit — a live-account scan with a hod_break row in the window and `peer_strategies={'bull_flag'}` classifies
the position `foreign` (no error log); with `peer_strategies=None` it is `suspect_over_exit` (old behaviour); with
`peer_strategies={'bull_flag','hod_break'}` it is `suspect_over_exit`. (b) integration — build the peer set from two
engine doubles with `spec=` whose `account_number` differ / match and assert the sets. Put them in
`tests/test_orphan_reconciler_peer_strategies_20261023.py`.

## 2. HOD force close: leg cancel still propagating → ERROR on attempt 1 (`trading/hod_break_engine.py`)
Evidence: 10/22 19:55:53–58 CELC and PHM: "a leg is still not terminal after 3s — selling only the N shares no leg has
sold" → Alpaca 40310000 `insufficient qty available … held_for_orders` → `FORCE CLOSE … FAILED` ERROR (+ `_notify`),
attempt 2 at 19:56:42 filled both (eod +$17.42 / +$39.61). Cause: `FC_LEG_POLL_S = 3.0` (l.2994) is shorter than the
paper OMS cancel propagation (5–7 s, `docs` of 10/17: paper queues ~7 s created→submitted; live is 0.01 s).
Fix: `FC_LEG_POLL_S = 10.0` (live cancels in ms, so this costs nothing live). In `force_close_all`'s `except` (l.3179
area): if the exception text contains `40310000` or `insufficient qty available` AND `pos.fc_attempts == 1` →
`logger.warning(f"{tag} FORCE CLOSE {sym}: shares still held by a cancelling leg (attempt 1) — retrying in
{FC_RESUBMIT_S:.0f}s")` and NO `_notify`; every other exception, and the same error on attempt ≥ 2, stays ERROR +
`_notify` exactly as today. Make sure the retry really happens on the next pass (it did on 10/22 — keep that path).
Tests in `tests/test_hod_force_close_held_qty_20261023.py`: attempt-1 held-qty → WARNING, no notify, position still open,
fc_attempts 1 → next pass resubmits; attempt-2 held-qty → ERROR + notify; a different exception on attempt 1 → ERROR.

## 3. StopMonitor escalation: held-for-orders retry window too short (`trading/stop_monitor.py`)
Evidence: 10/22 16:00:18–21 SNDQ: SL leg replaced 16:00:07, unfilled 10 s, escalate → `close position` 40310000 ×4 in
3 s (`_HELD_QTY_RETRY_BACKOFFS_S`, l.404) → ERROR "market close failed after retry" + CRITICAL "placed EMERGENCY
stop-market" (the emergency order then succeeded one second later: the cancel had just propagated). Booked
stop_loss_unconfirmed @ 18.32 −$18.93: correct, the two Telegram lines were noise.
Fix: extend `_HELD_QTY_RETRY_BACKOFFS_S` so the retries span ≥ 10 s (e.g. `(0.5, 1.0, 2.0, 3.0, 4.0)`); keep the ERROR +
CRITICAL for a TRUE exhaustion. Find the existing unit test for the backoff list and update its expectation; add one
asserting the sum ≥ 10 s.

## 4. ORB open-tick budget: first exhausted tick is not an ERROR (`trading/orb_engine.py` l.1420, `data_sources/alpaca_client.py` l.198)
Evidence: 10/23 13:31:27 `API call timed out after 15.3s + 0 retry: get_1min_bars_range_multi(155 symbols …)` ERROR,
then `ORB: open-tick budget (20.0s) exhausted — PARTIAL universe … 2597 of 2597 … deferred` ERROR. 13:33:09 the next
tick admitted 34 candidates; preplace and the 09:35:08 SK order ran normally. Self-healed in 100 s.
Fix: (a) the budget-exhausted line is `WARNING` on the first exhausted tick of the session and `ERROR` only when it is
the SECOND consecutive exhausted tick or the tick ends at/after 09:34:00 ET (the preplace needs the universe by
09:35:05). Keep a per-session counter on the engine, reset at session start. (b) In `alpaca_client` the timeout log
is `WARNING` when the caller allowed retries and this was not the last attempt, and `ERROR` only on the final attempt —
read how `+ 0 retry` is produced and honour it; do not change retry counts anywhere.
Tests in `tests/test_orb_open_tick_budget_level_20261023.py`: first exhaustion → WARNING; second consecutive → ERROR;
first exhaustion at 09:34:10 ET → ERROR; a successful tick resets the counter.

## Validation (one suite at a time, never two)
`python3 -m pytest tests/test_orphan_reconciler*.py tests/test_hod_force_close*.py tests/test_hod_oco_persistence.py
tests/test_stop_monitor*.py tests/test_orb_open_tick*.py tests/test_orb_preplace*.py -q -x --no-header -p no:randomly
--ignore=tests/integration` must be green, then the full `tests/` (same flags, `--ignore=tests/integration`) must be
zero failures. Write `docs/telegram_error_noise_fix_20261023_RESULT.md` (≤ 40 lines: files changed, test counts, the
account-number source you used for the peer set). Return ≤ 150 words. Do NOT commit.
