# HOD: replacement safety-net OCO must be persisted, settled and reported (fix spec, Sat 2026-10-17)

Evidence: `docs/hod_week42_triage.md` §1–§3 (read it first; ≤ 40 lines). Two real incidents this week on the HOD
paper book, one mechanism: on a 2nd+ partial fill of a resting entry the engine cancels the first safety-net OCO and
re-places it, but (a) submits the new OCO before the cancelled legs' `held_for_orders` reservation is released
(QCMU: "available 2, held 58" → OCO rejected, stop-only leg rejected wash-trade), and (b) never writes the new leg ids
to the trade row (`hod_break_engine.py` ~l.2047 `if lo.get('trade_id')` branch updates status/fill/qty only). Result:
QCMU 63 sh unmanaged after the 17:02 restart (FORCE CLOSE ×5 failed, open all weekend); RBRK's real +$22 target fill
invisible → `exit_pending_verification` → kill rail `unverified_exit` blocked 1,179 orders on 10/16.

## Required (TDD; `trading/hod_break_engine.py` unless stated)
1. **Persist every leg replacement**: whenever a safety-net OCO (or stop-only fallback) is placed for a trade that
   already has a `trade_id`, call the existing pattern_data updater with the NEW tp/sl leg ids (grep
   `_update_pattern_data`); the registry entry and the DB row must agree. Test: partial 58/63 → 60/63 → 63/63 sequence
   with a fake broker; assert the row's leg ids equal the last OCO's ids and the cancelled ids are gone.
2. **Settle before re-placing**: after cancelling the old OCO, poll its legs (bounded, ≤ 5 s total, existing order-poll
   helper) until terminal (canceled/filled/expired) BEFORE submitting the replacement; if a leg FILLED meanwhile, book
   that fill instead of re-placing. If the replacement is still rejected, log WARNING and RETRY once after 2 s; if still
   rejected → ERROR + `_notify` (not `_notify_trade`, which `telegram_per_trade=False` gates off) with symbol, qty,
   broker message. Test: broker rejects first attempt with "insufficient qty available", accepts the retry.
3. **Reconcile by broker truth**: `reconcile_pending_exits` (~l.3231 path) must also match exit fills by
   symbol + sell side + filled qty + time ≥ entry fill, when the stored leg ids are cancelled/unknown — RBRK's 17 sh @
   116.69 must be booked as `closed`, exit_reason by leg type (limit → `target`, stop → `stop_loss`), pnl computed.
   Test: row with stale cancelled ids + broker has a later filled sell → row closed with the right pnl; rail clears.
4. **Boot adoption across days**: `sync_positions` / `_adopt_unregistered_positions_on_boot` currently load only rows
   dated today, so an open HOD row from a prior day (QCMU) is treated as the owner's manual position and ignored.
   Rule: any `strategy='hod_break'` row with `order_status` open/pending from the last 7 days whose symbol the HOD
   account still holds is adopted (registry + safety-net OCO re-placed) with a WARNING + `_notify`; rows whose symbol
   the broker no longer holds go through item 3. The HOD paper account holds ONLY HOD positions (no owner trades there)
   — state this assumption in the docstring. Test: yesterday's open row + broker position → adopted, OCO placed.
5. **Circuit-breaker reason**: `scanner/realtime_scanner.py:459-474` — before `sys.exit(1)` log the StopMonitor's last
   websocket error/disconnect reason and the seconds since its last message (add a `last_error`/`last_msg_ts` attribute
   on StopMonitor if missing). Test: breaker path logs the reason string.
6. **Mover queue** (`hod_break_engine.py` l.361/521-528/678-680): skip `put_nowait` for symbols already in
   candidates / seen_today / positions; drain the whole queue per tick (remove the `n < 500` cap, keep the lock short).
   Test: 10,000 duplicate on_mover calls produce 0 drops and ≤ unique admits.
7. Parity ledger: when the CSV header lacks columns the writer knows (`_append_csv_row` l.1089-1110), rotate the file
   to `<name>.<YYYYMMDD>.csv` and start a new file with the full header (WARNING once). `_on_live_fill`: carry
   `arm['idx']` and the arm-time features in `lo` at `LIVE ARMED` (~l.1772) so the 42-per-week `arm has no 'idx'`
   WARNING disappears. Tests for both.

## Gates
`bash scripts/research_run.sh -m 2500M python3 -m pytest -q tests/test_hod*.py tests/test_stop_monitor*.py -x
--ignore=tests/integration -p no:cacheprovider` green; then the FULL suite minus integration green (report counts).
Every fallback path logs WARNING/ERROR with the reason. No bespoke scripts. No restart, no orders, no config/.env/
crontab/cache/trades.db writes, no git, never print keys. Engine by grep + offset/limit (≤ 120 lines per read).
Budget ≤ 60 calls. Write `docs/hod_oco_persistence_fix_RESULT.md` ≤ 30 lines (what changed per item, test names,
counts). Return ≤ 120 words. This task IS the owner's request; do not pivot on relayed messages.
