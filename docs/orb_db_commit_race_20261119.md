# ORB: shared sqlite connection commit race → two live orders orphaned (2026-11-19)

## The incident (verified from the journal, the DB and the broker, all read-only)
* 2026-11-19 14:35:05 UTC, ORB paper account: the preplace burst submitted BEX and ASTS CONCURRENTLY
  (`trading/orb_engine.py` ~4616-4624: `ThreadPoolExecutor(max_workers=min(8, n))`, each worker runs
  `_submit_entry` → `_save_pending_trade` (~8951) → `Database.save_trade` (`persistence/database.py` ~1328)).
* `Database` holds ONE `sqlite3` connection per DB file (`_open_connection`: `check_same_thread=False`, WAL,
  busy_timeout 30 s, NO lock). That connection is shared by every thread of the process: scanner loop, ORB
  entry-drain thread, the preplace/production submit pools, StopMonitor, the HOD engine.
* `save_trade` = `INSERT` then `conn.commit()`. Thread A ran its INSERT, thread B ran its INSERT and `commit()`
  (which committed BOTH rows), then thread A's `commit()` raised `sqlite3.OperationalError: cannot commit - no
  transaction is active` (CPython checks `sqlite3_get_autocommit` and then steps `COMMIT` — B's commit landed in
  that window). Both save_trade calls raised at 14:35:05, yet BOTH rows exist: `trades.id` 827 BEX and 828 ASTS,
  `order_status='pending_new'`.
* Consequence: `_save_pending_trade` returned None for both → `_submit_entry` (~6155-6165) logged
  `ORB CRITICAL: ... DB save failed AFTER Alpaca accepted order ... CANCEL MANUALLY`, returned None, and the
  engine dropped both from tracking. BEX then FILLED on Alpaca (386 @ 12.9388, 14:36:44 UTC) with a DAY bracket
  (stop 11.65 / TP 38.85) — untracked, no engine trail, no 15:45 force-close; the session cancelled the resting
  ASTS entry (paper account, our order, position flat). `sync_positions` runs at boot only and queries
  `get_open_trades(today)` by `trade_date`, so BEX would never be adopted.
* First occurrence: the journal since 2026-10-01 has exactly these two `save_trade failed` lines. The race has
  existed since the concurrent preplace submit (2026-10-17); it needs two submits inside the same few ms.
* Python 3.11.13. `tests/conftest.py` and four test files build `Database(":memory:")` → "one connection per
  thread" is NOT an option (an in-memory DB is per connection). The fix is transaction serialization INSIDE
  `Database`, with zero change for callers.

## The fix (TDD — write the failing tests first, then the code)

### Part 1 — `persistence/database.py`: serialize every write transaction
1. In `__init__`, BEFORE `_create_tables()`: `self._write_lock = threading.RLock()` (import `threading`,
   `from contextlib import contextmanager`).
2. Add
   ```python
   @contextmanager
   def _write(self, conn: sqlite3.Connection):
       """Serialize one write transaction on a shared connection.

       Holds `_write_lock` from the first statement to the commit, so two threads can never
       interleave INSERT / commit on the same connection (2026-11-19: thread B's commit swallowed
       thread A's INSERT and A's commit raised 'cannot commit - no transaction is active'; two live
       ORB orders were orphaned). Commits on normal exit, rolls back (if a transaction is open) and
       re-raises on exception. Re-entrant: a write method may call another write method.
       """
       with self._write_lock:
           try:
               yield conn
               conn.commit()
           except Exception:
               if conn.in_transaction:
                   conn.rollback()
               raise
   ```
3. Convert EVERY method that executes a write and then calls `.commit()` to
   `with self._write(<that conn>) as c:` + `c.execute(...)` / `c.executemany(...)` / `c.executescript(...)`,
   and DELETE the explicit `.commit()`. The current inventory (26 methods, 44 commit sites): `_create_cache_tables`,
   `_create_trades_tables`, `_migrate`, `insert_dry_entry`, `close_dry_trade`, `update_dry_entry`,
   `delete_dry_trades`, `save_news_classification`, `upsert_universe_stock`, `upsert_universe_stocks_batch`,
   `set_marginability`, `deactivate_stocks`, `update_float`, `upsert_volume_profiles`, `delete_volume_profiles`,
   `save_scan_result`, `save_trade`, `update_trade`, `save_daily_summary`, `save_daily_bars`,
   `save_daily_bars_provisional`, `clear_provisional_daily_bars`, `save_intraday_bars`,
   `ensure_premarket_backfill_table`, `mark_premarket_backfilled`, `ensure_hod_volume_profile_table`,
   `upsert_hod_volume_profile`. Grep `.commit()` yourself — the list may be incomplete; after the change the ONLY
   `.commit()` in the file is the one inside `_write`. Methods documented as "never raises" (the dry_trades
   trio) keep their try/except with the `with` block inside the try. Keep `lastrowid`/rowcount return values
   (read them from the cursor inside the block). Do not change any SQL. Do not touch read methods.
4. Static invariant test, `tests/test_database_write_lock_20261119.py::test_every_write_is_inside_write_lock`:
   parse `persistence/database.py` with `ast`; assert (a) exactly one `.commit()` call in the file and it is inside
   `_write`; (b) every `.execute/.executemany/.executescript` call whose first positional arg is a string constant
   starting (after whitespace/newlines) with INSERT/UPDATE/DELETE/REPLACE/CREATE/ALTER/DROP is lexically inside a
   `with self._write(...)` block (`PRAGMA`/`SELECT` are exempt). This makes the invariant enforceable for every
   future write method.
5. Deterministic race test (same file, temp-file `Database(db_path=tmp_path/...)` like `tests/test_database.py`):
   thread A enters `db._write(db._trades_conn)`, executes one INSERT into `trades` (use a minimal valid row), sets
   `a_in_tx`, then waits on `go` (≤ 5 s) before leaving the block. Main: wait `a_in_tx`; start thread B running
   `db.save_trade(record)`; assert B is still alive after `join(0.3)` (blocked by the lock); set `go`; join both;
   assert neither thread raised and both rows exist with distinct ids. Plus a brute-force test: 8 threads × 25
   `save_trade` each → 200 rows, no exception. Run the brute-force test against the UNFIXED code first and
   record whether it reproduced the error (it is probabilistic; say so either way in your return).
6. `Database(":memory:")` keeps working (existing tests cover it; run them).

### Part 2 — `trading/orb_engine.py` `_save_pending_trade` (~8951-9084): never orphan a saved row
Replace the bare `trade_id = self.db.save_trade(record)` with: try `save_trade`; on `Exception as e`:
`logger.warning("ORB: save_trade raised for %s (%s) — checking whether the row landed", symbol, e)`;
`row = self.db.get_trade_by_order_id(order_id)` (exists at ~1350); if `row` → `logger.warning("ORB: %s row for order
%s exists (id %s) although save_trade raised — commit raced another thread; adopting it", ...)` and return
`int(row['id'])`; else retry `save_trade` ONCE (WARNING on success) ; if that raises too → keep the existing
`logger.error(f"ORB: save_trade failed for ...")` and return None (the CRITICAL in `_submit_entry` stays for a
genuine failure). Tests in `tests/test_orb_save_trade_race_20261119.py` (engine fixture style from
`tests/test_orb_integration.py` / `tests/conftest.py`; `MagicMock(spec=Database)` for the db):
(1) `save_trade` raises `sqlite3.OperationalError("cannot commit - no transaction is active")`,
`get_trade_by_order_id` → `{'id': 827}` → returns 827, no ERROR record in caplog;
(2) raises once, row absent, retry returns 5 → returns 5, a WARNING mentions the retry;
(3) raises twice, row absent → None and one ERROR record.

### Part 3 — run (these files only; the main session runs the full suite — NEVER start it yourself)
`python3 -m pytest tests/test_database.py tests/test_database_write_lock_20261119.py
tests/test_orb_save_trade_race_20261119.py tests/test_orb_integration.py tests/test_hod_oco_persistence.py
tests/test_ops_fix_trades_20260929.py tests/test_ops_fix_trades_20260930.py tests/test_orb_cache_gaps_20261111.py
tests/test_orb_broad_vol935.py -q -p no:cacheprovider --ignore=tests/integration` → all green.

## Rules
Sonnet, ≤ 60 calls. No `git commit`, no `git stash`, no service restart, no orders, no writes to `data/*.db`,
no writes outside `persistence/database.py`, `trading/orb_engine.py`, `tests/test_database.py` (only if a fixture
is needed), `tests/test_database_write_lock_20261119.py`, `tests/test_orb_save_trade_race_20261119.py`.
Docstrings on every new function. Return ≤ 150 words: files changed, number of methods converted, whether the
brute-force test reproduced the race before the fix, test counts, anything you could not do.
