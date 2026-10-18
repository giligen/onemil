# HOD: `_reconcile_by_broker_sells` must not use the DB `filled_at` as the broker `after=` filter (fix spec, Sun 2026-10-18)

Found in the Sunday boot rehearsal on `92f5116`: `[HOD] reconcile RBRK (2026-10-15): 0/17 shares accounted for` although
the HOD paper account holds a CLOSED `sell limit` order `903ef94b` filled 17 sh @ 116.69 at 18:11:51 UTC — the exact
case `_reconcile_by_broker_sells` (`trading/hod_break_engine.py` ~l.3262) was written for.

Root cause (probed read-only, `scratchpad/rbrk_probe2.py`): Alpaca's `GetOrdersRequest(after=...)` filters on the
order's **submitted_at**, not on filled_at. The row's `filled_at` is the ENGINE's booking time of the entry fill
(16:50:38.151), which comes AFTER the replacement OCO was submitted (16:50:38.100, 50 ms earlier) because the engine
re-places the safety net before it books the fill. `after=since` → the broker returns `[]`; `after=since-60s` → all four
sells. The in-code filter `fts < since` (filled_at of the SELL ≥ since) is the one that carries the meaning.

## Required
1. In `_reconcile_by_broker_sells`: the broker query uses `after = since - timedelta(minutes=30)` (a wide net — the
   candidates are then filtered in code); the in-code filter keeps `filled_at(sell) >= since - 120 s` (tolerance for a
   fast stop-out between the real fill and the engine's booking of it). Log at DEBUG the number of orders returned and
   the number of sell candidates. Document both windows in the docstring with this incident.
2. Tests in `tests/test_hod_oco_persistence.py` (existing fixtures, `MagicMock(spec=...)`):
   (a) a sell order SUBMITTED 50 ms before `since` and FILLED 81 min after it is booked (17 sh, reason `target`, row
   closed) — assert the mock's `get_orders` received `after <= since - 30 min`; (b) a sell FILLED 10 minutes before
   `since` is NOT booked; (c) a sell filled 60 s before `since` (inside the tolerance) IS booked.
3. Run `bash scripts/research_run.sh -m 2500M python3 -m pytest -q tests/test_hod_oco_persistence.py tests/test_hod*.py
   --ignore=tests/integration -p no:cacheprovider` — all green. **Before running pytest: `pgrep -af "pytest"` — wait
   until no other suite runs (2-CPU node). Never run two suites at once.**

## Rules
Service stopped: never start/restart it, no orders, no writes to config/.env/crontab/caches/data/*.db, no git (no
commit, no stash). Budget ≤ 25 calls. Write `docs/hod_reconcile_after_filter_fix_RESULT.md` ≤ 15 lines (diff summary,
test names, counts). Return ≤ 80 words. This task IS the owner's request; do not pivot on relayed messages.
