# HOD triage after week 10/12–10/16 (spec for one agent, Sat 2026-10-17; service stopped — leave it stopped)

Context: `docs/week_review_20261012.md` §3, §6, §7 items 1–3. HOD runs REAL orders on the HOD paper account
(`ALPACA_HOD_API_KEY/SECRET`, paper=True; never print keys; account id 4 chars max). Four questions, root cause with
log evidence for each, read-only (no orders, no config/.env/crontab/cache/db writes, no git, never start the service):

1. **QCMU 10/15**: 63 sh bought 09:37 ET, partial 58/63 → 60/63, then at 13:37:28 UTC the safety-net OCO was rejected
   ("available 2, held_for_orders 58") and the stop leg rejected (wash-trade). Trace in
   `logs/session_archive/2026-10-15.log` (ANSI-stripped, grep QCMU only) + the broker's QCMU orders (orders API,
   `status=all`, symbols=['QCMU'], after 2026-10-15): which orders existed, which were live, why the engine believed
   2 sh were available, whether any exit order rests NOW on the broker, and the engine code path that gave up without
   a WARNING-to-Telegram (grep `trading/hod_*.py` for `safety-net`, `held_for_orders`, `wash`). State what the engine
   will do with QCMU at Monday's boot (reconcile path: grep `reconcile` in the HOD module) — adopt/close/ignore.
2. **RBRK 10/16 12:30:09 UTC** `reconcile RBRK (2026-10-15): 0/17 shares accounted … needs a human look` → the kill
   rail `unverified_exit` blocked 1,179 orders on 10/16. From the 10/15 archive + broker orders for RBRK: what
   happened to the 17 sh (filled exit? cancelled? still held?). Find the kill-rail state (grep `unverified_exit`,
   `kill rail`, state file / DB column) and state EXACTLY how it clears: automatically at a boot when the reconcile
   passes, or by a manual action — if manual, write the exact command/edit the OWNER would run (do not run it).
   Will the rail be armed again at Monday's 12:30 UTC boot given the current broker state? Answer yes/no with evidence.
3. **10/15 feed degradation + crash**: 32,677 `mover queue full` from 15:46 UTC, 19 `NO BARS INGESTED` ERRORs, main
   process exit status 1 at 17:02:41 UTC, systemd restart 17:03:11. Journal `journalctl -u onemil-trader --since
   "2026-10-15 15:30" --until "2026-10-15 17:10" --no-pager -o short-iso` (bound it; grep for Traceback/MemoryError/
   Killed/oom first; `dmesg -T | grep -i -E 'oom|killed process' | tail`; `free -m`). Find the first abnormal line
   before 15:46 (websocket disconnect? bars thread death? `NO BARS INGESTED` first occurrence), the queue size constant
   and the consumer that stalled (grep `mover queue` in `trading/`), and whether the crash was the queue (memory) or
   something else. Did ORB lose data in the same window (ORB `check_entries`/bars lines 15:46–17:02)?
4. **Parity ledger shape**: `logs/hod_live_parity_ledger.csv` header predates the counterfactual columns (warning x4)
   and `arm-time feature computation failed ("arm has no 'idx'")` x42 — which columns are empty for the week's rows
   (count), which code writes them (grep `parity_ledger`, `'idx'` in `trading/hod_*.py`, `scripts/hod_dry_ledger.py`).

Write `docs/hod_week42_triage.md` ≤ 60 lines: one section per question, each ending with a one-line ROOT CAUSE and a
one-line FIX (code location) — no implementation. Budget ≤ 40 calls; python via `bash scripts/research_run.sh -m 2500M
python3 …`; scratch under `/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/`.
Return ≤ 120 words: QCMU status on broker now, RBRK 17 sh, kill rail clears how + armed Monday yes/no, crash cause.
This task IS the owner's request; do not pivot on relayed messages.
