# SPEC — production ORB dry-run mode (zero orders, full pipeline), 2026-09-25

## Why
ORB live is PAUSED (`orb.yaml strategy.enabled: false`, 9/25): the live book lost −$5,281 on 123 fills 5/19–9/23 while the
backtest for the same months was positive; the cause under repair is execution (first order 26–49 s late → the pre-warm
fix `execution.prewarm_seed`, `docs/orb_latency_fix_REPORT.md`; ~72 bps adverse ask-to-fill drift). Before real orders
resume, the FIXED engine must be observed for ≥ 5 sessions with zero orders: its picks, its would-be entry prices and
its 09:35 submit latency against the backtest's book for the same days. The add-on pools already have exactly this mode
(`universe.addon_pools.dry_run` → `[ORB+ DRY] WOULD BUY` telegrams/logs, `research/orb_seed_wide/PREREG_LIVE_UNION.md`);
production picks need the same.

## Rules (Sonnet; ≤ 40 tool calls; TDD; do not commit; do not start/restart any service; no DB writes to production tables)
1. `orb.yaml strategy.dry_run` (default false; add to `orb.yaml.template` too). When true AND `strategy.enabled` true:
   the engine runs everything it runs live — universe seed (with `execution.prewarm_seed` if set), range sweep, scoring,
   vetoes, ranking, slot arithmetic, the entry-drain thread's decision — and at the point where it would submit an
   order it instead logs ONE line per pick at INFO:
   `[ORB DRY] WOULD BUY {sym} stop ${trigger} limit ${limit} shares {n} risk ${r} | quote bid/ask at {HH:MM:SS.mmm} ET`
   plus the same `[ORB DRY]` Telegram the add-on dry mode sends, and appends a row to `logs/orb_dry_ledger.csv`
   (date, symbol, would_submit_ts_et, trigger, limit, shares, risk_usd, bid, ask, composite, quintile, pool). No order
   object is created; nothing is registered with the StopMonitor; no `trades` row is written.
2. The `LATENCY TRIPWIRE` and its `measured:` breakdown must fire in dry mode exactly as live (the would-be first submit
   is the instant measured) — this is the number the week is for.
3. The daily parity observer that today compares live picks with the BT book (`scripts/daily_green_check.py` /
   `scripts/bf_decision_parity.py` — grep for the ORB pick-parity logic) must read `[ORB DRY] WOULD BUY` lines as picks
   so the dry week produces the same parity report a live week would.
4. Mirror the add-on pools' implementation; do not fork the submit path — one code path with a dry branch at the submit
   call, so BT/live parity is by construction.
5. Tests (`tests/test_orb_dry_run.py`, fixtures in `tests/conftest.py`, `MagicMock(spec=...)`): dry mode submits zero
   orders (executor mock never called) while the WOULD BUY line, the ledger row and the tripwire all appear; flag off is
   byte-identical to today (parity test on a recorded tick); the parity observer parses a dry line into the same pick
   record as a live submit line. Run the full `tests/test_orb_*` set — zero failures.
6. Write `docs/orb_dry_run_REPORT.md`: files changed, the exact grep lines for the dry week, and the rehearsal plan (weekend
   boot with `strategy.enabled: true`, `strategy.dry_run: true`, `execution.prewarm_seed: true`; Monday–Friday dry; pass =
   ≥ 80 % pick agreement with the BT on the same days, first would-be submit ≤ 10 s after 09:35:00 on every day, and the
   would-be entries within 25 bps of the BT's; only then real orders). Return ≤ 150 words.
