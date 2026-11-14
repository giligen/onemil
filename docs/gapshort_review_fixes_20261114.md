# GAPSHORT review findings (high-effort review 2026-11-14) — fix list, one Sonnet agent, TDD each, full suite before done
Files: scripts/gap_short_paper.py (P), scripts/gap_short_parity.py (Y), trading/gap_short_rules.py (R), scripts/locate_ledger.py (L).
Every item: failing test first, then the fix. No strategy constant changes (frozen test stays).

1. P _ask(): get_latest_quote raises AlpacaAPIError when no quote → catch per symbol, return None, so the "no ask → market cover"
   path runs and one halted symbol cannot abort --cover for the others.
2. P cancel/replace race: after broker.cancel(), RE-READ the order (status canceled/filled) and use its FINAL filled_qty for `left`;
   never over-cover. Also handle a cancel() that fails because the order is already terminal (filled/rejected/expired/canceled):
   treat as "proceed with position qty from the account", not as a step failure.
3. P client_order_id collisions: every cover/flatten/market order id carries a unique suffix (leg + attempt index + HHMMSS), so a
   retried market cover is never rejected for a duplicate id; score_row must then match covers by PREFIX gapshort-<leg>-<day>-<sym>.
4. P OPG deadline mid-loop: write a ledger row (entry_status=not_submitted, note=opg deadline) for EVERY remaining pick, not only
   the current one; parity's reason `opg_deadline` must be reachable for all of them.
5. P pause flag: ONE week definition (ISO week via a shared helper — import scripts/book_status.week_start if it exists, else move
   it to trading/gap_short_rules.py and use it in both places); the flag is re-evaluated on every --weekly over the CURRENT ISO week
   only and cleared when that week is green OR when 5 trading days have passed with no rows (the instrument must never die silently);
   each write/clear logs INFO/ERROR with the number; Telegram line in --weekly states PAUSED/ACTIVE.
6. P load_universe() (and L line ~69): require MAX(bar_date) == the previous NYSE session (use pandas_market_calendars if present,
   else the exchange calendar helper already used elsewhere — grep `previous_session|trading_day` in scripts/) else logger.error
   "daily_bars stale (<date>) — refusing to pick" and exit 0 with no orders.
7. R day_killed(): wire it — --cover computes realised P&L of today's covers and, if ≤ −$300, writes the pause flag for the rest of
   the day (no further picks are possible anyway; it must at least log ERROR once and mark the ledger day_killed=1); or, if the
   owner spec rail cannot bind intraday, delete the dead constant and amend the spec. Choose wiring.
8. P placebo_rows: the "10:30 ask proxy" bar must be the last 1-min bar STARTING < 10:30 ET (the 10:29 bar); fetch with
   get_1min_bars_range_multi in ONE batched call for all placebo names; partial failures keep the names that succeeded.
9. P build_candidates: if gappers and bars == {} → raise so with_retries engages (3 retries), then the documented skip-day WARNING.
10. P score_row: P&L and net use the entry order's filled_qty and the covered qty (partial fills), not the ledger qty.
11. P reconcile(): cancel resting buys in ANY non-terminal status (new, accepted, partially_filled, pending_new, held,
    pending_replace, pending_cancel) before placing a market cover; after covers, also check for and close any gapshort- LONG
    position (q > 0) left by an over-cover, logging ERROR.
12. P every fallback logs WHY (fresh ask None → market; locate_fee_bp blank → default LOCATE_BP; parity prev_close fallback).
13. P/R single source of truth: delete OPG_CUTOFF from P and use R.OPG_LAST_SUBMIT_HM; remove dead/alias constants in R
    (COVER_TIME, MAX_NAMES, MIN_PRICE, L1_CUTOFF, TOP_N, NOTIONAL, MIN_GAP, MAX_GAP) keeping ONE name each — update the frozen
    constants test to the surviving names (same values).
14. P make_client(): one shared `make_mom_paper_client()` (put it in data_sources/alpaca_client.py or a small helper module) used by
    P, Y and — if trivial — momentum_sleeve/eod_sections (only if their tests stay green; otherwise leave them and note it).
15. Y on api_error: do NOT write universe_diff rows; write one row reason=api_error for the date, Telegram says "api error", INFO
    not ERROR unless 2 consecutive days.
16. L make_session(): _REPO / '.env'; select_gappers() must apply R.is_todays_premarket_trade (import from trading.gap_short_rules).
17. P --score on a day with 0 ledger rows: INFO "0 picks today" when logs/gap_short_paper.log shows a PICK line for the date,
    WARNING only when no pick ran.
Then: `python3 -m pytest tests -q --ignore=tests/integration -x -p no:cacheprovider` green; dry-run --pick/--cover/--flatten/--score and
`python3 scripts/gap_short_parity.py --dry-run`; return ≤ 150 words: per item done/not + why, test counts, dry-run lines.
