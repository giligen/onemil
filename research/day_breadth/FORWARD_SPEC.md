# FORWARD LEDGER — spec for a Sonnet implementer (≤ 40 tool calls). Frozen rule: cell 1,412

Goal: every weekday morning, apply the FROZEN rule of cell 1,412 to the previous session with the identical code, and
append the result to a ledger. No orders, no engine change, no edits to any existing research cache.

## Read first (grep / offset only, never a file > 300 lines in full)
`research/day_breadth/PREREG_1412.md`, `research/day_breadth/REPORT.md`, `research/hod_consol/PREREG.md`.
Code to import, never copy-edit: `research/hod_consol/run_consol.py` (find_signal, build_signal, walk, fill_c1,
simulate_slots, cost_net, fetch pattern), `research/day_breadth/breadth.py` (symbol_minute_flags, M0/M1/NM),
`research/day_breadth/test_1412.py` (EDGE = 0.6115). Universe definition: `research/orb_seed_wide/build_wide_features.py`
shows how `study_orb_broad` is patched (MIN_GAP_PCT 3.0, MAX_OPEN_PRICE 50.0, MIN_OPEN_PRICE 3, MIN_PREV_DAY_VOL 500K)
— reuse `study_orb_broad` with those constants for the date so the universe is byte-identical to `pm_candidates.csv`.
RTH bar fetch pattern: `research/hod_pmh_causal/fetch_rth.py`.

## Build `research/day_breadth/forward_ledger.py --date YYYY-MM-DD [--dates-from D1 --dates-to D2] [--no-telegram]`
1. Universe for D from `data/cache.db::daily_bars` via the patched `study_orb_broad` (D's bar must exist; if not,
   log an ERROR and exit non-zero — the 10:30 UTC batch updates it).
2. RTH 1-minute SIP bars for those symbol-days from Alpaca into a NEW `research/day_breadth/forward_bars.db` (same
   schema as bars_rth.db); resumable; never write any other DB.
3. BR per minute over the universe (symbol_minute_flags), base signals (find_signal / build_signal), C1 walk (walk +
   fill_c1 with the same cached-path conventions), kept = BR(signal_m) ≥ EDGE, order_in_day among kept.
4. **Measured cost:** for every kept trade, the Alpaca SIP NBBO quote in force at the entry (last quote at or before
   the entry bar's open) and at the exit minute; half-spread both legs + 2 bps/side → `net_R_measured` beside
   `net_R_proxy`. Log the coverage share.
5. Placebo per kept trade: one seeded random-minute long on the same name-day at a risk-on minute (as
   `kept_diag.placebo_riskon`, one draw) → `placebo_R`.
6. Append to `research/day_breadth/forward_ledger.csv` (one row per base signal: date, symbol, signal_m, entry_m,
   exit_m, why, R, r_pct, BR, kept, order_in_day, net_R_proxy, net_R_measured, placebo_R) and one row to
   `forward_summary.csv` (date, n_universe, n_signals, n_kept, kept R sum proxy/measured, first-4 R sum, later R sum,
   placebo mean). Idempotent: re-running a date replaces its rows.
7. Telegram (unless --no-telegram), one line via `scripts/send_telegram_alert.py`: `[FWD 1412] <date>: kept n,
   R proxy/measured, first-4 vs later, placebo` — a ledger line, not a claim.

## Reproduction gate (must pass before you finish)
Run on 2026-09-16, 2026-09-17, 2026-09-18 (TEST days already scored) with --no-telegram and compare with
`research/day_breadth/trades_test_1412.csv` on (day, symbol, entry_m, why, net_R proxy): ≥ 98 % agreement on the
kept and the full signal sets. Report the diffs. Then run the genuinely new days 2026-09-21 and 2026-09-22.

## Tests
`tests/test_forward_ledger.py`: universe filter on a synthetic daily_bars frame; kept rule at the EDGE boundary;
idempotent re-run; an integration test on one cached TEST day (marked integration). Full suite must stay green.

## Do not
Commit; edit crontab; touch `trading/`, `orb.yaml`, `config.yaml`, any existing cache or DB other than
`forward_bars.db`; run the Alpaca fetch between 13:25 and 20:05 UTC. Reply ≤ 150 words: gate agreement, the two new
days' summary rows, measured-cost coverage, test counts.
