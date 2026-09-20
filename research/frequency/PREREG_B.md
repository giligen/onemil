# PREREG_B — Cell 1,297: bull flag re-entry at recalibrated cost

**Date**: 2026-09-20. **Hypothesis**: `BT_ALLOW_REENTRY=1` (multi-trade-per-symbol-per-day)
adds a positive-edge trade set once execution cost is recalibrated down from the
Stage-2 default assumption to setting M (entry 23.2 bps, exit at 50% of the
currently-assumed stop-exit slippage table), reversing the shipped verdict in
CLAUDE.md ("Empirically −$1,299/yr — DO NOT enable in prod", measured at the
OLD, higher cost assumption).

## Method
1. Stage-2 only, cache `data/bull_flag_cache_causal_full_20260905.csv`
   (`BT_CACHE_PATH_OVERRIDE`), `--start 2025-01-01 --end 2026-05-31
   --capital 50000 --risk 2000 --max-shares 10000`. Run twice:
   `BT_ALLOW_REENTRY=0` (baseline) and `=1` (re-entry). Live P1 knobs in
   config.yaml untouched (read-only).
2. Added set = rows keyed by (symbol, date) present in the `=1` output but
   absent from the `=0` output (baseline's one-trade-per-symbol-per-day is a
   subset of re-entry's rows for the same symbol-date, so a `(symbol,date)`
   that appears with a DIFFERENT trade in `=1` and did not in `=0` is not
   possible here — added rows are strictly extra trades on days the baseline
   already had a fill, or new symbol-days the baseline skipped once the first
   leg failed some other gate. Keyed on row identity, not just symbol+date,
   using entry_time_et to disambiguate multiple legs per day).
3. Cost recalibration to setting M, adapted from `research/exec_cost/recal.py`:
   - **Entry (exact)**: Stage-2 rows carry `planned_entry` (the causal
     breakout level) and `entry_price` = `planned_entry * (1 + entry_slip)`
     with `entry_slip=0.005` (batch_backtest.py default, confirmed via
     `trading.entry_slippage_pct` fallback). Re-price to M's target entry
     cost of 23.235 bps (same target `recal.py` used for BF-M):
     `new_entry_price = planned_entry * (1 + 0.0023235)`.
   - **Exit (approximate — CAVEAT)**: `recal.py`'s exact method rescales a
     real per-trade quote half-spread column (`sp_e`/`sp_x`) that does not
     exist in the Stage-2 per-trade CSV (`CSV_HEADERS` has no spread/quote
     columns — confirmed by grep). Lacking real quotes for this book, exit
     cost is recalibrated as 50% of the ASSUMED stop-exit slippage table
     `recal.py` itself uses for the "current" charge (`charged_x = 0.003 *
     exit_price` for `exit_reason` containing "stop"): recovered cost =
     `0.5 * 0.003 * exit_price` added back for stop-type exits only,
     0 elsewhere (target/EOD/bar-close exits are already modeled at 0
     slippage per `backtest.py` comments). This is a flat-rate proxy, NOT
     the real quote-spread recalibration — reported as a limitation, not
     hidden.
4. Per split (TRAIN 2025, VAL 2026-01-01..05-31; nothing ≥ 2026-06-01):
   n trades, total $, MDD, net R/trade (pnl/2000) with day-clustered t,
   green-week share, trades/week — computed for both the baseline and the
   re-entry-on book, plus the ADDED set alone.
5. Cadence: `python scripts/cadence_bar.py --trades <csv> --split VAL|TRAIN`
   on the re-entry-on book (columns `date,pnl_R,symbol`, `pnl_R=pnl/2000`).

## Pass bar (pre-committed, decided before any numbers are read)
- Added-set net R/trade ≥ +0.10 R on BOTH TRAIN and VAL.
- Stacked (baseline + added) total $ up on BOTH splits vs baseline alone.
- MDD (re-entry-on book) ≤ 1.25× baseline MDD.
- Trades/week up ≥ 30% (re-entry-on vs baseline).
All four must hold. Any one failing → verdict NO-GO, keep
`BT_ALLOW_REENTRY` at its current default-off status (this cell does not
touch config/cron/trading/ regardless of verdict — decision recorded only).

## Independent-check status
This is a single-implementation Stage-2 run against the shipped simulator —
no second reimplementation was built for this cell (budget: 20 tool calls).
The verdict below is therefore a SCREEN, not a ship decision: a PASS here is
necessary but not sufficient per `feedback_independent_check_before_claims`;
a real enable would still need trade-by-trade reproduction before touching
config.yaml.
