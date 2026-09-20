# exec_cost: PREREG MOC (cells 1,287 ORB / 1,288 BF)

QUESTION (pre-registered): what does the honest book gain if every TIME exit (ORB 15:45 force
close; bull flag 15:55 flat) is replaced by a market-on-close fill at the 16:00 official close,
which pays NO spread, instead of a marketable order at 15:45/15:55 that pays the half-spread?
Everything else unchanged. Stops/locks/trails that fire between the old exit time and 16:00 still
fire at their price (walk the 1-min bars from the old exit minute to 15:59 with the trade's stop
level as of that minute; a stop hit in that window exits at that bar's open, charged the
half-spread as before). Pass bar: net $ up AND MDD not worse in BOTH 2025 and 2026-01..05
(TEST >= 2026-06-01 sealed), per-trade Delta reported with day-clustered t.

INPUTS:
- ORB honest book: analysis_results/orb_bplus_book.csv (THIS is the honest ORB reference per
  docs/CLAUDE_HISTORY.md, $10K stage). Time exits = exit_reason 'eod' (pure) and 'scale_eod'
  (partial + eod remainder).
- Bull flag: research/bf_stage2_regen7_raw_20260905.csv (regen-7 Stage-2 output, honest book,
  $107,351/79tr reference per docs/CLAUDE_HISTORY.md).
- Closes: data/cache.db daily_bars.close (official auction print proxy).
- Half-spread: research/mature_method/frames14/f45_minute_table.csv, minute-of-day median FULL
  spread as % of price (med column); half-spread = med/2. Rows at clock_m=945 ("15:45*ORB flat")
  and clock_m=955 ("15:55*HOD/BF flat").
- Bars: data/cache.db intraday_bars_1min (timestamp column, UTC ISO strings).

METHOD CAVEAT DECLARED UP FRONT: the ORB honest-book CSV (analysis_results/orb_bplus_book.csv)
carries no entry_time/exit_time/stop_loss/shares columns (it is a features+pnl summary, not a
per-minute trade ledger) -- the full bar-walk for stops firing in the extra 15:45-16:00 window
cannot be reconstructed from this file. The bull-flag Stage-2 CSV DOES carry those columns. Scope
is declared BEFORE running: BF gets the full bar-walk spec; ORB gets a spread-swap-only estimate
(no stop-window walk) with the gap disclosed as the single reported caveat, not discovered after
an unfavorable result.

Report: research/exec_cost/REPORT.md, <= 60 lines, per book/split: n time-exits, old net $, new
net $, Delta $, Delta per trade (R and $) with t, stops fired in extra window (BF only; N/A ORB),
MDD old/new, share of daily-close != 15:59-close by >0.5%, the ONE caveat.
