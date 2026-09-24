# SPEC — implement and score `research/thermo/PREREG.md` (cells 1,420–1,422). Sonnet, ≤ 40 tool calls

Read `research/thermo/PREREG.md` first (frozen; every definition is there — do not change any). Write only under
`research/thermo/`. Do not commit. Grep/offset reads only; never read a file > 300 lines in full.
**Market-hours rule:** nothing that reads `data/cache.db` before 20:05 UTC (check `date -u`; if earlier, build and
unit-test the code first, then run the DB step). Use `nice -n 19 ionice -c3` for the book rebuild.

## Steps
1. `thermo.py` (docstrings, verbose progress) with small pure functions: `load_orb_population()` (three CSVs via
   `trading/orb_csv.read_orb_csv`, concatenated; assert no date overlap and report the day count per file),
   `thermometer(daily_outcomes, window=20, min_n=40)` (strictly prior days), `hot_flags(T, min_hist=60)` (expanding
   median of strictly earlier defined values), `split_stats(R, hot, day)` (hot − cold, OLS `R ~ 1 + hot` day-clustered
   t via statsmodels `cov_type='cluster'`, iid t, n per cohort), quintile table, ex-top-5 %.
2. `test_thermo.py` (pytest): a synthetic series proves T(d) never uses day d or later (change day d's outcome →
   T(d) unchanged, T(d+1) changes); the expanding median excludes the current day; min_n / min_hist give undefined.
3. Rebuild the 2025–26 ORB book exactly as the PREREG says (after 20:05 UTC). Consistency check vs
   `research/orb_seed_wide/out/runB_true.csv` restricted to 2025: n fills and total `_sized_pnl` within 5 % → else
   STOP, write the discrepancy to REPORT.md and return.
4. Cells 1,420 and 1,421 with every statistic and lens the PREREG lists (oracle same-day T, stale T ending 60 trading
   days before d, per-period legs, share of days hot per period, hot fills/week; HOD fills/week via
   `research/hod_consol/run_consol.simulate_slots` on the hot cohort). HOD: drop `split == 'TEST'` on read; T_HOD uses
   TRAIN+VAL signals only.
5. Cell 1,422 report-only: locate the bull-flag P1 Stage-2 trade files that `research/bf_2024/REPORT.md` scored
   (2024H2 and 2025–26; grep that report for paths), R per trade as that report defines it, split by T_ORB hot/cold.
6. `REPORT.md`: one table per cell, the frozen pass-bar verdict per cell (PASS / FAIL, each leg shown), the lenses,
   and a "Data" section with row counts, date ranges and dropped/undefined days.

## Orchestration (the main session launches nothing for you; you launch the script and return)
Build and unit-test the code now. Smoke the scorer on the pieces that need no DB (the HOD cell; the ORB cell on the
2023-24 books only, to a scratch output, never REPORT.md) so the unattended run cannot crash. Then write
`research/thermo/run_thermo.sh` (`nohup setsid`-safe, logs to `run_thermo.log`): waits while UTC is in [13:25, 20:05)
AND while `pgrep -f research/hod_ofi/rerun.sh` finds a process (one heavy job at a time); rebuilds the 2025–26 book
with `nice -n 19 ionice -c3`; runs the consistency check; scores all three cells into REPORT.md; ends with a
Telegram ping via `scripts/send_telegram_alert.py` carrying NO numbers. Launch it with
`nohup setsid research/thermo/run_thermo.sh >> research/thermo/run_thermo.log 2>&1 < /dev/null &` and return.

Return ≤ 150 words: tests passed, smoke ran clean (no numbers needed), script launched (pid), anything surprising.
