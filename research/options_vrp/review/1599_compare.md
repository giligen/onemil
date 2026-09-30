# 1,599 builder vs. rebuild comparison — BLOCKED, not a finding

Read in full: `PREREG_1567.md` + Amendments 2, 2a, 3 (Amendment 3 = the active data/cost rules — Databento
OPRA cbbo-1m NBBO, fills at bid/ask, no slippage constant, VOID rail from the cycle table, $150 spend cap).

## What was asked
Compare `cell_1599_cycles.csv` (builder) vs `rebuild_1599_cycles.csv` (rebuild), keyed by
`cell + sample + entry_date`: cycle-set Jaccard on `(expiry, short_strike, long_strike)`, share of common
cycles with `|pnl difference| ≤ $5`, whether the TRAIN-selected cell matches, and the VAL/EXTENSION mean
monthly-return differences on it.

## Why it can't be done yet
Both sides of the comparison are pre-data:

* **Builder** — `cell_1599_cycles.csv` is a **header-only** file (0 data rows, 114 bytes). Per
  `RESULT_1599.md` (written 2026-09-28 04:59 UTC), the BUILD stage never ran: `mondays.parquet` and
  `opt_cache/dbn/legs/*.parquet` did not exist at that time, so `cell_1599.py` exited before producing a
  single cycle. TRAIN selection is explicitly **NONE** for that reason (stated in the task, confirmed by
  the file).
* **Rebuild** — `rebuild_1599_cycles.csv` **does not exist on disk at all**. `rebuild_1599.py --stage all
  --workers 16` is running now (PID 3988487, started 05:09 UTC, still alive at check time, elapsed ~03:40).
  Its log (`rebuild_1599.log`) shows it is still in the Databento fetch/cost-check stage: as of the last
  read it had priced 50/133 Monday panel snapshots, with a mix of successful `get_cost` calls (now totaling
  **$0.293** against the $150 cap per `opt_cache/dbn/spend.json`) and repeated `422
  symbology_invalid_request` / `400 data_invalid_datetime_string` errors on specific Monday windows (SPY.OPT
  root symbol not resolving intraday for several dates; one malformed ISO timestamp in an early run). Only
  4 leg parquet files exist so far (`opt_cache/dbn/legs/SPY___2609...`), all for a single 2026-08-10 entry —
  nowhere near the 693-Monday ladder the fetch stage needs to walk before cycles, VAL or EXTENSION reads
  are possible.

Neither file has the entry-level rows the comparison needs. Any Jaccard, `$5`-band share, TRAIN-match
verdict, or VAL/EXTENSION monthly-return diff computed from what exists today would be computed from two
empty sets and would be a fabricated 0/1, not a measurement. Per the project's own rule (no research claim
ships without an independent, obtainable comparison), that is not reportable.

## What the (partial) evidence does say
* The `422 symbology_invalid_request` errors are worth flagging to whoever owns `rebuild_1599.py`/
  `fetch_dbn.py`: they recur on specific Monday intraday windows (e.g. 2024-02-19, 2024-05-27, 2024-09-02,
  2025-01-20, 2025-02-17, 2025-05-26) for the plain `SPY.OPT` parent-symbol cost probe, while the definition
  and other-Monday cbbo-1m probes succeed — looks like an intermittent resolution issue on that specific
  request shape, not a spend or logic problem. Not chased further here (out of scope / step budget).
* Spend is nowhere near the $150 cap ($0.29 used); the cap is not the blocker.
* The rebuild process was left running in the background per the task's own instructions ("long pulls run
  with nohup ... awaited with a background until-loop"); this comparison task's 12-call budget does not
  allow waiting out a 693-Monday ladder plus per-leg full-life fetches, so the run was left in place for the
  next pass rather than killed or rushed.

## Verdict
**BLOCKED, not a finding.** Re-run this comparison once `rebuild_1599.log` shows the mondays + legs stages
complete and `rebuild_1599_cycles.csv` exists with data rows (and once `cell_1599.py` has been re-run over
the same completed cache so `cell_1599_cycles.csv` is no longer header-only). Until then, `cycle_jaccard`,
the `$5`-band share, the TRAIN-selected-cell match, and the VAL/EXTENSION monthly-return diffs are all
**undefined**, reported here as 0 / false only to satisfy the caller's schema — not as measurements.
