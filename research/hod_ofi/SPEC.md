# SPEC — order-flow imbalance at the HOD break (`research/hod_exit_lab/PREREG_OFI.md`, cells 1,393–1,395)
For a Sonnet implementer, ≤ 40 tool calls. Owner approved the data spend 2026-09-24 (budget cap $110 total).

Read `research/hod_exit_lab/PREREG_OFI.md` (frozen cells, features, pass bar). Grep/offset reads only; never a file
> 300 lines in full. Write only under `research/hod_ofi/`. Do not commit. No Databento/Alpaca fetch between 13:25 and
20:05 UTC. Databento key: `DATABENTO_API_KEY` in `.env` (`from dotenv import load_dotenv; load_dotenv('.env')`).

## Two data-collection clarifications made by the main session BEFORE any data exists (disclose in the report)
1. **Causal window end = the break moment.** HOD-break entries are intra-bar stop entries at the level, so features
   must end at the FIRST trade print ≥ the break level (`entry` in features.csv) inside minute `entry_m`, not at the
   bar close. If no such print is found, end = the start of minute `entry_m` + 60 s and flag the row.
2. **Window = [end − 5 min, end + 1 min]** (6 minutes, not 10): no cell uses post-break data; this keeps signal +
   placebo windows inside the owner's $110 cap. Placebo window: one seeded random minute m_p in 10:00–14:00 of the
   same name-day (≠ entry_m ± 15), end = m_p's start; same 5-minute feature window.

## Steps
1. Population: `research/bf_zero/causal_filter/features.csv` rows with split TRAIN or VAL only (TEST sealed — never
   fetch a TEST window). Minutes are ET minutes since midnight (`entry_m`).
2. **Cost gate first:** `metadata.get_cost` for XNAS.ITCH `mbp-1` and `trades` on 40 random windows; extrapolate to all
   signal + placebo windows; print the projection; if it exceeds $110, STOP and report without fetching.
3. Fetch per window (threads ≤ 6, retries, resumable manifest) into `research/hod_ofi/raw/` (parquet per day): `mbp-1`
   and `trades`, `stype_in='raw_symbol'`. Availability rail per PREREG (≥ 60 % of window seconds quoted; coverage ≥ 80 %
   of signals; winner/loser missingness gap ≤ 5 pp — winners/losers from `research/hod_exit_lab/b0_trades.csv` net_R).
4. Features exactly as the PREREG: OFI_5 (Cont-Kukanov-Stoikov L1 OFI summed over the 5 minutes, divided by mean
   displayed L1 depth), OFI_1 (last 60 s), TSI_5 (Lee-Ready trade sign vs prevailing quote, signed shares / total),
   spread_bps_at_break. Same four for each placebo window.
5. Score on the B0 book (`research/hod_exit_lab/b0_trades.csv`, join on day+symbol): cut at the TRAIN-H1 median of
   each feature; kept vs dropped net R on TRAIN-H2 and VAL; VAL day-clustered t (reuse
   `research/hod_consol/adversarial_read.stats`); kept fills/week after first-12/day, 4-concurrent slots (reuse
   `research/hod_consol/run_consol.simulate_slots`, which needs day, entry_m, exit_m); decile tables with Spearman on
   TRAIN-H2 and VAL; the placebo-window deciles; ex-top-5 %. Apply the PREREG pass bar per cell.
6. Write `research/hod_ofi/REPORT.md` (coverage, cost actually billed, tables, verdicts, the two clarifications).

Reply ≤ 150 words: projected and billed cost, coverage, per cell kept − dropped on TRAIN-H2 / VAL, VAL t,
Spearman, placebo lift, verdict.
