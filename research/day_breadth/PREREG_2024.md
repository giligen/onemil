# PREREG — the risk-on base entry on a FRESH, UNTOUCHED half-year: 2024-07-02 .. 2024-12-31. Cells 1,413–1,414

Why: every split of the 2025–26 population has been used (1,400–1,412). The Databento EQUS.SUMMARY point-in-time
daily bars for 2024H2 (every US equity, delisted included — `data/research/databento/equs_daily_2024H2.parquet`)
give one half-year nobody has looked at, and a survivorship-free universe on top. Frozen before any 2024 minute bar
is fetched.

## Universe (parity with `pm_candidates.csv`, knowable by 09:35 ET)
Per day D in 2024-07-02..2024-12-31, from the EQUS 2024H2 daily bars: open_D ≥ 1.03 × close_{D−1}, open_D in [$3, $50],
volume_{D−1} ≥ 500,000, PIT `instrument_class == 'K'` (every matched 2025–26 universe symbol is class K), symbol not
matching `^Z[A-Z]ZZT$`; then, from the fetched minute bars, RTH volume 09:30–09:35 ≥ 15,000 (`study_orb_broad`'s
MIN_935_VOL_FLOOR). Availability rail: ≥ 80 % of universe symbol-days must have Alpaca SIP RTH bars, else VOID.

## Rules — frozen, identical code (`run_consol`, `breadth`, `test_1412.EDGE = 0.6115`)
Base signal, entry, stop, C1 exit, stop-at-open, proxy cost (15 bps half-spread both legs + 2 bps/side), fill walk,
BR over the 2024 universe (`breadth.symbol_minute_flags`), placebo (`kept_diag.placebo_riskon`, 8 draws).

| cell | book | derived from |
|---|---|---|
| 1,413 | all kept signals (BR(signal_m) ≥ 0.6115), C1 — the replication of 1,412 | frozen 2026-09-23 on 2025-H1 |
| 1,414 | kept signals with order_in_day ≥ 4 (skip each day's first four kept signals), all taken | the 2026 diagnostic (first 4 lost, later won on TEST / VAL) |

## Pass bar (each cell, on 2024H2)
Net ≥ +0.10 R per trade; trade-weighted day-clustered t ≥ 2; setup − risk-on placebo ≥ +0.10 R (the quantity that
held in all three earlier periods); kept − rest ≥ +0.10 R (1,413); weekly Sharpe (mean weekly R / sd weekly R) ≥ 0.30
and green weeks ≥ 55 % (the cadence of a basket book, dimension-free); fills ≥ 3/week. Reported beside it: the
first-4 / 12-a-day slotted book, worst day, worst week, share of R on the top 10 % of days.

## Verification of any pass
Universe causality trace on Opus first (every condition knowable by 09:35, no post-D information, no current-listing
filter), independent rebuild from this prose, measured NBBO re-score (Alpaca quotes at entry / exit minutes).

## Not allowed
Changing the edge, the exit, the order threshold, the universe thresholds or the cost after any 2024 number exists;
any other cell on 2024H2. Programme count 1,414.
