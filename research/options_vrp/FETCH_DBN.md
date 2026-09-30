# FETCH_DBN — PREREG_1567 v3 (Amendment 3) fetch stage report

Generated: 2026-09-30T14:01:26.355922+00:00

## Spend
Total charged: $53.6632 of $150.0 cap (5365 purchases logged, 0 skipped at the cap).

## SPY underlying
3390 sessions written to opt_cache/dbn/spy_prices.parquet. 691 sessions (2013-04-08..2016-01-03) are an UNFILLABLE GAP: Databento's equities datasets (ARCX.PILLAR, XNAS.ITCH, DBEQ.BASIC, EQUS.*) all start 2018-05-01 or later and Alpaca's stock history starts 2016-01-04 (verified empirically, empty response before that date). 2016-01-04..2023-12-31 sourced from Alpaca minute bars (10:00 / 15:59 ET); 2024-01-02..2026-09-25 reused from the existing Alpaca cache built by fetch_options.py (no re-pull, no re-charge).

## Monday ladders
38755 (entry_date, symbol) rows written to opt_cache/dbn/mondays.parquet.

## Per-leg full life
16057 unique OSIs in the superset plan (0.20-/0.30-delta strikes ± 3 strikes, and each one's $10-below partner); 20872 cached under opt_cache/dbn/legs/.

## Gaps / caveats
* Entry Mondays inside the SPY-price gap (2013-04-08..2015-12-31) have spot_10=NaN; the   Monday-ladder fetch for those weeks pulls the OPTION side only (definition + cbbo-1m are   available from 2013-04-01) but cannot select the delta-targeted superset without a spot   price -- resolving that (e.g. put-call parity from the same chain) is BUILD-stage work,   not fetched here.
* A cycle/Monday is only in mondays.parquet if both its definition and cbbo-1m pulls   succeeded and returned a non-empty 10:00 bar; anything else is logged as WARNING/ERROR   in fetch_dbn.log, never silently dropped.
* This run may have stopped partway through the plan if the $150 cap or the step budget   was reached first; rerun the same command to resume (already-cached Mondays/legs are   skipped).
