# PREREG — cells 1,607–1,609: the QUOTED SPREAD as a filter (is the expensive trade the winning trade?)

FROZEN 2026-09-28 16:20 UTC before any number. Programme count: 1,606 → 1,609. Owner 9/28: "maybe the ones with the
bigger spread are the winners? if they are the winners, maybe the spread is a filter?"

## What was seen (disclosed)
* ORB: the 150 bps spread gate skipped the monsters (BKKT +$20.8K at 153 bps quoted, XNDU +$11.9K at 267), 100–150 bps
  was the richest per-trade bucket ($1,299), > 300 bps was net-negative after costs (`research/orb_spread_gate_verdict.md`,
  2026-07). The 30 bps stop-limit cap's never-fills were winners (Q_fill: cap 50 = +15 %), 100 bps was worse.
* HOD: the 15 bps cap's unfilled breaks earned −0.57 R vs +0.29 R for fills (cell 1,423) — the expensive breaks were
  the losers there. The 1,478 feature set carried spread_bps_at_arm (no out-of-sample signal in the multi-feature model);
  a single-feature spread cut on the 9,911 fills with the standard cost has NOT been read.
* Live stops (trades DB, 91 exits): median slippage ≈ 0, p90 28–79 bps; ORB entry drift 72 bps mean.

## Cells
* 1,607 HOD-SPREAD: the 9,911 base fills of cell 1,438 (`causal_arming_causal.csv` status == fill; outcomes = the
  standard-cost net R of `model_1478_L3_predictions.csv`), the quoted spread at the arm instant in bps of price
  (`features_1478_A.csv` spread_frac_at_fill × 1e4 and, beside it, 2 × half_entry / fill — state which is the arm-time
  quote and use the causal one; the spread must be the quote BEFORE the fill print). Quintiles set on TRAIN-H2 only;
  per quintile and holdout: n, mean net R, day-clustered t, ex-top-5 %, fills/week at 12/4; the two candidate filters
  read on VAL: KEEP-WIDE = top two quintiles, KEEP-TIGHT = bottom two quintiles (both pre-declared; no other cut).
* 1,608 ORB-SPREAD: the ORB candidate/fill population with quotes: (a) the live trades DB 2026-05..09 (entry_quote_spread
  / entry price in bps, realized P&L in R = pnl / risk); (b) the backtest population with the 09:34:59 quoted spread
  where the nightly feature CSVs carry it (inspect `analysis_results/orb_features_*.csv` via `trading/orb_csv.read_orb_csv`
  for a spread column; if absent, the tick tape of cell 1,426 gives the NBBO at t* for the 410 replay signals). Buckets
  ≤ 50 / 50–100 / 100–150 / 150–300 / > 300 bps: n, mean R, t, win rate, the share of the book's P&L per bucket, on
  every sample available (state each sample's dates and n).
* 1,609 ORB-DRIFT-AS-SIGNAL (report-only, not causal at the decision instant): the realized entry drift
  (drift_ask_to_fill_bps) vs the trade's R on the 123 live fills — if the fastest bursts are the winners, the cost is
  the price of the edge and no cap should skip them; if not, a cap is a filter.

## Pass bar (frozen; VAL for 1,607; the live + backtest samples for 1,608)
1,607: a pre-declared filter passes if its kept VAL mean net R ≥ +0.15 with day-clustered t ≥ 2.5, ex-top-5 % > 0,
≥ 3 fills/week, TRAIN-H2 same sign t ≥ 1, dropped < kept on both holdouts. 1,608: a bucket rule (e.g. "skip ≤ 50 bps"
or "skip > 300") counts as a finding only if the dropped bucket's mean R < 0 on BOTH the live sample and the backtest
sample with n ≥ 20 each, and the kept book's mean R rises by ≥ +0.03 R; the existing 300 bps gate is the baseline.

## Independent check and consequences
Rebuild of 1,607 from the prose (quintile edges within 1 bps; kept means within 0.01 R); the refuter's first lens is
causality of the spread field (the quote before the arm/fill print, never the fill's own quote) and the cost double-count
(the standard cost already charges the half-spread once — the filter must not be judged on gross). PASS on 1,607 →
a `min_spread_bps` / `max_spread_bps` gate in the HOD engine for the dry run; PASS on 1,608 → the ORB gate values
change on the owner's word. FAIL → the spread is closed as a filter on both books; the ORB 300 bps gate stays.

## Not allowed
Choosing the quintile cut on VAL; more than the two pre-declared HOD filters; reading the HOD TEST.
