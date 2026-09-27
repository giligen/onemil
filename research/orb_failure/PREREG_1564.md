# PREREG — cells 1,564–1,566: the FAILED OPENING-RANGE BREAKOUT, declared at 10:30 ET

FROZEN 2026-09-27 07:40 UTC before any number. Programme count: 1,563 → 1,566. Owner 9/27: "Read what you said on ORB
failures and HOD failure prediction! Signal is there. Just define an hour to declare ORB 'failure signal'. Be an owner."

## What was seen (disclosed) and the deduction
`research/hod_entry/review/orb_x_hod_diagnostic.md` (9/27): on the same symbol-day, HOD-break fills after an ORB
LOSER earn −0.90 R (TRAIN-H2, n 30, t −7.0) and −0.54 R (VAL, n 38, t −2.7); after an ORB winner +0.19 / +0.11 R
(n 61 / 87, t ≈ 1). Two limits: the ORB exit time was not recorded, so causality was not provable, and the overlap is
1.3 % of HOD fills, so it cannot rescue the HOD book. The deduction: a gap-up name that breaks its opening range and is
stopped back through the range low has trapped the morning's buyers; its later breaks fail. That is a SIGNAL ON ITS OWN
POPULATION (every ORB candidate, not only HOD fills), with a mechanism, and frequency is the question. The declaration
hour makes it causal: whatever is known at 10:30 ET decides.

## Population and the signal
Every ORB candidate symbol-day in the cumulative ORB feature file the diagnostic used (`analysis_results/
orb_features_20260925_2054.csv` via `trading/orb_csv.read_orb_csv`, 2025-01-02..2026-09-25; the selection-chain flags
kept as features, the broad pool is the population; state the daily count). Opening range = the 5-minute range
09:30–09:35 (range_high / range_low as the ORB engine, `study_orb_features.py`). Minute bars: `data/cache.db`
intraday bars READ-ONLY (2025-01-02 onward). Events, all decided on bars through 10:30:00 ET:
* BREAK: a bar high ≥ range_high + $0.01 in [09:35, 10:30).
* FAILURE (the signal, cell 1,564): a BREAK followed by a bar low ≤ range_low − $0.01 before 10:30 (the ORB stop hit).
* SUCCESS (cell 1,566, the mirror): a BREAK with no bar low ≤ range_low − $0.01 before 10:30 and close(10:29) ≥
  range_high (the break held).
Declaration: at the open of the 10:30 bar. Report-only declaration hours 10:00 and 11:00 (no selection among hours).
Calibration line (report-only): the HOD-break base fills (`model_1478_L3_predictions.csv`) on FAILURE symbol-days
with fill_min ≥ 10:30 — their base outcome must reproduce the diagnostic's negative number; and on SUCCESS days.

## Trades (entry at the 10:30 bar open, the first obtainable price after the declaration)
* 1,564 FAILED-BREAK SHORT: short at the 10:30 open at the bid (entry half-spread charged from the NBBO at 10:30 —
  fetched per event from Alpaca quotes, resumable cache `quotes_1564/`; if unavailable the minute-of-day half-spread
  table valid 09:37–14:01), stop = the day's high through 10:30 + $0.01 (stop-limit standard on the exit), target =
  entry − 2 R (limit), cover 15:55 at the ask; shortable flag and SSR (prior close −10 %) excluded; borrow 3 %/yr
  pro rata; price ≥ $5. R as % of price reported (the rail at 0.5 %).
* 1,565 FAILED-BREAK SHORT, VWAP TARGET: as 1,564 with the target = the session VWAP at 10:30 (report-only if VWAP is
  below the entry by less than 0.5 %).
* 1,566 HELD-BREAK LONG (the mirror): long at the 10:30 open at the ask, stop = range_low − $0.01, target entry + 2 R,
  15:55 at the bid; costs as the base standard.
Path: `sip_rebuild.walk_path` semantics on minute bars (stop first on a bar touching both; gap-through at the open).
Splits: TRAIN = 2025, VAL = 2026-01..2026-09; no sealed TEST (disclosed; the forward dry run is the test).

## Report per cell, per split
n events, events/week, mean net R, day-clustered t, ex-top-5 % and ex-top-1 %, winner-capped +3 R, median R as % of
price, exit mix, the count-matched null (random ORB candidates of the same days that did NOT fail, same trade, 1,000
draws, seed 1564 — the control that separates the failure from the gap-day drift), the universe placebo (all ORB
candidates, same trade at 10:30), per-month table, and the calibration line above.

## Pass bar (frozen; VAL, per cell)
Mean net R ≥ +0.15, day-clustered t ≥ 2.5, ex-top-5 % > 0, winner-capped positive, ≥ 3 events/week, null percentile
≥ 99 (the failure must beat the non-failure control on the same days), TRAIN same sign t ≥ 1, median R ≥ 0.5 % of
price. Shorts: positive after borrow and exclusions.

## Independent check and consequences
Rebuild from this prose on the same bars (event-set Jaccard ≥ 0.99, ≥ 99 % of rows within 0.01 R). Refuters:
causality (every event field from bars through 10:30:00 only; the 10:30 open is the entry, never the 10:29 close),
obtainability (short at the bid with the spread; locate; SSR; halts), price scale, statistics (tails, day concentration,
the control, the calibration reproduction), and the ORB feature file's own selection (is the candidate pool itself
point-in-time — `read_orb_csv` and the nightly file are the live scanner's output, state it). PASS on 1,564 or 1,565
→ a short-book proposal for the owner (borrow, SSR, his manual shorts on the same account) with a dry ledger first;
PASS on 1,566 → a second ORB entry mode (the 10:30 confirmation) into the ORB dry week. FAIL → the failure signal is
closed as a trade with its calibration on record; it stays as a veto feature for any future long on those names.

## Not allowed
Moving the declaration hour after a number; selecting among 10:00 / 10:30 / 11:00; reading the HOD calibration as a
trade; more than the three cells.
