# RESULT — Form 4 insider open-market purchases (cells I1 CLUSTER, I2 OFFICER, I3 OPPORTUNISTIC)
Builder A. Spec: `research/insider_events/PREREG.md`. TEST (2024-01+) is built into signals_form4.csv /
trades_form4.csv (split='TEST') but excluded from this table and was not read for any cell.

| cell | arm | hold | split | n | mean_net% | t_clust | mo_pos% | ex_top5% | sig/wk | null_pctile | book_mo% | book_DD% | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | all | 5 | TRAIN | 2553 | -0.14 | -0.31 | 0.67 | -1.42 | 8.16 |  | 2.42 | -29.38 |  |
| I1 | all | 5 | VAL | 1103 | 0.07 | 0.22 | 0.5 | -1.02 | 10.59 |  | 0.19 | -42.43 |  |
| I1 | all | 20 | TRAIN | 2553 | 1.83 | 3.10 | 0.67 | -0.20 | 8.16 |  | 2.04 | -34.24 | YES |
| I1 | all | 20 | VAL | 1103 | 0.12 | 0.19 | 0.42 | -1.88 | 10.59 | 23.6 | -0.34 | -36.42 |  |
| I1 | 2018-23 delisted-included (named) | 20 | TRAIN | 1917 | 1.25 | 1.63 |  |  |  |  |  |  |  |
| I1 | 2018-23 delisted-included (named) | 20 | VAL | 1103 | 0.12 | 0.19 |  |  |  |  |  |  |  |
| I1 | 2018-23 survivors-only (named) | 20 | TRAIN | 1917 | 1.25 | 1.63 |  |  |  |  |  |  |  |
| I1 | 2018-23 survivors-only (named) | 20 | VAL | 1103 | 0.12 | 0.19 |  |  |  |  |  |  |  |
| I1 | 2018-23 delisted-included (price-avail) | 20 | TRAIN | 1917 | 1.25 | 1.63 |  |  |  |  |  |  |  |
| I1 | 2018-23 delisted-included (price-avail) | 20 | VAL | 1103 | 0.12 | 0.19 |  |  |  |  |  |  |  |
| I1 | 2018-23 survivors-only (price-avail) | 20 | TRAIN | 1917 | 1.25 | 1.63 |  |  |  |  |  |  |  |
| I1 | 2018-23 survivors-only (price-avail) | 20 | VAL | 1103 | 0.12 | 0.19 |  |  |  |  |  |  |  |
| I1 | all | 60 (report-only) | TRAIN | 2553 | 7.72 | 7.15 | 0.69 | 3.07 | 8.16 |  | 1.78 | -32.75 |  |
| I1 | all | 60 (report-only) | VAL | 1103 | 1.50 | 1.49 | 0.42 | -1.71 | 10.59 |  | -2.11 | -44.01 |  |
| I2 | all | 20 | TRAIN | 5114 | 1.31 | 2.55 | 0.72 | -0.46 | 16.34 |  | 1.49 | -32.73 | YES |
| I2 | all | 20 | VAL | 2101 | 0.25 | 0.58 | 0.46 | -1.44 | 20.17 | 70.7 | 0.38 | -22.12 |  |
| I2 | 2018-23 delisted-included (named) | 20 | TRAIN | 3767 | 0.56 | 0.82 |  |  |  |  |  |  |  |
| I2 | 2018-23 delisted-included (named) | 20 | VAL | 2101 | 0.25 | 0.58 |  |  |  |  |  |  |  |
| I2 | 2018-23 survivors-only (named) | 20 | TRAIN | 3767 | 0.56 | 0.82 |  |  |  |  |  |  |  |
| I2 | 2018-23 survivors-only (named) | 20 | VAL | 2101 | 0.25 | 0.58 |  |  |  |  |  |  |  |
| I2 | 2018-23 delisted-included (price-avail) | 20 | TRAIN | 3767 | 0.56 | 0.82 |  |  |  |  |  |  |  |
| I2 | 2018-23 delisted-included (price-avail) | 20 | VAL | 2101 | 0.25 | 0.58 |  |  |  |  |  |  |  |
| I2 | 2018-23 survivors-only (price-avail) | 20 | TRAIN | 3767 | 0.56 | 0.82 |  |  |  |  |  |  |  |
| I2 | 2018-23 survivors-only (price-avail) | 20 | VAL | 2101 | 0.25 | 0.58 |  |  |  |  |  |  |  |
| I3 | all | 20 | TRAIN | 4665 | 1.60 | 3.06 | 0.67 | -0.30 | 14.90 |  | 1.72 | -23.93 | YES |
| I3 | all | 20 | VAL | 1767 | 0.61 | 1.28 | 0.5 | -1.23 | 16.97 | 91.9 | 1.18 | -19.92 |  |
| I3 | 2018-23 delisted-included (named) | 20 | TRAIN | 3361 | 0.77 | 1.09 |  |  |  |  |  |  |  |
| I3 | 2018-23 delisted-included (named) | 20 | VAL | 1767 | 0.61 | 1.28 |  |  |  |  |  |  |  |
| I3 | 2018-23 survivors-only (named) | 20 | TRAIN | 3361 | 0.77 | 1.09 |  |  |  |  |  |  |  |
| I3 | 2018-23 survivors-only (named) | 20 | VAL | 1767 | 0.61 | 1.28 |  |  |  |  |  |  |  |
| I3 | 2018-23 delisted-included (price-avail) | 20 | TRAIN | 3361 | 0.77 | 1.09 |  |  |  |  |  |  |  |
| I3 | 2018-23 delisted-included (price-avail) | 20 | VAL | 1767 | 0.61 | 1.28 |  |  |  |  |  |  |  |
| I3 | 2018-23 survivors-only (price-avail) | 20 | TRAIN | 3361 | 0.77 | 1.09 |  |  |  |  |  |  |  |
| I3 | 2018-23 survivors-only (price-avail) | 20 | VAL | 1767 | 0.61 | 1.28 |  |  |  |  |  |  |  |

"pass" = clears every TRAIN or VAL bar item in the frozen PREREG for that split (day-clustered t via
statsmodels OLS clustered by entry session; ex-top-5% recomputed after dropping the top 5% of trade
returns; null = 1,000 count-matched draws, seed 1474, same session/n, random ELIGIBLE non-tainted symbol).

## Data notes
* SEC quarterly zips: 2016q1-2026q1 all downloaded (41/42); 2026q2 = 404 (not yet published by the SEC as
  of 2026-09-26, not a dataset-coverage gap). AFF10B5ONE (the 10b5-1 checkbox) only exists in the SEC
  schema from 2023q1 onward; pre-2023 purchases cannot be screened for planned 10b5-1 trades and were kept.
* Purchase-coded rows (P/A/Form4/no-10b5-1, valid ticker, valid filing_date): 402,664 across 2016-2026;
  49.1% have a ticker absent from the multiday panel's ~4,871-symbol tradable universe (mostly sub-$5 or
  sub-$1M-ADV/OTC/foreign filers) and were dropped there; a further ~15-27% of each cell's candidates were
  dropped for ineligibility/taint at the signal session. Final deduped signals/year: I1 ~320-600, I2
  ~650-1,200, I3 ~620-1,050 (2026 row is partial, through June).
* Survivorship arm is VOID for this population: 0 of the 2018-23 signal-symbols matched
  `research/multiday/data/delisted_names.parquet`, and 0 matched a second, independent price-availability
  proxy (`Panel.last_fin`, the panel's own last finite close) — every symbol that ever produced a Form4
  signal has a finite close all the way to ~the panel's last session, so this panel does not encode
  delisting as a stopped price series for this population. The mandated survivorship check could not
  discriminate here; do not read the "survivors-only" rows above as a pass — they are identical to
  "delisted-included" by construction, not evidence of a robust effect.
* PASSES: none. All three cells clear TRAIN alone (t 2.55-3.10, mean +1.3 to +1.8%/trade) but fail VAL on
  every VAL-only item (t 0.19-1.28 vs required >=2.0; book monthly -0.34% to +1.18% vs required >=+1.0%;
  null percentile 23.6-91.9 vs required >=99). I1 hold=5 is net NEGATIVE in TRAIN; I1 hold=60 (report-only)
  is far larger (+7.7% TRAIN) than hold=20/5 and ex-top-5% only turns positive at hold=60 -- a tail/top-trade
  pattern, not a stable per-trade edge (per-hold ex-top-5% is negative at 5 and 20 in both splits).
* Per PREREG "Independent check and consequences": FAIL -> closed as a passing claim; the Form4 purchase
  data build (`purchases.parquet`, 402,664 rows) stays available for any future information-event cell.

## Caveats
* Cross-quarter + joint-filer duplication before final dedup: ~6.3% of concatenated rows dropped as exact
  duplicates (same accession/owner/trans_date/shares/price appearing in adjacent quarterly dumps).
* I2's "flagged Officer or Director with a title containing CEO/CFO/President/Chair/Director" was
  operationalized as (Officer-or-Director relationship flag) AND (relationship+title text contains one of
  those 5 keywords) -- a bare Director always qualifies on the word "Director" itself; a bare Officer with
  no C-suite title in RPTOWNER_TITLE does not. This is a judgment call, not literal PREREG text; flag for
  the independent-check reviewer.
* I3's 12-month lookback uses TRANS_DATE history per (owner_cik, symbol) pair, not FILING_DATE -- standard
  Cohen-Malloy-Pomorski convention, but a different causal axis than the FILING_DATE-based signal timestamp.
* Book tie-break within a session (`value` descending) and the no-double-slot dedup spacing (hold=20 for
  every cell's spacing, even I1's 5/60 report-only variants) are both explicit, PREREG-silent choices —
  see the docstring in cells_form4.py for the full list of design decisions made where prose was ambiguous.
* Independent reimplementation (PREREG requirement #1, trade-by-trade on (day,symbol)) has NOT been done —
  this run is Builder A's own build only.
