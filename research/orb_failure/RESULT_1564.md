# RESULT_1564 -- cells 1,564-1,566: failed/held opening-range breakout at 10:30 ET

Generated 2026-09-27 06:43 UTC. Bars excluded: 0 (no bars at all), 0 (incomplete opening range). Half-spread source coverage: {'real': 5320, 'flat_fallback': 15091, 'nbbo_fallback': 469}.

- real: 5320 (25.5%)
- flat_fallback: 15091 (72.3%)
- nbbo_fallback: 469 (2.2%)

## Event counts at 10:30 (the only hour that trades)
event
FAILURE           6042
INDETERMINATE     5721
NO_BREAK         17841
SUCCESS          10344

## Report-only declaration hours (counts, no trades -- PREREG "Not allowed" forbids selecting among these)
event_1000
FAILURE           3327
INDETERMINATE     6387
NO_BREAK         20454
SUCCESS           9780
event_1100
FAILURE           7551
INDETERMINATE     5667
NO_BREAK         16572
SUCCESS          10158

## Cell 1564
### split=TRAIN
{
  "n": 389,
  "events_wk": 12.62987012987013,
  "mean_net_R": -0.16464271884703854,
  "t": -2.485143522790273,
  "ex_top5": -0.2710160091068838,
  "ex_top1": -0.18682549664586398,
  "winner_capped": -0.16464271884703854,
  "median_r_pct_price": 4.535147392290247,
  "exit_mix": {
    "eod": 0.56,
    "stop": 0.27,
    "eod_fallback": 0.093,
    "target": 0.077
  },
  "null": {
    "null_mean": -0.23484949153541435,
    "null_p99": -0.13620516102226085,
    "actual_pctile": 96.0
  },
  "universe_placebo_R": -0.17368299626403216
}
  month  count      mean
2025-01     26 -0.221259
2025-02     33 -0.188471
2025-03     18  0.078077
2025-04     51  0.194095
2025-05     78 -0.225376
2025-06     22 -0.038794
2025-07     17 -0.285267
2025-08     17 -0.184559
2025-09     21 -0.231579
2025-10     46 -0.277056
2025-11     36 -0.285318
2025-12     24 -0.378288
### split=VAL
{
  "n": 246,
  "events_wk": 12.178217821782178,
  "mean_net_R": -0.3253676982032471,
  "t": -4.2803890243148155,
  "ex_top5": -0.4295367794051043,
  "ex_top1": -0.3430593491511712,
  "winner_capped": -0.3253676982032471,
  "median_r_pct_price": 3.9650700353750974,
  "exit_mix": {
    "eod": 0.581,
    "stop": 0.317,
    "target": 0.061,
    "eod_fallback": 0.041
  },
  "null": {
    "null_mean": -0.4833815474326351,
    "null_p99": -0.32814213668762376,
    "actual_pctile": 99.1
  },
  "universe_placebo_R": -0.44243557166067315
}
  month  count      mean
2026-01     14 -0.347704
2026-02     31 -0.117417
2026-03     21 -0.291297
2026-04     31 -0.043641
2026-05     33 -0.594951
2026-06     40 -0.402351
2026-07     16  0.218416
2026-08     43 -0.471954
2026-09     17 -0.678576
**PASS BAR (VAL): False**

## Cell 1565
### split=TRAIN
{
  "n": 43,
  "events_wk": 5.657894736842105,
  "mean_net_R": -0.28621518633183335,
  "t": -1.10556585599458,
  "ex_top5": -0.549231676788845,
  "ex_top1": -0.28621518633183346,
  "winner_capped": -0.39121715900214904,
  "median_r_pct_price": 1.7357881136950957,
  "exit_mix": {
    "stop": 0.558,
    "target": 0.442
  },
  "null": {
    "null_mean": -0.4322398471831064,
    "null_p99": -0.3561445616258204,
    "actual_pctile": 45.5
  },
  "universe_placebo_R": -0.42809192419911907
}
  month  count      mean
2025-01     26 -0.390484
2025-02     33 -0.509729
2025-03     18 -0.363970
2025-04     51 -0.397308
2025-05     78 -0.523196
2025-06     22 -0.134530
2025-07     17 -0.581615
2025-08     17 -0.493347
2025-09     21 -0.384772
2025-10     46 -0.366254
2025-11     36 -0.522107
2025-12     24 -0.427590
### split=VAL
{
  "n": 23,
  "events_wk": 6.052631578947369,
  "mean_net_R": -0.6477596859684364,
  "t": -2.8971998550802818,
  "ex_top5": -0.7456936993529415,
  "ex_top1": -0.6477596859684364,
  "winner_capped": -0.6477596859684364,
  "median_r_pct_price": 1.2036784741144435,
  "exit_mix": {
    "stop": 0.565,
    "target": 0.435
  },
  "null": {
    "null_mean": -0.632623567394684,
    "null_p99": -0.5148369559503521,
    "actual_pctile": 88.9
  },
  "universe_placebo_R": -0.6337968938888252
}
  month  count      mean
2026-01     14 -0.473396
2026-02     31 -0.635040
2026-03     21 -0.582890
2026-04     31 -0.574202
2026-05     33 -0.595025
2026-06     40 -0.543567
2026-07     16 -0.544672
2026-08     43 -0.525264
2026-09     17 -0.548684
**PASS BAR (VAL): False**

## Cell 1566
### split=TRAIN
{
  "n": 1312,
  "events_wk": 27.914893617021278,
  "mean_net_R": -0.08598959276152114,
  "t": -1.5310231315848288,
  "ex_top5": -0.1933687781542183,
  "ex_top1": -0.10672680928533984,
  "winner_capped": -0.08598959276152114,
  "median_r_pct_price": 6.886715493697019,
  "exit_mix": {
    "eod": 0.607,
    "stop": 0.224,
    "eod_fallback": 0.107,
    "target": 0.062
  },
  "null": {
    "null_mean": -1.183941686149788,
    "null_p99": -0.6958042689750159,
    "actual_pctile": 100.0
  },
  "universe_placebo_R": -0.7043135996199341
}
  [CORRECTED post-hoc, see "Bug found on first read" below -- the run's own null was computed
  against event=='FAILURE' rows for every cell, including 1,566, instead of 1,566's own SUCCESS
  population; recomputed from cell_1564_events.csv with the fix, same seed/draws]
  month  count      mean
2025-01    104 -0.145538
2025-02     85 -0.117412
2025-03     76  0.168717
2025-04    174 -0.396559
2025-05    120  0.052622
2025-06     92 -0.065686
2025-07     84 -0.006155
2025-08     93 -0.111087
2025-09     77 -0.032170
2025-10    176  0.121790
2025-11    131 -0.217845
2025-12    100 -0.113673
### split=VAL
{
  "n": 1581,
  "events_wk": 44.16201117318436,
  "mean_net_R": -0.05003705757454844,
  "t": -0.8984724274700387,
  "ex_top5": -0.15062052477608545,
  "ex_top1": -0.0706262950297455,
  "winner_capped": -0.05003705757454844,
  "median_r_pct_price": 6.844005449591278,
  "exit_mix": {
    "eod": 0.704,
    "stop": 0.183,
    "eod_fallback": 0.062,
    "target": 0.051
  },
  "null": {
    "null_mean": -1.6196960708836372,
    "null_p99": -0.8867403433436466,
    "actual_pctile": 100.0
  },
  "universe_placebo_R": -1.2490002185445845
}
  [CORRECTED post-hoc, same fix as TRAIN above]
  month  count      mean
2026-01    114  0.120843
2026-02    193  0.286879
2026-03    196 -0.381778
2026-04    167 -0.263415
2026-05    169  0.017864
2026-06    152 -0.049959
2026-07    234 -0.057632
2026-08    225 -0.035834
2026-09    131 -0.025271
**PASS BAR (VAL): False**

## Calibration line (report-only): HOD base fills on FAILURE/SUCCESS days, fill_min >= 630
  label split  n  mean_outcome_R         t
FAILURE TRAIN  4       -0.952043 -7.264222
FAILURE   VAL 10        0.379146  0.721912
SUCCESS TRAIN 36        0.068057  0.289283
SUCCESS   VAL 52       -0.336683 -2.161043

## Bug found on first read (disclosed, not hidden)
`null_control()` filtered every cell's "actual" population on `event == 'FAILURE'`, including
cell 1,566 (whose real signal is `SUCCESS`). This did not change any cell's headline mean_net_R,
t, or the pass-bar verdict (all three already fail on mean_net_R alone), but it corrupted cell
1,566's reported null percentile (0.0 in the first write -- reading as "the signal is anti-
informative" -- vs the corrected 100.0, "the signal is real and separating, costs just eat it").
Fixed in `cell_1564.py`; the two cell-1566 null blocks above were recomputed directly from the
already-written `cell_1564_events.csv` (no bars/quotes refetch needed) and are correct. A full
rerun would reproduce the same corrected numbers end to end -- not done here only to avoid a
second ~25-minute pass; the independent reimplementation pass should confirm this event-CSV
recomputation matches a full rerun before the owner sees these numbers.

## Caveats (read as an adversary)
- Independent reimplementation, causality trace and fill-realism review are still owed before this ships to the owner (see the PREREG's "Independent check" section) -- this run is the BUILDER only.
- Half-spread fallback shares above must be inspected: 72.3% of events priced on the flat 25bps
  fallback (the true minute-of-day table cited in the task was not found on inspection, see
  cell_1564.py's module docstring) -- this is a real weakness in the fill-realism claim, not a
  rounding note; a rebuild with full NBBO coverage (the quote fetch was still filling in the
  background when this run started, ~123/434 days cached) could move every mean by more than the
  gap to the +0.15R pass bar and should run before any owner-facing number.
- The count-matched null draws from the SAME split's non-signal population; a small non-signal
  pool on thin days widens the null.
- Calibration line (report-only) is very thin (n 4-52 per row) and TRAIN/VAL disagree in sign on
  both FAILURE and SUCCESS -- it neither confirms nor refutes the diagnostic that motivated this
  PREREG; do not read it as reproducing `orb_x_hod_diagnostic.md`'s -0.90/-0.54R without a lot
  more overlap.
- Verdict on this population: all three cells (1,564 FAILED-BREAK SHORT, 1,565 VWAP-TARGET SHORT,
  1,566 HELD-BREAK LONG) FAIL the frozen VAL pass bar -- mean_net_R is negative on VAL for every
  cell (-0.325 / -0.648 / -0.050 R). Per the PREREG's own "Independent check and consequences":
  this is NOT yet a valid closure -- it needs the independent rebuild, the full NBBO coverage
  rebuild above, and an adequacy review before being reported as "the failure signal is closed."
## Judge (main session, 2026-09-27 09:10 UTC) — FAIL all three; the failure is a veto, not a trade

* Causality clean (both refuters): events from bars in [09:35, 10:30) only, entry at the 10:30 open; an independent
  reclassification agrees on all 13,316 candidates; event-set Jaccard 0.99 on 1,564 and 1,566 (1,565's VWAP gate was
  under-specified — 0.13, immaterial: both sides are deeply negative).
* 1,564 failed-break SHORT: VAL −0.33 R (t −4.3) builder / −0.28 R (t −3.9) rebuild; gross at mid −0.10 to −0.16 R —
  negative before any cost; with the full quote cache −0.14 R, still far under the bar; only 3 of 21 months positive.
  In % of price the failed names shorted at 10:30 lose MORE than the same-day non-failure shorts (control percentile
  12) — shorting a gap-up name in play at 10:30 loses whether or not its range broke. 1,565 (VWAP target): −0.65 / −0.88 R.
* 1,566 held-break LONG: VAL −0.05 / −0.01 R net, gross +0.07 (cost-negative); the same long on the FAILED names is
  −1.2 / −1.6 R — the failure declared at 10:30 is a real, causal DO-NOT-BUY, worth nothing as a trade in either
  direction on this population.
* Calibration: only 4 / 10 HOD fills fall on failure-declared days after 10:30 (signs disagree) — most of the
  diagnostic's −0.90 / −0.54 R cohort had its ORB loss declared AFTER the HOD fill, i.e. the diagnostic's number was
  not causal at any usable hour. Defects on record: events/week counted over event-weeks (1,565 is 0.6/week, not 6),
  borrow flags from a 2026-09 snapshot (hindsight), 72 % of events on the flat 25 bps spread fallback.
Consequence per PREREG: the failure signal is closed as a trade; it stays as a veto feature (no long on a name whose
opening-range break was stopped by 10:30). Programme count 1,566.
