# REPORT — multiday_catalyst: cells 1,285 / 1,286 (catalyst-day continuation, 3-day hold)

PREREG: `eabe3e7` (research/multiday_catalyst/PREREG.md), committed alone before any scoring. Score.py / analysis.py
in this directory reproduce every number below from `data/cache.db`, `orb_news_catalyst_nightly.csv`,
`orb_asset_class_map_20260711.csv`, `pit_listings.py`. Cells counted: 1,285 (signal) + 1,286 (control) = the two
pre-registered; no additional cell was scored.

## Verdict: REFUTED at VAL — net R is negative, not the pre-registered positive. Do not ship.

Signal net R is negative in BOTH non-sealed splits (TRAIN -0.061R, VAL -0.191R), signal-minus-control is
negative in both splits, and the cadence bar fails all seven criteria on both splits. Nothing here clears the
PASS BAR; nothing here supports "some edge, wrong size" either — the point estimate leans negative, most
clearly in VAL.

## Headline numbers

| Split | n (sig/ctl) | Signal net R (clustered t) | Control net R (clustered t) | Signal minus Control (t) | entries/wk (10-slot) |
|---|---|---|---|---|---|
| TRAIN 2025 | 139 / 13,288 | -0.061R (t=-0.45) | -0.042R (t=-0.78) | -0.019R (t=-0.13) | 2.89 |
| VAL 2026-01..05 | 53 / 6,870 | -0.191R (t=-1.26) | -0.095R (t=-1.30) | -0.095R (t=-0.57) | 2.89 |
| TRAIN halves | H1 n=50: -0.095R / H2 n=89: -0.042R | same-signed (both negative) | | | |
| 2024-07..12 | not runnable, either cell | news feed starts 2025-01-02; absence of a news record before that date is missing data, not confirmed "no news" -- cannot score signal OR control. (Corrects the PREREG addendum's "control-only" note -- a mid-run correction, recorded here.) | | | |
| TEST >=2026-06-01 | 32 / 4,798 | SEALED -- counted only, never aggregated or read. | | | |

$/week at $66K book, 1% risk ($660/trade): TRAIN approx 3.02/wk x -0.061R x $660 approx -$121/wk; VAL approx
2.94/wk x -0.191R x $660 approx -$371/wk.

MDE (day-clustered, 80% power): TRAIN 0.38R, VAL 0.42R -- both exceed the +0.15R pass bar, so this test cannot
tell a small positive effect from zero. It CAN tell that the point estimate itself is negative, more so in VAL.

## CADENCE BAR -- both splits fail all 7 criteria

```
CADENCE BAR  (multiday_catalyst, TRAIN, live config: 10 slots, N/A, R = $660)
C1 gap       median 38.0 wk  P90 38.0 wk        [fail]   gaps: [38]
C2 bleed     P90 -12.60 R     cycles net>0 0% [fail]
C3 reds      P10 -2.09 R  min -9.02 R  MDD 16.25 R   under-water max 47 wk   [fail]
C4 green     32%  null 50%                   [fail]
C5 fills/wk  2.62                             [fail]
C6 tail      C6 not audited
C7 power     cycles 1   bootstrap P90-gap 75% UB 22.9 wk    [fail]

CADENCE BAR  (multiday_catalyst, VAL, live config: 10 slots, N/A, R = $660)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []      (0 strong weeks -- no cycle ever closes)
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -2.00 R  min -3.98 R  MDD 13.10 R   under-water max 22 wk   [fail]
C4 green     23%  null 50%                           [fail]
C5 fills/wk  2.41                                     [fail]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB N/A wk    [fail]
```
TRAIN produced exactly one +5R week in 53 weeks; VAL produced zero. Green-week share (32% / 23%) is BELOW the
50% count-matched null on both splits -- the book is not even coin-flip-consistent, it is worse than a coin flip.

## Diagnostics

| | TRAIN signal | VAL signal | TRAIN control | VAL control |
|---|---|---|---|---|
| gross R | -0.031 | -0.157 | +0.079 | +0.019 |
| cost R (15:55 half-spread) | 0.030 | 0.034 | 0.121 | 0.114 |
| WR | 36.7% | 45.3% | 42.8% | 38.9% |
| avg win / avg loss (R) | +1.28 / -0.84 | +0.77 / -0.99 | +1.41 / -1.13 | +1.53 / -1.13 |
| ex-top-1% / ex-top-5% R | -0.109 / -0.287 | -0.232 / -0.312 | -0.129 / -0.297 | -0.180 / -0.353 |
| top-5 share | -274% (denominator ~0) | -82% | -15% | -11% |
| MDD (10-slot book) | -19.5R / approx -$12.9K | -16.0R / approx -$10.6K | -983R / approx -$649K (unlimited slots, not the live book) | -1177R / approx -$777K |
| holding profile D+1/D+2/D+3 (mean R) | -0.049 / -0.082 / -0.014 | +0.034 / -0.166 / -0.204 | +0.029 / +0.095 / +0.112 | +0.070 / +0.074 / +0.063 |
| wrappers | 0 of 139 (news-catalyst selection never hit an identified wrapper) | 0 of 53 | 2,090, mean -0.062R (worse than the 11,198 stocks at -0.038R) | 1,299, mean -0.137R (worse than the 5,571 stocks at -0.085R) |

Cost is NOT the story -- gross is already negative in both signal splits; the 0.03-0.15% half-spread is a
minor tax on a gross-negative book. Notable: control's D+1 to D+3 P&L climbs (continuation exists in the broad
"green day" population), but SIGNAL's D+2/D+3 in VAL turns sharply negative (-0.17R, -0.20R) -- a post-news
fade, not continuation.

## Checklist items

1. Independent reimplementation: not run this pass (single-agent budget; disclosed, not a substitute for the
   requirement -- a negative result going in front of the owner should still get one before any live decision
   is made on it).
2. Causality trace: universe gate (pit-listed + price + causal ADV20 + causal dollar-vol pctile), news flag,
   green-day flag all computable at/before close of D; entry uses D's own close (no look-ahead); stop/exit
   reference only D+1..D+3.
3. Price-scale check: `intraday_bars_1min` timestamps are UTC, not ET -- a first pass compared the wrong
   clock hour and reported 174/200 "mismatches"; corrected to convert to America/New_York and match the 16:00
   (or 15:59) print specifically. Corrected result: 183/200 signal trades had a matching 15:59/16:00 bar (17
   had none -- `intraday_bars_1min` covers only 7,941/13,620 symbols, populated selectively, not a full daily
   tape); of those 183, 58 (31.7%) disagree from the daily_bars close by >0.5% -- `daily_bars` close and the
   literal 16:00 minute-bar print are not tightly aligned about a third of the time, a real data-quality
   caveat on the "16:00 auction print" language; `daily_bars` close is used as the entry/exit basis regardless.
4. Fill realism: stop fills at min(open, stop) on a gap-through; half-spread charged on the entry auction and
   on a genuine stop fill, not on the D+3 close exit (also an auction print) -- disclosed choice, not hidden.
5. Tail dependence: ex-top-1%/5% reported above; C6 tail-obtainability audit not run (no trade cleared +3R in
   the signal cell to audit -- TRAIN top win +1.28R avg, nothing near a lottery-ticket monster).
6. Multiplicity: 2 cells scored (1,285 signal + 1,286 control), exactly the pre-registered count. One design
   decision changed mid-run and is disclosed above (2024H2 corrected from "control-only" to "not runnable,
   either cell").

## Split-rail exclusions

Signal: 246 candidates -> 224 raw trades (7 excluded for a >40% overnight return inside the D..D+3 window
[2.8%], 0 for a definition-feed 'M' corporate-action record, 2 for R<1% of price [0.8%]); TRAIN+VAL subset used
above (139+53=192; the remainder falls in the sealed TEST window or 2024H2). Control: 57,898 candidates ->
25,048 raw trades (38 split-excluded [0.07%], 9,516 R-skipped [16.4%]); 92 further dropped here for falling
before the 2025-01-02 news-feed start (the 2024H2 correction above). Corporate-action proxy
(`security_update_action == 'M'` in the PIT definition feed) never fired on any candidate -- disclosed as a
plausibly weak proxy (routine record updates, not necessarily splits/M&A); the 40% overnight-return rail is
the load-bearing filter.

## The one caveat that alone could most explain the headline

Low power. TRAIN/VAL signal n = 139/53 trades, day-clustered SE approx 0.135R/0.151R -> MDE approx 0.38R/0.42R,
larger than the -0.06R/-0.19R point estimates themselves. A wider window or more catalyst-days could sharpen
this number in either direction. It does NOT reverse the sign: both splits, and the halves-of-TRAIN, land
negative; VAL (the pre-registered pass-bar split) is 1.26 SE below zero with the worst D+2/D+3 holding-day
profile of any cell in this report. This is a genuine negative lean, not a coin flip dressed as noise -- but
the sample is too small to rule out "roughly zero, badly measured" as the true state.

## Coverage

`daily_bars` 2024-06-03..2026-09-18 (13,620 symbols); PIT listings 2024-07..2026-09 (2024-06 excluded, outside
coverage); news feed 2025-01-02..2026-09-18 (2024H2 not covered -- see above).
