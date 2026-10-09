# REPORT cell 1 (vol line): VIX term-structure sleeve, PREREG_1 - 2026-10-09

**VERDICT: VOID by the completeness gate (and FAIL on the numbers it does allow).** Alpaca daily bars start 2016-01-04
(SVXY) and 2018-01-18 (VXX); PREREG demanded 2011-01-03 and >= 99 %. Coverage over 3,965 NYSE days: SVXY 68.3 %
(1,258 LOST = 2011-01-03..2015-12-31), VXX 55.3 % (1,772 LOST = 2011-01-03..2018-01-17). No lost day after each
first bar. VIX/VIX3M 100 %. Nothing below is reportable as the PREREG result; it is descriptive on what exists.
TRAIN-A is therefore 2016-01-04..2017-12-29 (104 wk, SVXY leg only; the VXX leg cannot trade before 2018-01-18).
Setup: $5K slice, fixed notional, R = $50 (1 % of slice; PREREG left R undefined), close-to-close (c2c), SVXY era-1
scaled x0.5 (-1x through 2018-02-27 close; raw in results.csv). Expense ratio charged literally (double counts: ETP
prices are already net, <= $45/yr).

## Cost (15:58 ET first quote, SIP, last 60 sessions)
Half-spread bps: SVXY mean 0.82 p90 0.88; VXX mean 2.60 p90 2.86. 2x cost moves any total by < $150: cost is not the issue.

## Table (total $ over the half; t = week-clustered; xTop5 = weekly total after dropping the top 5 % of weeks)
| variant (c2c, scaled) | half | trades | total @1x | total @2x | t_iid(trades) | t_week | xTop5 @1x |
|---|---|---|---|---|---|---|---|
| base 1.05/0.95 | A | 10 | 4,394 | 4,385 | 1.33 | 2.63 | 2,571 |
| base 1.05/0.95 | B | 122 | 7,683 | 7,546 | 1.02 | 1.24 | **-8,457** |
| thr 1.03/0.97 | A / B | 7 / 111 | 4,462 / 5,439 | 4,456 / 5,301 | 1.30 / 0.80 | 2.63 / 0.88 | 2,622 / -10,086 |
| thr 1.10/0.90 | A / B | 14 / 131 | 3,760 / 7,546 | 3,748 / 7,419 | 1.16 / 1.05 | 2.38 / 1.41 | 1,968 / -6,369 |
| short-only 1.05 | A / B | 10 / 101 | 4,394 / 2,958 | 4,385 / 2,876 | 1.33 / 1.16 | 2.63 / 0.77 | 2,571 / -4,638 |
| hyst5 1.05/0.95 | A / B | 2 / 24 | 4,799 / 1,931 | 4,797 / 1,905 | n/a / 0.53 | 2.78 / 0.30 | 2,664 / -12,062 |
| PLACEBO base, r lagged 20d | A / B | 12 / 122 | 4,570 / -2,970 | 4,560 / -3,108 | 2.18 / -0.41 | 2.74 / -0.55 | 2,412 / -12,097 |
Next-open entry (15:58 proxy), base: A 4,311 / B 9,364 @1x, xTop5 2,506 / -6,033. Unscaled (-1x era-1): A 8,883, B 7,103.
Mean weekly $ (base c2c scaled): A +42.25 (104 wk), B +16.77 (458 wk).

## Tail lines (base rule, $5K, unrounded)
* 2018-02-05 close-to-close: **$0.00** - r(2018-02-02) = 0.98382 (inside 0.95..1.05) so the rule was flat. The crash was
  the hypothetical SVXY hold: 211.42 -> 144.04 (2/5, -31.87 % = -$1,593.5) -> 24.48 (2/6, -83.00 % = -$4,150.2).
* 2018-02-06 c2c: **-$86.35254** (long VXX entered 2/5 close at r = 0.75375, VXX fell).
* Next-open exit: 2/5 **-$270.67723** raw (-$135.63797 at the x0.5 scaling), 2/6 **-$315.01469** (VXX entered at 2/6 open).
* Worst day, half A: -$662.14 (2016-06-24, Brexit). Half B: -$1,032.14 (2025-04-09, long VXX into the tariff-pause reversal);
  short-only B worst -$845.75 (2020-06-11). Placebo B worst -$2,075 (2018-02-06 scaled), -$4,150 raw.

## Cadence-bar scorecard (base, c2c, scaled, R = $50; C6 not audited)
| | A (104 wk) | B (458 wk) |
|---|---|---|
| C1 gap median / P90 <= 3 / 6 | median 6 -> FAIL | median 9 -> FAIL |
| C2 bleed P90 >= -4R, cycle net>0 >= 75 % | PASS | FAIL |
| C3 P10 >= -2R, MDD <= 8R | P10 -2.87R, MDD 22.8R -> FAIL | P10 -4.81R, MDD 49.5R -> FAIL |
| C4 green >= 55 % and null + 10pp | 70.5 % vs null 50.3 % -> PASS | 59.0 % vs 50.0 % (+9.0pp) -> FAIL |
| C5 >= 3 fills/wk | 0.10 -> FAIL | 0.27 -> FAIL |
| C7 >= 10 cycles | 8 cycles -> FAIL | 31 cycles, bootstrap bound -> FAIL |
Same-signed in both halves: yes at the point estimate (A +42.25, B +16.77 $/wk), but B's ex-top-5 % weeks total is
-$8,457 (the whole B profit is a handful of weeks) -> the ex-top-5 % > 0 requirement FAILS in B for every variant.
Placebo (20-day stale r): A +4,570 (as good as the real rule: the 2016-17 SVXY bull run, signal not identified),
B -2,970 (c2c) / -168 (next-open): ~0 in B as required, but A fails to separate. C1-C3/C5 are mostly a unit effect of
R = $50 (a 5R strong week = 5 % of the slice); the dollar results above are unit-free.

## MDE (weekly mean detectable at t 2.5, sd of the weekly series)
A: $40.10/wk over 104 weeks (observed +42.25: at the MDE, t 2.63). B: $33.82/wk over 458 weeks (observed +16.77, t 1.24:
below the MDE). Half A holds 10 trades and one regime; it cannot carry a verdict.

## Own caveats
* Gate failure is structural (Alpaca history), not a fetch loss; the PREREG sample (2011-2015, incl. Aug-2011 and the
  2015 VIX events; old-series VXX before 2018-01-18) was never tested. Sourcing it is a new data choice for the owner.
* Pre-2018 VXX = different product (series B ETN from 2018-01-18); VXX leg in B only. 71-79 legs per run skipped (no ETP bar).
* Entry/exit at the close with the signal computed from the same close (c2c) is not executable at 15:58 with the CBOE close
  (published ~16:15); the next-open variant is the executable one (B +9,364 @1x), and its spread is the 15:58 one, not the open's.
* x0.5 scaling of -1x-era daily returns is an approximation of the -0.5x product (path-dependent compounding ignored).
* Hysteresis semantics (my choice): a state flips only after 5 consecutive closes of the new raw state, entries and exits.
* 5 variants x 2 execs x 2 scalings x 2 costs x 2 halves + placebo = 96 cells counted; full numbers in results.csv.
Files: run.py, fetch.py, trades.csv, weekly.csv, results.csv, run.log, data/.
