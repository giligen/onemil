# R8 - implementation E: F6-PDR under the LIVE ENGINE conventions

R8 - implementation E: the F6-PDR book re-scored under EXACTLY the LIVE ENGINE's conventions.

E = trading/red_to_green.py (the shipped spec module) + trading/hod_break.{entry_fill, walk_exit, run_book}:
  prior day      prior row of the daily panel (the study stand-in for the engine's daily_bars prior row;
                 R4 proved universe.csv and the panel agree on prior-day OHLC to 1e-6 on all 644,580 keys)
  precondition   o[0] of the FIRST RTH bar < prior close   AND   (prev_high-prev_low)/prev_low >= 8%
  level          prior close x 1.003                       (red_to_green.level_for, level_buffer=0.003)
  floor          (run_hi[i-1]-run_lo[i-1])/run_lo[i-1] >= 5%   -- denominator is the RUNNING LOW
  scan           the first bar i>=1 at which the floor holds AND h[i] >= level AND run_lo[i] < level;
                 bars that fail the floor are SKIPPED, not fatal (red_to_green.detect's `continue`)
  cut            m[i] > 840 on the SIGNAL bar aborts the day (detect returns None)
  stop           run_lo[i], lowest low 09:30 THROUGH the signal bar
  fill           the next bar that prints, at its open, iff open <= level x 1.006 (hod_break.entry_fill)
  floors         entry >= $5 (engine min_price), (entry-stop)/entry >= 1% (red_to_green.r_ok)
  exits          hod_break.walk_exit from the bar AFTER the fill bar: eod(m>=955)@open beats stop(l<=stop)
                 @min(stop,open)*0.999 beats target(c>=target)@target;  partial = half at +2R then stop->entry
  book           trading.hod_break.run_book(rows, 12, 4)
  cost           contract (c): half = 0.5*(spread_cc_bps/100)/max(r_pct,0.05); entry 0.25*half;
                 exit legs stop 0.875 / eod 0.412 / target 0.875 x half   (cc band on the ENTRY price/minute)


E candidates (pre-book): 13541   population: 273488 PDR>=8 symbol-days

## exit hold
split    n  tpw   meanR   gross     t   WR  stopP   wkR  green  worst     ex1     ex5    cap3  totR   mdd
TRAIN 1500 28.3 -0.0268  0.0182 -0.64 38.5   42.8 -0.76   0.51  -20.8 -0.1234 -0.2879 -0.1252 -40.2 -60.8
  VAL  664 30.2 -0.0117  0.0363 -0.17 38.6   45.2 -0.35   0.59  -26.7 -0.1349 -0.3060 -0.1433  -7.8 -37.0
 TEST  427 30.5 -0.1017 -0.0561 -1.58 37.7   41.9 -3.10   0.43  -18.1 -0.1625 -0.3172 -0.1562 -43.4 -55.5

monthly net R: 2025-01 -11.6 (129) | 2025-02 -16.0 (114) | 2025-03 -11.1 (127) | 2025-04 +7.3 (133) | 2025-05 +19.4 (122) | 2025-06 -29.2 (119) | 2025-07 +10.5 (126) | 2025-08 +2.9 (119) | 2025-09 -0.1 (122) | 2025-10 -23.8 (152) | 2025-11 +3.1 (113) | 2025-12 +8.3 (124) | 2026-01 -20.5 (131) | 2026-02 +6.7 (122) | 2026-03 -2.9 (142) | 2026-04 +28.1 (132) | 2026-05 -19.2 (137) | 2026-06 -10.2 (134) | 2026-07 -25.1 (141) | 2026-08 -14.1 (129) | 2026-09 +6.0 (23)
months green 9/21

## exit r2
split    n  tpw   meanR   gross     t   WR  stopP   wkR  green  worst     ex1     ex5    cap3  totR   mdd
TRAIN 1713 32.3 -0.0389  0.0077 -1.43 41.7   37.6 -1.26   0.43  -14.8 -0.0591 -0.1449 -0.0389 -66.6 -80.1
  VAL  761 34.6 -0.0606 -0.0115 -1.46 42.4   39.9 -2.10   0.55  -25.5 -0.0823 -0.1671 -0.0606 -46.1 -60.1
 TEST  491 35.1 -0.0979 -0.0509 -1.99 39.5   38.5 -3.43   0.43  -16.3 -0.1193 -0.2084 -0.0979 -48.1 -48.7

monthly net R: 2025-01 -31.7 (147) | 2025-02 -8.0 (124) | 2025-03 -17.3 (142) | 2025-04 -3.0 (156) | 2025-05 +11.5 (144) | 2025-06 -8.4 (132) | 2025-07 +15.9 (149) | 2025-08 -6.1 (136) | 2025-09 -2.9 (133) | 2025-10 -23.9 (169) | 2025-11 -1.8 (132) | 2025-12 +9.2 (149) | 2026-01 -13.2 (146) | 2026-02 +13.6 (141) | 2026-03 -20.1 (161) | 2026-04 -7.1 (156) | 2026-05 -19.3 (157) | 2026-06 -17.0 (157) | 2026-07 -22.5 (162) | 2026-08 -8.2 (145) | 2026-09 -0.5 (27)
months green 4/21

## exit partial
split    n  tpw   meanR   gross     t   WR  stopP   wkR  green  worst     ex1     ex5    cap3  totR   mdd
TRAIN 1532 28.9 -0.0381  0.0089 -1.18 42.0   44.9 -1.10   0.45  -15.0 -0.0951 -0.2166 -0.0698 -58.4 -76.8
  VAL  679 30.9 -0.0501 -0.0004 -0.97 41.5   47.3 -1.55   0.55  -24.4 -0.1178 -0.2428 -0.0932 -34.0 -45.1
 TEST  434 31.0 -0.0805 -0.0325 -1.44 40.6   44.0 -2.50   0.43  -17.6 -0.1196 -0.2425 -0.0931 -35.0 -42.6

monthly net R: 2025-01 -27.6 (134) | 2025-02 -10.7 (115) | 2025-03 -14.5 (130) | 2025-04 +1.6 (136) | 2025-05 +15.4 (123) | 2025-06 -19.4 (120) | 2025-07 +12.7 (129) | 2025-08 -0.2 (121) | 2025-09 -1.9 (123) | 2025-10 -24.2 (155) | 2025-11 -2.3 (115) | 2025-12 +12.6 (131) | 2026-01 -13.7 (132) | 2026-02 +8.2 (125) | 2026-03 -17.1 (144) | 2026-04 +9.5 (136) | 2026-05 -20.9 (142) | 2026-06 -3.6 (136) | 2026-07 -20.3 (144) | 2026-08 -11.7 (130) | 2026-09 +0.7 (24)
months green 7/21

