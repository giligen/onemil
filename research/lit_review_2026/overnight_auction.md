# Overnight family with auction costs (close→open), by book size and cost

universe: close >= $5, 20-day dollar volume >= $10M; volume-shock decile cut on TRAIN at 1.59x ADV20

## V volume shock — n available per day: TRAIN 308, VAL 351, TEST 301
  top-4   | TRAIN gross  +9.3 (t +0.7) net@2bp  +7.3 @5bp  +4.3 @10bp  -0.7 | VAL gross -22.2 (t -0.9) net@2bp -24.2 @5bp -27.2 @10bp -32.2 | TEST gross -23.1 (t -0.7) net@2bp -25.1 @5bp -28.1 @10bp -33.1
  top-10  | TRAIN gross  +2.5 (t +0.3) net@2bp  +0.5 @5bp  -2.5 @10bp  -7.5 | VAL gross -28.0 (t -1.9) net@2bp -30.0 @5bp -33.0 @10bp -38.0 | TEST gross -24.7 (t -1.2) net@2bp -26.7 @5bp -29.7 @10bp -34.7
  top-25  | TRAIN gross  +7.6 (t +1.5) net@2bp  +5.6 @5bp  +2.6 @10bp  -2.4 | VAL gross -17.3 (t -2.0) net@2bp -19.3 @5bp -22.3 @10bp -27.3 | TEST gross  -8.3 (t -0.7) net@2bp -10.3 @5bp -13.3 @10bp -18.3
  top-50  | TRAIN gross  +6.9 (t +2.0) net@2bp  +4.9 @5bp  +1.9 @10bp  -3.1 | VAL gross  -3.7 (t -0.6) net@2bp  -5.7 @5bp  -8.7 @10bp -13.7 | TEST gross  -7.6 (t -0.9) net@2bp  -9.6 @5bp -12.6 @10bp -17.6

## H new 252d high — n available per day: TRAIN 25, VAL 36, TEST 20
  top-4   | TRAIN gross +20.7 (t +1.2) net@2bp +18.7 @5bp +15.7 @10bp +10.7 | VAL gross +12.8 (t +0.5) net@2bp +10.8 @5bp  +7.8 @10bp  +2.8 | TEST gross +26.1 (t +0.8) net@2bp +24.1 @5bp +21.1 @10bp +16.1
  top-10  | TRAIN gross +20.4 (t +2.1) net@2bp +18.4 @5bp +15.4 @10bp +10.4 | VAL gross +12.0 (t +0.8) net@2bp +10.0 @5bp  +7.0 @10bp  +2.0 | TEST gross  +6.3 (t +0.3) net@2bp  +4.3 @5bp  +1.3 @10bp  -3.7
  top-25  | TRAIN gross +26.6 (t +4.6) net@2bp +24.6 @5bp +21.6 @10bp +16.6 | VAL gross +17.3 (t +1.9) net@2bp +15.3 @5bp +12.3 @10bp  +7.3 | TEST gross -11.6 (t -1.0) net@2bp -13.6 @5bp -16.6 @10bp -21.6
  top-50  | TRAIN gross +27.0 (t +5.6) net@2bp +25.0 @5bp +22.0 @10bp +17.0 | VAL gross +15.5 (t +2.3) net@2bp +13.5 @5bp +10.5 @10bp  +5.5 | TEST gross -13.2 (t -1.3) net@2bp -15.2 @5bp -18.2 @10bp -23.2

## B both — n available per day: TRAIN 22, VAL 31, TEST 17
  top-4   | TRAIN gross +21.4 (t +1.2) net@2bp +19.4 @5bp +16.4 @10bp +11.4 | VAL gross +12.6 (t +0.5) net@2bp +10.6 @5bp  +7.6 @10bp  +2.6 | TEST gross +26.0 (t +0.7) net@2bp +24.0 @5bp +21.0 @10bp +16.0
  top-10  | TRAIN gross +22.2 (t +2.2) net@2bp +20.2 @5bp +17.2 @10bp +12.2 | VAL gross +11.7 (t +0.8) net@2bp  +9.7 @5bp  +6.7 @10bp  +1.7 | TEST gross  +7.0 (t +0.4) net@2bp  +5.0 @5bp  +2.0 @10bp  -3.0
  top-25  | TRAIN gross +26.8 (t +4.4) net@2bp +24.8 @5bp +21.8 @10bp +16.8 | VAL gross +17.0 (t +1.8) net@2bp +15.0 @5bp +12.0 @10bp  +7.0 | TEST gross -11.3 (t -0.9) net@2bp -13.3 @5bp -16.3 @10bp -21.3
  top-50  | TRAIN gross +26.9 (t +5.2) net@2bp +24.9 @5bp +21.9 @10bp +16.9 | VAL gross +15.1 (t +2.1) net@2bp +13.1 @5bp +10.1 @10bp  +5.1 | TEST gross -10.4 (t -0.9) net@2bp -12.4 @5bp -15.4 @10bp -20.4

## what it is worth, honestly

N=25 TRAIN: mean weekly return on the book +0.16% (5 sessions) → $+95/week on $60K at 1x, weekly sd $782, weeks green 29/51
N=25 VAL: mean weekly return on the book -1.16% (5 sessions) → $-694/week on $60K at 1x, weekly sd $614, weeks green 6/22
N=25 TEST: mean weekly return on the book -0.62% (5 sessions) → $-373/week on $60K at 1x, weekly sd $669, weeks green 8/14
N=50 TRAIN: mean weekly return on the book +0.14% (5 sessions) → $+82/week on $60K at 1x, weekly sd $735, weeks green 30/51
N=50 VAL: mean weekly return on the book -0.42% (5 sessions) → $-251/week on $60K at 1x, weekly sd $477, weeks green 10/22
N=50 TEST: mean weekly return on the book -0.60% (5 sessions) → $-360/week on $60K at 1x, weekly sd $451, weeks green 6/14