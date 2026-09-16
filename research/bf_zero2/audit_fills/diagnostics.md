# Fill-audit diagnostics — pool 25,876 candidates, book 1,676 trades

## 1 — entry bar OPENS above the fill level (a stop-buy cannot fill below the open)
  pool: 8,632 of 25,876 (33.4%) | median gap-through 0.47% of the entry price, p90 2.92%, max 160.20%
  book: 924 of 1,676 (55.1%) | median gap-through 1.59% of the entry price, p90 4.45%, max 16.72%
  book gap-through by split: TRAIN 52.2% | VAL 58.8% | TEST 60.3%
  book mean net R, gapped vs not: TRAIN gapped +0.221 (n 520) clean +0.017 | VAL gapped +0.398 (n 240) clean +0.208 | TEST gapped +0.498 (n 164) clean +0.132

## 2 — the entry bar itself trades through the stop (the sim only looks from the next bar)
  pool: 309 of 25,876 (1.2%)
  book: 118 of 1,676 (7.0%)
  of those book trades, the study booked: {'stop': 96, 'target': 13, 'eod': 9}
  their mean net R as booked: -1.154 vs -1.099 if stopped at the entry bar

## 3 — stop fills
  stop exits in the book: 688 (41%), mean booked R -1.223 (a clean −1R exit would be −1.0 minus cost)
  mean rr_base on stop exits -1.155 → the min(stop, open) rule already books -15.5% of R beyond −1R on average (gap-downs ARE handled)
  extra slip 25 bps → mean rr on those trades -1.205
  extra slip 50 bps → mean rr on those trades -1.287
  extra slip 100 bps → mean rr on those trades -1.451

## 4 — target fills (conservative check)
  target exits in the pool 2,625; the rule needs a bar CLOSE >= entry+2R, so the bar traded through the fill price by construction (0 impossible fills).
  book trades whose HIGH reached +2R before the booked exit (a resting limit would have filled; the study did not): 592 of 1,676 (35.3%)
  of those, 38 exited at something other than the target, booking a mean -0.707R instead of +2R → the close-fill rule is CONSERVATIVE by ~+0.061R per book trade

## 5 — the 15:55 exit
  eod exits 419 (25%), mean net R +0.187; the half-spread IS charged on them (score3.py line 40: cost unless why == target)
  fill = the OPEN of the first bar >= 15:55; mean(open − close of that bar)/R over eod exits is reported in the variants table (eod10 = 10 bps worse)

## 6 — liquidity at $100 of risk per trade
  pool: median notional $1,763 | median $vol in the 5 min after entry $307,196 | median position as a share of it 0.53%
  book: median notional $2,626 | median $vol in the 5 min after entry $280,633 | median position as a share of it 0.98%
  book trades above 1% of the 5-min $ volume: 833 of 1,676 (49.7%) | their mean net R +0.059 vs +0.357 for the rest
  book trades above 2% of the 5-min $ volume: 643 of 1,676 (38.4%) | their mean net R +0.059 vs +0.302 for the rest
  book trades above 5% of the 5-min $ volume: 358 of 1,676 (21.4%) | their mean net R +0.035 vs +0.256 for the rest
  NOTE: at $100 risk/trade the book is tiny. Scaling: the same trades at $2,000 risk multiply the position by 20 → 88% of book trades would then exceed 1% of the 5-min $ volume.

## 7 — tape that ends before 15:55 (halts / no prints)
  pool: exits taken at the LAST bar because the tape ends before 15:55: 675 of 25,876 (2.6%)
  book: exits taken at the LAST bar because the tape ends before 15:55: 15 of 1,676 (0.9%)
  their mean net R as booked +0.307 (n 15), contribution +4.6R of the book total +349.4R
  book trades with a >= 5-minute hole in the tape between entry and exit: 568 of 1,676 (33.9%)
  book trades with a >= 15-minute hole (a plausible halt): 200 of 1,676 (11.9%)
  the >=15-min-hole trades book -0.028R mean, -5.5R total
  book trades whose symbol has no bar at or after 15:55 at all: 116 of 1,676 (6.9%)

## 8 — costs
  the study charges 0.5 x 40 bps / r_pct in R units on every non-target exit;
  median r_pct in the book 3.81% → median charge 0.053R, mean charge 0.067R
  REAL NBBO at the book's own fill minute (n 837): median full spread 1.519% of price, mean 2.934%, p75 3.777%, p90 7.217%
  by price band: (5.0, 10.0] med 2.029% mean 3.308% (n 288) | (10.0, 20.0] med 1.392% mean 3.263% (n 235) | (20.0, 50.0] med 1.188% mean 2.505% (n 202) | (50.0, 1000000000.0] med 1.170% mean 2.055% (n 112)
  the study assumes a 40 bps full spread → half 20 bps. Half the REAL median is 76 bps, half the real MEAN is 147 bps.
  entry: the study fills 30 bps through the level. Half the real spread exceeds 30 bps on 84% of sampled trades → the 30 bps entry slip does NOT double-count the exit half-spread; on those trades it UNDER-charges the entry.