

# 6 — the fill model: touch-inside-the-signal-bar vs the live capped limit at the NEXT bar open

book trades re-simulated: 1676 of 1676 (bars missing for 0)
split      n  as-is meanR  as-is R/wk |  next-open cap 0.6% meanR    R/wk  fill rate |  cap 1.0% meanR    R/wk
TRAIN    996        0.130        2.44 |                    -0.169   -3.17      45.8% |          -0.174   -3.27
VAL      408        0.322        5.97 |                    -0.155   -2.87      34.1% |          -0.147   -2.73
TEST     272        0.350        6.80 |                    -0.083   -1.62      32.0% |          -0.093   -1.81

  ("no fill" is scored 0R and still consumes the slot, which is what the live book does.)
  t of the next-open-fill book: TRAIN -5.73 VAL -3.39 TEST -1.55