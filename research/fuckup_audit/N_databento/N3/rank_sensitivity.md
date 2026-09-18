# Top-20 selection sensitivity to the ADV translation

The paper filters on 1M CONSOLIDATED shares; we hold Nasdaq-venue volume only, so the
threshold is scaled by the measured venue share.  Overlap = share of the primary
top-20 picks that also appear in the alternative threshold's top-20 for the same day.

| ADV_min (XNAS shares) | implied consolidated | picks | overlap with primary |
|---|---|---|---|
| 118,424 | 1.0M @ share .118 | 25,135 | 100.0% |
| 200,000 | 1.7M @ share .118 / 1.0M @ .20 | 25,125 | 60.4% |
| 300,000 | 2.5M @ share .118 / 1.0M @ .30 | 25,120 | 37.3% |
