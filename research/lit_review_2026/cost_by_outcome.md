# Cost by OUTCOME — correcting the flat round-trip assumption

A target exit rests on a limit and pays NO exit spread. A stop or a 15:55 close crosses the spread at the EXIT minute.
Costs are in R, i.e. as a fraction of the trade's own risk.

## by exit type
          n  s_entry_bps  s_exit_bps  entry_R  exit_R  total_R
why                                                           
eod     293         37.8        14.0    0.055   0.023    0.083
stop    293         40.1        29.2    0.115   0.073    0.194
target  288         40.0        27.8    0.126   0.000    0.126

## spread widening from entry to exit (median ratio s_exit / s_entry)
why
eod       0.412
stop      0.875
target    0.708

## by price band and exit type (median total cost in R)
why       eod   stop  target
pb                          
$10-20  0.072  0.191   0.102
$20-50  0.071  0.191   0.096
$5-10   0.125  0.278   0.111
$50+    0.070  0.185   0.161

## the number that replaces the old one
observed exit mix: {'eod': 0.335, 'stop': 0.335, 'target': 0.33}
BLENDED expected cost = 0.135 R per trade (weighting each exit type by how often it happens)
the OLD flat assumption (one full entry-minute spread on every trade) = 0.191 R
so the flat model was OVERstating the true cost by 0.057 R per trade.