# REBUILD 1,735 (independent, prose + paths/signals parquet; TRAIN 7,390 + VAL 4,745, TEST untouched)
Mismatch rows vs 1735_per_signal.csv at 0.001 R (12,135 matched on day,symbol): **b0 0, V1 0, V4 0** (max |diff| 9e-16).
B0 also equals hod_exit_lab/b0_trades.csv (0 mismatches; 125 rows there have NaN net_R).

Mean net R, my build: B0 TRAIN -0.2212, VAL -0.3043.
Paired dR vs B0 (mine = file): V1 TRAIN +0.1763, VAL +0.2169; V4 TRAIN -0.0337, VAL -0.0055.

Spec gaps resolved to match (first pass of mine differed; resolved after seeing the file's numbers, so NOT a blind match):
1. Stop/target are tested from the bar AFTER the entry bar (entry bar ignored).
2. EOD exit = 15:55 bar open; if that bar is missing, the last bar's close.
3. Bps cost = 2 bp x (entry + exit price) / R, not 4 bp x entry (matters for big winners, up to 0.002 R).
4. Paths have missing minutes (71 % of paths): V4 decides on the LAST bar with minutes_since_entry <= 20 and
   exits at the open of the next row. Literal "bar at minute 20" misses ~650 signals.
Recommend PREREG_1735 state rules 1-4 explicitly.
