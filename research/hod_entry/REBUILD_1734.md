# REBUILD 1,734 (independent, from PREREG prose only)
Rebuilt 5,506 fills (dedupe fill_id), SPY daily_bars, prior-session-close features -> 1734_rebuild_fills.csv.
Cut points (229 base-book days, np.quantile 1/3, 2/3): spy_ret20 0.0066911 / 0.0283342; spy_ret1 -0.0013769 / 0.0039842.

Compared on fill_id (5,506 base rows; 306 FORWARD rows in 1734_fills.csv not compared):
* fill_id sets, base_R, prior date: 0 mismatches. Feature values (1e-9) ret20 / below_50 / ret1: 0 / 0 / 0.
* Labels: spy_ret1 0 mismatches; spy_ret20 18 rows (one day, 2025-12-11): its value sits ON cut 1; first impl. puts it in bottom, mine (x>=cut, as for cut 2) in mid. spy_below_50 labels: first impl. top = above 50d mean (naming only).
* The PREREG does not fix the tie rule (a cut equals a real day's value). A cut-day tie moves 2025-10-30 (33 rows) and others between buckets; freeze it.

spy_ret20 mean base_R (n), first impl. | mine:
| read | bottom | mid | top |
|---|---|---|---|
| TRAIN-H2 | +0.1825 (601) | -0.0875 (1007) | -0.0957 (741) |
| TRAIN-H2 mine | +0.1847 (583) | -0.0841 (1025) | -0.0957 (741) |
| VAL both | +0.0034 (1504) | -0.1340 (448) | -0.0318 (1205) |
Verdict: 18 label mismatches, TRAIN-H2 bottom/mid differ by 0.002 R (tie rule only); VAL and top exact. Bottom-tercile edge reproduces either way.
