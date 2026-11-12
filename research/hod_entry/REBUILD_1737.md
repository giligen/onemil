# REBUILD 1,737 — M1 (retest bid), independent from the PREREG prose
Bars: bars_sip.db read-only. Floor read from config.yaml `hod_break.min_r_pct: 1.5` -> 1.5 %. Output: `1737_rebuild_m1.csv` (8,245 arms; 8,061 are in file 1, 184 are not and none would fill).

**Mismatches vs 1737_orders.csv (M1, 8,061 common rows, 3,653 filled in both):** fill flag 0, c_m 0, fill_m 0, fill_px 0, stop 0, half_spread 0, R 0. net_R under the PREREG-literal cost (stop leg = half_spread + 13 bp, NO extra 2 bp; R = fill x 1.0002 - stop): 2 (NVA 2025-10-03, PRE 2026-03-11 — stop hit on the fill bar itself; file 1 exits below the stop, I exit at the stop). Under the brief's wording (stop leg ALSO pays 2 bp; R = fill - stop) net_R differs on all stops by ~0.01 R (convention, not a bug).

| M1, 1.000 rule (strict low < level) | n | mean net R | day-t | iid t | stop share |
|---|---|---|---|---|---|
| TRAIN (brief cost / PREREG cost) | 1,564 | +0.183 / +0.190 | 3.09 / 3.22 | 4.75 | 49 % |
| VAL | 2,089 | +0.181 / +0.186 | 3.63 / 3.73 | 5.40 | 49 % |
| 10 bp through-trade, TRAIN | 1,456 | +0.147 / +0.156 | 2.42 / 2.57 | 3.67 | 50 % |
| 10 bp, VAL | 1,964 | +0.140 / +0.144 | 2.84 / 2.92 | 4.06 | 50 % |

Fill share of placed orders 45.7 % / 45.1 % (42.5 / 42.4 % at 10 bp). Mean excluding the top 5 % of fills: +0.087 / +0.086 (not tail-carried).

**Adverse selection (B0 net R):** FILLED -0.096 TRAIN / -0.160 VAL; UNFILLED-with-C -0.334 / -0.379; never-confirmed (no close >= level) -1.37. The bid fills the better B0 arms, no adverse direction on B0.

**Capped book (8 fills/day, 4 concurrent, first-come, fill_m order):** 1.000: TRAIN n 853 +0.102 R t 1.6; VAL n 784 +0.129 R t 1.9; fills/wk 33 / 37 (uncapped 60 / 99). 10 bp: TRAIN +0.041 t 0.66; VAL +0.094 t 1.5. The cap halves the edge and kills significance.

**Suspicious / caveats:** queue position ignored (a strict one-tick-through fill is optimistic; 10 bp costs ~0.04 R); fill-bar stop-vs-fill order unobservable (variants moved the mean by < 0.015 R: stop incl. the break bar +0.172 VAL vs +0.181); look-ahead check passed (every fill/cancel decision re-derived on bars truncated at that bar, 0 differences over 8,245 arms).
