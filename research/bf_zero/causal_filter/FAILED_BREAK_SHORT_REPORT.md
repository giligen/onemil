# FAILED_BREAK_SHORT_REPORT — PREREG_FAILED_BREAK_SHORT.md (cells 1357-1358)

Population (TRAIN+VAL, TEST untouched): 12135. Shortable-proxy excluded: 43.72%. SSR-proxy excluded: 0.38%. Kept for simulation: 6797.

**AVAILABILITY CAVEAT (read first):** `bars_sip.db` has 1-min bars for only ~40% of (symbol, day) pairs
in the kept population (confirmed by direct query: 323/800 random rows had zero rows in `bars`, all 344
trading days are present in the db but per-symbol-day coverage inside those days is partial — this db was
evidently built for a narrower symbol set than the current `features.csv`). `no_bars` accounts for 59.4%
of the "no trade" share below. This is BELOW the 80% availability rail in CLAUDE.md's research protocol.
The 1,958 real trades below are the trades computable from what IS in the cache, not a random sample of
the true population — if bars availability correlates with anything (e.g., liquidity, volatility, how
recently a symbol was added to the universe), the realized sample is biased and the result is NOT
reportable as a clean population estimate until the missing bars are backfilled or the gap is shown to be
uncorrelated with outcome. Treat everything below as informative-only pending that check.


## Cell 1357

No-trade reasons (share of kept population): no_bars 59.4%, r_too_small 6.3%, no_failure 5.5%

NBBO coverage on real trades: 99.9%

| split    |    n |   tpw |   meanR_gross |   meanR |    sd |   t_iid |   t_clust |   WR |    ex5 |   cap5 |   share_stop |   share_target |   share_eod |   green |
|:---------|-----:|------:|--------------:|--------:|------:|--------:|----------:|-----:|-------:|-------:|-------------:|---------------:|------------:|--------:|
| TRAIN    | 1152 | 21.74 |         0.42  |   0.055 | 2.539 |    0.73 |      0.23 | 32.5 | -0.357 | -0.087 |        0.645 |              0 |       0.355 |    0.43 |
| VAL      |  805 | 35    |         0.273 |  -0.176 | 2.74  |   -1.83 |     -1.3  | 24.7 | -0.653 | -0.368 |        0.718 |              0 |       0.282 |    0.35 |
| TRAIN-H1 |  619 | 11.68 |         0.597 |   0.236 | 2.493 |    2.36 |      0.58 | 38.6 | -0.157 |  0.101 |        0.575 |              0 |       0.425 |    0.26 |
| TRAIN-H2 |  533 | 10.06 |         0.215 |  -0.156 | 2.578 |   -1.4  |     -1.24 | 25.3 | -0.589 | -0.306 |        0.726 |              0 |       0.274 |    0.17 |


Placebo means:
| split   | placebo                       |   n |   meanR |
|:--------|:------------------------------|----:|--------:|
| TRAIN   | D1 (time-shuffle, same sym)   | 635 |  -0.768 |
| TRAIN   | D3 (symbol-shuffle, same day) | 231 |  -0.459 |
| VAL     | D1 (time-shuffle, same sym)   | 437 |  -0.927 |
| VAL     | D3 (symbol-shuffle, same day) | 198 |  -0.562 |

VAL week-by-week P&L at R=$100:
wk
2025-12-27/2026-01-02      19.0
2026-01-03/2026-01-09    -648.0
2026-01-10/2026-01-16    2173.0
2026-01-17/2026-01-23    1351.0
2026-01-24/2026-01-30    1306.0
2026-01-31/2026-02-06   -1964.0
2026-02-07/2026-02-13   -4455.0
2026-02-14/2026-02-20    -820.0
2026-02-21/2026-02-27   -2768.0
2026-02-28/2026-03-06   -1961.0
2026-03-07/2026-03-13    1169.0
2026-03-14/2026-03-20    -148.0
2026-03-21/2026-03-27    -955.0
2026-03-28/2026-04-03   -5369.0
2026-04-04/2026-04-10    2739.0
2026-04-11/2026-04-17   -2695.0
2026-04-18/2026-04-24    3972.0
2026-04-25/2026-05-01   -2113.0
2026-05-02/2026-05-08   -1106.0
2026-05-09/2026-05-15     325.0
2026-05-16/2026-05-22    -186.0
2026-05-23/2026-05-29   -2058.0

Cadence bar (VAL):
```
CADENCE BAR  (HOD-S-1357, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median 2.0 wk  P90 5.2 wk        [pass]   gaps: [1, 1, 6, 4, 2]
C2 bleed     P90 0.00 R     cycles net>0 60% [fail]
C3 reds      P10 -27.61 R  min -53.69 R  MDD 183.95 R   under-water max 17 wk   [fail]
C4 green     33%  null 50%                   [fail]
C5 fills/wk  36.59                             [pass]
C6 tail      C6 not audited
C7 power     cycles 5   bootstrap P90-gap 75% UB 7.1 wk    [fail]
diagnostics  ex-top-5% -181.65 R   capped -239.03 R   top-5 share -28.0%   weekly P&L histogram: [-53.7, -44.6, -27.7, -26.9, -21.1, -20.6, -19.6, -19.6, -11.1, -9.5, -8.2, -6.5, -1.9, -1.5, 0.2, 3.3, 11.7, 13.1, 13.5, 21.7, 27.4, 39.7]

```

### Pass-bar criteria — cell 1357: **FAIL**

- 1. net>=+0.10R TRAIN & VAL, VAL t>=2: FAIL
- 2. both TRAIN halves > 0: FAIL
- 3. ex-top-5% > 0 both splits: FAIL
- 4. beats D1 & D3 by >=0.10R on VAL: PASS
- 5. >=3 trades/week VAL: PASS
- VAL D1 mean -0.927 R, VAL D3 mean -0.562 R

## Cell 1358

No-trade reasons (share of kept population): no_bars 59.4%, r_too_small 6.3%, no_failure 5.5%

NBBO coverage on real trades: 99.9%

| split    |    n |   tpw |   meanR_gross |   meanR |    sd |   t_iid |   t_clust |   WR |    ex5 |   cap5 |   share_stop |   share_target |   share_eod |   green |
|:---------|-----:|------:|--------------:|--------:|------:|--------:|----------:|-----:|-------:|-------:|-------------:|---------------:|------------:|--------:|
| TRAIN    | 1152 | 21.74 |         0.396 |   0.023 | 1.529 |    0.51 |      0.13 | 47   | -0.079 |  0.023 |        0.515 |          0.445 |       0.04  |    0.3  |
| VAL      |  805 | 35    |         0.214 |  -0.243 | 1.549 |   -4.45 |     -3.47 | 40.1 | -0.36  | -0.243 |        0.579 |          0.391 |       0.03  |    0.3  |
| TRAIN-H1 |  619 | 11.68 |         0.561 |   0.192 | 1.502 |    3.17 |      0.66 | 52.8 |  0.099 |  0.192 |        0.456 |          0.494 |       0.05  |    0.17 |
| TRAIN-H2 |  533 | 10.06 |         0.204 |  -0.173 | 1.538 |   -2.6  |     -2.28 | 40.2 | -0.286 | -0.173 |        0.583 |          0.388 |       0.028 |    0.13 |


Placebo means:
| split   | placebo                       |   n |   meanR |
|:--------|:------------------------------|----:|--------:|
| TRAIN   | D1 (time-shuffle, same sym)   | 635 |  -0.646 |
| TRAIN   | D3 (symbol-shuffle, same day) | 231 |  -0.576 |
| VAL     | D1 (time-shuffle, same sym)   | 437 |  -0.759 |
| VAL     | D3 (symbol-shuffle, same day) | 198 |  -0.569 |

VAL week-by-week P&L at R=$100:
wk
2025-12-27/2026-01-02     154.0
2026-01-03/2026-01-09   -1804.0
2026-01-10/2026-01-16     475.0
2026-01-17/2026-01-23   -1711.0
2026-01-24/2026-01-30     379.0
2026-01-31/2026-02-06   -1005.0
2026-02-07/2026-02-13   -2205.0
2026-02-14/2026-02-20    -635.0
2026-02-21/2026-02-27   -1588.0
2026-02-28/2026-03-06   -1609.0
2026-03-07/2026-03-13    -261.0
2026-03-14/2026-03-20    -892.0
2026-03-21/2026-03-27     -70.0
2026-03-28/2026-04-03   -3753.0
2026-04-04/2026-04-10     794.0
2026-04-11/2026-04-17   -1736.0
2026-04-18/2026-04-24     367.0
2026-04-25/2026-05-01      36.0
2026-05-02/2026-05-08   -2271.0
2026-05-09/2026-05-15   -1778.0
2026-05-16/2026-05-22     166.0
2026-05-23/2026-05-29    -587.0

Cadence bar (VAL):
```
CADENCE BAR  (HOD-S-1358, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -21.65 R  min -37.53 R  MDD 196.89 R   under-water max 21 wk   [fail]
C4 green     29%  null 50%                   [fail]
C5 fills/wk  36.59                             [pass]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB 11.0 wk    [fail]
diagnostics  ex-top-5% -203.29 R   capped -198.29 R   top-5 share -4.1%   weekly P&L histogram: [-37.5, -22.7, -22.1, -18.0, -17.8, -17.4, -17.1, -16.1, -15.9, -10.1, -8.9, -6.4, -5.9, -2.6, -0.7, 0.4, 1.5, 1.7, 3.7, 3.8, 4.7, 7.9]

```

### Pass-bar criteria — cell 1358: **FAIL**

- 1. net>=+0.10R TRAIN & VAL, VAL t>=2: FAIL
- 2. both TRAIN halves > 0: FAIL
- 3. ex-top-5% > 0 both splits: FAIL
- 4. beats D1 & D3 by >=0.10R on VAL: PASS
- 5. >=3 trades/week VAL: PASS
- VAL D1 mean -0.759 R, VAL D3 mean -0.569 R

## Overall: cell 1357 FAIL, cell 1358 FAIL

Runtime: 0.5 min.
## Verdict (main session, 2026-09-22) — VOID on the availability rail, NOT a negative result
Only **7,090 of the 15,656** feature symbol-days (45.3 %) exist in `bars_sip.db`; `no_bars` is 59.4 % of the kept
population. CLAUDE.md rail 7 requires ≥ 80 % coverage, so cells 1,357–1,358 are VOID and the numbers below are
informative only. The missing pairs are not random: the db was built for a narrower symbol set, so the walked
1,958 trades are a coverage-selected subset. `research/bf_zero/backfill_bars_sip.py` is fetching the 8,566 missing
symbol-days from Alpaca SIP (append-only, resumable); the cells are re-scored on the complete population before
any verdict is recorded. Cell numbers stay reserved for the re-run — the programme count does not advance here.

### The one result that does NOT depend on coverage: the population drifts UP
Placebo D1 (same name-day, random 10:00–14:00 minute, prior-10-min-high stop) is **−0.93 R** and D3 (random other
symbol that day) is **−0.56 R** on VAL, against the real failed-break short's −0.18 R. Shorting this population at
a random time loses roughly half an R to a full R; the failed break identifies the least-bad shorts. Two
consequences, both to be re-tested on the complete data: (1) the short side is fighting an intraday upward drift in
gapper names, so a short book here needs a much larger edge than the long side to clear cost; (2) the long book's
−0.2 R gross is therefore an entry/exit-mechanics problem, not a population that goes down — which points the next
long-side pass at the exit, not at more selection filters.
