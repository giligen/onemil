# RESULT 1,766 - F5 sector sympathy (EXPLORATORY: sector map below the 80 % rail)

**First caveat: only 64.7 % of population symbol-days (TRAIN 65.3 %, VAL 60.9 % of HOD trades) carry a sector, under the 80 % rail; the cell is EXPLORATORY, not VOID.** The asset-class map has no sector column (symbol, asset_class, name), so sector = 2-digit SIC (secondary 3-digit) from the 1764 EDGAR submissions; wrappers (26.8 % of symbol-days) carry none. Unmapped names are the delisted tail, so heat is undercounted.

Universe: honest gapper book, 7,487 symbol-days, TRAIN 27 wk / VAL 22 wk, HOD comparator trades (NBBO-cost net R). Break minute = the regular rule on every covered symbol-day (matches the comparator entry minute on 100 % of trades). Heat = earlier breaks by other same-sector names. TEST sealed.

Fills per week (TRAIN / VAL): SOLO 21.4 / 22.8, ONE 5.2 / 6.0, LEADER 1.1 / 1.3, LAGGARD 3.6 / 3.2, unmapped 16.6 / 21.4.

Net R: SOLO TRAIN -0.23, VAL -0.15; LAGGARD TRAIN -0.24 (ex-top5 -0.36, fast losers 24 %), VAL -0.29 (t -1.5).

- LAGGARD - SOLO: H1 +0.09, H2 -0.00, TRAIN -0.01 (t -0.06), VAL -0.14 (t -0.71). Signs split; VAL t far below 2.
- Heat 0/1/2/3+ on TRAIN: -0.23, -0.34, -0.17, -0.27; not monotone.
- Filter "LAGGARD only": dR/wk +7.0 on VAL against placebo p95 +8.3 (TRAIN +10.9 vs +11.7); the gain is mechanical dropping from a negative book. LAGGARD fills 3.2/wk.
- Other-sector heat placebo: VAL mean +0.10, p95 +0.32; the real effect (-0.14) sits inside it.
- SIC3 secondary: TRAIN +0.09 (t 0.4), VAL -0.11 (t -0.4), 2.4 fills/wk; also fails.

**Structure: NO.** Only the fill count and the other-sector null pass. Caveats: 35 % unmapped, SIC2 coarse, LEADER small (n 31 / 28). Cells: 1,766 plus the SIC3 read.

CHILDREN:
1. Theme heat from same-day headline keywords instead of SIC, covering the unmapped delisted tail.
2. First-breaker sympathy as a gross-of-cost return predictor for same-sector names before their own break.
3. Market-wide break clustering by minute as a day-regime read.
