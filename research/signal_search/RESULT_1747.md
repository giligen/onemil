# RESULT 1,747 — dividend ex-day / dividend-month pressure (G3), Stage 1 rejection-only — KILL

Sources: Hartzmark-Solomon JFE 2013 (sciencedirect S0304405X13000585); NBER w30688. TEST (2024-26) sealed, never read.

**Data.** Alpaca data API `GET /v1/corporate-actions?types=cash_dividend` (free): earliest ex_date 2016-01-04 (nothing in
2015), so TRAIN is the full 2016-20. No announcement date in the feed -> B2 uses ex-10 trading days. Raw 372,939 events ->
323,624 regular (rate>0, not special, USD, not foreign issuer, no return_of_capital) -> 317,682 after the 3x-trailing-rate
rule. Bars: 3,177 of 5,814 name-eligible payers returned (LOST 2,637, delisted/unknown to Alpaca = survivorship; Stage 1
is rejection-only). Dividend converted to split-adjusted units via raw/split close at ex-1. Universe: 20d median $vol >= $20M
(primary) at ex-2, price >= 5; payer-universe equal-weight mean as the second benchmark.

**B1 (primary), liquid tier, gross excess vs matched control (same name, +-30 d, no dividend within 20 d):**
| split | n | /wk | raw bp | excess bp | t iid | t day-clustered | ex-top-5% | years |
|---|---|---|---|---|---|---|---|---|
| TRAIN | 13,889 | 53 | +6.8 | -4.8 | -1.95 | -0.62 | -37.5 | 16:+17 17:+16 18:-13 19:+4 20:-49 |
| H1 16-18 | 8,311 | 53 | +15.3 | +5.6 | 2.40 | 1.11 | -18.9 | |
| H2 19-20 | 5,578 | 53 | -6.0 | -20.3 | -3.99 | -1.15 | -62.9 | |
| VAL | 8,274 | 53 | +17.1 | +12.2 | 4.57 | 1.73 | -16.6 | 21:+5 22:+10 23:+21 |
MDE80 on VAL 19.8 bp. Tail >5 % share 3 %. Green months VAL 67 %. Versus the universe mean VAL +12.1 bp (t day 5.9), TRAIN +4.8.
Cost (400 VAL events, NBBO 15:55-16:00, first 1,000 quotes): half-spread entry 2.5 / exit 2.4 bp mean (median 1.7); round
trip with spread + 2 bp/leg = 8.9 bp, MOC-only 4 bp -> VAL net ~ +3 bp (excess) / +8 bp (MOC); not the binding kill.

**KILL (literal):** VAL day-clustered t 1.73 < 2; TRAIN halves opposite sign (+5.6 / -20.3); ex-top-5 % on VAL -16.6 <= 0.
Passed: VAL excess > +10 bp, 2022 positive (+9.8), 53 events/week. 2020 is -49 bp. The effect is ~12 bp, sub-MDE and one-tailed.

**B2 run-up (ex-10 -> ex-1):** VAL excess -36 bp (t day -1.4), TRAIN -25: wrong sign vs paper's +17. **B3 post-ex short (40 d):**
VAL short +19 bp vs control (n 347, t 0.2), TRAIN +287 driven by 2020 (n small); short leg vs universe -30 bp; ETB share not
computed (assets file has no shortable column). Neither promoted.

Caveats: mid tier ($5-20M) VAL is contaminated by a bad corporate-action rate (MCHB, 6.5e5 bp yield); not used. Cells: 1,747 (programme 1,747).
Files: 1747_fetch.py, 1747_events.parquet, 1747_score.py, 1747_summary.csv, 1747_cost.csv.
