# RESULT -- cell 1,700g: beat SPY in MOST years -- risk-adjusted / vol-scaled momentum

PREREG_1700g.md (FROZEN). 9 cells, U2 weekly equal-weight, cost=5bps/side+half-spread (capped 20bps), null=300 draws/cell (PREREG allows 1,000 only if the whole run clears 10min; 9 cells x 508 weekly periods extrapolates over that budget -- stated, not hidden, matching 1700d/e precedent). Windows: whole 2017-01-01..2026-09-30, H1 ..2021-12-31, H2 2022-01-01..

## By-year RETURN %, all 9 cells + SPY (2026 partial, through Sep)

| cell | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|
| SPY | +21.3% | -4.3% | +29.2% | +19.3% | +28.6% | -18.0% | +24.7% | +27.9% | +16.5% | +12.8% |
| V1_N20 | +12.4% | -5.5% | +32.3% | +68.0% | -11.9% | -15.7% | +20.6% | +68.9% | +59.8% | +49.8% |
| V1_N50 | +18.6% | -3.0% | +25.4% | +64.6% | +1.9% | -16.6% | +16.2% | +49.8% | +44.4% | +31.8% |
| V2_N20 | +19.9% | -6.1% | +14.8% | +80.7% | +16.5% | -1.9% | +15.2% | +50.5% | +36.6% | +64.7% |
| V5_N50 | +19.1% | -5.4% | +22.9% | +61.2% | +14.2% | -11.3% | +15.4% | +42.8% | +27.5% | +40.2% |
| V3_N20 | +12.4% | -6.0% | +23.3% | +11.8% | +2.2% | -6.7% | +15.2% | +38.0% | +17.2% | +15.0% |
| V3_N50 | +18.6% | -3.2% | +21.5% | +10.1% | +4.6% | -10.6% | +14.7% | +35.4% | +16.2% | +12.6% |
| V4_N20 | +19.9% | -5.7% | +12.1% | +10.7% | +12.5% | -1.5% | +15.1% | +40.8% | +16.7% | +22.1% |
| V6_N20 | +12.1% | -7.3% | +20.3% | +85.6% | -4.8% | -22.3% | +12.5% | +70.7% | +75.0% | +7.2% |
| V6_N50 | +14.2% | -6.5% | +29.9% | +51.8% | -0.3% | -14.6% | +16.0% | +44.2% | +49.4% | +19.5% |

## By-year $, compounding from $50K (continuous across years, not reset)

| cell | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|
| SPY | $60,668 | $58,063 | $75,035 | $89,512 | $115,113 | $94,402 | $117,727 | $150,577 | $175,411 | $197,855 |
| V1_N20 | $56,177 | $53,109 | $70,244 | $118,028 | $103,939 | $87,584 | $105,624 | $178,451 | $285,153 | $427,098 |
| V1_N50 | $59,286 | $57,524 | $72,110 | $118,700 | $120,903 | $100,816 | $117,149 | $175,496 | $253,493 | $334,091 |
| V2_N20 | $59,941 | $56,295 | $64,643 | $116,804 | $136,026 | $133,499 | $153,829 | $231,494 | $316,149 | $520,706 |
| V5_N50 | $59,560 | $56,356 | $69,288 | $111,679 | $127,586 | $113,161 | $130,576 | $186,514 | $237,889 | $333,582 |
| V3_N20 | $56,177 | $52,781 | $65,061 | $72,707 | $74,337 | $69,384 | $79,932 | $110,299 | $129,272 | $148,629 |
| V3_N50 | $59,286 | $57,376 | $69,738 | $76,811 | $80,316 | $71,793 | $82,382 | $111,570 | $129,650 | $146,009 |
| V4_N20 | $59,941 | $56,545 | $63,380 | $70,146 | $78,945 | $77,790 | $89,505 | $125,997 | $147,075 | $179,647 |
| V6_N20 | $56,042 | $51,942 | $62,473 | $115,935 | $110,367 | $85,723 | $96,434 | $164,630 | $288,043 | $308,887 |
| V6_N50 | $57,118 | $53,423 | $69,418 | $105,342 | $104,981 | $89,700 | $104,078 | $150,103 | $224,248 | $268,065 |

## 9-cell summary

| cell | years_beat_spy | ann_net | spy | excess_H1 | excess_H2 | alpha_t | sharpe | maxDD_book | maxDD_spy | worst_year | worst_yr_ret | weeks_cash | turnover | null_pct_hit | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V1_N20 | 6/10 | +24.8% | +15.1% | -1.7% | +20.1% | 0.60 | 0.72 | -63.0% | -31.8% | 2022 | -15.7% | +0.0% | +16.8% | 96.7 | fail |
| V1_N50 | 6/10 | +21.7% | +15.1% | +1.3% | +10.5% | 0.45 | 0.75 | -49.8% | -31.8% | 2022 | -16.6% | +0.0% | +14.7% | 99.0 | fail |
| V2_N20 | 5/10 | +27.4% | +15.1% | +3.8% | +18.4% | 1.17 | 0.86 | -41.6% | -31.8% | 2018 | -6.1% | +0.0% | +20.6% | 88.0 | fail |
| V5_N50 | 5/10 | +21.7% | +15.1% | +2.5% | +9.2% | 0.82 | 0.83 | -35.6% | -31.8% | 2022 | -11.3% | +0.0% | +16.1% | 92.3 | fail |
| V3_N20 | 4/10 | +11.9% | +15.1% | -8.2% | +3.2% | 0.06 | 0.62 | -30.1% | -31.8% | 2022 | -6.7% | +39.2% | +16.8% | 99.7 | fail |
| V3_N50 | 3/10 | +11.7% | +15.1% | -6.7% | +1.2% | -0.25 | 0.63 | -33.4% | -31.8% | 2022 | -10.6% | +27.3% | +14.7% | 95.3 | fail |
| V4_N20 | 4/10 | +14.1% | +15.1% | -7.1% | +6.1% | 0.45 | 0.73 | -34.3% | -31.8% | 2018 | -5.7% | +28.4% | +20.6% | 98.3 | fail |
| V6_N20 | 3/10 | +20.7% | +15.1% | -0.5% | +10.8% | 0.21 | 0.67 | -65.1% | -31.8% | 2022 | -22.3% | +0.0% | +21.8% | 43.3 | fail |
| V6_N50 | 6/10 | +19.0% | +15.1% | -1.5% | +8.7% | 0.04 | 0.71 | -50.6% | -31.8% | 2022 | -14.6% | +0.0% | +19.0% | 99.3 | fail |

## Pass list (0/9): NONE -- no cell clears years-beat-SPY>=7/10 + excess>0 both halves + maxDD<=1.25xSPY + hit-rate null>=95%.
Multiplicity: 9 cells (stated, PREREG). No parameter tuned after seeing numbers.

## Methodology notes
- Universe U2: price>=$10 at the signal-day close, ADV20>=$200M, >=273 trading days of history (book/name earliest eligibility ~2017-02 given the panel starts 2016-01-04) -- a few early-2017 weeks may run under-N, logged as WARNING, shared with 1700d/e's own panel.
- V2/V4/V5 risk-adjusted rank = 12-1 return / 252-day daily-return std, unannualised (the sqrt(252) scale factor is identical across names so it cannot change the cross-sectional rank).
- V3/V4 exposure = min(1, 20% / (126-day trailing daily std of the UNSCALED book's own equal-weight close-to-close return x sqrt(252))) at each week's signal date; cash (the unexposed remainder) earns 0%; weeks lacking 126 days of the book's own history default to 100% exposure (fail-safe, logged WARNING) -- unavoidable in ~the first 5 months of 2017.
- hit-rate null: for the SAME 300 same-N draws, compound within each calendar year and count years beating SPY; percentile = share of the 300 draws with hit-count <= the real count.
- Cost 5bps/side + half the (high-low)/close spread proxy (capped 20bps) on traded dollars; exposure scaling adds no extra cost (cash<->equity transfer is not modeled as a trade).
- Reuses the already-vetted 1700c/1700d/1700e panel/universe/cost machinery without a fresh price-adjustment audit (prior cells in this chain already cleared that check).
