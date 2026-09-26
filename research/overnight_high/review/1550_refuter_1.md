# PREREG_1550 refuter 1: price scale, survivorship, look-ahead

Code: `review/refuter1_fetch_ca.py` (Alpaca corporate actions: 7,036 split/spin/stock-div records 2018-2026 and 71,562
cash dividends for book symbols) and `review/refuter1_pricescale.py`. Output: `refuter1_stats_v2.csv`,
`refuter1_audit_v2.csv`, the builder books `refuter1_builderbook_*`, the corrected books `refuter1_correctedbook_v2_*`,
and the log `refuter1_run_v2.log`. (`refuter1_stats.csv` is v1. Its d/e rows are INVALID because the price filter ran
on back-adjusted closes. Ignore it.)

## Verdict: the FAIL stands, but the builder's numbers are wrong

The builder reproduces exactly: 4.75/5.94 on EXTENSION, 11.56/11.72 on PANEL TRAIN+VAL and -2.01/-9.23 on TEST. Two
defects in that build change the magnitudes:

1. **Outcome filter (look-ahead).** `cell_1550.base_universe` drops every name-night with |ret_on_next| > 50 %. That
   filters on the RESULT. It removes genuine squeeze nights: GME 2024-05-13 +113 %, VIRX +215 %, CODX, SPRT, NURO and
   OCGN. None of those nights is a split. This is the real cause of the builder/rebuild gap in
   `1550_compare.md`, not the ADV min_periods convention.
2. **Databento zero rows.** `panel_2024_2026.parquet` has 227,954 rows of 0/0/0/0/0 OHLCV, placeholder rows after a
   delisting or around a halt. The builder keeps them:
   - They drag ADV20 down, so volume shocks fire that are not real.
   - Their zero next open gives a -100 % night. There are 12 such nights in the N10 book, all cash-merger completions:
     FARO, EXAS, HOLX, CFLT, FYBR and others.
   - Those -100 % nights are hidden only by defect 1.
   - There is also one gap night in the book: CLBR on 2025-07-15, whose next row is 2026-03-27, at -41.6 %. It exists
     because `shift(-1)` takes the next row, not the next session.

## Price basis
Both sources are raw and on the same basis:
- 96 % of the in-window splits show the raw gap (1,411 on Alpaca, 1,580 on Databento).
- Spot checks: AAPL 2020-08-31 went 499.23 -> 127.39, NVDA 2024-06-10 went 1208.88 -> 120.37, and AVGO 2024-07-15
  went 1700.67 -> 170.00.
- No builder book night straddles a split ex-date. One N25 night straddles a spin-off.
- Cash dividends were not credited. That understates the book by +0.73 bps/fill on EXTENSION and +0.3 to +0.5 bps on
  PANEL, which is immaterial.
- Splits inside the 252-day window do not fabricate signals in any material way. Book membership moves by about 3 %
  under a split-adjusted high252/ADV, and the combined corrections move it by 303 of 9,362 fills.

## Look-ahead
All three fields are causal:
- high252 = shift(1).rolling(252) of closes, so it uses closes through t-1 and is compared against close_t.
- ADV20 = shift(1) of volume, so it uses t-1.
- dvol20 = shift(1), so it uses t-1.

The only look-ahead is the ±50 % outcome filter.

## Corrected book
The corrected book is split-adjusted, credits dividends, drops zero rows, requires the next row to be the next session,
applies the $5 filter on the RAW close and has no outcome filter.

| sample | N | net5 bps | t day | ex-top5 % | years + | placebo margin (t) | ex-±30 % nights |
|---|---|---|---|---|---|---|---|
| EXT 2020-24H1 | 10 | +17.6 | 2.31 | -46.9 | 3/5 (2022 -24.5, 2023 -4.4) | +16.1 (2.22) | +1.7 (t 0.67) |
| EXT | 25 | +14.0 | 1.78 | -36.1 | 3/5 | +9.3 (1.65) | +4.5 (t 0.95) |
| PANEL TRAIN+VAL | 10 | +30.8 | 1.77 | -44.7 | - | +28.8 (1.65) | +16.3 (t 1.62) |
| PANEL TRAIN+VAL | 25 | +20.0 | 1.59 | -35.5 | - | +14.8 (1.46) | +12.9 (t 1.37) |
| PANEL TEST (spent) | 10/25 | -10.4 / -11.1 | | | | | |

Every variant fails the frozen bar on three counts:
- ex-top-5 % is below 0.
- Fewer than 4 years are positive.
- The EXTENSION t is below 2.5 and the PANEL t is below 2.

Of the corrected N10 EXTENSION mean, 16 of the 17.6 bps come from 43 genuine nights beyond ±30 %, mostly the 2020-21
squeezes.

## Survivorship and comparability
- **The survivorship sign is not established.** The builder says the bias is upward, which would make the FAIL
  conservative. The test here: inside the PIT panel, drop the 2,106 names that died before 2026-09. That moves N10
  from +30.8 to +15.9 and N25 from +20.0 to +15.1. In that window the dying names HELPED the book, so the extension's
  missing pre-2024-07 delistings could bias it DOWN.
- **The verdict does not depend on that sign.** The ex-top-5 % and years criteria fail by wide margins.
- **The two universes are comparable:**
  - Eligible names per day: 2,290-2,890 on EXTENSION and 2,780-3,480 on PANEL.
  - Median dvol20: $44-49M and $50-62M.
  - Book names per day at N25: 9-20 and 21.
  - Universe overnight return: -5 to +12 bps/yr on EXTENSION and +4.6 to +10.6 on PANEL.
