# RESULT -- cell 1,700f: reconciliation of 1,700d vs the other bot's A1/A2/A3 table

NOT a strategy claim -- a diagnostic to see which universe construction reproduces the other bot's 2017-2022 numbers. 12-1 momentum (skip 21), weekly Mon-open rebalance, equal weight, 5bps/side ONLY (flat -- matches the other bot's stated cost, not 1700d's +half-spread model). $50,000 compounding from 2017-01-02. Universes: (a) PIT = price>=$10 & adv20>=$200M evaluated at each signal date (=1700d U2); (b) TODAY-FIXED-400 = the 400 highest-adv20 names (price>=$10) on 2026-09-29, membership frozen for all periods (deliberate look-ahead + survivorship); (c) TODAY-FIXED-400 + full-273-day-history == (b) by construction here -- sig_by_date already requires history_ok (>=273 trading days) for every row shared by (a) and (b), so adding that requirement to (b) changes nothing; not separately run.

Two ETP/name-pattern exclusions applied (shared with cell 1,700d -- same panel, same assets file): (1) company-name regex ETF/ETN/FUND/TRUST/WARRANT/UNIT/PREFERRED/RIGHT (case-insensitive); (2) test-ticker regex `^Z[A-Z]ZZT$`.

## N=10

| Year | PIT (a) | TODAY-FIXED-400 (b) | Their claim |
|---|---|---|---|
| 2017 | +6.4% | +12.1% | NA% |
| 2018 | -15.1% | -14.3% | NA% |
| 2019 | +34.1% | +62.2% | NA% |
| 2020 | +88.8% | +169.9% | +164.0% |
| 2021 | +1.3% | +27.2% | +79.7% |
| 2022 | -18.7% | +3.2% | NA% |
| 2023 | +26.4% | +40.5% | NA% |
| 2024 | +100.6% | +135.9% | NA% |
| 2025 | +43.2% | +105.8% | NA% |
| 2026 | +56.8% | +126.3% | NA% |

End-2026 $ from $50,000 (2017-01-02): PIT = $536,412; TODAY-FIXED-400 = $4,263,695; their claim = NA (not given for this N).

**Conclusion N=10**: PIT does NOT match (max |diff| 78.4pt); TODAY-FIXED-400 does NOT match (max |diff| 52.5pt) within +-10pt/yr over [2020, 2021].

## N=20

| Year | PIT (a) | TODAY-FIXED-400 (b) | Their claim |
|---|---|---|---|
| 2017 | +13.9% | +18.0% | +27.9% |
| 2018 | -3.7% | -2.5% | -29.4% |
| 2019 | +34.3% | +66.7% | +43.6% |
| 2020 | +70.9% | +112.4% | +119.7% |
| 2021 | -10.5% | +29.8% | +34.0% |
| 2022 | -14.4% | -5.7% | -0.4% |
| 2023 | +23.2% | +43.6% | +18.8% |
| 2024 | +71.6% | +106.9% | +62.0% |
| 2025 | +62.3% | +95.9% | +51.5% |
| 2026 | +51.4% | +105.1% | +63.0% |

End-2026 $ from $50,000 (2017-01-02): PIT = $500,765; TODAY-FIXED-400 = $2,978,320; their claim = $967,704.

**Conclusion N=20**: PIT does NOT match (max |diff| 48.8pt); TODAY-FIXED-400 does NOT match (max |diff| 44.9pt) within +-10pt/yr over [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026].

## N=30

| Year | PIT (a) | TODAY-FIXED-400 (b) | Their claim |
|---|---|---|---|
| 2017 | +14.4% | +20.6% | NA% |
| 2018 | -0.6% | +0.3% | NA% |
| 2019 | +31.7% | +54.7% | NA% |
| 2020 | +65.8% | +78.2% | +108.7% |
| 2021 | -5.5% | +25.9% | +37.7% |
| 2022 | -17.5% | -2.9% | NA% |
| 2023 | +26.8% | +45.8% | NA% |
| 2024 | +64.7% | +103.0% | NA% |
| 2025 | +52.8% | +89.9% | NA% |
| 2026 | +51.0% | +66.2% | NA% |

End-2026 $ from $50,000 (2017-01-02): PIT = $466,850; TODAY-FIXED-400 = $1,903,745; their claim = NA (not given for this N).

**Conclusion N=30**: PIT does NOT match (max |diff| 43.2pt); TODAY-FIXED-400 does NOT match (max |diff| 30.5pt) within +-10pt/yr over [2020, 2021].

