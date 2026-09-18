# Multi-day program — DATA stage

Built 2026-09-18, ~1h50m wall on the trading node (prices 44 min, EDGAR submissions 24 min, EDGAR XBRL
16 min, the rest minutes). 1.7 GB on disk. Nothing here looks at the return of any strategy: this is the
panel the 26 pre-registered cells in `PLAN.md` will run on, plus the coverage facts that decide which of
them can be believed.

Code (all committed, all resumable): `data/build_universe.py` · `data/fetch_prices.py` ·
`data/partition_prices.py` · `data/price_scale_check.py` · `data/survivorship.py` ·
`data/edgar_common.py` · `data/edgar_earnings.py` · `data/edgar_eps.py` · `data/summarize.py`
(`summarize.py` reprints every number below from the artifacts, so this page cannot drift from the panel).
Bulk artifacts are gitignored: `universe.parquet`, `prices_by_year/{raw,all}/year=*.parquet`,
`earnings_events.parquet`, `eps_facts.parquet`, `delisted_names.parquet`, `assets_all_active.parquet`.

---

## 1. Universe — 12,131 symbols

Every ACTIVE, tradable US equity asset on Alpaca (14,355 returned; 877 untradable and 1,347
warrants/units/preferred/rights dropped by the shared `AlpacaClient._is_common_stock` name rule;
0 test tickers survived `pit_listings.is_test_ticker`). Leveraged/inverse wrappers are KEPT and flagged
(`kind='wrapper'`), the 2026-09-05 owner rule, so a family can drop them without a re-pull.

| by exchange | n | | by kind | n | | flag (today) | share |
|---|---|---|---|---|---|---|---|
| NASDAQ | 4,778 | | fund | 6,281 | | marginable | 97.3% |
| ARCA | 2,709 | | common | 4,870 | | shortable | 41.7% |
| NYSE | 2,425 | | wrapper | 980 | | easy_to_borrow | 41.7% |
| BATS | 1,635 | | | | | fractionable | 61.6% |
| OTC | 309 | | | | | | |
| AMEX | 275 | | | | | | |

`shortable` / `easy_to_borrow` / `marginable` are recorded per symbol — the F1–F6 short legs are gated on
them, with the caveat in §7.

## 2. Prices — 18,221,178 rows × 2 panels, 2016-01-04 → 2026-09-18

Alpaca SIP daily bars for the whole universe, pulled TWICE (`adjustment='raw'` and `adjustment='all'`),
61 batches of 200 symbols, `feed='sip'`. Stored as `prices_by_year/<adj>/year=YYYY.parquet`, symbol as a
dictionary column, OHLC/vwap float32, volume + trade_count float64 (828 MB per pair of panels; peak memory
one year, ~150 MB).

| year | rows | symbols | | year | rows | symbols |
|---|---|---|---|---|---|---|
| 2016 | 1,095,145 | 4,527 | | 2022 | 1,812,346 | 7,465 |
| 2017 | 1,168,769 | 4,853 | | 2023 | 1,921,438 | 8,046 |
| 2018 | 1,261,570 | 5,219 | | 2024 | 2,119,382 | 8,917 |
| 2019 | 1,356,378 | 5,599 | | 2025 | 2,394,383 | 10,378 |
| 2020 | 1,465,420 | 6,133 | | 2026 (to 09-18) | 1,978,947 | 11,821 |
| 2021 | 1,647,400 | 6,988 | | **total** | **18,221,178** | **11,823** |

The two panels are row-identical (same keys, every year). The symbol count RISING from 4,527 in 2016 to
11,821 in 2026 is the survivorship hole in one line: these are today's names, back-filled.

**Price-scale check (200 random keys, `price_scale_check.py`, PLAN §1 item 3).** The cumulative
corporate-action factor is rebuilt by chaining daily return ratios forward from the key date to the
symbol's last session — it never reads adj/raw on the key date — and must reproduce the adjusted close:

* `raw_close × cum_factor` vs adjusted close: **0.00% of keys fail at 0.01%** (0 of 200); median relative
  error 1.0e-06, p99 4.0e-05 (float32).
* factor on each symbol's last session = 1.0: **100%**.
* per-field (open/high/low vs close) factor disagreement > 0.01% on 40.5% of keys — **this is penny
  rounding, not a tape mismatch**: the implied ABSOLUTE price error is median $0.0028, max $0.0096, i.e.
  ≤ half a cent. It matters only below ~$5 (§7, gap 4).

## 3. Survivorship — the panel is today's names, and here is what that costs

### (a) Databento point-in-time definitions, all exchanges, 2024-07 → 2026-09
Names carrying a `security_type='C'` (common stock) definition record in that month that are absent from
Alpaca today. `absent_broker` = gone from Alpaca entirely (the honest number); `absent_universe` = also
counts names we filter out by class.

| month | listed | absent_broker | % | | month | listed | absent_broker | % |
|---|---|---|---|---|---|---|---|---|
| 2024-07 | 4,842 | 889 | 18.36 | | 2025-09 | 4,627 | 431 | 9.31 |
| 2024-10 | 4,782 | 791 | 16.54 | | 2026-01 | 4,614 | 313 | 6.78 |
| 2025-01 | 4,711 | 688 | 14.60 | | 2026-05 | 4,606 | 173 | 3.76 |
| 2025-05 | 4,646 | 556 | 11.97 | | 2026-09 | 4,578 | 43 | 0.94 |

18.36% of the 2024-07 common-stock listings are gone from Alpaca 2.21 years later → **8.75% per year**.
The monotone decay across the 27 months is the signature of a real attrition process, not a data defect.
(All-instrument-type rows are in `data/survivorship_pit.csv`.)

### (b) XNAS.ITCH daily tape on disk, 2018-05 → 2024-06
`research/fuckup_audit/N_databento/N3/xnas_daily.parquet` + `R_daily/xnas_daily_2024H1.parquet`. Symbols
there come from Databento's DAILY symbology map, which was repaired once in N3 (`fix_symbol_map.py`: ITCH
re-assigns instrument_ids every session, a stale map silently renames symbols); a residual mis-map would
appear as a name that never existed, so these are an UPPER bound. ITCH suffix classes (`+` warrant,
`=` unit, `^` preferred, …) are excluded here; `liquid` = ≥ 20 sessions that year and ≥ $100K average
daily dollar volume.

| year | symbols | absent today | % | liquid symbols | liquid absent | % | implied rate/yr (liquid) |
|---|---|---|---|---|---|---|---|
| 2018 | 9,137 | 4,267 | 46.7 | 4,928 | 1,700 | 34.5 | 5.0% |
| 2019 | 9,583 | 4,317 | 45.1 | 5,093 | 1,691 | 33.2 | 5.4% |
| 2020 | 10,168 | 4,434 | 43.6 | 5,971 | 2,007 | 33.6 | 6.4% |
| 2021 | 12,348 | 5,679 | 46.0 | 7,431 | 2,705 | 36.4 | 8.3% |
| 2022 | 12,641 | 5,322 | 42.1 | 6,282 | 1,690 | 26.9 | 7.2% |
| 2023 | 12,542 | 4,479 | 35.7 | 5,744 | 1,188 | 20.7 | 7.0% |
| 2024 (H1) | 11,459 | 3,027 | 26.4 | 5,488 | 779 | 14.2 | 6.7% |

**Two independent overlaps agree on the order of magnitude: 5–9% of listed names disappear per year.**
The all-symbol column is inflated (the Nasdaq venue tape carries OTC and odd-class names Alpaca never
lists); the `liquid` column is the one to quote.

`delisted_names.parquet` = the union, **9,126 names** with their last-seen date (8,124 from the ITCH tape,
1,002 from the point-in-time definitions) — the PIT re-run population for every survivor cell. Prices for
the 2018→2024 part of it exist in the two XNAS parquets; for the 2024-07→now part there is no price
source on disk yet (§7, gap 8).

## 4. Earnings calendar — 142,695 events, SEC EDGAR, point-in-time

Every 8-K / 8-K/A whose `items` contains **2.02** (Results of Operations), for the 6,028 distinct CIKs that
6,357 of our symbols map to via `company_tickers.json` (older pages in `filings.files` pulled too, so
coverage reaches 2015). 143,331 raw filings on 3,914 CIKs → 142,695 events on **4,094 symbols** after the
CIK→symbol join and the 2016-01-04 floor.

* share of the whole universe with ≥ 1 event: **33.7%** (most of the universe is funds, which do not file 8-Ks)
* share of **common stocks** with ≥ 1 event: **78.1%** (3,806 of 4,870; a further 288 funds/wrappers file one); median 43 events per symbol
* 100% of event symbols have prices in the panel

| year | events | | year | events | | year | events |
|---|---|---|---|---|---|---|
| 2016 | 10,222 | | 2020 | 12,577 | | 2024 | 15,400 |
| 2017 | 10,625 | | 2021 | 13,562 | | 2025 | 15,873 |
| 2018 | 11,176 | | 2022 | 14,473 | | 2026 (to 09-18) | 12,268 |
| 2019 | 11,592 | | 2023 | 14,927 | | | |

**Acceptance time (`acceptanceDateTime`, the only point-in-time stamp in the record):**
post-close **56.10%**, pre-open **36.45%**, intraday **7.45%**.

**Event session** = the first trading session whose CLOSING AUCTION is strictly after the acceptance
instant, from the Alpaca exchange calendar (early closes use the real 13:00 ET close, not a hardcoded
16:00). Verified on the head of the table: 16:08 ET acceptance → next session. An 8-K accepted 07:00 ET
trades that same session.

## 5. EPS facts — 361,268 quarterly facts, and the availability column

`companyconcept/us-gaap/EarningsPerShareDiluted` (1,232 facts fell back to `…Basic`) for the 3,909 CIKs
with an Item-2.02 8-K → 413,412 raw facts on 3,714 CIKs → **361,268 quarterly facts on 3,706 CIKs**, of
which **46,098 are derived Q4** (FY − Q1−Q2−Q3 of the same fiscal year, carrying the ANNUAL filing's
`filed` date, `source='derived_q4'`). Amendments are kept as separate facts with their own `filed`.

Nothing is computed from them here — SUE is a family-stage computation — but the availability audit is a
COLUMN on `earnings_events.parquet`, not a hope: `eps_prior_filed`, `eps_prior_end`,
`eps_prior_filed_date`, `n_prior_qfacts`, each built from facts filed STRICTLY BEFORE that event's
acceptance instant.

* events with ≥ 1 prior EPS fact: **86.2%**
* events with **≥ 9 prior quarterly facts** (what the 8-difference SUE σ needs): **77.6%**
* symbols whose last event is SUE-ready: 86.7%
* per year: 74.9 / 74.2 / 73.8 / 73.6 / 73.7 / 69.5 / 69.0 / 78.1 / 86.5 / 88.4 / 87.7 (2016→2026) —
  the 2021–2022 trough is the SPAC/IPO cohort with no filing history.

## 6. Industry — 2-digit SIC on 48.9% of the universe, 92.1% of common stocks

From the same submissions record (`sic`), written into `universe.parquet` as `sic`, `sic2`, `sic_desc`.
Coverage: common 92.1%, fund 21.6%, wrapper 8.3%. Largest 2-digit buckets: 28 chemicals/pharma (808),
73 business services (670), 67 holding/investment (548), 60 depository institutions (419), 36 electronics
(274), 38 instruments (274). F4's industry-adjusted reversal runs on common stocks, where coverage is
92.1% — the 7.9% without a SIC must be dropped from the industry mean, not bucketed as "unknown".

## 7. Gaps that would bias a family, and in which direction

1. **Survivorship (the big one).** The price panel is today's listed names: 5–9%/yr of the cross-section
   is missing, and it is the WRONG 5–9% — delisted names are disproportionately the losers (bankruptcy,
   sub-$1, forced delisting) with a minority of winners (cash acquisitions). Direction: long-only legs are
   biased UP for every family, short legs are biased DOWN (the best shorts are the missing names), so the
   published L−S spread is biased UP at both ends. Worst exposed: **F3 (12-1 momentum)** and **F5
   (52-week-high)**, whose formation windows are a year long and whose D1 decile is exactly the delisting
   cohort; then **F4**. Least exposed: **F1/F2**, which are event-anchored and hold 20–60 sessions.
   Mitigation is pre-registered: re-run every survivor on the point-in-time Nasdaq panel using
   `delisted_names.parquet`.
2. **Earnings coverage is US-domestic-filer only.** 21.9% of common stocks have no Item-2.02 8-K: foreign
   private issuers file 6-K (never an 8-K item), recent IPOs have no history, and 5.3% of common stocks
   have no entry in `company_tickers.json` at all. F1/F2 therefore measure PEAD on US domestic filers;
   ADRs — a cohort with its own documented drift — are absent. Direction: unknown sign, but the result
   must not be stated as "US equities".
3. **SUE availability is 77.6%, and the missing 22.4% is not random** — young companies, SPAC de-SPACs and
   loss-makers lack 9 quarters. Published PEAD is strongest in small, young, illiquid names, so F1's
   decile sort runs on a maturer subsample: direction **conservative** (biases the measured drift DOWN).
   The missingness table per year is in §5 and must be re-cut per split before F1 is scored.
4. **The adjusted panel is rounded to the cent.** Half a cent is 50 bps on a $1 stock and 10 bps at $5.
   **F4 (1-week reversal)** and **F6 (overnight vs intraday)** difference small returns, so below ~$5 the
   rounding noise is the same size as the effect. The PLAN's raw-close ≥ $5 gate handles it; any cell that
   relaxes that gate must report the rounding as a cost.
5. **`easy_to_borrow` / `shortable` are TODAY's flags** applied to a 2016–2026 short leg. That is both a
   look-ahead (a name hard to borrow in 2016 may be ETB now) and survivorship (names that became
   unborrowable often delisted, so they are not in the panel at all). Direction: **overstates short-leg
   feasibility**. The flag is a disclosure, not a simulation — every short cell states the share of its
   trades that are ETB today, as Stage O did.
6. **308 universe symbols have no SIP daily bar at all** — 307 of them OTC (253 common, 54 fund). They are
   silently absent from the panel; no family trades OTC, so this is a disclosure, not a bias.
7. **Ticker recycling and renames pollute both overlaps in opposite directions**: a symbol reassigned to a
   new company counts as "still listed" (attrition understated), a renamed survivor counts as delisted
   (overstated). Magnitude is small relative to the 5–9% band but it is why §3 quotes a range.
8. **No prices for the delisted names after 2024-06.** `delisted_names.parquet` names them, but the only
   price source on disk covers 2018-05→2024-06 (Nasdaq venue). The PIT re-run is therefore possible on the
   Nasdaq panel 2018→2024 — which is what PLAN §Survivorship pre-registers — and NOT on 2024-07→2026 or on
   NYSE/ARCA-listed delistings. Any survivor whose edge lives only in 2024-2026 cannot be survivorship-checked
   with what we own.
9. **Two pulls, one moving tape.** The raw and adjusted panels were fetched ~30 min apart on a live trading
   day; the first raw pull was missing 530 symbol-rows for 2026-09-18 that the later adjusted pull had. Both
   panels were topped up (`fetch_prices.py --topup-from`) and are now row-identical. Any future re-pull on a
   live day must do the same or drop the last session.
