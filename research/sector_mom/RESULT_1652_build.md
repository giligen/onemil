# RESULT — cells 1,652-1,654: sector-ETF 12-1 momentum (build)

Built 2026-09-29T06:37:55.032847+00:00Z by cell_1652.py from PREREG_1652.md (frozen). Read window 2016-01..2023-12 pooled. TEST 2024-01+ NOT computed (sealed).
Panel: 83/96 nominal months usable (13 skipped, see Caveats). Top-3 set changes month to month in 54.9% of processed decisions.

## BLOCKER: Alpaca returned no bars before 2016-01-04 on this account, not 2015-01-01 as fetched
Every `StockBarsRequest` was sent with `start=2015-01-01`, matching the PREREG and the parent
instruction exactly. Alpaca's response for **every legacy sector (XLK, XLF, XLE, XLV, XLI, XLP,
XLY, XLU, XLB, XLRE) and SPY** starts at **2016-01-04**, not 2015-01-01 -- 2684 daily bars each,
uniformly. This is not a per-symbol inception limit: XLRE's true inception is 2015-10-07 (per the
PREREG's own data note) and it is truncated to 2016-01-04 exactly like the 1998-vintage sectors,
while XLC (true inception 2018-06-18) correctly comes back from its real 2018-06-19 start. The
only explanation consistent with all 12 symbols is an **account/data-plan historical cutoff at
2016-01-01** on this Alpaca subscription, not a coding error in this script and not a per-symbol
data gap. Net effect: the earliest decision with a valid D-12 anchor is 2017-01 (not 2015-12), so
**all of calendar 2016 (12 months) plus 2017-01 are unobtainable** -- 13 of the nominal 96 months,
not the ~1 month a fetch-start/eligibility rounding edge would cost. The per-year table below and
every "years positive" count therefore span **2017-2023 (7 years), not 2016-2023 (8 years)** as
the frozen PREREG's per-year table calls for. This should be resolved (checking whether a
different data plan/endpoint on this account can reach 2015-2016, or amending the PREREG's sample
start to 2017-01) before this build is treated as a full read of the frozen 2016-2023 window --
flagging for the owner/parent rather than silently deciding it myself.

## Pass-bar table (frozen bar: mean excess >= +0.25%/mo net & NW t >= 2.5, OR >=6/8yr positive & MDD no worse than benchmark+5pp & worst month no worse than benchmark worst+3pp; AND ex-top-5% excess >= 0)

| Cell | N mo | Mean net/mo | Mean excess/mo | NW t | Ann. excess | MDD cell | MDD bench | Worst cell | Worst bench | Yrs+ (of 7, see BLOCKER) | Ex-top5% excess | MDE/mo | Turnover/mo | $/mo @$60K | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1652 | 83 | 1.147% | 0.134% | 0.68 | 1.63% | 15.83% | 22.65% | -9.65% | -14.43% | 2/7 | -0.142% | 0.550% | 42.2% | $688 | FAIL |
| 1653 | 83 | 0.201% | -0.811% | -1.16 | -9.31% | 24.76% | 22.65% | -13.45% | -14.43% | 2/7 | -1.818% | 1.931% | 82.7% | $121 | FAIL |
| 1654 | 83 | 1.043% | 0.030% | 0.13 | 0.36% | 15.83% | 22.65% | -8.56% | -14.43% | 2/7 | -0.397% | 0.772% | 43.0% | $626 | FAIL |

## Per-year excess sign table (return-month year, % sum of monthly excess) -- 2017-2023 only, 2016 missing (see BLOCKER)
* **1652**: {2017: -0.732, 2018: -1.063, 2019: -5.374, 2020: 8.365, 2021: -7.935, 2022: 20.294, 2023: -2.399}
* **1653**: {2017: -16.227, 2018: -1.505, 2019: -28.553, 2020: 5.449, 2021: -44.521, 2022: 47.506, 2023: -29.465}
* **1654**: {2017: -0.732, 2018: -1.063, 2019: -7.473, 2020: 3.449, 2021: -7.935, 2022: 16.847, 2023: -0.574}

## Pass-bar detail
* **1652**: primary(mean_excess>=0.25% & t>=2.5)=False, alt(>=6/8yr pos & mdd<=bench+5pp & worst<=bench_worst+3pp)=False, ex_top5%>=0=False
* **1653**: primary(mean_excess>=0.25% & t>=2.5)=False, alt(>=6/8yr pos & mdd<=bench+5pp & worst<=bench_worst+3pp)=False, ex_top5%>=0=False
* **1654**: primary(mean_excess>=0.25% & t>=2.5)=False, alt(>=6/8yr pos & mdd<=bench+5pp & worst<=bench_worst+3pp)=False, ex_top5%>=0=False

## Stacking line
Capital window = all month (the sleeve holds through every session, unlike the day books). Collision = the day books' overnight margin usage and the TOM sleeve's four nights/month (both draw on the same buying power). $/month is mean NET return x $60,000, not excess.

## Caveats (read as an adversary)
1. **Skipped months** (13): 2016-01: only 0 eligible sectors at decision 2015-12 (need >=3); likely the 2015-01-01 fetch start leaves D-12=2014-12 unavailable; 2016-02: only 0 eligible sectors at decision 2016-01 (need >=3); likely the 2015-01-01 fetch start leaves D-12=2015-01 unavailable; 2016-03: only 0 eligible sectors at decision 2016-02 (need >=3); likely the 2015-01-01 fetch start leaves D-12=2015-02 unavailable; 2016-04: only 0 eligible sectors at decision 2016-03 (need >=3); likely the 2015-01-01 fetch start leaves D-12=2015-03 unavailable; 2016-05: only 0 eligible sectors at decision 2016-04 (need >=3); likely the 2015-01-01 fetch start leaves D-12=2015-04 unavailable; 2016-06: only 0 eligible sectors at decision 2016-05 (need >=3); likely the 2015-01-01 fetch start leaves D-12=2015-05 unavailable; 2016-07: only 0 eligible sectors at decision 2016-06 (need >=3); likely the 2015-01-01 fetch start leaves D-12=2015-06 unavailable; 2016-08: only 0 eligible sectors at decision 2016-07 (need >=3); likely the 2015-01-01 fetch start leaves D-12=2015-07 unavailable; 2016-09: only 0 eligible sectors at decision 2016-08 (need >=3); likely the 2015-01-01 fetch start leaves D-12=2015-08 unavailable; 2016-10: only 0 eligible sectors at decision 2016-09 (need >=3); likely the 2015-01-01 fetch start leaves D-12=2015-09 unavailable; 2016-11: only 0 eligible sectors at decision 2016-10 (need >=3); likely the 2015-01-01 fetch start leaves D-12=2015-10 unavailable; 2016-12: only 0 eligible sectors at decision 2016-11 (need >=3); likely the 2015-01-01 fetch start leaves D-12=2015-11 unavailable; 2017-01: only 0 eligible sectors at decision 2016-12 (need >=3); likely the 2015-01-01 fetch start leaves D-12=2015-12 unavailable
2. **Turnover simplification**: turnover = sum(|target_weight_new - target_weight_prior|) using the PRIOR REBALANCE-DAY target weights, not weights drifted by one month of price action between rebalances. Understates cost slightly for persisting names; entry/exit cost (the dominant driver) is exact.
3. **1653's "excess"** is computed against the same long-only equal-weight benchmark as 1652/1654 per the PREREG template, but 1653 is market-neutral by construction (0% net exposure) while the benchmark carries full sector-equity beta -- this is a structural mismatch, not an edge measurement; read 1653's raw monthly net return as the primary number.
4. **MDE** uses the frozen PREREG denominator sqrt(96), not the achieved N, per the PREREG's own frozen formula -- it is a pre-committed power line, not refit to this build's sample.
5. **Dividends**: adjustment='all' back-adjusts for both splits and dividends, so monthly closes are total-return-like; not independently verified against a second dividend source in this build.
6. **No independent reimplementation yet** -- this is the FIRST build from the PREREG prose; per CLAUDE.md protocol this number is not owner-reportable until a second agent rebuilds it blind and holdings agree (Jaccard >= 0.98).
7. **Price-scale check**: not run in this build (no |t|>6 daily cells here since everything is monthly, but the daily-bar cache itself was not independently diffed against a second source).