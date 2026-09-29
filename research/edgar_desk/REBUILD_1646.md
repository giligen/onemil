# REBUILD 1,646-1,648 — independent rebuild from PREREG_1646.md prose only

Built blind to the first build (`cell_1646.py`, `events_1646.csv`, `RESULT_1646_build.md`) — not opened
until this file and `events_1646_rebuild.csv` were written. Code: `research/edgar_desk/rebuild_1646.py`.

## Method
Bars: `alpaca_daily_2019_2024H1.parquet` + `panel_2024_2026.parquet` concatenated (no date overlap),
zero/NaN-OHLCV dropped (1,083,653 of 19,369,528 rows), dedup on (symbol,bar_date). Master trading
calendar = SPY's own session dates (1,929 sessions, 2019-01-02..2026-09-04). Split-guard: flag any
single-day |return|>60% and drop the symbol's event if that bar falls in [E-6,E+1]; **caveat: this only
catches splits more extreme than ~1.67:1 — a plain 2:1 (-50%) or 3:2 (-33%) unadjusted split would NOT be
flagged.** Gap-guard: every session E-6..E+1 must have a bar for the symbol, else drop.

Events: `events_raw.csv` filtered to `form=='8-K'` exactly (excludes `8-K/A`, `8-K12B`, etc.) with `2.02`
as an exact token in `;`-split `items` (128,645 rows). Acceptance UTC parsed, `tz_convert('America/
New_York')` for ET date/time; after-close = ET time >= 16:00. Same-(cik,et_date) duplicates dropped,
keep earliest acceptance (12,153 dropped).

E estimator (per cik, per filing D, using only its own prior filings): L1 = prior filing closest to
D-365d within ±45d (else skip, no L1 found: 15,932). If an L2 exists (closest to L1-365d within ±45d),
drift = (L1-L2 days) - 365; skip if |drift|>7d (7,738 skipped). **E = L1 + 365 + drift** (E is ~1y AFTER
L1, not L1 itself — an early bug in this rebuild set E = L1+drift with no +365, which the causality guard
caught immediately: 9,816/9,825 "violations" in a 400-CIK debug run, since E-6 landed days before L1's own
acceptance. Fixed before any number below was produced). Causality guard (L1/L2 accepted before session
E-6) passes 100% once fixed, as expected since L1/L2 sit ~1y before E. E rolled forward to the next SPY
session. Firm-quarters where the actual release D preceded session E-5 (estimate too late to trade as
pre-announcement) are dropped (7,673 of 80,591 survivors-of-drift-filter). Universe (price>=$3, dvol20>=
$5M at E-6): 29,330 dropped. Test tickers: 0 matched. Symbol used = the ticker on filing D's own row
(approximation if a ticker changed between E-5 and D — rare, undocumented in this data).

Costs: 5 bps/leg, 10 bps round trip, subtracted from raw close/open return. SPY-adjusted = net minus
SPY's raw (no-cost) return over the identical entry/exit sessions. 1648 mechanism = mean volume over L1's
OWN [-5,+1] window / adv20 at L1's own E-6 (fully causal, symmetric definition to the live signal).
TEST (2024H2+, 17,190 candidate firm-quarters) is carried in the event registry (for the Jaccard set
check) but **no return, cost, t-stat or tercile number was computed for TEST** — sealed.

## Results (TRAIN 2019-2022 n=19,511; VAL 2023-2024H1 n=10,686)

Hit rate: **[E-1,E+1] TRAIN 48.4% / VAL 51.2%; [E-3,E+3] TRAIN 69.8% / VAL 74.2%; median |gap| = 2.0 / 1.0
sessions.** Well under the frozen 70% bar on [E-1,E+1] in both splits — this looks like the estimator's
real precision (drift-adjusted ±1yr calendar matching), not obviously a bug: the causality-guard trap
above shows a real bug would have been loud (near-100% drop), not a quiet miss rate.

| split | cell | mean net bps | day-clust t (n days) | ex-top5% | ex-top1% | cap+30% | SPY-adj mean | MDE (t2.5) |
|---|---|---|---|---|---|---|---|---|
| TRAIN | 1646 | +28.9 | -0.51 (723) | **-68.2** | -3.4 | +24.0 | +3.4 | 13.7 |
| TRAIN | 1647 | +47.5 | -0.48 (723) | **-84.5** | +5.7 | +35.6 | +6.5 | 18.3 |
| VAL   | 1646 | +12.6 | +1.06 (368) | **-71.5** | -16.3 | +9.1 | -2.2 | 15.5 |
| VAL   | 1647 | +28.0 | +1.51 (368) | **-93.4** | -12.0 | +17.8 | -14.7 | 22.5 |

**Ex-top-5% is negative in all four rows** — the entire positive raw mean is carried by the top ~1-5% of
events; day-clustered t never exceeds 1.51. This FAILS the frozen pass bar (day-clust t>=2.5, ex-top5%>0)
decisively on this rebuild, before even reaching the SPY-adjusted or hit-rate bars. Early-arrival share
(1646 exits before E-1 close): TRAIN 19.5%, VAL 17.8%.

1648 tercile (mean net 1647 bps by prior-year announcement-window volume ratio):
TRAIN low/mid/high = 63.6 / 36.4 / 62.0 (**not monotone**, n=6,257/6,256/6,257); VAL = 2.2 / 12.6 / 69.5
(monotone increasing, n=3,558/3,557/3,557). Not monotone on both halves -> fails that pass-bar leg too.

## Caveats an adversary would raise
No market-cap data on disk -> the "<=$1B vs larger" size split was NOT computed (would need a fetch
outside SPY-only scope; flagged, not silently dropped). Split-guard gap noted above (2:1/3:2 unadjusted
splits not caught) is the most likely source of residual tail contamination given how concentrated the
edge is in the extreme upper tail — this should be checked before trusting even the raw (pre-guard) sign.
No delisting flag on disk: a delisted name simply stops appearing in bars and is dropped by the gap-guard
(NOT kept at -100% as the refuter asks) — likely makes this rebuild optimistic, not pessimistic.
