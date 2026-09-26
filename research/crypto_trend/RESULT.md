# RESULT — crypto trend (cells C1-C4), PREREG frozen 2026-09-26

| Cell | Split | Weeks | Mean %/wk (net) | NW t (4 lag) | Green% | Null green% | MaxDD% | Ex-top5% mean%/wk | Time-in-mkt | Round trips | Fees % (cum.) | BH-BTC mean%/wk | BH-BTC MaxDD% | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 (SMA100 trend) | TRAIN | 331 | 1.27 | 2.02 | 31.1% | 41.4% | 79.18% | -0.09 | 41.5% | 47 | 16.6 | 1.25 | 82.88% | -* |
| C1 | VAL | 78 | 3.98 | 1.06 | 34.6% | 47.8% | 54.16% | -0.59 | 61.1% | 19 | 5.55 | 1.43 | 24.97% | **FAIL** |
| C2 (Donchian 20/10) | TRAIN | 331 | 1.67 | 2.43 | 39.3% | 42.9% | 76.09% | 0.17 | 44.2% | 46 | 14.8 | 1.25 | 82.88% | -* |
| C2 | VAL | 78 | 0.32 | 0.36 | 23.1% | 37.5% | 42.93% | -0.69 | 36.8% | 12 | 4.2 | 1.43 | 24.97% | **FAIL** |
| C3 (28d momentum) | TRAIN | 331 | 2.06 | 3.01 | 36.3% | 43.0% | 80.11% | 0.56 | 43.8% | 81 | 26.8 | 1.25 | 82.88% | -* |
| C3 | VAL | 78 | 0.47 | 0.48 | 26.9% | 44.4% | 42.36% | -0.59 | 51.3% | 24 | 7.5 | 1.43 | 24.97% | **FAIL** |
| C4 (C1 ^ C3) | TRAIN | 331 | 1.64 | 2.62 | 28.1% | 35.0% | 74.50% | 0.22 | 32.5% | 59 | 21.1 | 1.25 | 82.88% | -* |
| C4 | VAL | 78 | 0.05 | 0.05 | 21.8% | 41.6% | 46.39% | -0.83 | 44.0% | 22 | 7.15 | 1.43 | 24.97% | **FAIL** |

\* TRAIN rows are the cross-check corroboration input to the VAL bar (own-sign, t≥2 — all four pass this alone), not a scored split; the frozen pass bar applies to VAL only.

**Data notes.** BTC/ETH/SOL daily closes, Binance public klines (inception 2017-08-17 / 2017-08-17 / 2020-08-11) merged with Alpaca `CryptoHistoricalDataClient` daily bars from 2021-01-01 (Alpaca preferred on any overlapping date, the executable venue); median |Alpaca/Binance-1| on 2,095/2,095/1,677 overlap days = 0.045%/0.045%/0.062% per coin, all < the 0.3% PREREG bar. TEST (2025-07-01 onward) was fetched into `bars_daily.csv` but never scored, per the PREREG seal. VAL is exactly the pre-registered 78 weeks: a week is scored only if BOTH its Monday decision date and its Monday return-realization date fall inside one split, so the one week whose return would have touched sealed-TEST price data (week of 2025-06-30) is dropped from VAL and from TEST, not leaked across the boundary.

**Which cells pass.** Zero of four. Every cell fails VAL on t (0.05-1.06, need ≥2.0), green-week share (21.8-34.6%, need ≥55% and all four sit BELOW their own count-matched null of 21.8-47.8% — the rules produce fewer green weeks than random long/flat scheduling at the same exposure), max drawdown (42.9-54.2%, need ≤30% and all four are worse than buy-and-hold BTC's 25.0%, failing the "must beat BH on drawdown" check), and ex-top-5%-weeks mean (-0.09 to -0.83%/wk, need ≥0 — the whole VAL point estimate is tail-carried, a lottery ticket by the PREREG's own tail-dependence test). The large VAL mean returns (0.05-3.98%/wk) are driven entirely by a handful of huge weeks, consistent with the failed tail check.

**Caveats.** (1) Donchian (C2) high/low bands are computed on the CLOSE series only (per the Data section's close-only scope), not intraday high/low bars — a disclosed interpretation, not the PREREG's literal wording. (2) Fee model charges 30bps only on a coin's own 0->1/1->0 flip, not on weight drift caused by another coin joining/leaving the equal-weight book — a disclosed simplification. (3) This is the BUILDER pass only: PREREG's Independent-check section (a second agent rebuilding from prose, ≥99% coin-week agreement, VAL mean within 0.05%/wk) has NOT been run and must precede any owner action, per CLAUDE.md's no-claim-without-independent-check rule. (4) Per PREREG consequence: FAIL on all four closes this population; no further crypto cell without a new mechanism.

## Judge's verdict (2026-09-26) — FAIL on every cell; population CLOSED per the PREREG
VAL (78 weeks, 2024-01 → 2025-06): C1 +3.98 %/wk t 1.06, C2 +0.32 t 0.36, C3 +0.47 t 0.48, C4 +0.05 t 0.05; green-week
share 22–35 % — BELOW each cell's count-matched null (38–48 %); max drawdown 43–54 % vs buy-and-hold BTC 25 %; ex-top-5 %
weeks negative in all four (tail-carried). Independent rebuild: C1/C3/C4 position agreement 100 %, C2 76 % (the PREREG did
not fix the Donchian channel field — close vs high/low — and the rule is stateful; under BOTH readings C2 fails, mean
+0.29 / +0.20 %/wk, t < 0.5); fee-model reading differed (flat 30 bps per flip vs weight-scaled) by 0.03–0.09 %/wk, which
cannot move any t to 2. The adequacy critic: the null is adequate; the C2 gate is formally unmet and is documented here.
No further crypto cell without a new mechanism.
