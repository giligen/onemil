# RESULT 1,640-1,642 -- VIX basis carry (build + score)

Run 2026-09-28T20:25:44.972305Z. TEST (2024-01..2026-09) is SEALED -- not read.

## Data provenance
- VIX spot: 9281 rows 1990-01-02..2026-09-25 <- cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv
- VX futures monthly settlements: 178 contracts usable (target ~181, 2011-09..2026-10) <- cdn.cboe.com/data/.../VX/VX_<expiry>.csv (2013+) and cdn.cboe.com/resources/futures/archive/volume-and-price/CFE_<code>_VX.csv (2011-2013); both need Referer: https://www.cboe.com/
- SVXY: 2684 daily bars 2016-01-04..2026-09-04 <- Alpaca adjustment=ALL
- SVIX: 1113 daily bars 2022-03-30..2026-09-04 <- Alpaca adjustment=ALL
- UVXY: 2684 daily bars 2016-01-04..2026-09-04 <- Alpaca adjustment=ALL
- VIXY: 2684 daily bars 2016-01-04..2026-09-04 <- Alpaca adjustment=ALL

## Price-scale check (|daily return| > 40%)
- SVXY 2018-02-06: -83.0% -- real: Volmageddon vol spike (2nd day)
- UVXY 2016-06-24: 43.7% -- plausible real: Brexit referendum result shock
- UVXY 2018-02-05: 66.7% -- plausible real: Volmageddon (long-vol ETP jumps up)
- UVXY 2018-02-08: 50.3% -- plausible real: Volmageddon 2nd aftershock (2018-02-08 was a large down day)
- UVXY 2020-06-11: 50.2% -- plausible real: COVID 'Black Thursday' mini selloff (SPX -5.9%)
- UVXY 2024-08-05: 58.3% -- plausible real: Aug-2024 global vol spike / yen-carry unwind
- VIXY 2024-08-05: 43.1% -- plausible real: Aug-2024 global vol spike / yen-carry unwind

## Per-cell / per-split stats (in-market unless noted)
| block | days | %in-mkt | bps/day | NW t | ann.ret | worst day | worst mo | max DD | $750 P&L | $5K P&L | ex-w5 bps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1640_TRAIN_era1(-1x) | 540 | 86.11% | 29.8 | 1.967 | 71.91% | -18.26% | -26.11% | -35.96% | $1,645 | $10,965 | 47.4 |
| 1640_TRAIN_era2(-0.5x) | 465 | 68.17% | 3.5 | 0.423 | 3.95% | -8.22% | -11.07% | -20.36% | $56 | $371 | 14.2 |
| 1640_VAL(-0.5x) | 1006 | 74.95% | 5.1 | 0.755 | 5.91% | -16.90% | -13.60% | -32.88% | $193 | $1,288 | 12.8 |
| 1642_TRAIN_era1(-1x) | 540 | 100.00% | 13.9 | 0.505 | -27.15% | -83.00% | -89.48% | -93.07% | $-370 | $-2,464 | 48.0 |
| 1642_VAL(-0.5x) | 1006 | 100.00% | 7.8 | 1.031 | 12.23% | -19.50% | -38.86% | -62.18% | $439 | $2,925 | 15.6 |
| 1641_VAL | 439 | 76.08% | 21.9 | 1.250 | 36.93% | -13.34% | -17.49% | -39.23% | $547 | $3,645 | 39.5 |

## Basis-decile table, next-day SVXY return (mechanism check)
TRAIN corr(decile,ret)=0.540  VAL corr(decile,ret)=-0.299
- TRAIN era1 (decile1..10, bps): -167 39 50 -5 20 67 32 23 33 54
- VAL (decile1..10, bps): -8 25 17 2 34 5 1 2 1 -2
- VAL SVIX (decile1..10, bps): 110 52 58 -85 61 -19 80 11 9 3

## Pass bar (VAL, cell 1,640, -0.5x era) -- item by item
- [PASS] net bps/day in market >= +4: 5.148466075750813
- [FAIL] NW t >= 2.5: 0.7549486908851025
- [PASS] >= 40% days in market: 0.7495029821073559
- [PASS] TRAIN -1x era same sign, t>=1: 1.9674077899035751
- [FAIL] decile table monotone, both halves (corr>0): (0.5400693798266165, -0.29865400554828353)
- [PASS] worst day >= -25%: -0.16903006239364715
- [PASS] max drawdown >= -35%: -0.32877131621464306
- [PASS] gate worst day better than always-in: (-0.16903006239364715, -0.19498289623717213)
- [PASS] gate max DD better than always-in: (-0.32877131621464306, -0.6217616580310881)

**OVERALL: FAIL** (7/9 items met)

## Caveats (read as an adversary)
- **BLOCKER, not a coding bug**: this Alpaca account's historical data plan starts ~2016-01-04 for every symbol tested (confirmed via a direct SPY-2011 probe returning 0 rows) -- SVXY/UVXY/VIXY 2011-10..2015-12 are UNAVAILABLE here. TRAIN era1 (-1x) is therefore scored on 2016-01-04..2018-02-27 (~2.1 yr), not the full 2011-10..2018-02-27 (~6.4 yr) the PREREG samples from. VAL (2020-2023, the pass-bar split) is unaffected. No unauthorized substitute data source was used.
- Day-indexing convention (entry/hold/exit P&L formula) is this script's own literal reading of the PREREG -- not independently specified there; an independent rebuild must match it exactly (see module docstring) or the Jaccard/bps check will disagree for a structural, not a coding, reason.
- NW t is computed on the IN-MARKET daily P&L subsequence only (trade order preserved, flat days dropped), paired with 'mean bps/day in market' -- not on the full including-zeros calendar series. This is a judgment call the PREREG does not disambiguate.
- VIX 20-day mean uses a trailing window that INCLUDES day t (pandas default).
- Contract-fetch failures (see data provenance count vs ~181 target) leave basis gaps on those dates; F30 is simply not computed on a gap day (no entry/exit can trigger off it).
- SVIX TRAIN block is empty/near-empty by construction (inception ~2022) -- report-only, not part of the pass bar, per PREREG.
- Monotonicity is checked via Pearson corr(decile index, mean next-day return) > 0, a looser bar than strict staircase monotonicity; read the raw decile values above too.
- This is an ETP daily-return backtest, not a futures-notional backtest: SVXY/SVIX daily resets already embed their own cost/borrow drag, which is NOT separately itemized here.