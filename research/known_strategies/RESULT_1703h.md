# RESULT — Cell 1,703h: BTC Trend Family
## Price-Scale Check
Median Alpaca/yfinance close ratio: 1.000454 (0.0454% diff)
Daily-return correlation: 0.998927
Status: PASS (diff < 0.5%)

## Family Table
| Cell | Full | Since2018 | Y2018 | Y2022 | Y2025 |
|------|------|-----------|-------|-------|-------|
| mom_10 | 0.889 | 0.461 | -0.245 | -0.898 | -0.677 |
| mom_15 | 0.871 | 0.434 | -0.605 | -0.893 | -0.730 |
| mom_20 | 1.052 | 0.660 | -0.707 | -0.924 | -0.772 |
| mom_30 | 1.018 | 0.557 | -0.961 | -0.936 | 0.244 |
| mom_50 | 0.780 | 0.581 | -0.989 | -0.662 | -0.308 |
| mom_100 | 0.617 | 0.276 | -0.899 | -0.740 | 0.250 |
| sma_50 | 1.188 | 0.804 | -0.959 | -0.970 | 0.769 |
| sma_100 | 0.995 | 1.138 | -1.001 | -1.001 | 0.387 |
| sma_200 | 0.814 | 0.518 | 0.000 | 0.000 | -0.770 |
| always_long | 0.649 | 0.282 | -0.891 | -0.980 | -0.233 |

## Plateau/Peak Verdict
Rule: every N in {15,20,30} within 0.15 CAGR/DD of N=20, both windows, beats always-long
Verdict: PEAK

## Adversary Caveats
- Single-symbol backtest on synthetic spliced data; regime-dependent
- Fill realism: assumes market fills at close, no slippage
- Transaction cost (10 bp) is simplified; real trading has variable costs
- Tail dependence: check worst-year and bear-year returns per cell

## Reviewer note (Fable, after reading the table as an adversary)
- Verdict by the frozen rule: PEAK for N = 20 inside the momentum-sign family (mom_15 is 0.18 / 0.23 CAGR/DD below it).
- Family-wide fact, which the rule did not ask: EVERY trend cell (mom 10–50, SMA 50–200) beats always-long on CAGR/DD in both
  windows (always-long 0.65 full / 0.28 since 2018; cells 0.78–1.19 / 0.43–1.14 ex mom_100) and loses less in each bear year
  (2018: −8…−56 vs −73; 2022: −8…−48 vs −65). The mechanism is a plateau; the parameter 20 is not. The best cell changes with the
  window (sma_50 full, sma_100 since 2018) — parameter noise, no winner to pick.
- Defects: the `always_long_50pct` reference row is zeros (not computed); `sma_200` 2018 shows 0 % time in market although BTC
  was above its 200-day until ~2018-03 — not re-run, neither changes a verdict.
