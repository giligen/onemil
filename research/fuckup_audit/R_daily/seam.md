# R_daily — universe set, venue-share constant, seam and price-scale checks

## 1. Universe reference set (point-in-time listings, 2024-07..2026-09)

- `exchange == XNAS` and `security_type == C`, test tickers removed: **3,257 distinct symbols**, median **2551 per month**.
- EQUS era (2024-07 on): membership is taken PER MONTH.
- ITCH era (2018-05..2024-06): membership is the UNION of the bought months — a survivorship filter on that era, quantified in §4 and controlled by the secondary universe (every ITCH symbol, no PIT requirement).

## 2. Venue-share constant (ITCH volume -> consolidated-equivalent)

Overlap month 2024-09, Nasdaq-listed common stocks with EQUS dollar volume >= $10M (15,430 symbol-days, 1,229 symbols):

| median | p25 | p75 | p10 | p90 |
|---:|---:|---:|---:|---:|
| **0.3694** | 0.3016 | 0.4332 | 0.2247 | 0.5081 |

The panel multiplies every ITCH-era volume by **1/0.3694 = 2.707** so that one `20-day median dollar volume >= $10M` rule means the same thing on both sides of the seam.  The dispersion (p25..p75) is real per-name variation in Nasdaq market share; it makes the ITCH-era liquidity gate noisier than the EQUS-era one, never look-ahead.

## 3. Price-scale check

**ITCH vs EQUS, same symbol-day, overlap month** (52,802 keys): median abs difference **0.0000%**, within 0.1% **66.1%**, off by > 0.5% **23.19%**.  For a Nasdaq-listed name the ITCH close is the Nasdaq closing cross, i.e. the official close — which is why the universe is restricted to Nasdaq-listed names (for an NYSE-listed name the ITCH open/close would be an arbitrary off-primary print).

**ITCH vs Alpaca `cache.db::daily_bars`** (0 random keys from an ITCH-era row group, Nasdaq-listed commons):

- **no usable overlap**: `daily_bars` holds 19 rows before 2024-07-01 (the cache was built from 2025 on), so the ITCH era cannot be checked against Alpaca directly. The chain that does cover it is ITCH -> EQUS.SUMMARY (§3, 188,793 keys in the overlap month) -> Alpaca (N2 checked EQUS against `daily_bars` on 200 keys: 99.0% within 0.01%).

## 4. Splits (the files are UNADJUSTED — flagged, never silently adjusted)

Overnight close ratios <= 0.55 or >= 1.80 on Nasdaq-listed commons over the ITCH era: **1,872 events**, of which **568** sit within 5% of an exact 2:1 or 3:1 ratio (`split_candidates.csv`).  These fabricate both signals and exits in an unadjusted panel, so the run reports a control book with every trade whose hold window contains one of them removed.

