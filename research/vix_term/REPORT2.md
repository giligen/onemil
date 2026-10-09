# REPORT2 — cell 2, vol line: SPVXSP rebuild from free CBOE VX settlements (2026-10-09)

**VERDICT: VOID by the PREREG_2 tracking gate (R^2 < 0.95 on BOTH legs). Per PREREG_2 the backtest was NOT run:
no trades2.csv / weekly2.csv, no strategy table, no cadence-bar scorecard. Cell 2 produced no strategy evidence.**

## Data (free, no purchase)
* Per-contract VX monthly files, `cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_<expiry>.csv`
  (contracts expiring 2013-01..2026-12; 2013-14 files start in 2013 so earlier rows were merged from the archive) and the
  archive `cdn.cboe.com/resources/futures/archive/volume-and-price/CFE_<code><yy>_VX.csv` (contracts 2010-12..2014).
  193 monthly contracts, none lost (`data/vx/`). No consolidated `VX_History` file is reachable (403).
* Index: constant-30-day, daily roll front->second in proportion to business days remaining (w_front = bdays t..E1 /
  bdays E0+1..E1, set at the prior close), excess return, level 100 at 2010-12-01 (`index_rebuild.csv`, `index2.py`).
* Legs: long = idx ret - 0.89 %/yr/252 (VXX); short = -0.5 x idx ret - 0.95 %/yr/252 (SVXY), second column -1.0x to the
  2018-02-27 close (era-correct).

## Completeness (gate: >= 99 % of NYSE days 2011-01-03 -> 2026-10-08, 3,965 days)
VIX 100 % . VIX3M 100 % . rebuilt index 100 % (0 NaN-return days; no CBOE-only holiday rows survive). PASS.
Caveat: 820 of 35,767 contract-day rows (655 in 2013 contracts) had Settle = 0 in the CBOE file and use the day's Close
instead (listed by expiry year in `run`/fetch log). That is a settle-vs-last-trade approximation, mostly in 2011-2014.

## Tracking validation (daily returns, Alpaca adjusted bars; IEX and SIP feeds returned identical closes)
| leg | window | beta | alpha bps/d | R^2 | n | mean TE bps/d | sd TE bps/d |
|---|---|---|---|---|---|---|---|
| VXX vs +1x idx | 2018-01-19..2026-10-08 | 0.823 | -2.1 | **0.855** | 2192 | -0.2 | 197.6 |
| SVXY vs era leg (-1x then -0.5x) | 2016-01-05..2026-10-08 | 0.664 | +1.8 | **0.489** | 2706 | -1.7 | 255.2 |
| SVXY, -1x era only | 2016-01-05..2018-02-27 | 0.470 | +2.9 | 0.261 | 541 | -10.1 | 562.8 |
| SVXY, -0.5x era only | 2018-02-28..2026-10-08 | 0.971 | +0.6 | 0.958 | 2165 | +0.4 | 48.2 |
| SVXY vs forced -0.5x all years | 2016-01-05..2026-10-08 | 0.962 | +1.1 | 0.555 | 2706 | +0.8 | 213.5 |
Gate: R^2 >= 0.95 on both legs required. VXX 0.855 and SVXY 0.489 -> **VOID** (`tracking.csv`, `gate.json`).

## Diagnostics (NOT a pass path; shown so the VOID is not misread)
* Two days carry the failure: 2018-02-05 index +97.1 % vs VXX +33.5 % and SVXY -31.9 %; 2018-02-06 index -26.0 % vs
  VXX -1.7 % and SVXY -83.0 %. VX settles at 16:15 ET, the ETPs' close is 16:00: the 02-05 after-close VIX spike
  shows in the futures settle but in the ETP only the next day (my reading of the pattern, not separately proven).
  Two-day compound: SVXY -88.4 % real vs -96.4 % synthetic -1x leg.
* Excluding 02-05/06: VXX 0.919, SVXY 0.929 (still < 0.95). Excluding 2018: VXX 0.935, SVXY 0.940. 2019-2026 only:
  VXX 0.935, SVXY 0.964. So the 16:15-vs-16:00 offset (and VXX ETN premium/discount noise, per-year VXX R^2 0.78 in 2022,
  0.88 in 2020) caps tracking below the PREREG bar even outside the 2018 episode; only SVXY in the -0.5x era clears it.
* The -1x era (2016-2018) is the worst fit (R^2 0.26, beta 0.47): the pre-2018 SVXY was -1x on the 16:00 close vs a 16:15
  settle, with the front-month roll mix differing from the index weights.

## Tail lines requested (arithmetic on $5,000 of the SYNTHETIC leg, NOT a strategy result)
The rule is flat at the 2018-02-02 close (VIX3M/VIX = 0.984), so the base 1.05/0.95 rule held nothing across 02-05/06.
Held through, $5K of the synthetic -0.5x leg: 02-05 -$2,428.570283806344, 02-06 +$649.9464412154515; synthetic -1x leg:
02-05 -$4,857.140567612688, 02-06 +$1,299.892882430903. Real SVXY (-1x then): 02-05 -31.87 %, 02-06 -83.00 % (-$1,593.5105477249078 and -$4,150.236045542904 on $5K of the real SVXY close-to-close).

## MDE / cadence-bar / strategy table
Not computed (gate stop). Carried over: cell 1 B-half (2018-2026, 458 wk) MDE $33.8/wk.

## Own caveats
* A VOID here is a claim about THIS rebuild, not about the strategy: the gate measures my index vs 16:00 ETP closes. A
  closer test would use 16:00 VX prices (not in free CBOE data) or an index series with matching timestamps; a looser gate
  (e.g. R^2 on 2-day returns, or era-restricted to 2019+) is a new PREREG, not a post-hoc rescue.
* Roll-weight convention (business days incl. t..E1) and the Settle=0 -> Close fallback are my implementation choices;
  an independent rebuild was not run. Weekly VX contracts excluded (monthlies only, as SPVXSP).
* Alpaca SIP and IEX daily bars were identical for both ETPs (checked on the whole history), so feed is not the cause.
Files: fetch2.py, fetch_sip.py, index2.py, run2.py, index_rebuild.csv, tracking.csv, gate.json, data/vx/ (193 files), data/*_daily_sip.csv.
