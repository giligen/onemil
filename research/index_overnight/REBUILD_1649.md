# REBUILD — cells 1,649-1,651 (independent, from PREREG_1649.md prose only)

Cost: 0.5bp/leg all-in (2 legs = 1bp round trip, matches "1bp generous" parenthetical). Early closes: no
pandas_market_calendars installed → detected empirically from bar data (closing-auction volume spike moves
from tod=960(16:00) to tod=780(13:00); verified on 2018-12-24 known half day) using v780>3*v960 & v780>1e5,
>=2/3 ETFs agree. Result, 2016-2023: **15 dates** — Jul3 2017/18/19/23, Nov23/24/25/26/27/29 2016-22 (one
each yr), Dec24 2018/19/20. Cross-checked against hand NYSE-calendar reasoning (July-4/Dec-25 weekday
alignment): exact match. A night/event touching an early-close date on either leg is dropped; any leg
>=2024-01-01 is dropped (TEST sealed).

## 1,649 unconditional overnight (2016-2023 pooled)
n=5,988 ETF-nights (1,996 nights x3). **Pooled mean = 3.221 bps/night, night-clustered t=1.79** (bar: >=2bps,
t>=2.5 informational — mean passes, t is short of the bar but PREREG marks it informational).
Per ETF (mean bps / t / ON-Sharpe / BuyHold-Sharpe / ex-worst5% / maxDD / worst night @$20K):
SPY 2.02/1.21/0.43/0.79/11.9/-32.9%/-$2,150; QQQ 2.77/1.44/0.51/0.89/14.2/-28.6%/-$1,894;
IWM 4.88/2.43/0.86/0.50/16.2/-31.4%/-$1,862. Sleeve (equal-wt 3ETF): maxDD -30.3%, worst night -$1,969,
ann.ret 7.6%, ex-worst5% 13.8bps.
Per-year sign (pooled avg): only 2016 (barely, +0.01bps) and 2022 negative → **7/8 years positive**; per-ETF
worst is SPY 6/8. **Alternative pass rule mixed: years/ETFs-positive legs pass, but overnight Sharpe only
beats buy-and-hold Sharpe for IWM (0.86>0.50) — SPY and QQQ FAIL that leg (0.43<0.79, 0.51<0.89)**: intraday
is not adding pure noise for the two largest-cap names, contra the documented mechanism.
Stacking: ~$6.4/night x21 nights/mo x3 ETF sleeve ≈ **$135-400/mo at $20K** (single-name to 3-name deployment).
Collision: a gap-down open hurts every overnight long the same morning the day-books (BF/ORB) are also long
into the open — correlated, not diversifying, tail risk.

## 1,650 conditional overnight (s_t=15:30-16:00/09:30-16:00 vol, heavy=s_t>=1.25x trailing-60 median)
TRAIN 2016-19: n=576 (400 night-clusters), mean=2.99bps t=0.92, ex-worst5=10.4, excess-over-complement=
-1.48bps t=-0.43 (heavy UNDERPERFORMS). Tercile(next-night bps) T1/T2/T3 = 4.04/4.78/3.69 — not monotone.
**VAL 2020-23: n=464 (324 clusters), mean=8.65bps t=1.95** (bar +8bps ok, t<2.5), ex-worst5=18.6, **excess=
+6.86bps t=1.34** (bar alt needs t>=2, fails). Tercile T1/T2/T3 = 4.75/-1.15/5.16 — not monotone either half.
Per-ETF VAL heavy mean: IWM 8.4/QQQ 8.2/SPY 9.4 (all positive). **Verdict: FAILS the pass bar** — VAL excess-t,
TRAIN t>=1, and both tercile-monotonicity conditions all miss; only raw VAL mean and "all 3 ETFs positive"
pass. Stacking: heavy nights ≈9.7 ETF-nights/mo in VAL x8.65bps x$20K ≈ **$170/mo**, on top of (not instead
of) 1,649 — same gap-down collision.

## 1,651 turn-of-month (2016-2023 pooled)
n=279 (93 month-clusters x3, 6 fewer than a no-exclusion 285: Jun/Nov-2019 events hit a half-day leg).
**Pooled mean=32.00 bps/event, month-clustered t=1.39.** 6/8 years positive (neg: 2016, 2018), 3/3 ETFs
positive (IWM 30.1, QQQ 30.9, SPY 35.0) → alternative rule (mean+years+ETFs) numerically passes.
**Tail check bites hard: ex-top-5%=5.27bps — 83% of the pooled mean is carried by the top 5% of events.**
ex-worst5%=60.4bps. Worst single event: IWM 2020-03-31->04-03, -841 bps (COVID). **Without 2020: n=243,
mean=14.40bps, t=0.64** — mean less than half, t-stat gutted. This reads as a lottery ticket riding 2020,
not a stable calendar edge; ex-top-5% and no-2020 both bite per the CLAUDE.md tail-dependence rule.
Stacking: 32bps x$20K x1 event/mo/ETF ≈ $64/event; 3 concurrent ETF legs ≈ $190/mo gross of the tail-risk
above (single-name rotation: ~$64/mo). Collision: TOM entry-nights are ALSO 1,650-heavy on **151/279 (54%)**
of events — the two sleeves fire together more than half the time, not independent.

## Refuter checklist (1,651)
1. Month boundary: first 3 SPY events = Jan29->Feb3-16, Feb29->Mar3-16, Mar31->Apr5-16 (holiday-aware via
   the daily-bar trading calendar itself, not a naive weekday count).
2. MOC fill = daily-bar close, NOT the 16:00 minute bar. Mean|official close - 16:00 bar close| on event
   days = **6.42 bps (n=558), max 80bps (IWM)** — meaningful; using the minute bar would misprice this.
3. Dividends: bars are `adjustment=all` (already total-return). Ex-div dates NOT independently verified here
   (no unadjusted series/div calendar cached) — from general knowledge SPY/QQQ/IWM pay quarterly, ex-div
   ~3rd Friday of Mar/Jun/Sep/Dec, which structurally sits mid-month, away from the TOM window. **Flagged as
   an unverified rebuild limitation**, not a computed result.
4. TOM-entry-night ∩ 1,650-heavy-night = 151/279 = 54% (see stacking above).
5. Tails: covered above — ex-top5%=5.27bps, ex-worst5%=60.4bps, worst=-841bps, no-2020 mean=14.4bps/t=0.64.
6. Per-year: 2016 -1.3, 2017 +19.7, 2018 -22.2, 2019 +32.8, 2020 +150.9(!), 2021 +35.4, 2022 +5.8, 2023 +35.6
   (pooled avg/yr) — 6/8 positive, but 2020 is >4x every other positive year.

## Independent-check comparison vs first build (nights_1649.csv, tom_1651.csv)
**1,649**: Jaccard=0.9925 (5,988 mine ⊂ 6,033 orig; the 45 orig-only rows = exactly my 15 excluded
early-close dates x3 ETFs — orig build appears not to exclude early closes). Matched n=5,988: mean|diff|=
0.34bps, **median|diff|=0.003bps**, corr=0.9979 — near-exact on the shared population. 4 isolated large
diffs (up to ~170bps: IWM/QQQ 2018-11-21, SPY 2018-12-21, QQQ 2020-12-23) not resolved within budget, but
immaterial given the near-zero median.
**1,651**: Jaccard=0.9789 (279 mine ⊂ 285 orig; the 6 orig-only = Jun-2019 and Nov-2019 events, both hitting
a half-day leg — Jul3-2019 is literally the 3rd session of July, Nov29-2019 is literally Nov's last session).
Matched n=279: mean|diff|=0.018bps, median=0.014bps, **corr=1.0000** — essentially exact agreement on
mechanics. Both Jaccards sit just under the 0.98 bar for one fully-understood, single-cause reason
(early-close exclusion policy), not a methodology or coding disagreement.
