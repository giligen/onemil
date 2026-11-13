# RESULT 1,760 — G1 ORB/HOD breakout on Alpaca crypto, hourly (Stage 1)

**Verdict: KILL** (primary S2 long, BTC+ETH, taker, VAL). TEST never fetched.

**Data.** BTC/ETH/SOL hourly 2021-01..2023-12. Missing hours: BTC 16, ETH 18, under 0.2 %. **SOL VOID**: feed ends 2023-07-06, 49.7 % of VAL missing. Scored BTC+ETH.

**Spec choices (PREREG silent, causal).** Break = bar close beyond the level, entry at that close (one bar late), first break per day per side, stop before target, 10 bp slip on stop exits, 5 bp round-trip spread, zero-hold entries skipped.

**Scale.** Median R: S1 0.86 %, S2 1.12 % (VAL). Taker round trip costs 0.52–0.67 R, maker 0.35–0.43 R. Gross edge is about zero, so cost decides.

**VAL 2023, BTC+ETH, taker: net R, t iid, t day-clustered, trades/wk.**
- S1 long: -0.79, -12.1, -9.7, 7.4. S1 short: -0.89, -15.5, -12.5, 7.9.
- S2 long: -0.64, -8.8, -7.6, 5.5. S2 short: -0.73, -10.1, -8.7, 4.9.
- Gross R: S1 long +0.088, S1 short -0.062, S2 long +0.005, S2 short -0.119.

**S3 (S2 long, VAL).** Weekend n 85, net -0.90; weekday n 202, net -0.53. No weekend effect; both gross about 0.

**Other reads.** Maker S2 long VAL net -0.42 (t -6.0), same sign. Ex-top-5 % below zero everywhere (S2 long VAL -0.77). TRAIN halves S2 long: 2021 -0.19, 2022 -0.51. Green weeks 13 %, positive months 1 of 12. MDE at 80 % power 0.20 R.

**Kill rules hit:** VAL net <= 0, t < 2, ex-top-5 % <= 0. Frequency passes; edge does not.

**Scope of the null.** Hourly bars, close-confirmed entry, R about 1 % of price, fee-paying execution, BTC and ETH, 2021–23. It says nothing about finer bars or maker-only resting orders with a fill rate.

CHILDREN:
1. G1b: 5-minute bars, same 13:30 UTC range, to test whether a tighter R leaves gross edge (probably dead on 0.5 % cost).
2. G1c: daily-bar crypto trend (20-day breakout, wide stop) where cost per R is under 0.1.
