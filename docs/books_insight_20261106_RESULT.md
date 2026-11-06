# Books insight since "real" running — result 2026-11-06 (ledger tally, read-only)

Sources: `data/trades.db` (paper, opened read-only), `logs/momentum_sleeve*_weekly.csv`, state files, IEX last trades at
~10:59 ET 11/6, SPY/IWM IEX daily bars. R = pnl/total_risk. 11/6 HOD is a PARTIAL day (4 fills still open).

## HOD paper (>= 10-19, closed, n 137)
| ISO wk | fills | $ | W/L | stops | sumR | meanR | top $ | ex-top1 | ex-top5 |
|---|---|---|---|---|---|---|---|---|---|
| W43 | 45 | +74 | 19/26 | 22 | +1.55 | +0.034 | 104 | -30 | -365 |
| W44 | 53 | -112 | 18/35 | 29 | -9.49 | -0.179 | 103 | -215 | -609 |
| W45 (to 11/6 am) | 39 | -185 | 13/26 | 16 | -12.41 | -0.318 | 106 | -291 | -693 |
Totals: 137 fills (~49/wk), -$224, -20.3 R, mean -0.149 R, median -0.89 R, 36 % wins, 49 % stopped; 15 days, 53 % green,
day-clustered t -0.43, max DD -$420, ex-top-1 % -$433, ex-top-5 % -$944; avg risk $38/trade. Excluding the 11/6 partial
day (-$297): n 128, +$73, mean -0.053 R, t +0.17, W45 +$112 (+0.038 R). Reference -0.026 R net / +0.075 gross: the book sits at
or below net reference; the pooled mean -0.15 R is within ~0.5 SE of it (2SE = 0.25 R).

## ORB paper
Spec filter (`closed` only) vs the full ledger — 11 fills sit as `filled` with exit_price, pnl and `force_close` set
(status never advanced to closed), net +$2,061, so the spec filter is WRONG for ORB:
| window / filter | fills | fills/wk | $ | sumR | meanR | wins | stops | days green | day t | 2SE(R) |
|---|---|---|---|---|---|---|---|---|---|---|
| >=9/25 closed only | 20 | 4.0 | -2,197 | -13.1 | -0.656 | 10 % | 15 | 10 % | -3.88 | 0.19 |
| >=9/25 closed + filled-with-exit | 31 | 6.0 | -136 | +0.1 | +0.002 | 29 % | 15 | 21 % | -0.06 | 0.56 |
| >=10/29 closed + filled | 11 | 12.0 | -1,048 | -6.5 | -0.588 | 18 % | 7 | 0 % (3 d) | -3.55 | 0.32 |
Full-ledger weekly ($ / fills / sumR): W40 -203/2/-1.1; W41 +164/4/+0.7; W42 -693/5/-3.7; W43 -215/4/-1.7;
**W44 +1,696/8/+11.6 (10/27 alone +1,859: CRDU +949, LABX +497, NBIG +284, NEBX +144)**; W45 -885/8/-5.8.
Ex-best-day (10/27): -$1,995; ex-top-5 %: -$1,581; max DD -$1,048. Stops 6 of 8 in W45.
Kill rails: weekly kill tripped 11/3 16:34 UTC (-$882 <= -$750) and again 11/4 (-$885); rails then rescaled to
-1,250/-1,875/-3,750 (log 11/5, 11/6). No kill trips found 10/19-11/2.
Versus replica (+0.1 R/fill, 2.4 fills/wk): fills/wk 6.0 (2.5x), mean R +0.002 (-0.1 R below, inside MDE 0.56 R); since 10/29 -0.59 R.
Also 12 `pending_new` rows (never filled) and 2 `stale_closed` — unfilled counterfactual.

## MOM (marks 11/6 ~10:59 ET, IEX)
| | start | now | return | SPY | IWM |
|---|---|---|---|---|---|
| paper (from 10/2, $20,000) | 20,000 | 13,814 | -30.9 % | -3.8 % | -3.1 % |
| live (from 10/19, $10,000) | 9,995 | 7,722 | -22.8 % | -3.4 % | +0.5 % |
Paper weekly equity: 10/5 20,103 (peak) / 10/12 19,317 / 10/19 17,829 / 11/2 16,416 / 11/6 13,814. Live: 10/19 9,995 / 11/2 9,174 / 11/6 7,722.
Live vs paper, same dates (10/19 -> 11/6): paper -22.5 %, live -22.7 %: tracking gap 0.2 pp. 11/2 -> 11/6: both -15.8 %.
Rails: paper peak $20,103 (10/5); -25 % line $15,077 — paper is $1,263 below it (-31.3 % from peak); -40 % kill line
$12,062, $1,752 (12.7 %) away. LIVE: rail $7,500 (0.75 x $10,000), equity $7,722 = **$222 (2.9 %) above**; checked at the next
rotation Mon 11/9 14:45 UTC. Broker positions match state (20/20, no qty diffs) on both accounts.

## What the data can / cannot say (MDE = 2SE of mean R)
* HOD: n 137, MDE 0.25 R, 15 days — cannot separate -0.15 R from the -0.026 reference; can say no gross edge visible, W45 worsening is noise-level.
* ORB: n 31, MDE 0.56 R — cannot say anything about +0.1 R; can say P&L is one Monday (10/27) and the post-Amendment 11 fills are -0.59 R.
* MOM: 5 weekly marks, one regime (a -4 % SPY / -4 % IWM crash day 11/5); can say live tracks paper and the beta/loss is ~6-8x the indices.

## Anomalies
1. ORB status field: `filled` rows with exit + pnl (11) are outside the `closed` filter; the EOD/ramp tallies reading `closed` only would show -$2.2K instead of -$0.1K.
2. MOM 10/26 rotation crashed in both paper and live (get_all_assets: paper timeout traceback, live pydantic ValidationError on Asset) — no week-44 row; sleeves held week-43 names through 10/19 -> 11/2.
3. Live MOM rail $222 away with a Monday rotation pending; paper is already through its -25 % line (paper does not liquidate).
