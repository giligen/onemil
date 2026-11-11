# HOD + ORB paper books, 9/30 → 11/10 (30 sessions) — Opus build 2026-11-11

(Agent build; the write was blocked for the agent, saved verbatim by the session. Rebuild: `REBUILD.md`.)

**Corrections to the PREREG's stated facts**
- HOD fills are in trades.db: 288 closed `hod_break` paper rows in the window, 293 since 9/29.
  - The ledger is $158 below the broker, and all of that is on 10/02 partial-fill rows where `shares` < `filled_qty`.
  - `total_risk` is wrong on those rows, so R = pnl / (|fill − stop| × filled_qty) was used.
- Amendment 1 counts from 9/25, not 10/30.

**Finding 1: the ramp checker**
- Stage start 9/21 on any account gives 3 live fills (−$119) plus 32 paper fills (−$88).
- Its monster CRDU +$949 includes $560 from the 861-share accident: the ledger booked the broker's 1,460 shares at its own 5.56 entry.
- DEMOTE comes from losing streak 8 ≥ 5.
- "Sessions 25" counts weekdays, not days with a trade.
- `--stage-start 2026-09-25` gives a paper-only window, but the accidents still have to be removed by hand.
- The gate today: 32/40 symbol-days and 15 sessions with a fill. Realized ex-accidents is −$614 and ex-monster is −$1,110, so the remaining 8 symbol-days need more than +$139 each. Projected around 11/20.

**Per book**

| | ORB ledger | ORB ex-accidents | HOD |
|---|---|---|---|
| fills / sessions | 31 / 14 | 31 / 14 | 288 / 29 |
| fills per week | 5.2 | 5.2 | 48 |
| realized $ | +8 | −517 | −150 ledger (broker +7) |
| mean R / median R | +0.03 / −0.55 | −0.09 / −0.55 | −0.008 (robust) / −0.72 |
| win rate | 32 % | 32 % | 39 % |
| ex-top-1 / ex-top-5 % $ | −941 / −1,437 | −1,014 / −1,403 | −257 / −1,700 |
| t iid / day-clustered | 0.10 / 0.06 | −0.47 / −0.27 | −0.11 / −0.10 |
| MDE on mean R | 0.78 R | 0.54 R | 0.28 R |
| stop rate | 52 % (BT 31 %) | 52 % | 52 % (anatomy 44 %) |
| entry slip | ask-to-fill +43 bps (model 30) | | +2.5 bps vs trigger |
| stop slip | −27 bps (−$187 total) | | −4.7 bps (stop-market BT was 35) |

**Fill rates**
- ORB: 74 % (BT 75 %).
- HOD: 155 filled of 203 that crossed on the tape, 76 % (BT 82 %). The 12/day cap bound on 11 of 29 days.

**Execution defects, priced from broker fills**
- CRDU 10/27: +$603.
- LABX 10/27: +$497, naked to the force close. This is the best fill, and its outcome under the rule is unknown.
- BEX 11/9: the add-on made +$3; the ledger mis-booked the trade by −$34.
- TXN 10/27: classed as EXT, not ORB.
- Not priced: the weekly-kill halt 11/4–6, and 11/10 with no ranges.

**Cross-check against the accounts**
- ORB account: equity +$1,285 = TOM +$1,166 + ORB +$89 + EXT −$3, residual +$33.
- HOD account: equity −$4 against broker realized +$7, residual −$12.

**Weekly ($ / R)**
- ORB ex-accidents: −106/−0.7, +164/+0.7, −693/−3.7, −215/−1.7, +1,136/+7.7, −885/−5.8, +82/+0.7.
- HOD: −375/−7.1, −76/+5.9, +265/+5.8, +74/+2.0, −112/−9.7, −22/−1.0, +95/+1.7.

**Cadence bar**
- Both books pass ≥ 3 fills/week.
- Both fail green weeks (3/7 each) and weekly P10 (ORB −5.8 R, HOD −9.7 R).
- Six weeks cannot settle the strong-week gap, the bleed, the null percentile or the cycle count.

**Ranked explanations, ORB**
1. Edge ≈ 0 out of sample and the book is tail-carried, consistent with REPORT_1709 (+0.02 R).
2. Stop excess: 52 % vs 31 %. About 6.6 extra stops at roughly −1 R covers the whole deficit.
3. SK-Hynix wrapper clustering in W45: 5 of 8 fills, all stopped.

**Ranked explanations, HOD**
1. At its reference (−0.026 R net).
2. Better execution produced no edge: 1,466/1,467 gave −0.05/−0.08 R.
3. Tail and risk-size noise: ex-top-5 % is −0.12 R, and 17 fills risked less than $10.

**Recommendations (agent)**
- ORB: score the gate with `orb_ramp_check.py --stage-start 2026-09-25`, removing CRDU −$560 and LABX's naked +$497 by hand. If realized or ex-monster is ≤ 0, shelve; there is no appeal. Ignore today's DEMOTE, and change no knobs (code frozen).
- HOD: keep running unchanged to 500 fills (about 12/10). Pause and run cell 1,723 early if equity since 9/30 is ≤ −$750. Before 1,723 reads: use broker fills or the robust R; fix `scanner_qualified_at_arm`, which is 0 on all 1,436 parity rows.
- HOD: add no exits or filters; those were already rejected.
