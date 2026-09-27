# PREREG_1567 refuter 1: obtainability, look-ahead and data

Verdict: the builder's overall **FAIL stands**, and this check makes it stronger. The builder's TRAIN
selection (cell 1574) and its headline TRAIN statistics are **not reportable**, because they come from
outcome-correlated VOIDs.

Scripts, all read-only on the cache: `review/refuter1_probe.py` (checks each Monday),
`review/refuter1_legs.py` (audits the legs and diagnoses the VOIDs), `review/refuter1_stale.py` and
`review/refuter1_pairaware.py` (a sensitivity run of `cell_1567.main()` with only strike selection
changed; output in `review/pairaware/`).

## D1: VOID survivorship drives the 1574 selection (verdict-relevant for the selection, not for FAIL)
`select_strikes` picks the short strike with the nearest delta among strikes that printed. It then VOIDs the
cycle when the long leg exactly W below has no 10:00 print. It never tries the next-nearest valid pair. SPY
lists every strike, so a live engine would trade the round pair.

Crash-window Mondays that passed the IV gate but were VOID in cell 1574 (Δ0.15, W10, hold to expiry):
* 2025-02-24: IV 15.4 %. The nearest-delta short was 561 and 551 had no print. 560/550 both printed (Δ 0.159).
  Expiry 4/4, SPY closed at 505.28. Result: **−$909** (max loss).
* 2025-03-03: IV 17.4 %. The nearest-delta short was 549 or 550 and its long leg had no print. 550/540 both
  printed. Expiry 4/17. Result: **−$922**.

Pair-aware rerun (the same code with only this change, still causal and still using the 10:00 window):

| cell 1574 TRAIN metric | Builder | Pair-aware |
|---|---|---|
| Cycles | 19 | 29 |
| Monthly Sharpe | 7.60 | 0.21 |
| Win rate | 100 % | 93 % |
| Max drawdown | $0 | $1,673 |
| April 2025 | +$158 | −$1,673 |

With pair-aware selection, TRAIN selects cell **1568** (Δ0.15, W5, management A, gate on). Its VAL read:

* Mean monthly return on B: 0.00 %
* Monthly Sharpe: 0.00
* Ex-top-5 % P&L: −$149

That is a **FAIL**. About 35 % of the gated TRAIN weeks in cell 1574 were VOID, and the VOIDs removed exactly
the two losers of April 2025. The "thin-sample artifact" caveat in RESULT understates this: the cause is
survivorship in the data, not luck.

The rebuild's 1586 VAL figure (+9.98 %/mo) is not reproduced. Under the builder's code with pair-aware
selection, 1586 gives +2.95 %/mo on VAL with Sharpe 0.96 (FAIL), and its TRAIN has three −$800 cycles
(Feb–Mar 2025 entries). `1567_compare.md` attributes the gap to sizing (9 contracts against 2).

## D2: the fill standard cannot be verified with this cache
* The cache holds option trade prints only (OHLCV), with no NBBO. Whether "mid − $0.03/leg" is achievable
  cannot be measured here.
* Leg audit of the 35 cycles in cell 1574:
  * 49 % of cycles used a fallback bar (outside 10:00–10:05) on at least one leg. Some of those bars are at
    10:06–10:09, after the fill window, and the strike choice uses them. That is a mild look-ahead.
  * Median print volume in the 15-minute window is 9 contracts on the long leg and 14 on the short leg.
  * The two legs printed a mean 2.4 min apart. SPY moved up to $1.48 between the leg prints, a mean credit
    error of about $0.02/sh. On 2024-08-05 SPY moved $1.40 between legs, about $0.10 on a $0.98 credit.
* Scale: credits run $0.70–1.10. Each extra $0.02/leg costs about $4/cycle against a mean of $66/cycle.
  This cannot flip FAIL.

## D3: management timing (management A only; not the builder's selected cell, but it is the pair-aware selection)
* profit_50 and dte21 exits fill at the same daily close that triggered them, not on the NEXT session as the
  PREREG requires. 0.8 % of the 493 such exits price a leg from a forward-filled older close.
* STOP exits fill at the next day's daily OPEN print, which is noisy:
  * The 4/17 550P printed an open of 1.67 on 2025-04-07 and closed at 43.62.
  * The 4/25 515P opened at 33.20 and closed at 22.47.

  All 239 stop fills are exposed to bad opening prints.

## D4: 2024-08-05 and April 2025, leg by leg (cell 1574)
**2024-08-05:** no 1574 spreads were open. The gate skipped the July entries (IV 10–13 %). The 8/5 entry
(455/445, credit 0.977, IV 23.7 %) expired worthless.

**April 2025:**
* Two builder spreads were open:
  * 515/505, expiring 4/25.
  * 525/515, expiring 4/30.
* On the 4/8 closes they marked at 9.39 and 6.12, so about −$1,400 of combined mark-to-market loss was near
  max loss. Held under management B, they recovered to +$82 and +$76.
* Closing cost at 10:00 on 4/4 and 4/7 cannot be measured: there are no minute bars on non-entry days, and
  the daily opens are chaotic.
* Monthly drawdown is measured on REALIZED exit-month P&L, so mark-to-market troughs are invisible.
  "Max DD $0" is not a risk statement.

## D5: assignment, ex-dividend and settlement
* Management B holds deep-ITM short puts. Example: the 525P was about $28 ITM on 4/8 with 22 DTE, so early
  assignment was unlikely given its time value.
* Settlement at intrinsic ignores physical settlement and pin risk. It also ignores Alpaca's expiry-day
  liquidation of ITM positions that lack buying power: 100 SPY shares are $50–75K per contract, against a
  $65K account.
* Put ex-dividend risk is minor.

## D6: other data points
* Fixed equity of $65K: B/6 = $1,083, which buys 1 contract at W10, so only about 85 % of B is used. This is
  benign.
* The spy_daily bar for 2026-02-02 has a low of 69.005, a bad print. The simulation does not use it, but the
  cache is dirty.
* 95 % of entry-minute attempts are absent, many of them unlisted expiries.
* Only $1 strikes are in the grid.
