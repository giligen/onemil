# Refuter 1: cells 1,591–1,598 (v2, tick-priced). Lens: obtainability, look-ahead, data

Verdict: **REFUTED.** The build's arithmetic is right. The reported VOID rail and the selected cell's VAL headline are not.

## 1. The VOID rail is reported as 0 %, but every cell is 60–69 % VOID (the verdict changes)
* `run_cell` (cell_1591.py:382–407) `continue`s on every VOID and never writes a VOID row. `cell_stats` then counts
  `void_reason != ''`, which is always 0. So `void_cycles=0` and "VOID share 0.0 % PASS" are both artefacts.
* Per-cell counts from the run log (full_run_1591.log:484–635): 1597 has 41 cycles and 92 VOID of 133 Mondays
  (69 %). 1593 is 84/133, 1595 is 91/133, and the gated cells are 38–45 VOID against 18–25 cycles (about 60–70 %).
  **All 8 cells breach the frozen 10 % rail, so the v2 result is VOID, not "FAIL with rails passing".**
* Most VOIDs come from strike selection. `select_strikes` still needs a 09:55–10:10 print on the chosen strike AND on
  K−10, so the "never VOID for a missing print" intent of Amendment 2 is not met. The ticks DB (495 legs) has
  124 ok / 224 fallback_5m / 147 void.

## 2. VOIDs correlate with losses in VAL (the v1 defect again)
Proxy: every Monday gets a short strike at 1597's median moneyness (−2.45 %), $10 wide, and SPY close at +45 days
(review/refuter1_1591_voidcorr.csv).
* VAL: included Mondays have a proxy loss rate of 5.6 % and a mean settle of $0.15/sh. VOID Mondays: 10 %, $0.92/sh (6×).
  The four VAL losers **2026-01-26, 02-02, 02-09, 02-17** (settle 9.7 / 10 / 10 / 7.2) are ALL VOID. 1597's
  "100 % win rate, worst month $0, Sharpe 4.25" exists because those entries were dropped.
* TRAIN: the 2025-04-04-expiry entries 2025-02-18 (SPY −17 % to expiry) and 02-24 (−12 %) are both VOID (full
  losses in the proxy). So is 2025-01-21 (full loss) and 02-03 (full loss). Loss rates are similar (13.7 % vs 13.0 %).
* Filling every Monday by proxy (median credit $1.73, ladder cap ignored): 1597 VAL comes out ≈ +6.9 %/mo on B,
  Sharpe 1.8, worst month −$2,452 (2026-03). TRAIN comes out ≈ +4.3 %/mo, Sharpe 0.84, worst −$3,240 (2025-03).
  This is not evidence (crude proxy). It shows the VOID filter moves the book both ways by several %/mo, so
  neither "FAIL" nor the 3.17 %/mo headline is a reading of the specified ~6-open ladder.

## 3. Tick fills (obtainability)
* Of 1597's 41 cycles, only 10 have BOTH legs traded in 10:00:00–10:00:30. The other 31 use the 5-minute fallback.
  The two legs' prints are a median of 71 s apart (p75 186 s, max 295 s), so the spread price mixes two SPY prices.
* Recompute: all 41 of 1597's rows were re-priced from ticks_state.db plus the SPY 16:00 minute bar, with 0 mismatches
  (credit and P&L to $0.5).
* The 15:59 OPRA snapshot on 2026-09-25 (Nov-06 expiry) shows a 30Δ/23Δ pair (756/746) with $0.13 spreads on each
  leg, so the half-spread is $0.065 per leg, above the $0.03 headline. For the 20Δ pair (741/731) the spreads are
  $0.10/$0.09. On that sample, last trades sat −0.135 and +0.235 from the leg mids. The $0.10 rail (1597 VAL
  2.91 %/mo) brackets this, so cost does not change the verdict. The fills are noisy, not biased in a known direction.

## 4. Look-ahead, exits, assignment, equity
* Strike selection and the IV gate use `entry_mid` (the mean of the 10:00–10:04 bar closes). The fallback ran 4,155
  times: it takes the nearest bar in 09:55–10:10, which can be as late as 10:10. That is minor (≤ 10 min) and is not the verdict driver.
* M=A exit fallback: `price_exit` calls `daily_asof(sym, sess)`, which returns the exit session's CLOSE, not the OPEN
  that the label and Amendment 2a describe (240 fallback legs). The fill is mis-timed, but it is not look-ahead. It
  only affects the A cells, not 1597. The trigger is evaluated on the daily close and acted on the next session, which is correct.
* The M=B hold-to-expiry and 16:00 intrinsic settlement are correct. Early assignment of a deep ITM short put in
  2025-04 (six rungs ITM means about 6 × $55K of stock) would exceed a $65K account's buying power. The loss stays
  defined, but the account needs a rule to close before expiry. Dividends make early put exercise less likely.
* Fixed equity $65K: every cycle is 1 contract with a worst case of about $830, and the budget assertion holds.
* 2024-08-05 and 2025-04-07 (Δ0.30) are both VOID. Both would have been winners (SPY +10 % / +16 % to expiry).

## Required fix
Write VOID rows and count them. Select strikes without requiring prints (from the listed grid by BS delta on a
surface or the neighbouring strikes). Recompute the rail. If VOID stays above 10 %, the frozen rule makes the test
VOID/inadequate, not a closure of defined-risk selling.
