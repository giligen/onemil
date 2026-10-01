# RESULT 1,687 -- ORB frequency/exit/sizing ideas 25, 33, 41, 42, 43, 44

PREREG: research/orb_freq/PREREG_1684.md. Book: production, real entry minutes, n=478 fills (recon={'no_bars': 5, 'no_range_or_breakout': 0, 'bad_R': 0, 'below_R_floor': 0, 'ok': 478}). Bar: paired dR>=+0.05R, day-clustered t>=2.5, ex-top5%>0 in BOTH years (TRAIN2025 & VAL2026), same-signed OOS2024H2 (1679's bar).

**CAVEAT (checked, not a bug):** `analysis_results/orb_bplus_book.csv` (the production book with real entry minutes that 1693/1679 reconstruct) spans 2025-01-02..2026-09-28 only -- it is a live/BT-tracking ledger, not a full-history backtest, so it has ZERO 2024H2 fills. Confirmed independently (date histogram on the raw CSV) and cross-checked against 1693_reads.csv: every production-slice pool there (e.g. gap_size_5-7%) also shows n=0 in OOS2024H2; only 1684/1685's separate daily-bar admission pools (a different data source, point-in-time universes) have 2024H2 coverage. Every OOS2024H2 read below is therefore UNTESTED, not a null -- reported as such, never silently folded into a pass.

## idea41_no2R_half3R_trailMFE1R
- TRAIN2025: dR +0.036R (iid_t 0.41, day_t 0.71, n212, ex_top5 -0.065, wkP10 -2.89, $/yr@375 +2832)
- VAL2026: dR -0.200R (iid_t -1.86, day_t -1.89, n259, ex_top5 -0.295, wkP10 -8.02, $/yr@375 -27261)
- OOS2024H2: dR +nanR (day_t nan, n0, $/yr@375 +nan)
- **verdict: fails 1679's bar (both years >=+0.05R, day_t>=2.5, ex_top5>0); OOS2024H2 UNTESTED (no fills on this book)**

## idea42_powerHour
- TRAIN2025: dR +0.000R (iid_t nan, day_t nan, n212, ex_top5 +0.000, wkP10 +0.00, $/yr@375 +0)
- VAL2026: dR +0.000R (iid_t nan, day_t nan, n259, ex_top5 +0.000, wkP10 +0.00, $/yr@375 +0)
- OOS2024H2: dR +nanR (day_t nan, n0, $/yr@375 +nan)
- **verdict: fails 1679's bar (both years >=+0.05R, day_t>=2.5, ex_top5>0); OOS2024H2 UNTESTED (no fills on this book)**

## idea43_addUnit1R
- TRAIN2025: dR +0.135R (iid_t 1.39, day_t 0.79, n212, ex_top5 -0.122, wkP10 -2.48, $/yr@375 +10753)
- VAL2026: dR +0.281R (iid_t 2.39, day_t 2.00, n259, ex_top5 -0.062, wkP10 -2.37, $/yr@375 +38251)
- OOS2024H2: dR +nanR (day_t nan, n0, $/yr@375 +nan)
- **verdict: fails 1679's bar (both years >=+0.05R, day_t>=2.5, ex_top5>0); OOS2024H2 UNTESTED (no fills on this book)**

## idea44_rangeFloorSizing
- TRAIN2025: dR +0.000R (iid_t nan, day_t nan, n212, ex_top5 +0.000, wkP10 +0.00, $/yr@375 +0)
- VAL2026: dR +0.000R (iid_t nan, day_t nan, n259, ex_top5 +0.000, wkP10 +0.00, $/yr@375 +0)
- OOS2024H2: dR +nanR (day_t nan, n0, $/yr@375 +nan)
- **verdict: fails 1679's bar (both years >=+0.05R, day_t>=2.5, ex_top5>0); OOS2024H2 UNTESTED (no fills on this book)**

idea44 detail: share of fills affected (range<0.75% of price) -- TRAIN2025 0.0%, VAL2026 0.0%, OOS2024H2 nan%. At the stated 13bps/0.75% floor the actual cost/R ratio achieved is 0.173R, not the idea text's 0.1R target (logged in 1687_cells.log) -- the 0.75% floor is authoritative, 13bps/0.1R is the rationale. $ at live sizing ($2000 risk/trade): multiply $/yr@375 above by 2000/375=5.33x.

## idea25_secondChance30minRange (ADDITIVE -- no production counterpart, not a paired delta)
- TRAIN2025: own mean R -0.408 (iid_t -1.83, day_t -1.16, n26, fills/wk ADDED 0.50, ex_top5 -0.653, wkP10 -1.02, $/yr@375 -3981)
- VAL2026: own mean R -0.183 (iid_t -1.19, day_t -1.71, n46, fills/wk ADDED 1.24, ex_top5 -0.343, wkP10 -1.89, $/yr@375 -4416)
- OOS2024H2: own mean R +nan (iid_t nan, day_t nan, n0, fills/wk ADDED 0.00, ex_top5 +nan, wkP10 +nan, $/yr@375 +nan)
- **verdict: fails the pool pass bar in both years (own mean R>=+0.05, day_t>=2.5, ex_top5>0)**

## idea33_earningsSplit (cohort split, not a paired delta)
- calendar: rebuilt from research/edgar_desk/events_raw.csv (8-K/8-K-A, item 2.02), approx event-session = filing_date or +1 day. Diagnostic: only 109/478 fill-symbols have ANY earnings event on record, and only 5/478 fills fall within a WIDENED +/-3 calendar-day window of one -- ORB's gap admission rarely coincides with a scheduled earnings filing; the split is likely **underpowered regardless of mapping precision**.
- earnings: TRAIN2025 +1.517R (tnan n1) | VAL2026 +nanR (tnan n0) | OOS2024H2 +nanR (n0)
- non_earnings: TRAIN2025 +0.264R (t1.50 n211) | VAL2026 +0.318R (t2.14 n259) | OOS2024H2 +nanR (n0)
- **verdict: INSUFFICIENT N to test the split (earnings cohort n=1 TRAIN / 0 VAL, both well under the MDE floor) -- non-earnings cohort alone is just 'most of production' (it IS the book, mean R +0.264/+0.318R, matching production's own +0.105R-ish live-config edge); no drop decision can be made** (approx event-session mapping: filing_date or +1 day -- see log)

