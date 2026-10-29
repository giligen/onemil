# REBUILD 1,709a/b — independent rebuild from PREREG_1709.md prose (Sonnet, 2026-10-29)
Code `rebuild_1709.py`; rows `rebuild_1709a.csv` / `rebuild_1709b.csv` (date, symbol[, delay_s]); full log `rebuild_1709_tables.txt`.
Conventions: R = (exit-entry)/(entry-stop) = pnl/(shares x risk); stops/lock-stops fill min(stop, bar open) x (1-35 bps);
Rule M at breakout-bar close (exit = close), Rule D at bar b1 (exit = entry-0.5 range), lock arm 1.75 / stop 0.5 x range_size from the NEXT bar,
force_close = OPEN of the first bar >= 15:45 ET (ledger shows 19:45:01 UTC = 15:45 ET, PREREG text says 15:55 — ledger used; px diff 0 bps).
Entry bar can stop on its low (base) vs never (`_nes` sensitivity; moves nothing > 0.02 R). Day-clustered bootstrap 4,000 reps, 95 %.
Stop = range_low unless pattern_data holds atr14 (PREREG rule). Opening range (09:30-09:34 bars) equals pattern_data range on 100 % of live rows.

## 1,709a (live ledger; n 121 with bars+pattern_data of 137; 14 fills after 9/28 excluded)
| | ledger | re-sim (i) BT entry rh x 1.003 | re-sim (ii) live fill |
|---|---|---|---|
| stop share | 38.0 % (41.6 % on all 137) | 37.2 % | 33.9 % (33.1 % entry-bar-no-stop) |
| lock / tag / force share | 9.9 / 31.4 / 20.7 | 11.6 / 24.8 / 26.4 | 10.7 / 28.9 / 26.4 |
| mean R [CI] | -0.091 [-0.31,+0.16] | -0.154 [-0.39,+0.13] | -0.083 [-0.33,+0.22]; net of row half-spread -0.119 |
* Excess ledger-over-BT: 13.2 pp (41.6 vs 28.4 fixed) or 9.6 pp on this n; (ii)-(i) = -3.3 pp -> explained -34 % (own-(i) basis: ledger 38.0 vs (i) 37.2 is only +0.8 pp, so (i) already reproduces the live stop share).
* Confusion (ledger rows x sim cols; force/lock/stop/tag): force 23/0/0/2; lock 0/11/1/0; stop 1/0/38/7; tag 8/2/2/26. Class agreement 81.0 % (stop vs not-stop 90.9 %; (i) 82.6 %).
* Ledger stop == my stop on 100 % of rows (only 8 rows carry atr14 in pattern_data; 96 % of ledger stops are plain range_low).
* **H1: FAIL** (explained < 50 %, agreement 81 % >= 70 %): the slow-stop / extra-stop excess is NOT a fill-price artefact on these bars.

## 1,709b (replay fills, 301 filled at 0 s / 212 at 30 s; stop = range_low base, ATR-floor sensitivity)
| segment (delay 0) | filled | covered | win-lose cov gap | mean R [CI] | ex-top5 | cap 3R | stop share | covered fills/wk |
|---|---|---|---|---|---|---|---|---|
| 2023-01..24H1 | 53 | 0 % | n/a | not computable | | | | 0 |
| 2024H2 | 34 | 100 % | 0.0 pp | +0.581 [-0.20,+1.67] | -0.030 | +0.131 | 29.4 % | 1.31 |
| 2025H1 | 42 | 100 % | 0.0 | +0.660 [+0.02,+1.36] | +0.149 | +0.340 | 38.1 % | 1.62 |
| 2025H2-2026-09 | 172 | 99 % | -0.6 | +0.335 [-0.00,+0.72] | -0.087 | +0.072 | 26.5 % | 2.79 |
| in-regime 2025-01..26-09 | 214 | 99 % | | +0.400 [+0.09,+0.73] | -0.022 | | | |
Delay 30 s: 2024H2 +0.714 [-0.33,+2.17] ex-top5 -0.163 (n 24); 2025H1 +0.771 [-0.15,+1.88] (27); 2025H2-26 +0.192 [-0.14,+0.57] ex-top5 -0.186 (126); in-regime +0.295 [-0.02,+0.66] ex-top5 -0.111.
ATR-floor sensitivity (floor from Databento daily parquet, binds on 49 % of fills): in-regime +0.387 [+0.05,+0.77] ex-top5 -0.061; 2024H2 +0.427 [-0.41,+1.60] ex-top5 -0.194. +40 % @ 3R scale-out (engine setting, not in PREREG): in-regime +0.342; 2024H2 +0.429.
BT-entry cross-check (re-sim at the BT population entry vs BT exit reason, n 528): class agreement 93.8 % range_low stop, 96.2 % WITH the ATR floor — the BT book used the floor, the live ledger mostly did not.

## Pass-bar lines (frozen PREREG)
* H2 out-of-regime mean R >= +0.10 with CI lo > 0: **FAIL** (only 2024H2 is coverable: +0.58 [-0.20,+1.67], n 34; 2023-01..24H1 has NO bars -> 0 % coverage).
* Same sign in both TRAIN halves: pass (+0.66 / +0.34 at 0 s; +0.77 / +0.19 at 30 s). Ex-top-5 % > 0: **FAIL** (2024H2 -0.03, 2025H2+ -0.09, in-regime -0.02). >= 3 fills/wk: **FAIL** (1.3-2.8 covered/wk).
* Availability rail: 2023-24 block 34 of 87 covered = 39 % (< 80 %) -> VOID for that period; in-regime 99 %, winner/loser gaps <= 2 pp.
## Caveats
* Mean R is carried by the top 5 % of fills (max +16.8 R); the in-regime CI excludes 0 but ex-top5 does not. Rule M/D exits charged no extra slippage (bar close / rule price); only 1,709a has a net-of-half-spread line.
* Late-fill Rule M: decided at the close of the fill bar when the fill is <= 15 min after the breakout bar, skipped beyond. Replay entry minute = 09:35:00 + max(t*, delay). Wrapper/unadjusted-split checks not repeated (replay R from raw 1-min bars).
* Agreement with the first build (agreement bar: stop share 2 pp, mean R 0.02, exit reason 95 %) is to be checked by the caller; this file never read the first implementation.
