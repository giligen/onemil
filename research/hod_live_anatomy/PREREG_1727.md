# PREREG — cell 1,727: double down on the 14:00 runner, at power (owner 2026-10-30: "keep nothing to the 500; $50 is a proxy")

Frozen before any number. The live book (21 runner-side positions) cannot resolve the add-on; the HOD backtest population
on the research bar store can. The dedicated "trade only the runners from 14:00" strategy (owner item 3) IS the add
leg evaluated on its own, so one cell answers items 2 and 3.

## Population
Cell 1,681's live-rule fills (TRAIN-H2 n ≈ 2,349 / VAL n ≈ 3,157; locate the per-fill file its script wrote — fill
minute, fill price, stop, target, R per share; `research/hod_entry/cell_1681*.py`). Bars: `research/bf_zero/bars_sip.db`
(`bars(symbol, day, t, o, h, l, c, v)`, read-only, never copied — 27 GB). Exclude fills after 13:30 ET (no afternoon to
speak of) and fills whose live-rule exit occurred before 14:00 (stop/target already hit) — the eligible set is "open at
14:00", exactly as in the live cells. TEST (2026-06 onward if the 1,681 split defines one) stays sealed.

## State at 14:00 (bars through the 13:59 close only)
open P&L in R; post-entry high; R1 = open P&L ≥ +0.5 R AND 13:59 close within 0.5 R of the post-entry high
(the 1,725 definition, unchanged). R2 as in 1,725 is reported but not a ship candidate.

## Measurements
1. Uncensored path 14:00 open → 15:55 close for R1 and non-R1: mean/median bps, R, MFE/MAE after 14:00, share up;
   per split; day-clustered t.
2. **B1 double down (the ship candidate)**: at the 14:00 bar open + 12 bps buy the ORIGINAL share count on R1 names;
   add-leg stop = 13:59 close − 1.0 R (original R per share), filled at stop − 12 bps on a bar low ≤ stop; the add
   exits with the position: if the original leg's 2 R target prints (bar high ≥ target) both legs sell at the target;
   otherwise both sell at the 15:55 close − 12 bps. Report the add leg alone (risk unit = 1 R of the original R per
   share): n, fills/week, mean R, day-clustered t, ex-top-5 %, winner-capped (+2 R), per split, both TRAIN halves
   (by alternating weeks). Also the combined position Δ vs the live rule.
3. Dollar translation: add-leg $ per month at $50 R and at $150 R per original position, from the VAL rate.

## Pass bar (frozen)
Add-leg mean ≥ +0.10 R in TRAIN-H2 AND in VAL; day-clustered t ≥ 2 on VAL; ex-top-5 % > 0 on VAL; winner-capped > 0;
both TRAIN halves same sign; ≥ 3 adds/week at the live fill rate (scale the BT rate by the live/BT fill ratio and say so).
PASS → spec a flagged paper add-on in `trading/hod_break_engine.py` (default OFF, paper only), weekend rehearsal,
Monday boot. FAIL → the add-on is closed on this population; report the CI and MDE, no parameter changes.

## Not in scope
Any other 14:00 rule, thresholds other than +0.5 R / 0.5 R / 1 R stop, any exit change (35 closed in 1,681).
