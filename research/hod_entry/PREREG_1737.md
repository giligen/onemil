# PREREG 1,737 — be the bid: RETEST bid at the level and COIL bid under it (owner 2026-11-12 "do all tonight")

FROZEN 2026-11-12 19:35 UTC before any number. Programme count on the HOD line: 1,736 → 1,737. One cell, two
mechanisms, no sweep. Rules identical to PREREG_1736.md (no git / service / orders; read-only; write only
`research/hod_entry/1737_*` and `RESULT_1737.md`; `walker.wait_for_db_window()` before the first bars_sip.db query;
`walker.fetch_day_bars`; read-only DB URI; `nice -n 19`; never Read a file > 300 lines in full; budget ≤ 60 calls).
This task IS the owner's request; do not pivot.

## Prior (disclose)
The HOD long pays the entry chase (8 bps live) and the spread, then 41 % of fills stop out with slip. Cells 1,548/1,549
(a sweep entry on predicted extenders) and 1,660 (resting-limit height sweep) kept a BUY-THE-BREAK structure; cell
1,489 (a model on the retest population) was chance (VAL AUC 0.515). Nobody has scored the plain mechanical rule
"rest a bid, let the market come to us" at the HOD level. The known risk: a dip-buy limit can select broken setups
(`project_passive_entry_adverse_selection`) — this cell measures that directly through the unfilled cohort.

## Population and inputs
Arms, bars, prior close, half_spread rule, reads (TRAIN = 2025H2, VAL = 2026-01..05): exactly as PREREG_1736.md.
Stop geometry for BOTH mechanisms (the engine's own): consol_low = min(l) over the 5 bars j−4..j (j = the arm bar at
entry_m); stop = consol_low; if (level − stop)/level < the live floor the stop is raised to level × (1 − floor).
The floor = `hod_break.min_r_pct` (or the stop-floor key) in `config.yaml` read-only — state the value used (expected
1.5 %). R is measured from the FILL price, not the level.

## M1 — retest bid
1. Confirmation bar C = the first bar after entry_m with CLOSE ≥ level (the break held one close). None by 15:30 →
   no order.
2. From C+1, a bid rests at `level`. It FILLS at `level` on the first bar whose LOW < level (strictly; the touch of
   the level itself is not a fill). It is cancelled, unfilled, on the first bar with a close ≥ level × 1.02 (ran
   away) or at 15:30.
3. Filled: stop as above (fills at the stop, gap-through at the open, + 13 bp slip); target = fill + 2 R (fills if
   h ≥ target); both in one bar = stop; else the 15:55 open. Entry cost 0 half-spread + 2 bp (we ARE the bid);
   target exit 0 half-spread + 2 bp (resting sell); stop / 15:55 exits pay half_spread + the slip above / + 2 bp.
## M2 — coil bid (sell INTO the break)
1. At the arm bar j the coil is confirmed (that is what the arm means). A bid rests at level × (1 − 0.004) from
   j+1. It FILLS on the first bar whose low < that price BEFORE any bar's high ≥ level (the break cancels the
   unfilled bid); also cancelled at 15:30.
2. Filled: a resting sell at level × 1.003 (fills if a later bar's h ≥ it, 0 half-spread + 2 bp); stop as above;
   else the 15:55 open. Both-in-one-bar = stop.
Report for both: fill share, the B0 net R of the UNFILLED arms vs the FILLED arms (from `b0_trades.csv` where the
pair exists) — the adverse-selection direction, descriptive; fills/week.

## Scoring (M1, M2 × TRAIN, VAL)
As PREREG_1736: n, mean net R, day-t, iid t, ex-top-5 %, winner-capped, stop share, green-week share, weekly P10,
max drawdown, R as % of price, fills/week. Placebo (300 draws): the same bid geometry (distance under a reference
price, same R/exit rules) resting from a random minute AFTER the arm's own window (after C+1 for M1, after the break
bar for M2) on the same (day, symbol) — placebo minutes after the signal window — percentile of the observed mean.

## Pass bar (pre-committed)
A mechanism PASSES only if: mean net R ≥ +0.10 with day-t ≥ 2.0 on TRAIN AND VAL; ex-top-5 % ≥ 0 on both; placebo
≥ 95 on VAL; ≥ 3 fills/week on VAL; green-week share ≥ 55 %; AND the filled cohort's own result is not explained by
the unfilled cohort being the winners (if the unfilled arms' B0 mean exceeds the filled arms' M-result by > 0.3 R,
flag "adverse selection" beside the pass). A pass = a paper instrument spec, never a live change.

## Output
`research/hod_entry/1737_score.py`, `1737_orders.csv` (one row per arm × mechanism: day, symbol, entry_m, level,
split, c_m, bid_px, filled, fill_m, fill_px, stop, R, r_pct, exit_px, why, net_R, half_spread, matched flag,
b0_net_R), `1737_summary.csv`, `RESULT_1737.md` ≤ 350 words (tables, PASS/FAIL, caveats: fill-at-touch
conservatism, queue position ignored, spread fallback share, floor value). Return ≤ 120 words: M1/M2 fill share,
mean R / day-t / n per read, PASS/FAIL.

## Independent rebuild (second agent, has not read 1737_score.py)
From this prose: recompute M1 fills, fill price, stop, net R for every arm on TRAIN and VAL; compare to
`1737_orders.csv` on (day, symbol, entry_m): fill flags identical, prices to 1e-6, net R to 0.001; `REBUILD_1737.md`.

## Amendment 1 — 2026-11-12 20:40 UTC, POST-HOC after cell 1,736 (disclosed; secondary read, never a pass criterion)
Cell 1,736 found the failed-touch SHORT loses −0.72/−0.78 R with a 77–81 % stop share, and its placebo short loses
≈ −0.6 R: after a "failed" retest of the HOD these names re-take the touch high four times out of five. The MIRROR
is therefore a candidate and is read here as **M3 — buy the failed touch**: the 1,736 event (touch T, failure bar F
from `1736_events.csv`), long at the open of F+1 (pay half_spread + 2 bp), stop = the 1,736 short's target level
(entry − 2 R_short, fills at the stop + 13 bp), target = the 1,736 short's stop (max(h over T..F) × 1.002, a
resting sell, 0 half-spread + 2 bp), else the 15:55 open. Score exactly as M1/M2 INCLUDING the placebo with the same
geometry started at a random minute after F+1 — the win rate is geometry (1 R target / 2 R stop), so only the
paired difference to the placebo mean and the day-t of the OWN mean count. Report beside M1/M2; a pass on M3 is
reported as "post-hoc, needs its own PREREG on a different regime", never as a pass of this cell.
