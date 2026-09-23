# PREREG — "Early in time": enter INSIDE the base under the high, before the HOD break. Cells 1,400–1,402

Owner 2026-09-23: everyone enters at the HOD break (the crowd's minute, worst fills); be in before them. This is a
NEW population, not an HOD-break variant: the entry is a long taken while the stock is basing just under its high of
day, with the stop under the base. The HOD break, if it comes, is what we are positioned for; bases that never
break are the cost. Frozen before any bar is walked.

## Universe (knowable at 09:30:00 ET)
`research/hod_pmh_causal/pm_candidates.csv` — the ORB wide seed from daily bars: gap ≥ 3 % vs prior close at the
open, open $3–50, prior-day volume ≥ 500K. Nothing after 09:30 decides membership. TEST (≥ 2026-06-01) excluded.
RTH 1-minute bars: `research/bf_zero/bars_sip.db` ∪ `research/hod_pmh_causal/bars_rth.db` (loader:
`build_pmh_causal.fetch_day_bars_dual`). Availability rail: ≥ 80 % of universe symbol-days have RTH bars, else VOID.

## Signal (evaluated at the close of each 1-min bar t, 10:00 ≤ t ≤ 14:00 ET; FIRST qualifying t per symbol-day)
* H_t = high of day through bar t; τ = the latest bar with high == H_t. **Age:** t − τ ≥ 20 (no new high for 20 min).
* B_t = min low of bars t−19..t (the base). **Depth:** (H_t − B_t) / H_t ≤ 6 %.
* **Pressing the high:** close_t ≥ B_t + 0.67 × (H_t − B_t) (top third of the base).
* **In play:** H_t ≥ 1.01 × the 09:30 bar open (an intraday high, not the opening print).
* **Cost gate (R-must-exceed-spread):** close_t − B_t ≥ 1.5 % of close_t (evaluated on close_t, causal).
* Entry = OPEN of bar t+1. Stop = B_t (touch → fill at B_t; a bar opening below B_t fills at its open). R = entry − B_t
  (if the entry opens at or below B_t the trade is a stop at the open).

## Cells (one exit each; always flat at the 15:55 bar open; stop checked before target inside a bar)
1,400 C1: target entry + 2 R · 1,401 C2: no target · 1,402 C3: breakeven lock (high ≥ entry + 1 R → stop = entry
from the next bar), no target.

## Cost
No measured NBBO exists for these minutes: base = 15 bps half-spread on BOTH legs + 2 bps per side, all exits,
gross reported beside net, cost-in-R median stated (≈ 0.23 R at the 1.5 % floor). **A cell that passes on this proxy
is re-scored on measured Alpaca NBBO at the entry minute before it is reported as a pass.**

## Exhibits first (reported before the cells)
1. Drift exhibit in the format of `research/hod_exit_lab/DRIFT.md` (MFE/MAE/minutes-to-MFE/give-back by split and
   time bucket; minute-since-entry excursion table) — directly comparable with the HOD population.
2. **The owner's question, DECOMP.md:** share of base entries whose stock later breaks H_t (outcome label, used
   descriptively only); mean R of base entries that later break vs never break; and, for the ones that break, the
   R of the base entry vs an HOD-break entry on the same name-day (next open after the first close > H_t, same
   stop B_t, same C1 exit) — "entering an R earlier" measured, with the cost of the bases that fail included.

## Placebos (TRAIN and VAL; the cell must beat both by ≥ +0.10 R on VAL)
D1: same name-day, seeded random minute in 10:00–14:00, stop = min low of the prior 20 bars, same 1.5 % floor
(≤ 5 redraws, else drop), C1 exit. D3: a random OTHER universe symbol-day of the same date, entered at the signal's
clock minute with its own prior-20-bar-low stop and floor, C1 exit.

## Splits, statistics, pass bar
TRAIN 2025 (halves H1/H2), VAL 2026-01..05. Per cell: n, trades/week, gross and net mean R, iid and day-clustered t,
win rate, ex-top-5 %, capped at +5 R, exit mix, halves, fills/week under first-12/day and 4 concurrent, bar-density
rule (≥ 80 % of minutes entry→15:55 present; conservative arm: a missing minute = stop), cadence block on VAL.
**Pass (all):** net ≥ +0.10 R on TRAIN and VAL; VAL day-clustered t ≥ 2; both halves > 0; ex-top-5 % > 0 on both
splits; ≥ 3 fills/week on VAL at the slot rule; D1 and D3 beaten by ≥ +0.10 R on VAL; cadence C3/C4 on VAL.
**Verification of any pass:** universe + signal causality trace on Opus FIRST, then an independent rebuild from this
prose (≥ 98 % trade agreement), then the measured-NBBO re-score. Programme count 1,402.

## Not allowed
Changing any threshold after seeing results; stacking cells; any TEST row; reading HOD-break signals to define the
universe or the entry (the PMH lesson).
