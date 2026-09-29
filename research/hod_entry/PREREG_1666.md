# PREREG — cell 1,666: independent rebuild of cell 1,488 (the no-withdrawal pyramid) at the measured cost (FROZEN 2026-09-29 17:56 UTC)

Why: cell 1,662's re-read of the programme makes 1,488 the ONLY HOD cell that clears the mechanical bar (+0.34 / +0.35 R,
t 2.66 / 2.77, n 181 / 232, ex-top-5 % > 0 on both halves, 7–11 fills/week) — but its raw R was reconstructed through a
target = +2 R / stop = −1 R convention and joined for the stop distance. The 9/26 build (RESULT_1487) had the pyramid at
+0.25 R own mean and FAILED its paired bar. Nothing reaches the owner from this line before an independent rebuild.

## Rule (verbatim from PREREG_1487 §1,488)
Base entry at 1/3 risk; at the end of minute fill_min + 15, if no dip below level − $0.01 has occurred and the position is
open, add 2/3 at the ask of minute fill_min + 16, move the whole position's stop to level − $0.01; target 2 R from the
ORIGINAL fill for the whole position; book in original-R units, paired vs the base (same fills), stop-limit standard.

## Rebuild protocol
* The builder has NOT read any 1487 / 1488 / 1662 script or CSV; it builds from this prose alone.
* Population: the 1,438 fill book — `fills_1658.csv` joined to `causal_arming_causal.csv` (status == fill) on
  (day, symbol): fill, stop, level, fill_min, split. Primary: stop ≥ 1.5 % (r_pct); unfloored reported beside.
* Path after the fill: 1-min bars from `research/bf_zero/bars_sip.db` (bars(symbol, day, t, o, h, l, c, v)). The "ask of
  minute fill_min + 16" = the OPEN of that bar plus the 7-bps entry cost. Withdrawal = any bar low ≤ level − $0.01 in
  bars fill_min + 1 … fill_min + 15. Exits checked bar by bar, stop before target inside a bar; EOD = the close of the
  15:55 ET bar. If the position stops or targets before minute 15 there is no add (book = base at 1/3 size).
* Cost (measured, 9/29): 7 bps on each entry leg, 6 bps on stop exits, 0 on target fills, 11 bps on the EOD exit.
* Units: original-R = full-size shares × (fill − original stop). The paired base = the same fill at 1× size under the
  standard rule (original stop, 2 R target, same EOD), same cost.

## Reads (both halves, always)
Own mean, iid t, day-clustered t, ex-top-5 %, fills/week, MDE beside every t; paired ΔR vs the base, its t and its
ex-top-5 %; the share of fills that add; the dollar exposure at the add (3× the base) and the worst day in R.

## Pass bar
PREREG_1487's 1,488 bar (paired ΔR ≥ +0.05 on both halves, VAL t ≥ 2.5, VAL book mean ≥ +0.10) AND PREREG_1662's
synthesis bar (net ≥ +0.05 R, t ≥ 2.5 both halves, ex-top-5 % > 0 both, ≥ 3 fills/week). Then a third party compares
this build with 1,662's per-fill file row by row (withdrawal flag, sign of pyr_R): ≥ 95 % agreement before any number
reaches the owner. Programme count on the HOD line: ≥ 1,700.

## Not allowed
Reading the first implementation; tuning the 15-minute window, the add size, the stop move or the target; any read
conditioned on the exit type; pooled-only numbers.

## Output
`research/hod_entry/RESULT_1666.md`, `1666_per_fill.csv` (fill_id, day, symbol, split, r_pct, withdrew, added, base_R,
pyr_R, delta_R, exit_type), `1666_rebuild.py`. The agent returns ≤ 150 words.
