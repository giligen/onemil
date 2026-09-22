# PREREG — HOD-break PASS 2: trade-level filters, entry mechanics, overnight runner, slots. Cells 1,380–1,388

Written 2026-09-22 while pass 1 (`PREREG.md`, cells 1,359–1,379) runs; frozen before any pass-1 result is read.
Depends only on pass-1 ARTIFACTS (`paths.parquet`, `signals.parquet`, `b0_trades.csv`), never on its verdicts.
Baseline, cost, splits, fill physics, paired-ΔR bar, verification: exactly as `PREREG.md`. Each cell is ONE change
from B0; nothing stacked. Mechanism stated first, because the entry-feature scan already failed at chance.

| cell | rule | mechanism (why this is not "another selection filter") |
|---|---|---|
| 1,380 T1 | drop signals with `r_pct` (stop distance as % of price) < 1.5 % | the R-must-exceed-spread law: cost in R = 0.3 % / r_pct; below 1.5 % the round trip costs > 0.2 R. A COST filter, not a prediction |
| 1,381 T2 | same at 2.5 % | dose-response check for T1 |
| 1,382 M1 | symbol had ≥ 1 HOD-break signal in the prior 5 sessions (any outcome) | stocks in play stay in play: multi-day attention persistence (cross-day, not the same-day n_prior the model saw) |
| 1,383 M2 | symbol's prior-5-session HOD-break signals were net POSITIVE under B0 (known at signal time) | winner persistence at the symbol level |
| 1,384 E1 | retest entry: after the break, enter at the next open after the first 1-min bar whose LOW ≤ the break level within 15 min; no retest → no trade | the chase-guard finding cuts both ways (memory: passive entry can select adversely) — measure it here, with the unfilled counterfactual reported |
| 1,385 O1 | overnight runner: if the B0 trade is still open at 15:55 with close ≥ entry + 1 R AND the close is in the top 20 % of the day's range, HOLD and exit at the next session's 09:30 open (daily_bars open); else B0 | close-at-high continuation (overnight drift concentrates in winners); TEST-safe because the next open is the outcome, the decision uses day-t only |
| 1,386 O2 | O1 with exit at the next day's 10:00 bar open instead of 09:30 | avoids the auction; dose check |
| 1,387 S1 | slot rule 8 concurrent instead of 4 (same first-12-per-day cap) | frequency for the cadence bar; selection among simultaneous signals is random-equivalent, so more slots = more of the same edge |
| 1,388 S2 | daily cap 20 instead of 12, 4 concurrent | same |

Report-only: T1/T2 dose curve by r_pct decile; M1 by count of prior signals; E1's unfilled-counterfactual cohort
(the B0 outcome of signals that never retested) — this line decides E1's reading; O1's overnight gap distribution
incl. the worst 1 % (the tail the owner accepts only if it is not once a quarter); S1/S2 fills/week and weekly P10.

Pass bar: `PREREG.md` §pass bar. For T/M cells the D/W clause applies (kept ≥ 3 fills/wk on VAL, dropped cohort
< 0 both splits). For E1 the paired comparison is on the INTERSECTION (signals that retested) AND the full-book
comparison (retest book vs B0 book, unpaired, with the unfilled counterfactual shown). For O1/O2 the overnight
tail is reported at P1 and the worst night; weekly MDD clause applies. Programme count after this pass: 1,388.
Verification of any pass: the same three refuter lenses + independent rebuild.
