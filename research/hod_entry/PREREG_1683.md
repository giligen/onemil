# PREREG — cell 1,683: the most profitable exit strategy across all 58 rules, tail included, shown week by week on the sealed quarter (FROZEN 2026-09-30 21:30 UTC)

Owner 21:25 UTC: "Get me the most optimized strat across all rules that is profitable, ignore the long tail. Show me
week by week of the past quarter."

## Protocol (the optimization is in-sample by design; the quarter is the honest read)
1. Pool: the floored HOD book 2025-07..2026-05 (1,663 join, r_pct ≥ 1.5 %, n 5,506) and the 58 exit rules H1–H58
   in `research/hod_entry/1681_hypotheses.py` (per-fill R already in `1681_per_fill.csv` for both halves; H36/H57/H58
   are joints). Objective: mean net R per fill (paired ΔR vs the base), NO tail gate, NO consistency gate.
2. Search: (a) best single rule on the pooled book; (b) greedy forward selection of ≤ 3 compatible rules (different
   triggers; precedence = selection order) maximising pooled mean ΔR; (c) a one-step threshold grid (±1 step of each
   selected rule's main threshold, e.g. 30→20/45 min, +1 R→+0.75/+1.25 R, 50 %→25/75 %) on the pooled book — the
   tuned version; (d) the same search under the live cap (first 12 fills per day). Report every candidate's pooled
   mean ΔR, t, fills/week and — reported, not used — ex-top-5 % ΔR.
3. Sealed read: apply the base rule, the best single, the best joint and the tuned joint UNCHANGED to the forward
   population `research/hod_entry/forward_2026q3/causal_arming_causal.csv` (2,872 fills, 2026-06-01..09-04; bars in
   the store; the per-fill path cache must be built for these fills the same way `1681_paths.parquet` was), floored
   r_pct ≥ 1.5 %, uncapped and under the 12-per-day cap.
4. Week-by-week table (14 ISO weeks) for base vs optimized: fills, sum R, mean R, $ at $150 risk per fill, green week
   flag, worst day; then totals: mean R per fill, iid and day-clustered t, MDE at n, ex-top-5 % (reported), green-week
   share, weekly P10, max drawdown, $ for the quarter at $150 risk and at the live cap.
5. Also the cost line: every exit rule executes at the next bar's open − 6 bps, partials in original-R units.

## What passes
Nothing "passes" a research bar here by construction (the selection is in-sample). The deliverable is the sealed
quarter's table. Decision rule (act-as-owner, fixed now): if the optimized strategy's sealed-quarter mean ΔR ≥ +0.05 R
with day-clustered t ≥ 2.0 and ≥ 3 fills/week under the cap, it goes to the owner as an exploration-tier PAPER candidate
(positive point estimate, mechanism, bounded downside, resolves in a quarter) with the engineering it needs listed;
otherwise the table stands as the answer and the in-sample optimum is recorded as such.

## Multiplicity
The search touches ~58 + 3 × 57 + grids ≈ 400 in-sample reads; the sealed quarter is read 4 × 2 times. Programme
count on the HOD line: > 5,900.

## Not allowed
Any use of the forward months in the search; changing the objective after seeing the quarter; dropping the R floor.

## Output
`research/hod_entry/RESULT_1683.md` (the week table FIRST, base vs optimized, capped and uncapped), `1683_weeks.csv`,
`1683_search.csv`, `1683_forward_per_fill.csv`, `1683_optimize.py`, `1683_optimize.log`. The agent returns ≤ 200 words.
