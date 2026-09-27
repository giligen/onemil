# Refuter 2 — cells 1,591–1,598 (v2, tick-priced): STATISTICS AND RISK lens

Recomputed from `cell_1591_cycles.csv` / `cell_1591_monthly.csv` (script `review/refuter2_1591_stats.py`, table
`review/refuter2_1591_cellstats.csv`) plus a VOID-selection probe (`review/refuter2_1591_voidprobe.py`). All builder
numbers reproduce exactly (means, Sharpe, green share, worst month, max DD, slip rails).

## Verdict: REFUTED — the reported VAL risk profile of the selected cell is a VOID artifact, and every cell breaches the VOID rail

1. **VOID share misreported as 0.0 %.** `run_cell` never writes VOID rows, so `cell_stats` counts `void_reason != ''`
   over traded rows only. The run log (`full_run_1591.log` line 634) says cell 1597 = **41 cycles, 92 VOID of 133
   Mondays (69 %)**; every cell is 58–69 % VOID (1591/1593: 84 of 133; 1595: 91; gated cells 38–45 of 63 un-skipped).
   Amendment 2 rail: > 10 % VOID → the cell is VOID. **Every one of the 8 cells is VOID**; "FAIL at 3.17 %/mo" is not a
   measurement.
2. **The VOID set removed every VAL loser.** Settling every Monday's 30-delta $10 spread at intrinsic (rebuild's
   strikes for all Mondays, SPY close at expiry, credit imputed at the builder's median $1.73 — a probe, not a claim):
   VAL traded Mondays 18/18 winners (imputed mean +$173); VAL VOID Mondays 41, 4 losers — 2026-01-26, 02-02, 02-09,
   02-23 entries (expiring March/April 2026), −$89, −$827, −$827, −$827. So "win rate 100 %, worst month $0, max DD
   $0, Sharpe 4.25" is the v1 April-2025 defect again, in VAL. Same bias on TRAIN in the other direction (traded mean
   +$16 vs VOID mean +$92 per cycle).
3. **Selection is uninformative.** Spearman TRAIN→VAL monthly Sharpe across the 8 cells = −0.17 (mean return −0.26).
   The TRAIN-best Sharpe cell (1592, 0.63) is VAL-worst (rank 8); the selected 1597 was TRAIN rank 5 at Sharpe −0.04
   and happens to be VAL rank 1. The eligibility rule (≥ 12 cycles, ≥ 55 % green) admits only the two gate-none
   M=B cells, so "selection" = pick the less-negative of two. The rebuild selects 1594 instead.
4. **Neighbour check** half-empty: Δ-neighbour 1593 VAL +1.78 %/mo (same sign; TRAIN −0.45 %, same as 1597's TRAIN
   sign); no W-neighbour exists in v2 (W fixed $10). M-neighbour 1595 VAL +0.87 %, TRAIN −0.69 %.
5. **Month concentration / tails (as-traded):** VAL top month 18 % of P&L, top-2 36 %, top-5 % cycles 6.6 % — flat,
   because it has no losers (see 2). TRAIN ex-top-5 % = −$14 (≈ 0). Spike months (1597 TRAIN): 2024-08 −$263,
   2025-03 −$1,586, 2025-04 −$806 → the whole TRAIN max DD $2,392 (0.37 B) with ≤ 5 spreads open.
6. **Budget assertion** holds in every cell (max concurrent worst case: 1597 $4,166; 1593/1594 $5,329; B $6,500), but
   trivially: 1 contract per cycle and 69 % VOID → average deployed worst case ≈ $1.5 K ≈ 23 % of B. "Return on B" is a
   quarter-deployed book. A VOID-free ladder at 1 contract/week reaches 8 open = $6,616 > B, so the budget WOULD bind.
7. **Monthly-series truncation:** exits after the split's last month are dropped from both splits (1597: +$541.82
   July-2025 exits of June-2025 entries belong to no split; VAL drops +$177.94 Sep-2026). TRAIN mean −0.09 % → +0.40 %
   with them. Does not change the selection.
8. **Comparison lines.** Naked put is sized at the SAME contract count (not "sized to the same premium", and ~12×
   the margin): naked VAL +$14,865 vs spread +$3,063; the Mar-3-2025 cycle naked −$4,039 vs spread −$806. SPY
   buy-and-hold on B: VAL +22.7 % ($1,477), DD −9.1 %, ret/DD 2.49; full window 2024-02..2026-09 +54.6 % ($3,552),
   DD −19.0 %, ret/DD 2.88. Cell 1597 as traded, full window: $3,507 total / $2,392 DD = 1.47 → **loses to SPY per
   unit of drawdown**; VAL ratio is undefined (DD 0 is the artifact). VOID-free probe: full window 2.6 vs SPY 2.88.
9. **Honest monthly $ at B = $6,500 (as traded):** TRAIN −$6/mo ($0.03) … −$22/mo ($0.10); VAL +$206 … +$189;
   pooled 31 months incl. truncated exits +$113 … +$95/mo. VOID-free probe: TRAIN +$271, VAL +$496 but with a
   −$1,743 month (2026-03), pooled +$381/mo, worst month −$2,389 (2025-03), max DD $4,524 (0.70 B).
10. **Scaling to $1,000/month:** as-traded pooled $113/mo → ×8.8 → B ≈ $57 K (equity ≈ $570 K at 10 %), worst month
   ≈ −$14 K, max DD ≈ −$21 K, construction worst month −$57 K. VOID-free probe $381/mo → ×2.6 → B ≈ $17 K (equity
   ≈ $170 K), worst month ≈ −$6.2 K, max DD ≈ −$11.9 K. Neither is reachable at the $65 K account.

## What must be fixed before any number is relayed
Write VOID rows; report VOID share against 133 Mondays; resolve why 58–69 % of Mondays have no 10:00–10:05 print on
30/20-delta SPY puts (likely strike/expiry selection pointing at illiquid strikes or a fetch defect — the rebuild saw
36 % VOID); re-run with the rail enforced. Until then the v2 status is **VOID (not measured)**, not FAIL-at-3.17 %.
