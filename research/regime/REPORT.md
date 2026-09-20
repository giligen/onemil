# REPORT: ORB per-regime multipliers, rule-regime vs HMM-regime

Book: `analysis_results/orb_bplus_book.csv`, entered==1 filled rows (165 of 218).
PnL: `_sized_pnl`. R denom: `_rp_position` is CONSTANT $3,333.3337 across every
trade (risk-parity sizing already normalizes risk) -> 1R = $3,333.3337.
TRAIN 2025 n=84, VAL (2026-01..05) n=40, TEST (>=2026-06-01) n=41 SEALED, untouched.

## TRAIN 2025 per-state (n, net R, t_iid, t_clustered, H1, H2 -> fitted mult)
Rule regime:
- A : n=70 netR=1.10 t_iid=2.01 t_clust=1.69 H1=0.21 H2=0.90 -> **1.5**
- B : n=6  netR=0.50 t_iid=2.57 t_clust=1.70 H1=0.50 H2=0.00 -> 1.0 (mixed, pooled+)
- C1: n=4  netR=0.14 t_iid=1.30 t_clust=1.30 H1=0.14 H2=0.00 -> 1.0 (mixed, pooled+)
- C2: n=4  netR=0.26 t_iid=1.26 t_clust=1.63 H1=0.28 H2=-0.02 -> 1.0 (mixed, pooled+)

HMM regime:
- hmm0: n=74 netR=1.36 t_iid=2.32 t_clust=2.13 H1=0.48 H2=0.88 -> **1.5**
- hmm1: n=7  netR=0.51 t_iid=2.72 t_clust=2.05 H1=0.51 H2=0.00 -> 1.0 (mixed, pooled+)
- hmm2: n=3  netR=0.13 t_iid=0.98 t_clust=0.98 H1=0.13 H2=0.00 -> 1.0 (mixed, pooled+)

Both systems land on the SAME shape: one dominant calm/bull state (A / hmm0,
~85-90% of TRAIN n) clears both-halves-positive + t>=1.5; every other state
is thin (n<=7), defaults to 1.0. No state hit the 0 or 0.5 branch.

## VAL 2026-01..05: per-regime vs flat-1.0
Rule (A=1.5, rest=1.0): flat $6,386.42 / MDD -$489.17 -> per-regime $6,930.04 /
MDD **-$733.75** (worse). netR 1.92 -> 2.08. States exercised: A n=25 (mult
1.5), C1 n=10 (1.0), C2 n=5 (1.0), B n=0. **PASS BAR: FAIL** ($ up, but MDD
worse).

HMM (hmm0=1.5, rest=1.0): flat $6,386.42 / MDD -$489.17 -> per-regime
$9,424.48 / MDD **-$608.29** (worse). netR 1.92 -> 2.83. States exercised:
hmm0 n=33 (1.5), hmm1 n=7 (1.0). hmm2: **0 VAL days** (called out per PREREG).
**PASS BAR: FAIL** (same reason, MDD worse -- smaller miss than rule).

Both fail the MDD leg by construction: the lever is 1.5x on the state holding
~80% of VAL trades, scaling winners AND losers together -- $ improves but
drawdown widens. Green-week share unchanged (41.18%, 17 wks) both systems --
the multiplier resizes trades, not timing.

## cadence_bar.py --split VAL (C1-C5), per-regime books
| | C1 gap | C2 bleed | C3 reds | C4 green | C5 fills/wk |
|---|---|---|---|---|---|
| RULE | fail | fail (0% cyc>0) | pass (P10 -0.08R, MDD 0.13R) | pass (100%/null 49%) | fail (1.82/wk) |
| HMM  | fail | fail (0% cyc>0) | pass (P10 -0.06R, MDD 0.15R) | pass (100%/null 49%) | fail (1.82/wk) |

C1/C2/C5 fail on both: a 40-trade/17-week slice has no cycle boundaries for
this book, not a regime-specific defect.

## Verdict
**NO-SHIP for both regime systems**, on the pre-committed bar (MDD leg fails
on both). Rule-regime and HMM-regime independently discover the identical
underlying pattern (one big calm state, everything else too thin to size on)
-- the agreement is reassuring about the pipeline, not about the edge. HMM
comes closer (smaller MDD delta, bigger $ lift) but still fails.

## Caveat
n=40 VAL trades (25-33 in ONE state) isn't powered to separate "regime helps"
from "the dominant state was a good stretch"; MDD fails by construction
(scaling the mode-state's winners up), so this test can't isolate a real
regime edge from plain leverage-on-the-mode-state. TEST (41, sealed) untouched.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PKSsd3LjBkXYzPECLNYR5W
