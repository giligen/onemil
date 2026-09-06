# ORB V1 — era-consistent feature vetoes through the honest pipeline (pre-registered 2026-09-06 21:10 UTC)

**Question.** The raw ORB breakout has no edge (entered candidates, fixed-exit proxy: 2025 −0.18R, 2026 −0.04R); the pipeline's selection is the whole edge — the same structure as BF. On BF, three hard era-consistent vetoes on raw features beat the fitted layers. Do ORB's era-consistent worst buckets, applied as hard no-refill vetoes AFTER the pipeline's ranking, improve the honest B+ book?

**Candidates** (bottom/top quintile of the entered raw candidates, worst in BOTH 2025 and 2026; thresholds fixed from the scan, not tuned):
| id | veto (drop pick when …) | 2025 R | 2026 R |
|---|---|---|---|
| V1a | `range_size_pct <= 2.221` (smallest opening ranges) | −0.43 | −0.30 |
| V1b | `avg_daily_range_pct_20d <= 5.01` (quietest names) | −0.31 | −0.28 |
| V1c | `spy_3d_range_pct >= 1.484` (choppiest tape) | −0.27 | −0.31 |
| V1d | `range_avg_bar_range_pct <= 0.943` (smallest range bars) | −0.44 | −0.26 |
| V1e | `return_volatility_20d <= 3.798` (lowest 20d vol) | −0.20 | −0.25 |
| V1all | a + b + c (the three with the largest both-year magnitude; d overlaps a, e overlaps b) | | |

**Mechanics.** `ORB_EXP_FEAT_VETO` (trading/orb_experimental_rules.py, post-selection, slot stays empty — the PDR/catalyst form; refill was refuted). Honest B+ pipeline at the live stage sizing from orb.yaml ($10K / 3 / $375), features `analysis_results/orb_features_20260905_1304.csv` (entered-inclusive), exits from the candidate dump with identical exit physics (`ORB_BT_RESIM_CACHE`). Outputs to `research/orb_veto_study/` only — never the nightly `analysis_results/orb_bplus_book.csv` / `orb_monthly_static_lock.csv`.

**Pass rule (pre-committed, per veto vs baseline).** KEEP only if ALL hold:
1. total P&L ≥ baseline total (the veto must not cost money over 21 months);
2. MDD not worse than baseline;
3. each era (2025H1, 2025H2, 2026) ≥ baseline era − $100 (no era made materially worse; $100 ≈ noise at $10K sizing);
4. red months ≤ baseline red months.
Anything else = REJECT, and the candidate is closed (no threshold search). If V1all passes but a component fails, ship nothing — components must stand alone (the per-rule audit pitfall).

**Outcome goes to REPORT.md; a KEEP becomes a proposal for the owner (a new veto = a live rule change), not a ship.**
