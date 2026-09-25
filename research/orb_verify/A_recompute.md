# Lens A — Independent recomputation of the ORB OOS-edge claim

Run 2026-09-25. Own code only (no import of `research/hod_consol/adversarial_read`
or `research/thermo/thermo.py`). CSVs read via `trading/orb_csv.read_orb_csv`
(`keep_default_na=False`, ticker `NA` preserved). R = `_sized_pnl` / 375
(risk_per_trade_usd confirmed in `orb.yaml.template:271`). Script:
`/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/recompute.py`

## Verdict: headline numbers REPRODUCE. Significance is thin and concentrated in one half-year; tail concentration is real and worse than the claim's own wording suggests.

## 1. Claim vs. independent recompute

| Metric (claim) | Claim value | My recompute | Match? |
|---|---|---|---|
| n fills | 554 | **554** | exact |
| mean R/fill | +0.077 | **+0.0769** | match |
| day-clustered t | 2.71 | **2.714** | match |
| total $ | $15,969 | **$15,969.10** | match |
| ex-top-5% mean R | -0.013 | **-0.0130** | match |
| top-5% (28 trades) "carry the whole P&L" | — | top 28 fills sum R=49.45 vs total R=42.58 → **116% of total P&L**, rest (526 fills) net **-6.86 R** | confirmed, slightly understated by claim's wording |
| half-years positive | 6 of 7, 2024H2 exception -0.007 | **6 of 7**, 2024H2 = **-0.0066** | match |

iid t for the full 554 = 3.016 (claim correctly reports the more conservative clustered 2.71, not the inflated iid figure — good practice).

## 2. Data-integrity checks (all clean)
- **Duplicates**: 0 duplicate (symbol,date) rows within any of the 3 files.
- **Cross-file overlap**: 0 overlapping (symbol,date) pairs between 2023file/2024file/thermofile.
- **Missing `_sized_pnl` on entered==1**: 0 in all three files (106/59/473 entered rows all populated).
- **Date bounds**: 2023file 2023-01-12→2024-06-28; 2024file 2024-07-05→2024-12-31; thermofile 2025-01-02→2026-09-23. All within the claim's stated windows.
- **Fill-count reconciliation**: 106 (2023file) + 59 (2024file) + 389 (thermofile rows with date≥2025-07-01) = **554**, matches claim exactly. Thermofile in-sample half (2025-01-01..06-30) excluded = 84 rows, correctly dropped.

## 3. Half-year breakdown (own computation, day-clustered t via statsmodels OLS cluster-robust)

| Half-year | n | mean R | total $ | t (iid) | t (day-clustered) | ex-top5% mean R |
|---|---|---|---|---|---|---|
| 2023H1 | 34 | +0.117 | $1,495 | 1.11 | 1.14 | +0.048 |
| 2023H2 | 28 | +0.133 | $1,399 | 1.05 | 1.00 | +0.051 |
| 2024H1 | 44 | +0.039 | $648 | 0.45 | 0.49 | -0.052 |
| 2024H2 | 59 | -0.007 | -$147 | -0.09 | -0.09 | -0.062 |
| 2025H2 | 127 | +0.048 | $2,298 | 1.06 | 0.89 | -0.044 |
| **2026H1** | 174 | **+0.136** | **$8,903** | **2.54** | **2.37** | +0.009 |
| 2026H2 | 88 | +0.042 | $1,373 | 0.81 | 0.54 | -0.037 |
| *2025H1 (excluded, in-sample)* | *84* | *+0.193* | *$6,079* | *2.86* | *2.79* | *+0.091* |

## 4. Adversarial findings

**(a) The pooled significance is carried almost entirely by one half-year, 2026H1.**
2026H1 alone contributes $8,903 of the $15,969 total (55.7%) at mean R=+0.136 — roughly
1.8x the pooled mean — and is the *only* individual half-year whose clustered t clears 2.0
(2.37). Every other OOS half-year has |t_clustered| ≤ 1.14. Dropping 2026H1 entirely leaves
380 fills, mean R=+0.0496, $7,067, **clustered t = 1.54** (not conventionally significant).
So: "positive in 6 of 7 half-years" is true but doesn't communicate that the claimed t=2.71
depends heavily on one half-year — remove it and t falls from 2.71 to 1.54, though the sign
stays positive. This is a claim about magnitude of confidence, not a reversal.

**(b) Tail concentration is a genuine lottery-ticket pattern (CLAUDE.md rule #5).**
Ex-top-5% mean R = -0.013 means the bottom 95% (526 of 554 fills) are net *unprofitable*
in aggregate (sum ≈ -6.86 R). The claim's own phrase "top 5% of fills carry the whole P&L"
is accurate and if anything conservative — the top 28 fills carry **116%** of the net P&L,
i.e. more than the entire realized edge. Under any per-trade cap this book is flat-to-negative.

**(c) The excluded in-sample half (2025H1) is hotter than any OOS half.**
2025H1 mean R=+0.193 / t=2.79 exceeds even 2026H1 (+0.136 / t=2.37), consistent with
ordinary in-sample inflation from the z-param fit — expected, not a red flag by itself, but
it means the strongest OOS half (2026H1) is the one closest in magnitude to the fitted
half and warrants a causality/regime check (Lens B/C territory) before leaning on it.

**(d) Minor undocumented completeness gap at the 2023/2024 file boundary.**
2023file's last row is 2024-06-28, 2024file's first is 2024-07-05. Business days
2024-07-01, 07-02, 07-03 (07-04 is the July 4 holiday) have **zero rows in either file**
— no candidates, no entered, no skipped-and-logged row. Given the ~500 trading-day span
this is unlikely to move the headline numbers, but it's an undocumented 3-day hole at
exactly the seam between the two source projects and should be confirmed as "no signal
fired" rather than "not fetched" before being called complete.

## 5. What did NOT break
No duplicate/overlap/missing-value defects, no ticker-NA corruption (read_orb_csv used
throughout), fill-count and dollar totals reconcile exactly, and the claim used the
correct (clustered, not iid) t-stat. This is a clean recomputation — the coding is not
where the risk is. The risk is entirely in (a) concentration in 2026H1 and (b) tail
dependence in (b) above, both of which the claim's headline "+0.08 R, t=2.71" presents
as a stable pooled result rather than "positive most periods, significant in one, and
carried by the top 5% of trades."
