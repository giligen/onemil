# RESULT 1,700r -- residual (beta-adjusted) momentum (PREREG_1700r.md, frozen)

REF reproduces: CAGR 27.18% / max DD -44.5% / end $507,823 (target 27.18% / -44.5% / $507,823). Engine = 1700p daily engine copied (ONLY the ranking score changes); sim 2017-02-06..2026-09-28, $50K, band cost.
Sanity 2024-12-31 (SPY-only beta, W252/W504): NVDA 2.65/2.32, KO 0.05/0.23; SPY own beta 1.0000, SPY residual score nan (0/0 noise expected).
Causality: scores for signal date 2022-06-03 recomputed with all later data deleted -> identical ranks and scores (all 8 score matrices): True.

| cell | CAGR | max DD | ratio | end $ | Sharpe | beta | alpha/yr (t) | corr REF | overlap | halves r/REF | ep1..5 | improves |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REF | 27.2% | -44.5% | 0.61 | 507,823 | 0.85 | 1.29 | +10.2% (1.1) | 1.00 | 100% | 1.00/1.00 | -45% -37% -34% -29% -29% | ref |
| SPY_12-1_N20_W252 | 24.5% | -44.1% | 0.56 | 414,578 | 0.93 | 0.92 | +11.4% (1.6) | 0.93 | 76% | 1.22/1.06 | -44% -27% -11% -20% -18% | - |
| SPY_12-1_N20_W504 | 23.6% | -39.6% | 0.60 | 385,979 | 0.88 | 0.99 | +9.9% (1.3) | 0.95 | 76% | 1.34/0.92 | -40% -30% -14% -21% -20% | - |
| SPY_12-1_N30_W252 | 20.6% | -46.2% | 0.45 | 303,676 | 0.83 | 0.97 | +7.1% (1.1) | 0.95 | 61% | 1.04/0.91 | -46% -30% -13% -19% -16% | - |
| SPY_12-1_N30_W504 | 22.3% | -39.8% | 0.56 | 348,630 | 0.86 | 1.05 | +7.7% (1.1) | 0.96 | 60% | 1.21/0.85 | -40% -32% -17% -19% -19% | - |
| SPY_6-1_N20_W252 | 18.9% | -45.0% | 0.42 | 265,258 | 0.77 | 0.87 | +7.4% (1.0) | 0.86 | 36% | 1.18/0.47 | -45% -22% -26% -13% -21% | - |
| SPY_6-1_N20_W504 | 18.9% | -36.2% | 0.52 | 265,373 | 0.76 | 0.94 | +6.4% (0.9) | 0.88 | 36% | 1.57/0.33 | -36% -25% -28% -15% -26% | - |
| SPY_6-1_N30_W252 | 15.6% | -50.2% | 0.31 | 202,023 | 0.68 | 0.96 | +3.0% (0.4) | 0.89 | 32% | 1.37/0.20 | -45% -29% -31% -14% -20% | - |
| SPY_6-1_N30_W504 | 17.5% | -42.2% | 0.41 | 236,408 | 0.74 | 0.99 | +4.2% (0.6) | 0.90 | 32% | 1.75/0.26 | -39% -27% -32% -14% -24% | - |
| 3F_12-1_N20_W252 | 19.8% | -38.5% | 0.51 | 284,860 | 0.80 | 0.84 | +8.5% (1.2) | 0.89 | 67% | 1.50/0.75 | -38% -21% -15% -17% -12% | - |
| 3F_12-1_N20_W504 | 23.0% | -33.1% | 0.69 | 367,663 | 0.90 | 0.85 | +11.0% (1.5) | 0.90 | 67% | 1.56/0.88 | -33% -25% -17% -17% -14% | - |
| 3F_12-1_N30_W252 | 17.5% | -45.5% | 0.39 | 237,101 | 0.75 | 0.96 | +4.6% (0.7) | 0.93 | 56% | 1.12/0.58 | -45% -26% -19% -17% -12% | - |
| 3F_12-1_N30_W504 | 21.7% | -39.4% | 0.55 | 332,840 | 0.87 | 1.00 | +7.7% (1.2) | 0.94 | 56% | 1.32/0.79 | -39% -30% -18% -18% -13% | - |
| 3F_6-1_N20_W252 | 9.3% | -52.2% | 0.18 | 117,695 | 0.48 | 0.78 | -0.2% (-0.0) | 0.82 | 34% | 0.72/0.10 | -48% -24% -19% -7% -6% | - |
| 3F_6-1_N20_W504 | 12.1% | -43.3% | 0.28 | 150,017 | 0.58 | 0.78 | +2.2% (0.3) | 0.83 | 34% | 1.06/0.11 | -38% -23% -20% -9% -9% | - |
| 3F_6-1_N30_W252 | 15.3% | -40.8% | 0.37 | 196,554 | 0.70 | 0.91 | +3.1% (0.5) | 0.88 | 31% | 1.42/0.29 | -37% -26% -24% -8% -7% | - |
| 3F_6-1_N30_W504 | 16.3% | -37.3% | 0.44 | 214,895 | 0.74 | 0.92 | +3.8% (0.6) | 0.89 | 31% | 1.46/0.40 | -35% -27% -25% -10% -10% | - |

Pass rule (DD >= 8 pts better AND CAGR >= 22% AND ratio >= REF 0.61+0.15): improves 0/16 (need 12); ratio beats REF in BOTH halves 1/16 (need 12); median cell by ratio SPY_12-1_N30_W252 cuts 4/5 episodes (need 3).
## Family verdict: **FAIL**

Median cell SPY_12-1_N30_W252: 20.6% / -46.2% / $303,676, beta 0.97, alpha +7.1%/yr (t 1.1), corr with REF 0.95, holdings overlap 61%, paired weekly diff vs REF -0.155% (t -1.9, ex-top-5% -0.392%).
50/50 blend median+REF (daily-rebalanced, informational): 24.2% / -45.2% / $403,341. No recommendation (family not REAL).

## Caveats (adversary)
- Cost model is the 1700c band (not measured NBBO); residual names may differ in liquidity from REF, so cost parity is assumed, not shown.
- W504 early-sample: the panel starts 2016-01, so in 2017 the 504-day window is truncated; presence rule applied to min(W, days available) -- W504 cells differ from W252 partly by sample at the start.
- One beta set per name-day applied to the whole score window (in-sample for the 12-1 window when W covers it); betas are noisy for names with <1 yr of history.
- U2 eligibility and the 40-name pre-cut are NOT applied to residual cells (they rank all eligible names, REF ranks the top-40 by sigV2 then takes 20): the candidate pool differs, not only the score.
- Close prices are as stored in the panel (adjustment status inherited from 1700j); no per-trade price-scale check was run on the residual cells; max |daily return| logged in 1700r.log.
- Overlap = mean share of the cell's N names also in REF top-20. Blend is informational only. Multiplicity: +16 cells on this line; single sample, 2017-2026 bull-heavy, halves are not independent of the theme.
- SUSPICIOUS: max |daily return| in the candidate returns is 447 (+44,700%): unadjusted split / bad bar somewhere in the panel; it hits REF too (same panel) but a residual score divides by residual std, so one bad bar can distort a name's score. Not resolved here; the FAIL verdict does not rely on it (0/16 improve, none close).
- KO SPY-only beta 0.05 (W252) / 0.23 (W504) on 2024-12-31 is below the 0.3-0.7 prior (NVDA 2.65/2.32 slightly above 1.5-2.5): plausible for 2024 (KO decoupled) but unverified externally. SPY's own score is NaN because its residual std is ~0 (beta 0.99999998) and the sd<1e-9 gate blanks it; i.e. residual ~0 as expected.
- Drawdown relief of the SPY-only N20 cells is real in the 2022-26 episodes (ep3-5 -11..-20% vs -34..-29%) but is paid for in CAGR; same one-for-one trade as every earlier repair.
