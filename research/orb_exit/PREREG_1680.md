# PREREG — cell 1,680: does the 50 % partial at +1 R buy CONSISTENCY on ORB? (FROZEN 2026-09-30 17:30 UTC)

Owner 17:25 UTC: "net is zero, but green weeks? green days? must be better, right?" — the consistency objective
(owner 9/6: 30–50 %/yr delivered monthly beats headline P&L; reshape books on a pre-committed consistency bar) and the
cadence bar (`docs/cadence_bar.md`, scorer `scripts/cadence_bar.py`).

## Inputs
`research/orb_exit/1679_per_fill.csv` — per-fill R under the actual exit (base) and under each rule, on L2 (the BT
book 2025-01..2026-09, n 478, R-floored) and L1 (live, n 123). Rule of interest (d) 50 % out at +1 R + the live rule
on the rest; comparators (e) the live lock reference and the base. Dollar series at $375 risk per fill (L2) and actual
shares (L1). Days and ISO weeks from the fill date.

## Reads (L2 by year 2025 / 2026 and whole; L1 whole)
1. Per-fill: mean R (must be within ±0.03 R of the base — the EV is not the question), win rate, SD of R, ex-top-5 %.
2. Per-day series (sum of R per trading day with ≥ 1 fill): green-day share, daily P10, worst day, max drawdown in R
   and $.
3. Per-week series: green-week share, weekly mean / SD (weekly Sharpe), weekly P10, worst week, max drawdown; the
   cadence bar via `scripts/cadence_bar.py` (strong week ≥ +5 R, median gap, P90 gap, bleed between strong weeks,
   green weeks vs the count-matched null of 1,000 draws); the same for (e) and the base.
4. Compounding read: geometric growth per week at $375 risk on $65K and at the ramp's next rung ($750), base vs (d).
5. Stability: the same tables on L1 (live) with the caveat that L1 is the damaged-execution period.

## Consistency bar (fixed now)
(d) "buys consistency" if, on L2 in BOTH years: mean R within ±0.03 of the base; green-week share ≥ base + 5 pp;
weekly P10 ≥ base's weekly P10 + 0.5 R; weekly Sharpe ≥ base × 1.15; AND the cadence bar's strong-week median gap does
not worsen by more than 1 week; AND the count-matched null percentile of green weeks ≥ 95. If it passes → independent
rebuild of the per-fill rule book from prose (a second agent walks the bars) → ORB PAPER with the partial as the ONE
mechanics change of its session (after the target-resting-limit session), forward read at 100 fills. If it fails one
clause, report which and by how much; no orb.yaml change.

## Multiplicity
3 rules × 3 splits × ~12 metrics; one pre-declared bar. Not allowed: tuning the partial size or level; dropping the
R floor; pooled-only numbers.

## Output
`research/orb_exit/RESULT_1680.md` (≤ 90 lines: the bar's clause table first), `1680_weekly.csv`, `1680_daily.csv`,
`1680_consistency.py`, `1680_consistency.log`. The agent returns ≤ 120 words.
