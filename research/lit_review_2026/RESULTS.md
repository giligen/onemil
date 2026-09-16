# RESULTS — literature-driven hypothesis tests (state at 2026-09-16 06:30 UTC; the cheaper model appends from here)

Conventions: splits TRAIN 2025 / VAL Jan–May 2026 / TEST Jun–Sep 2026 (2025–26 stores) or IS 2016–23 / OOS 2024→ (ETF store);
costs as in RUNBOOK §0; book = 4 concurrent, 12/day, first-come. "Cells" = distinct rule variants looked at for that row.

| # | M-id | rule | n (per split) | gross | net | t | hit | weekly R / weeks green / worst | cells | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | M6 | Zarattini SPY noise area, faithful (paper costs) | IS 2010 days / OOS 678 | IS 10.9 bps/day, OOS 0.6 | at 1 bp/leg: IS 4.0, OOS −7.2 | IS 3.2 / OOS 0.09 | 43% | IS SR 1.03, OOS −0.14 (1×) | 7 (ablations) | reproduces in-sample, dead on SPY since 2024 |
| 1b | M6 | same on QQQ | IS / OOS 678 | IS 12.8, OOS 10.1 bps/day | — | 3.7 / 1.6 | 44–45% | OOS 8.4%/yr, SR 0.99 unlevered | 1 | marginal: ≈ $5K/yr on $60K unlevered ≈ 1R/week; needs a live dry run before belief |
| 2 | M1 M2 M3 M12 | last-30-min sign timing SPY, 4 variants | 2010 / 678 | −2.0..−3.0 bps/day | same | −2.4..−3.6 | 41–45% | negative every year 2016–2026 | 4 | dead (anti-momentum in the 0DTE era) |
| 3 | M22 | volume-shock overnight, top-4 book, $5+, $2M+/day | 960 / 408 / 268 | +1.7 / +12.1 / −14.5 bps | −26 / −16 / −43 | −2.9 / −1.2 / −2.4 | 32–33% | −4.9% / −3.0% / −8.3% per week; 19/51, 9/22, 3/14 | 2 | dead at a 4-name book |
| 4 | M7 | 5-min ORB TQQQ, 0.1 ATR stop, hold to close | 1852 / 635 | +7.4 / +3.6 bps/day | at 1 bp/leg | 1.9 / 0.7 | 19–20% | SR 0.71 / 0.44; 80% of days stopped | 2 (QQQ ≈ 0) | not significant OOS; ≈ $110/week on $60K |
| 5 | M9 | RV monotonicity of ORB/HOD P&L, our universe | 2–3K per bucket | ≈ 0R in every RV bucket with the causal floor | — | — | — | — | 4 | the paper's monotonicity is absent |
| 6 | M36 M37 | large-loser / intraday-component reversal (top-4 book) | 847 / 403 / 252 and 960 / 408 / 268 | −86 / −94 / +10 and −156 / −73 / −118 bps | −105 / −113 / −9 and −174 / −93 / −137 | −3.6 / −2.4 / −0.2 | 43–48% | negative | 4 | dead as run (VIX-regime split and news filter still TODO per RUNBOOK) |
| 7 | M20 | end-of-day loser reversal ≤ −8% at 15:00 → 15:30→MOC | 11.7K / 6.7K / 5.9K | −0.7 / +40.2 / −3.2 bps | −21 / +20 / −23 (20 bps entry) | −10.6 / +8.2 / −8.0 | 41 / 51 / 44% | 10/51, 13/22, 4/14 green | 2 (+ winner control: +0.5 / +17 / −5.6) | dead: VAL's positive is period-wide (winners too); deepest tail +20 bps gross < cost |
| 8 | M8 | stocks-in-play ORB OOS, liquid universe, paper spec | 3470 / 1652 / 1083 fills | −0.12 / −0.25 / −0.17R | at 1 bp: −0.19 / −0.31 / −0.22R | — | 10–11% | −13.7 / −22.9 / −16.7R per week; 12/47, 5/22, 4/14 | 1 | dead out of sample; no RV monotonicity; both sides negative |
| — | bf_zero2 | 26 breakout/pullback families on ≥5%-range days, causal floor | 1–3K per config | ≤ +0.18R gross | best +0.007R | — | 34–50% | none ≥ 1R/week | 275 | dead (research/bf_zero2/REPORT.md) |

Verdict so far (rows 1–8 = HYPOTHESES.md priority items 1–8 done, except the VIX/news refinements of row 6): nothing in the
2022–2026 literature that our data can test survives out of sample at retail costs, except a marginal QQQ noise-band sleeve
(≈ 1R/week, t 1.6). The queue continues at RUNBOOK §4 rows 9–15 (gap table, open fade, SPY overnight, vol-gated timing, VWAP flip,
large-cap overnight continuation, daily add-ons); none has a realistic prior above 2–3R/week in HYPOTHESES.md §4.
