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

## Queue rows 6r, 9–15 (run 2026-09-16 07:00–08:10 UTC)

| # | M-id | rule | result | verdict |
|---|---|---|---|---|
| 9 | M16 | single-stock gap table (measurement): P(close>open), open→close, gap-fill by gap size × dollar-volume band | **0 of 27 cells** have mean open→close ≥ +50 bps with n ≥ 200 in BOTH years. Gap-ups above +10% run −60 to −190 bps open→close in the $2–10M band; full same-day fill 35–55%, half-fill 55–75% (`daily_queue.md`) | no gap-continuation edge; the table is the reference for how gappers behave and it says fade, not chase |
| 10 | M18 | open fade on prior-day attention names (top-20 by abs(return) × volume ratio at t−1), vs a rank-100–120 control, measured on fetched day-t minute bars (8,050 symbol-days, `attention.db`) | attention names, open→10:30: **−60 / −26 / −20 bps** (TRAIN/VAL/TEST); minus control **−46 bps (t −4.9)** on TRAIN, −19 (t −1.1) VAL, −8 (t −0.4) TEST; the deepest fade is in $2–10M names, and open→close is −67 bps on TEST | CONFIRMED as a veto, not a book: never buy a prior-day attention name in the first hour; the level is negative in all three splits, the excess over control is significant only in 2025 |
| 11 | M11 M12 | SPY overnight premium; conditional reversal after a bottom-quintile intraday day; slope of next overnight on the last-30-min return | M11 IS +0.55 bps (t 0.3), **OOS +3.63 bps (t 1.52, SR 0.93)**; M12 conditional OOS +8.8 bps (t 1.25, n 123); slope −0.54 (corr −0.24) IS but −0.04 (corr −0.01) OOS | fails the t ≥ 2 bar; the overnight drift is directionally back in 2024–26 but underpowered, and the reversal relationship died out of sample |
| 12 | M5 | volatility-gated last-30-min timing (top-tercile 09:30–10:00 realized vol, causal 250-day threshold; 602 IS / 236 OOS days) | first-half-hour sign −3.07 / −1.72 bps; 15:00→15:30 sign +1.08 (t 0.5) / −2.39 (t −1.3) | dead; gating does not rescue intraday momentum. NOTE: the first run of this row had a silent bug (the vol field was zero for every day, so the gate passed everything); fixed and re-run before recording |
| 13 | M10 | QQQ VWAP flip, 1 bp/leg | −20.9 bps/day IS, −28.5 OOS, 33 flips/day | dead, as the review predicted |
| 14 | M29 | cross-sectional overnight continuation, large caps ($50M+/day), rank on the trailing-20-day mean overnight return | top decile net **+7.4 (t 5.1) TRAIN, +6.7 (t 3.5) VAL, −13.4 (t −5.5) TEST**; top-4 book +42.7 / +72.5 / −40.4 bps | fails: two splits strongly positive, the read-once split strongly negative. This is exactly the shape a decayed effect makes; do not revive it without new data |
| 6r | M36 | large-loser reversal split by market-volatility regime (trailing 20-day SPY vol tercile, cut on 2025) | low/mid/high regimes: TRAIN −198 / −128 / +17 bps net; VAL −83 / −95 / −206; TEST −39 / +82 / −145 | no regime is consistently positive; the row-6 refinement does not rescue it. The news filter remains untested (needs a news pull) |
| 15 | M30 M31 M38 M39 M40 M41 | beta night-day; tug-of-war; high-MAX weekly losers; speculative Thu→Fri; SPY turn of month; new 252-day high on volume | M30 t ≤ 0.8 everywhere; M31 positive in all three but t ≈ 1.0 on 30–100 holdings; M38 −88 / +389 / +232 bps, t ≤ 0.9; M39 mixed, t ≤ 1.1; M40 n = 4–12 events, meaningless; M41 hold-10d +104 (t 2.0) / +389 (t 3.4) / **−183 (t −2.3)** | none passes; M41 is a second sign-flip on the read-once split |

**Cells counted in this batch:** 27 gap cells (measurement, not selection) + 2 (M18) + 5 (M11/M12) + 4 (M5) + 2 (M10) + 4 (M29) + 9 (M36r) + 10 (add-ons) = 63.

**Standing after the whole queue.** Fourteen of fifteen rows are dead or underpowered. Two things survive as usable knowledge,
neither of them a book: (a) the QQQ noise-band sleeve, ≈ 10 bps/day OOS at t 1.6 ≈ 1R/week at our size; (b) the open-fade veto,
prior-day attention names are worth −20 to −60 bps in the first hour, which is a rule about when NOT to enter. Two rows
(M29 overnight continuation, M41 new-high momentum) were positive with t > 2 on TRAIN and VAL and flipped hard on the
read-once split — the decay pattern the meta-literature predicts, and the reason the split discipline exists.
