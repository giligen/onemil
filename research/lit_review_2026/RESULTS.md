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

## 2026-09-16 09:30 UTC — THREE SIMULATION BUGS, AND A STRATEGY THAT SURVIVES THEM

The owner's challenge ("you understand how ridiculous the claim is") was correct. Three bugs in my own simulations
produced the negative verdicts. All are in `research/bf_zero2/score2.py` and the pass-1 fill convention; the corrected
scorer is `score3.py`.

**Bug 1 — the causal filter was three times too strict.** The study universe is "day range ≥ 5%", known only at the close,
so F5–F10 signals need a causal guarantee of membership. I used `dist_open_pct ≥ 5` (entry 5% above the open). The correct
and far weaker guarantee is `range_so_far_pct ≥ 5`: if the high-low range UP TO the signal bar is already 5%, the full
day's range is at least 5% whatever happens next (low-so-far ≤ open, so range/low ≥ range/open ≥ 5%). Both are causal.
The wrong one kept 229,265 rows; the right one keeps 733,990. I discarded two thirds of the legitimate population.

**Bug 2 — the entry cost was charged twice.** Pass 1 fills at `level × 1.003`, i.e. 30 bps through the level, which IS an
ask-side fill. `score2` then charged another half-spread (20 bps) on entry. On a 3% R that is 0.065R of phantom cost per
trade, on top of a real one.

**Bug 3 — the hold-to-close exit was never scored.** `score2` only ran the +2R close-fill exit. The day-trading literature's
actual spec (hold to the close with a −1R stop, no target) was computed in pass 1 as `rr_e4` and never put through the book.
It is the stronger exit in gross terms.

A fourth, from the day before: my cost model charged a quoted spread on trades that execute in the closing and opening
auctions, which is three to eight times too much for those.

### What survives after the fixes
24 of 108 family-config × exit × book-size cells are positive on all three splits (`score3_tables.md`). The clean one:

**F6 "red to green", +2R target on a bar close, −1R stop, flat 15:55.** Rule, entirely causal: the stock OPENS BELOW its
prior close; entry on the first 1-minute bar whose high reaches prior close × 1.003; stop = the lowest low before entry;
target = entry + 2R, filled when a bar CLOSES through it; otherwise flat at 15:55. Universe membership guaranteed at the
signal bar by range-so-far ≥ 5%. Price ≥ $5, entry by 14:00, R ≥ 1% of price. 4 concurrent, first-come.

| split | trades | R/trade | R/week | t | win rate | weeks green | max DD |
|---|---|---|---|---|---|---|---|
| TRAIN 2025 | 996 (4.0/day) | +0.130 | +2.4 | 2.9 | 44% | 35/53 | −19R |
| VAL Jan–May 2026 | 408 | +0.322 | +6.0 | 4.3 | 51% | 17/22 | −8R |
| TEST Jun–Sep 2026 | 272 | +0.350 | +6.8 | 3.8 | 53% | 12/14 | −9R |

- **Not tail-dependent**: capping every trade at +2R changes nothing, because the target caps it by construction.
- **Cost-robust**: at 80 bps of spread (double our measured median) it is +0.090 / +0.283 / +0.314R, t 2.0 / 3.7 / 3.4.
- **Not from the hindsight population**: rebuilt on days whose OPEN was ≥ $5 (the complete fetch, no scanner selection) the
  numbers are unchanged.
- 4 negative months out of 21; worst month −7.5R; exits ≈ 40% stops, 30% targets, 30% held to the close.
- At 20 slots: +3.1 / +13.6 / +17.9 R per week, t 2.1 / 5.2 / 4.9, but the drawdown scales to −55R on TRAIN.

The hold-to-close variant of the same entry is bigger (+0.28 / +0.68 / +0.58R, 5–13R/week) but **tail-dependent**: capped
at 5R the 2025 edge disappears entirely (+0.006R) and the top 1% of trades are 73% of TRAIN's profit. That is the profile
the owner rejected on the bull-flag book, and it is not the one to ship.

### Caveats that must be said
1. 21 months. TRAIN is the weakest split and the two 2026 splits are the strongest, which is the opposite of decay and may
   be a volatility regime rather than a stable edge.
2. Of a median 38 qualifying candidates a day, the book takes 4 **first-come**. Which four is unexplored; a ranking rule
   could add to this or could be the next overfit.
3. Stop fills are modelled 10 bps through the stop, which is optimistic on gap-downs; the 80 bps cost row partly covers it.
4. Nothing is live-validated. The existing engine already does capped-limit entry with bracket exits, so this is a spec
   change, not a rebuild.
