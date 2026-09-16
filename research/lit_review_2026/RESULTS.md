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

## 2026-09-16 11:00 UTC — THE RED-TO-GREEN BOOK IS DEAD. The entry fill was not the engine's.

An adversarial statistical audit (`research/bf_zero2/audit_stats/`) reproduced my numbers exactly from the trade file and
then re-simulated the same 25,876 signals under the fill convention the LIVE engine actually uses. The sign flips.

| entry convention | TRAIN | VAL | TEST | weeks green |
|---|---|---|---|---|
| fill at the touch, `level × 1.003` (what I reported) | +0.131R, +2.5/wk, t 2.95 | +0.326R, +6.1/wk, t 4.37 | +0.356R, +6.9/wk, t 3.90 | 34/53, 17/22, 12/14 |
| stop order filled at the NEXT bar's open | −0.247R, −4.6/wk | −0.160R, −3.0/wk | −0.110R, −2.1/wk | 11/53, 8/22, 6/14 |
| **live capped limit, no chase (`trading/hod_break_engine.py`)** | **−0.277R, −5.2/wk, t −7.0** | **−0.315R, −5.8/wk, t −4.8** | **−0.295R, −5.7/wk, t −3.5** | 11/53, 3/22, 2/14 |

Why: one minute after the trigger the price is a median **+48 bps** above it (mean +69 to +79, p90 +209 to +255). Pass 1
assumed +30 bps and assumed it was always obtainable. Only 57–60% of signals are obtainable at ≤ +60 bps, and on those
that DO fill the mean is still **−0.112 / −0.066 / −0.157R** — so this is adverse selection, not a no-fill artefact. The
signals you can actually buy at the level are the ones that did not continue.

The mistake is precisely the one the project already knew about: `trading/hod_break.py::entry_fill` (the spec) fills at the
NEXT bar's open or not at all, and `research/bf_zero/REPORT.md` §6 says so. But `build_candidates.py`, the pass-1 family
scanner every one of these searches is built on, fills at the touch. Every number derived from pass 1 inherits an entry
that cannot be obtained.

Three further findings from the same audit, all of which stand independently:
- **The "strategy" was the selection, not the setup.** Four RANDOM qualifying candidates a day return +0.015 / +0.049 /
  −0.020R; the whole qualifying population returns −0.006 / +0.080 / +0.017R. Taking the FIRST four of the day was doing
  100% of the work, and it was never counted as a search dimension — it entered as the book's queueing discipline.
- **That selection is mostly a tight-stop proxy.** Median stop distance rises monotonically with entry rank; inside
  stop-size strata the effect collapses. The 1–2% stop bucket is +0.32 / +0.49 / +0.47R and the 4.5–7% bucket is
  −0.04 / +0.04 / −0.06R. A tight-R book is maximally leveraged to exactly the fill assumption that turned out to be wrong.
- **97–100% of the trades are between 09:31 and 09:35.** The "entries until 14:00" rule was vacuous.
- Multiplicity was NOT the problem (search-adjusted p = 0.0009 over ~1,200 cells, measured by permutation). TRAIN sat at
  1.04× its own minimum detectable effect, so it was one underpowered observation and two short ones even before the fill.

**Standing rule added:** pass 1 must be rebuilt with the live entry convention (capped limit, next-bar open, no chase) as
the only fill model before any family search is run again. Until that rebuild exists, no number from
`research/bf_zero2/candidates*.csv` may be reported.

### Second audit, independent — same verdict, plus a cost correction that affects everything
`research/bf_zero2/audit_fills/REPORT.md`. It rebuilt the 25,876-candidate pool and re-walked every trade (1,641 of 1,676
identical to the published book, `rr_e1c` reproduced to 4.4e-6), then applied each fill correction separately.

| correction | TRAIN | VAL | TEST |
|---|---|---|---|
| base, as published | +0.124 | +0.320 | +0.353 |
| **entry = max(level × 1.003, the entry bar's OPEN)** | **−0.150** | **+0.057** | **+0.086** |
| stop slippage 50 bps | +0.069 | +0.264 | +0.305 |
| exit cost at the MEASURED spread | +0.073 | +0.272 | +0.309 |
| all corrections together | **−0.066 (t −1.7)** | +0.233 | **+0.050 (t 0.6)** |
| all corrections + 75 bps entry slip + mean spread | −0.143 | +0.143 | −0.014 |

- **55% of the book's trades have the entry bar OPENING above the fill price**, by a median 1.59% of price. That is a
  median 0.58R of free entry edge per gapped trade, and those trades supply 93 / 73 / 85% of the book's total R. The
  non-gapped candidates are flat to negative at pool level. This is the same defect the statistical audit found, measured
  a different way and costing about 0.27R per trade on every split.
- **Costs were understated by two to four times.** Fresh SIP NBBO pulled for 872 of the book's own trades, in their own
  fill minute, gives a median full spread of **1.52% of price** (0.86% restricting to actively quoted names) against the
  0.40% the scorer assumed. The 30 bps entry slip does not double-count the exit half-spread as I stated earlier; it
  UNDER-charges the entry on 84% of trades. My "bug 2" correction was therefore wrong in direction, and the real cost
  model for a 09:31–09:35 book on gap-down names is far harsher than 40 bps.
- The target fill is genuinely conservative (zero impossible fills; a resting limit would have done ~0.06R/trade better).
- Liquidity is harmless at $100 risk and fatal at scale: at $2,000 risk per trade, 88% of the book would exceed 1% of the
  5-minute dollar volume.

**Standing corrections for all future work**: (a) pass 1 must fill at the next bar's open under a cap, never at the touch;
(b) the cost model for first-five-minutes books must use the measured spread for THAT population, not the 40 bps median
from the HOD-break signal study, which was a different and more liquid population.

### Third audit — the fill was not merely optimistic, it was impossible. And the book failed its own pre-registered gates.
`research/bf_zero2/audit_data/` (the agent could not write REPORT.md; its narrative is reproduced in the commit and the
supporting CSVs and scripts are on disk). It re-derived all 25,876 candidates from bars (25,876 of 25,876 reproduced).

**Finding 1, FATAL, and nobody had seen it.** The family code gates on a bar's HIGH reaching the level and then books the
fill at that level, with no requirement that the bar's LOW be at or below it. So a trade can be filled below the low of the
bar that fills it — at a price the market never offered.

| | all 1,676 trades | entry ≤ 09:32 |
|---|---|---|
| filled BELOW the signal bar's low (impossible) | **41.2%** | 43.6% |
| level already traded through before the signal bar | 44.8% | 51.8% |
| +2R target already exceeded by the signal bar | 18.4% | 21.0% |

Possible fills earn −0.033 / +0.064 / +0.013R. Impossible fills earn +0.397 / +0.646 / +0.707R and are **101% of the
book's total profit**. Example: SMX on 2026-02-10, the book's fifth largest winner, booked in at $13.94 with a $14.25
target when the stock had already traded to $15.48. The defect is in shared code (`build_candidates.py` lines 110, 139,
149, 159, 167) and therefore in EVERY candidate row in `research/bf_zero/` and `research/bf_zero2/`. F6 won the scan
because its level, yesterday's close, is the stalest of all of them.

**Finding 2, FATAL.** "4 concurrent" in `score3.py` also meant 4 trades per day, and it bound on every single day. The
eligible population earns −0.006 / +0.080 / +0.017R. At the 12/day cap that `DESIGN.md` pre-registered, the book earns
+0.068 / +0.213 / +0.061R with TEST at t 1.03. The 4/day cap was a proxy for "take only 09:31–09:32", which is exactly
where the fill defect is worst.

**Finding 3, MAJOR.** The cost model in `score3.py` charges nothing on entry and 40 bps on exit. `DESIGN.md` pre-registered
half a spread in AND out. At the repo's own measured mean spread of 57 bps, applied as pre-registered, TRAIN falls to
+0.014R (t 0.31) before any fill correction.

**Finding 4, MAJOR.** The configuration fails the gates it was presented as surviving: `DESIGN.md` requires TRAIN ≥ +10R
per week (actual +2.3) and VAL ≥ +7R (actual +5.9). And TEST was used as a filter — "positive on all three splits: 24",
then the best of the 24 — which gate 3 forbids.

**The honest number**, applying nothing but `DESIGN.md` as written: **−0.127 / −0.123 / −0.093 R per trade**, about −3.9,
−3.9 and −3.2 R per week. Under the single most generous correction that merely makes the fills physically possible:
−0.098 / +0.097 / +0.115R, TRAIN t −2.42.

Clean: no price-scale mismatch (the daily open matches the 09:30 minute open; the 45 disagreeing trades are worth +5.6R of
+355.6R), no corporate-action artefact, no merge or ticker-reuse problem, survivorship residual 0.51% of the population.
A real acausality was found and is minor: `range_so_far_pct` included the signal bar's own extremes.

**The original bf_zero2 conclusion was right.** `score3` did not find an edge the first run missed. It correctly loosened
the causal floor and simultaneously halved the cost model, changed the book cap and kept an impossible fill.

`research/bf_zero2/build_candidates3.py` is the rebuilt pass 1: the fill is the next bar's open under the 0.6% cap (a real
printed price), `range_so_far` uses bars strictly before the signal, spreads are per price band measured on this
population, and the file header states the scorer's contract — half spread in and out, `run_book(rows, 12, 4)`, and the
pre-registered gates with TEST read once. Smoke-tested on three days; not yet run in full.

### Fourth audit — the arithmetic is correct, and that is the lesson
`research/bf_zero2/independent_r2g.py`. An agent that never read my implementation rebuilt the book from a prose spec.
Two full 419-day runs. Result: **1,673 of 1,676 trades match**, and on the matched set the entry minute, exit minute, exit
reason and net R agree to 1.6e-7. My published statistics are reproduced to 0.006R.

Its own summary of what that certifies: **"arithmetic, not edge."** The reimplementation worked from MY specification, and
my specification contained the impossible fill. So it faithfully reproduced a rule that cannot be traded. This is the
limit of the technique and it must be written down: **an independent reimplementation catches coding errors and cannot
catch specification errors.** Only a comparison against what the live engine can actually obtain catches those, which is
what audits 1 to 3 did.

Three small real defects it did find:
- The $5 floor is applied in code to the day's 09:30 open, while the written spec says the entry price. A documentation
  defect worth 0.006R, not a P&L defect — but the two must be made to agree.
- The 5.000% causal-range test needs an explicit floating-point tolerance; two trades sit exactly on it.
- The `NA` ticker was eaten by a default `pd.read_csv` in the auditor's own first run, the same defect class this repo has
  now been bitten by three times. Every CSV read in the research tree needs `keep_default_na=False`.

## STAGE 2 RESULT — the cost curve. The search space was never viable, and now we can prove it per segment.
`research/lit_review_2026/cost_curve.md`, from 2,570 real NBBO samples in the signal minute, stratified by price band ×
time of day × liquidity. `min_edge_R` is the gross R per trade a strategy must earn in that segment merely to pay a round
trip (half a spread in, half out).

| price band | 09:30–09:35 | 09:35–10:00 | 10:00–11:00 | 11:00–13:00 | 13:00+ |
|---|---|---|---|---|---|
| $5–10 | 0.185 | 0.186 | 0.120 | **0.118** | 0.111 |
| $10–20 | 0.220 | 0.229 | 0.125 | **0.105** | 0.140 |
| $20–50 | 0.290 | 0.229 | 0.182 | 0.148 | 0.127 |
| $50–200 | 0.364 | 0.348 | 0.182 | 0.141 | 0.143 |
| $200+ | 0.323 | 0.425 | 0.412 | 0.219 | 0.317 |

**The best honest gross edge this project has ever measured, across every family, study and literature replication, is
about 0.1R.** Every segment in the table demands at least that much just to break even, and the segment where all of our
"winning" books concentrated — the first five minutes — demands 0.19 to 0.36R. That is the whole story of the last month
in one table: the entry fee exceeds the prize, so the only books that looked profitable were the ones whose fills were
unobtainable.

Two findings worth carrying:
- **Expensive movers are worse, not better.** $200+ names show 54–76 bps spreads against 1.6–1.8% R, the worst ratio in
  the table. These are volatile movers, not blue chips; price is not a proxy for liquidity here.
- **The first five minutes is the worst window in every band.** Every strategy this project has built entered there.

**What this implies for stage 3, quantitatively.** We need segments where spread/R is below roughly 0.05. Two routes:
much tighter spreads (SPY's spread is under 1 bp; on a 0.5% move that is spread/R ≈ 0.003, forty times better than the
best stock segment) or much larger R (multi-day holds, where R is measured in whole percent). Both point away from
single-name intraday breakouts and toward liquid instruments or longer horizons — the same conclusion I reached from the
QQQ noise-band sleeve, now with a number attached instead of an intuition.

Caveat: this sample was drawn while the corrected scan was still writing, so it covers 2025-01 to 2025-08. It will be
re-run over the full period when the scan finishes; spreads may differ in 2026 but not by enough to move a 2-to-4x gap.

## STAGE 2b — the cost curve was wrong, and the correction changes the design constraint
The owner's objection: "the spread on losers will be different." Correct, and the flat model was wrong twice — a winner
exiting on a resting limit pays NO exit spread, and a loser crosses the spread at the exit minute, not the signal minute.
Measured on 874 real trades with NBBO pulled at BOTH the entry and the actual exit minute
(`research/lit_review_2026/cost_by_outcome.md`):

| exit type | entry spread | exit spread | entry cost | exit cost | **total cost** |
|---|---|---|---|---|---|
| target (resting limit) | 40.0 bps | 27.8 bps | 0.126R | **0.000R** | 0.126R |
| 15:55 close | 37.8 bps | 14.0 bps | 0.055R | 0.023R | **0.083R** |
| stop | 40.1 bps | 29.2 bps | 0.115R | 0.073R | **0.194R** |

**Blended cost 0.135R per trade, against the 0.191R I published. The flat model was 30% too harsh.** The conclusion does
not flip — our best honest gross edge is about 0.1R — but the gap is 0.035R, not 0.09R, and I had overstated it.

The owner's intuition is confirmed with a twist. Stops ARE the expensive outcome, 1.5× a target and 2.3× a close. But not
because the spread widens into a stop: the exit-to-entry spread ratio is 0.875 for stops, 0.708 for targets, 0.412 for
closes, so spreads narrow through the session for everything and stops merely narrow least. Stops are expensive because
they pay an exit spread at all AND because stopped trades have small R.

**Which exposes the real lever, and it is not the segment — it is the stop width.**

| stop width | cost in R | P(stop) | P(target) |
|---|---|---|---|
| 1–2% of price | **0.190R** | 50.6% | 38.9% |
| 2–3% | 0.115R | 42.3% | 21.2% |
| 3–5% | 0.071R | 29.1% | 10.8% |
| 5–8% | 0.053R | 17.1% | 5.4% |
| 8%+ | **0.026R** | 12.5% | 4.0% |

The spread in basis points is flat across all of them, 35 to 47 bps. What changes is the denominator. A tight stop is
penalised twice: it costs seven times more in R *and* it is hit four times as often. That is the mechanism behind the
audits' finding that our "edge" was a tight-stop proxy which evaporated under an honest fill — tight-R trades are where
an unobtainable fill flatters the result most and where real costs bite hardest.

**Design constraint for stage 3, restated:** a viable intraday book needs a stop of at least 3–5% of price, where costs
fall to 0.05–0.07R, and must accept the low target-hit rate that comes with it, meaning it lives on the 15:55 close
rather than on a fixed target. That is a different strategy shape from anything this project has built, all of which used
1–2% stops.

## FINAL — score4, the first result with every correction applied. Nothing passes, and now the null is trustworthy.
`research/bf_zero2/score4_tables.md`. Built on `candidates3.csv` (2.94M rows, rebuilt over 420 days with the live fill),
446,012 of them in the scoring population after the causal universe rule, price ≥ $5, entry by 14:00 and R ≥ 1%.

Every correction from the week is in this one number:
- fill = the NEXT bar's open under the 0.6% cap, never the touch (audits 1–3);
- cost per OUTCOME, measured: half a spread on entry always; a target exit rests on a limit and pays nothing; a stop pays
  half a spread at 0.875× the entry spread; a 15:55 close at 0.412× (the owner's correction, `cost_by_outcome.md`);
- spread = the measured per-price-band figure, not the 0.40% that had been assumed;
- book = 12/day and 4 concurrent, as `DESIGN.md` pre-registered, not the 4/day that was acting as a selection rule;
- causal range test on bars strictly before the signal.

**Result: 0 of 52 cells clear gate 1.** The best family-config in the entire search is F8, the 30-minute opening-range
break, at **−0.106R per trade and −2.2R per week on TRAIN**, with 23% of weeks green. Not one configuration is positive.
TEST was never consulted, because nothing reached it.

The realised exit mix on honest fills is 46% stops, 34% closes, 20% targets — against 33/34/33 under the old touch fill.
Making the entry obtainable moves a third of the winners into the stop and close buckets, which is exactly the adverse
selection audit 1 measured directly.

**This is the honest answer to the question the project has been asking since 9/13.** Twenty-six entry families across
ten shapes, on the whole point-in-time universe, on a provenance-checked consolidated tape, with fills the live engine
can obtain and costs measured on the trades themselves: intraday long breakout and pullback books on ≥5%-range US equities
do not pay. The earlier positive numbers were, in order, a hindsight universe, a one-publisher tape, an impossible fill,
and a flat cost model. Each was found and each is now fixed in code.

What remains true and useful: the cost curve and the stop-width table say where a viable book would have to live — a stop
of 3–5% of price and an exit on the close rather than a fixed target, in instruments whose spread is a small fraction of
R. That is a different shape from anything tested here, and it is testable with the machinery now built.
