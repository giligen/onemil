# Meta-labelling the ORB selection — REPORT

Run 2026-09-18, one python process at a time, `nice -n 15`, `ulimit -v 1500000`,
xgboost `n_jobs=1` on every fit (peak RSS < 700 MB, no BLAS thread grab).
Everything written is under `research/meta_label/` plus one default-inert hook in
`study_orb_pipeline_static_lock.py`. `data/cache.db` / `data/trades.db` were not
opened at all. No config, order, service, cron or production artefact touched.
**TEST was never scored by any model** (see §8).

Executes `research/meta_label/PREREG.md` as written: the 9 cells, the purged and
embargoed walk-forward, the splits, the gates, the shuffled-label control, the
ablation and the ship bar. Nothing was added to the design; two things were
*specified* by the harness where the PREREG was silent and both are named as such
(the M4 veto threshold, §6; the first scorable month, §3).

---

## 0. The prior this result sits in, stated before the numbers

The PREREG justifies meta-labelling as the principled version of our hand-built
veto stack (López de Prado 2018, ch. 3). That is a **mechanism** argument and it
stands. It is **not** an empirical prior: the 2023–2026 literature sweep
(`research/multiday/LIT_REVIEW_2023_2026.md`) found **no independent
peer-reviewed out-of-sample equity replication of meta-labelling** as of
2026-09 — one JFDS framework paper and one MDPI crypto pairs study, i.e.
advocacy literature with a citation gap where the evidence should be. The
technique is widely advocated and thinly validated.

Two consequences, applied throughout:

* The controls are the only evidence in play. If the shuffled-label control or
  the walk-forward were ambiguous, the correct output is "unreadable", not a
  number. They are not ambiguous (§5) — but that is what is carrying the report.
* A null here is **unsurprising and consistent with the absence of independent
  evidence for the technique**. It is not a surprise, and it is not a failure of
  the ORB data or of the D1 population.

Calibration, for anything that had come back positive: Chen & Velikov put 204
published anomalies at ~4 bps/month net post-publication. An effect large
relative to that earns extra leakage scrutiny, never celebration. D1's own
+0.415R on this same population was an availability leak — which is why §1 runs
before any model is fit.

---

## 1. The availability audit — run BEFORE any model was fit

`audit_availability.py` → `availability_audit.csv`. Every one of the 24 features
carried by `candidates_dump.csv` is traced to its construction in
`study_orb_features.py::compute_features` and to the latest timestamp its inputs
carry, then measured for coverage and for **missingness conditioned on the
outcome** (the D1 leak signature: a field populated only for symbol-days that
went on to do something).

Population hygiene first (F6-reconciliation standing rule): **0 NASDAQ test
tickers** (`^Z[A-Z]ZZT$`) in the 13,033 rows; the population is the ORB features
CSV, which is built off `daily_bars`, so no symbol without a `daily_bars` row can
enter. `entry_price` is confirmed to be the **09:35 order level**, not a realised
fill: `entry_price / range_high = 1.003` to 2.2e-16 on all 7,402 covered rows.

| | result |
|---|---|
| coverage | **100.0% on all 24 fields**, all three splits — no NaN anywhere |
| latest input timestamp | 09:35:00 (opening-range + SPY 5-min blocks), 09:30:00 (gap, price-vs-20d-high, SPY gap), prev close (prev-day + 20-day blocks), calendar |
| missingness vs outcome | max abs(entered − no-fill) gap in NaN rate = **0.0 points** |
| sentinel (0.0-on-missing) vs outcome | max gap **0.3 points** on the nine sentinel-bearing fields |

**DROPPED — 2 of 24:**

| field | reason |
|---|---|
| `spy_range_pct_5min` | 0.0 **missing-data sentinel** on 13.7% of TRAIN, 52.5% of VAL and **100% of TEST** (every month from 2026-04 on; the sentinel is day-level, all-or-nothing on 100% of sessions). The value in this dump is a cache-coverage marker, not the quantity the live engine computes from the tape at 09:35. Fitting on it fits the coverage era. |
| `spy_return_5min_pct` | same field pair, same sentinel, same rate |

The drop rule was applied mechanically: drop iff (a) the value is not computable
at 09:35:00 from data timestamped ≤ 09:35:00 — **0 fields**; or (b) NaN/sentinel
rate differs by > 5 points between entered and no-fill rows — **0 fields**; or
(c) a sentinel-bearing field carries the sentinel on > 20% of any split — **the
2 above**. Two notes on honesty: rule (c) reads a *coverage* statistic on TEST,
never an outcome; and `bars_green_in_range` / `last_bar_green` were *not* dropped
even though their zero rate differs between entered and no-fill rows by 6.1 and
16.9 points, because for those fields zero is a legitimate value (no green bar, a
red last bar) and that difference is signal — a red 09:34 bar makes the 09:35
breakout less likely to fire — not missingness.

**22 features survive.** Outcome columns (`pnl`, `pnl_pct`, `exit_reason`, `win`,
`entered`, `_rp_pnl`, `_rp_position`) never enter the matrix. Note in particular
that `entered` — whether the stop-limit filled — is an outcome and is not a
feature; the model must pick a candidate without knowing whether it will fill,
exactly as live does.

---

## 2. The label, the book, and the fill/cost model

Label = the candidate's own realised R under the shipped exit spec (static lock
1.75R → +0.5R, ATR stop floor, 40%@+3R scale-out, touchgo M/D, 15:45 flat):

    R = (pnl_pct / 100) * entry_price / (range_high - range_low)

which is share-count invariant and therefore identical to Stage Q's
`_sized_pnl / (shares * range)` before the quintile multiplier. A no-fill
candidate books R = 0 and still spends its slot.

**Fill and cost.** The PRIMARY arm is Stage Q's `meas_cost`: a capped limit at
`range_high × 1.003`, fill at `min(ask, cap)`, `ask > cap` ⇒ the order rests at
the cap and fills there if the NBBO ask returns before the 10:35 time stop and
never otherwise, with the measured per-trade NBBO on both legs and the resting
+3R scale leg free. The SECONDARY arm is `asis` (D1's — every elected order fills
at the cap), reported for every cell because it is the arm the M0 gate is stated
in. Both arms are read straight from `Q_fill/dump_{arm}.csv`; the labels the
model is trained on come from the same arm the book is scored in.

Books: **8 slots** (D1's dose-response optimum), account $26,666.67 so the
per-position cap stays the $10K-stage $3,333.33 and **binds on 100% of picks** —
which is what makes "the model may reorder and veto, never resize" enforceable:
every pick in every cell is the same dollar size as in M0, the quintile
multiplier is untouched, and no cell can win by sizing.

---

## 3. M0 — the reproduction gate on my own harness

`bash run_book.sh M0 asis` runs `study_orb_pipeline_static_lock.py` with the D1
environment and **nothing else**:

```
picks 215   P&L 14428.616990972434   ->  $14,428.62
```

byte-for-byte D1's `book_n8_q1on.csv` (`14428.616990972434`), to the cent, with
the meta hooks compiled in and inert. The same harness on the primary arm gives
**$11,050.32 / 215 picks**, which is Stage Q's `meas_cost` 8-slot number
($11,050) to the dollar. Two independent parity anchors, both clean. A third
fell out of `build_dataset.py`: recomputing the composite and the Q1 gate from
the `orb.yaml` literals leaves **5,771** candidates in the ranked pool —
D1's pool size exactly.

**The walk-forward hook, and the months it cannot reach.** The model needs 50
trailing training sessions before it may score a month, so **2025-01 .. 2025-03
are UNSCORED in every cell** and the pipeline falls back to the shipped ranking
there (all-NaN days tie on the meta key and the `(quintile, composite)`
tie-break decides — verified). Those 21 picks are byte-identical to M0 in all
nine cells and are reported separately as `PRE`; the TRAIN comparison is
**2025-04 .. 2025-12**, where a model is actually deciding. M0's own PRE/TRAIN
split is shown so nothing is hidden by the choice.

---

## 4. The nine cells

Walk-forward: train on a trailing **26-week** window (the window
`scripts/orb_weekly_refit.py` already uses), predict the next calendar month,
**embargo the 5 sessions immediately before the test month**, **purge** any
training row whose outcome window overlaps the test month. The purge removed
**0 rows across all 14 scored months** — and that is a fact about this book, not
a claim that purging is unnecessary: the ORB outcome window is 09:35 → 15:45 of
the candidate's own session, so no row dated before the test month can overlap.
The embargo is doing the real work. **No random K-fold was run at any point.**

Hyper-parameters: the single declared grid (depth 3/5 × 200/500 trees, lr 0.05,
min_child_weight 20), selected **on TRAIN only** by TRAIN R/pick, per model ×
flag-set, then frozen (`grid_train.csv`, `chosen_config.json`; 5 of 6 chose
depth 5 / 200 trees). M4 reuses M1's fitted model — it adds no fit.

`M4 flags=…` = the M1 classifier used **only to veto** (the shipped ranking
stands), at the p < 0.5 cut written into the harness. `M4b` = the same model at
the ranked pool's own base rate of a winner (0.145, §6).

### PRIMARY arm — `meas_cost` (Stage Q's measured fill + measured cost)

`d$` and `t` are the **day-paired** difference against M0 over the split's
sessions (the book is a daily object: slots, dedup and the no-refill vetoes all
live inside one day).

| cell | TRAIN picks | TRAIN P&L | TRAIN R/pick | TRAIN MDD | worst mo | red mo | d$ vs M0 | t | VAL picks | VAL P&L | VAL R/pick | VAL MDD | worst mo | red mo | d$ vs M0 | t |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **M0 (shipped)** | 84 | **5,692** | **+0.533** | −489 | +72 | **0** | — | — | 53 | **3,931** | **+0.483** | −470 | +172 | **0** | — | — |
| M1 flags=0 | 107 | 6,518 | +0.385 | −756 | −90 | 3 | +826 | 0.45 | 65 | 3,737 | +0.363 | −453 | +39 | 0 | −194 | −0.49 |
| M1 flags=1 | 108 | 6,892 | +0.415 | −756 | −90 | 2 | +1,199 | 0.69 | 64 | 3,829 | +0.382 | −453 | +39 | 0 | −102 | −0.25 |
| M2 flags=0 | 106 | 4,606 | +0.321 | −721 | −263 | 3 | −1,087 | −0.97 | 59 | 3,040 | +0.318 | −780 | −351 | 1 | −891 | −1.16 |
| M2 flags=1 | 103 | 5,946 | +0.400 | −819 | −90 | 2 | +254 | 0.17 | 58 | 2,590 | +0.276 | −1,056 | −490 | 1 | −1,341 | −1.75 |
| M3 flags=0 | 100 | 5,449 | +0.384 | −760 | −90 | 1 | −243 | −0.16 | 55 | 2,779 | +0.296 | −675 | −471 | 1 | −1,152 | −0.76 |
| M3 flags=1 | 107 | 6,394 | +0.401 | −560 | −90 | 1 | +702 | 0.39 | 61 | 2,427 | +0.228 | −675 | −471 | 1 | −1,504 | −1.00 |
| M4 flags=0 (p<0.5) | 2 | 546 | +1.626 | 0 | −10 | 1 | −5,146 | −2.00 | 3 | 1,128 | +2.562 | −119 | −97 | 2 | −2,803 | −1.26 |
| M4 flags=1 (p<0.5) | 2 | 546 | +1.626 | 0 | −10 | 1 | −5,146 | −2.00 | 3 | −119 | −0.419 | −22 | −97 | 2 | −4,050 | −1.61 |
| *M4b flags=0 (p<0.145)* | 58 | 3,375 | +0.546 | −563 | −419 | 2 | −2,317 | −1.87 | 36 | 3,838 | +0.690 | −477 | −190 | 1 | −93 | −0.17 |
| *M4b flags=1 (p<0.145)* | 58 | 3,410 | +0.549 | −508 | −364 | 2 | −2,283 | −1.74 | 36 | 4,348 | +0.808 | −399 | +172 | 0 | +417 | 2.23 |

### SECONDARY arm — `asis` (D1's fill model)

| cell | TRAIN P&L | TRAIN R/pick | TRAIN d$ | VAL P&L | VAL R/pick | VAL d$ |
|---|---|---|---|---|---|---|
| **M0** | **5,029** | **+0.487** | — | **6,386** | **+0.862** | — |
| M1 flags=0 | 6,366 | +0.384 | +1,337 | 6,138 | +0.666 | −249 |
| M1 flags=1 | 6,800 | +0.419 | +1,771 | 6,220 | +0.687 | −166 |
| M2 flags=0 | 4,427 | +0.318 | −602 | 5,476 | +0.655 | −910 |
| M2 flags=1 | 5,384 | +0.371 | +355 | 4,978 | +0.614 | −1,408 |
| M3 flags=0 | 5,178 | +0.372 | +149 | 5,122 | +0.644 | −1,264 |
| M3 flags=1 | 6,075 | +0.386 | +1,046 | 4,748 | +0.539 | −1,638 |
| M4 flags=0 | 530 | +1.569 | −4,499 | 1,138 | +2.580 | −5,248 |
| M4 flags=1 | 530 | +1.569 | −4,499 | −116 | −0.419 | −6,503 |
| *M4b flags=0* | 3,190 | +0.533 | −1,839 | 4,598 | +0.886 | −1,789 |
| *M4b flags=1* | 3,243 | +0.538 | −1,786 | 5,130 | +1.008 | −1,256 |

### G1 — the TRAIN gate: **0 of 8 cells pass** (0 of 10 including the M4b pair)

G1 is "the walk-forward book on TRAIN beats M0 by ≥ **+0.10 R/pick** at t ≥ 2".
The best model cell on TRAIN is **M1 flags=1 at +0.415 R/pick — that is
−0.118 R/pick *below* M0's +0.533**, not above it. Every reordering cell is
between −0.21 and −0.12 R/pick against M0. The sign is wrong before the
magnitude or the t-statistic matter, on both arms. **G2 is therefore never
reached and TEST stays sealed.**

**What the models actually do, since "it loses" is not a mechanism.** Every
reordering cell takes **more** picks than M0 (100–108 vs 84 on TRAIN; 55–65 vs
53 on VAL) at **lower** average quality. That is structural, not incidental: the
five shipped vetoes fire POST-ranking with NO refill, so a slot M0 spends on a
candidate that is later vetoed stays empty. The model, which is rewarded for
predicted R and is (in the flags=1 cells) handed the veto flags themselves,
learns to rank veto-surviving candidates up — it converts M0's empty slots into
filled ones. The extra picks are real trades and they are mildly profitable in
dollars on TRAIN (+$254 to +$1,199, t ≤ 0.69), but they dilute R/pick, and on
VAL the whole effect reverses (−$102 to −$1,504 on every reordering cell,
t between −0.25 and −1.75). A book whose gain is "spend the slots the veto
emptied" is the refill form that ORB has already tested and rejected twice
(PDR refill: MDD +42%; the pool-widening cells of D1).

The shape numbers say the same thing more bluntly than the means: **M0 has zero
red months in both splits on the primary arm; every reordering cell has 1–3, and
every one of them has a worse MDD and a worse worst-month on TRAIN.** No cell is
close to the PREREG's ship bar (beat M0 in every split with no worse MDD or worst
month).

---

## 5. The shuffled-label control — the harness does not manufacture edge

Identical pipeline, with the training-window label vector permuted at every
monthly refit (fixed seed per month; test labels untouched, books scored on real
P&L). `cells_shuffled.csv`:

| cell (shuffled) | TRAIN R/pick | TRAIN d$ vs M0 | t | VAL R/pick | VAL d$ vs M0 | t |
|---|---|---|---|---|---|---|
| M0 reference | +0.533 | — | — | +0.483 | — | — |
| M1 flags=0 | +0.246 | −2,805 | −2.32 | +0.455 | −64 | −0.15 |
| M1 flags=1 | +0.220 | −2,866 | −1.98 | +0.483 | +406 | 0.66 |
| M2 flags=0 | +0.136 | −3,845 | −1.91 | +0.390 | −362 | −0.41 |
| M2 flags=1 | +0.177 | −3,659 | −2.32 | +0.392 | −150 | −0.26 |
| M3 flags=0 | +0.307 | −1,532 | −1.00 | +0.551 | +644 | 0.64 |
| M3 flags=1 | +0.303 | −1,000 | −0.54 | +0.544 | +963 | 1.08 |
| M4 flags=0/1 | **0 picks** | −(all of M0's) | — | **0 picks** | — | — |

**Reading, and it is the load-bearing one: no shuffled cell produces a positive
edge over M0 on either split.** On VAL — the split where the real cells are also
flat — the shuffled difference is **−$362 to +$963, abs(t) ≤ 1.08, i.e. zero**,
which is exactly what the control has to show for the rest of the report to be
readable. On TRAIN the shuffled books are *significantly worse* than M0
(−$1,000 to −$3,845, t down to −2.32): randomising the ranking destroys value,
which is the correct sign and is independent evidence that the shipped composite
carries real information. The shuffled M4 vetoes **100% of scored picks** (its
p never clears 0.5 under a shuffled label), leaving only the 78 PRE/TEST picks
the model never scored.

The real cells beat their own shuffled twins on TRAIN by +0.08 to +0.22 R/pick —
so the models **are** learning something from the features — and they still lose
to M0. That is the whole finding in one line: a gradient-boosted secondary model
extracts real but *weaker* information than the hand-built composite it was meant
to replace, and none of it survives to VAL.

---

## 6. The M4 threshold — a specification defect I am naming, not hiding

The PREREG defines M4 ("train the model ONLY to veto") without a threshold. My
harness wrote **p < 0.5**. Against a base rate of **14.5%** winners in the ranked
pool over the model-active TRAIN window, that cut is degenerate: it vetoes
**137 of 215** picks and leaves **2 TRAIN and 3 VAL** picks standing. The `M4`
rows above are reported because they are what was pre-registered and run; they
are evidence about the threshold, not about meta-labelling.

`M4b` is the repair, and it is declared as post-hoc: the cut moves to the pool's
own base rate (0.145), i.e. "veto unless the model says this pick beats an
average pick's chance of winning". It is **counted in the cell count and is not
gate-eligible**. Its result is the same verdict anyway: TRAIN **+0.546 / +0.549
R/pick vs M0's +0.533** — a +0.014 R/pick difference, one seventh of the G1
threshold and one thirty-sixth of the split's MDE — bought by giving up
**$2,283–$2,317 of TRAIN P&L (t −1.7 to −1.9)**, 26 picks, and a worst month that
goes from +$72 to −$364. On the secondary arm it is worse on both splits. A veto
that removes 31% of the book to move R/pick by 0.014 is a capacity cut, not an
edge.

---

## 7. Per-feature-family ablation

Drop one family from the feature matrix, refit the whole walk-forward, rebuild
the book; the table is the change in R/pick vs the same cell with all features
(`ablation.csv`, 6 cells × 5 families × 2 splits = 60 cells).

| family dropped | mean Δ R/pick TRAIN | mean Δ R/pick VAL | worst single cell (VAL) |
|---|---|---|---|
| opening_range (10 fields) | **−0.109** | **+0.027** | −0.151 |
| calendar (2) | −0.077 | +0.051 | +0.157 |
| market — SPY gap, SPY 3-day range (2) | −0.059 | +0.027 | −0.033 |
| prev_day (4) | −0.048 | +0.025 | −0.036 |
| twenty_day (4) | −0.040 | −0.014 | −0.141 |

Every family contributes on TRAIN (removing it costs R/pick — that is fitting,
not evidence) and **four of five families are worth ≈ 0 or slightly NEGATIVE out
of sample**: dropping them *improves* VAL R/pick on average. No family moves VAL
by more than 0.05 on average, against a VAL MDE of 0.815 R/pick. The purpose of
this table in the PREREG was to stop a single leaky column hiding inside an
aggregate — it shows no such column: there is no family whose removal collapses
the result, because there is no result to collapse. The largest TRAIN
contributor is the opening-range block, which is also the block the shipped
composite already uses.

---

## 8. TEST is sealed, and stayed sealed

No `FREEZE.md` was written, because nothing passed G1, let alone G2.
`run_study.py test` refuses to run without one. Every walk-forward in this report
was invoked with `--last-month 2026-05`, so **no model ever produced a score for
a 2026-06+ session** and no model cell has a TEST book. Two disclosures, because
the seal is about outcomes and I want the record exact: (i) M0's own TEST numbers
appear in my run output (215-pick book, TEST $986 on the primary arm) — those are
the *shipped* book's, already published in `Q_fill/REPORT.md`, and no comparison
was made against them; (ii) the availability audit and the first MDE run printed
TEST-split *dispersion and coverage* statistics (R standard deviation, the SPY
sentinel rate) — no P&L decision was taken on them, and `mde.py` has since been
restricted to TRAIN/VAL.

---

## 9. Minimum detectable effect

At 80% power, two-sided 5% (z = 2.802), on M0's own book — `mde.py`:

| split | picks | R sd | **MDE per pick** | days | daily sd | **MDE per day** | over the split |
|---|---|---|---|---|---|---|---|
| PRE (2025-01..03) | 21 | 1.18 | 0.723 R | 18 | $167 | $110 | $1,985 |
| TRAIN (2025-04..12) | 84 | 1.65 | **0.505 R** | 53 | $362 | $139 | **$7,384** |
| VAL (2026-01..05) | 53 | 2.12 | **0.815 R** | 42 | $392 | $169 | **$7,112** |

(secondary arm: 0.495 R TRAIN, 1.082 R VAL — the `asis` VAL book is even more
tail-dominated.)

**The smallest per-pick effect this study could have resolved is ~0.5 R on TRAIN
and ~0.8 R on VAL.** G1's own bar (+0.10 R/pick) is *below* that — the gate is
stricter than the test is powerful, so a cell could in principle have cleared G1
on TRAIN by chance without the effect being resolvable. That is moot here: no
cell cleared it in the right direction. It is the honest statement of what this
population can support: **at 84 and 53 picks per split, an 8-slot ORB book cannot
see a selection improvement smaller than half an R per pick.** Nothing in this
study — positive or negative — should be read as evidence about effects below
that size. The measured differences (−0.12 to −0.21 R/pick against M0) are of the
same order as the MDE, so even the *negative* result is "not detectably better,
plausibly somewhat worse", not "proven worse".

---

## 10. Cells looked at

| group | cells |
|---|---|
| availability audit | 24 fields × (coverage, entered/no-fill, 3 splits) — descriptive, 0 search |
| **grid selection (the only SEARCH cells)** | 3 models × 2 flag-sets × 4 configs = **24**, all on TRAIN only |
| the 9 cells | 9 × 2 arms × 2 splits = 36 |
| M4b sensitivity (post-hoc, non-gating) | 2 × 2 arms × 2 splits = 8 |
| shuffled-label control | 8 × 2 arms × 2 splits = 32 |
| ablation | 6 cells × 5 families × 2 splits = 60 |
| **total reported** | **160 config × split cells, of which 24 were search cells** |

No veto threshold, z-param, quintile cutoff, adaptive mult, slot count or
`orb.yaml` value was tuned or moved. Sizing is identical in every cell by
construction (the $3,333.33 per-position cap binds on 100% of picks).
Permutation p and the tail tests (ex-top-1%/5%, +3R cap) are not reported:
PLAN §1 computes them for cells clearing G2, and none did.

---

## 11. Verdict — in PLAN §1 phrasing

> **In THIS universe (the 13,033 entered-inclusive ORB candidates of 2025-01-02
> → 2026-09-16, 5,771 of them inside the shipped ranked pool), at THIS horizon
> (a single 09:35 decision per candidate with the shipped static-lock exit), at
> THIS book size (8 slots, $10K stage, a per-position cap that binds on 100% of
> picks), over THIS window (a purged, embargoed, 26-week rolling walk-forward
> over 14 monthly refits), at THIS cost (Stage Q's measured capped-limit fill and
> measured per-trade NBBO), no gradient-boosted secondary model — classifier,
> pairwise ranker, R-regressor, or pure meta-label veto, with or without the five
> shipped veto flags as inputs — was detectably better than the shipped composite
> + quintile + five-veto stack. 0 of 8 pre-registered cells passed G1; the best
> was 0.118 R/pick BELOW M0, with the sign wrong before the magnitude. The
> smallest per-pick effect the test could have seen is 0.505 R on TRAIN and
> 0.815 R on VAL, so this is "not detectably better, plausibly somewhat worse",
> never "no edge exists". TEST was never read.**

Three things this does and does not say:

* **It does not say the features are worthless.** The models beat their own
  shuffled twins on TRAIN by +0.08 to +0.22 R/pick. There is learnable structure
  in the 22 causal features; the hand-built composite is simply extracting more
  of it, and D1 already showed that refitting *that* selection weekly pays.
* **It does not say meta-labelling is refuted.** One population, one book size,
  one model family, one exit spec, 84 + 53 picks. Given the prior in §0 — no
  independent OOS equity evidence for the technique — a null was the expected
  outcome and this is one more null, not a refutation.
* **It does say the binding constraint is not the algorithm.** Every cell's
  headline effect is smaller than the split's own MDE. At 8 slots and ~0.6 picks
  per trading day this book cannot resolve a selection improvement below half an
  R per pick, so the next honest lever is *more picks* or *more window*, not a
  better secondary model. That is the same conclusion `research/bf_zero2` reached
  from the other end.

**Recommendation: ship nothing.** The meta-label hook in
`study_orb_pipeline_static_lock.py` (`ORB_META_RANK_COL`, `ORB_META_VETO_COL`,
`ORB_META_VETO_THR`) is default-inert, verified byte-identical to D1 with the env
unset, and is left in place as research instrumentation — it is the seam any
future selection experiment plugs into without rewriting the pipeline. No
`orb.yaml` change, no engine change, no weekly-refit change is proposed;
`scripts/orb_weekly_refit.py` keeps refitting the selection exactly as it does
today, which remains the only selection lever with evidence behind it.

---

## 12. Where this could be wrong

1. **Power.** §9 — the study cannot see anything below ~0.5 R/pick. A real +0.2
   R/pick meta-label would have been invisible here.
2. **One book size.** Everything is measured at 8 slots. A model that only pays
   at 12+ slots (where D1 shows marginal picks going to +0.042 R) was not tested,
   and the slot count was fixed before the run, not searched.
3. **The label is the candidate's own R, not the book's.** A ranker optimising
   per-candidate R is not optimising the day's 8-slot outcome under family /
   super-group dedup and the no-refill vetoes. M2's per-day grouping is the
   closest approximation available and it is not the same objective.
4. **The no-refill interaction.** The single clearest model behaviour — filling
   slots M0 leaves empty — is an interaction with the veto layer, not a
   prediction quality. A design that froze the *number* of picks would measure
   selection more cleanly and is not what was pre-registered.
5. **14 monthly refits.** The walk-forward has 14 out-of-sample months, 9 of them
   in TRAIN. Config selection consumed TRAIN, so TRAIN is not clean for the
   chosen cells; VAL is, and VAL is where every cell is negative.
6. **The dropped SPY pair.** Dropping two features on a coverage argument is a
   judgement. It cannot have created the null (they are 100% sentinel on TEST and
   52% on VAL, so keeping them would have fed the model an era marker), but it is
   a deviation from "all 24" and is named as one.
7. **Two arms.** Reporting both `meas_cost` and `asis` doubles the display. The
   verdict is identical on both, which is the point of showing them, but the arm
   was not a pre-registered cell dimension.

## 13. Files

```
research/meta_label/
  PREREG.md                 the pre-registration (written before any fit)
  REPORT.md                 this file
  audit_availability.py     the feature-availability audit -> availability_audit.csv
  build_dataset.py          22 features + 5 shipped veto flags + R labels (2 arms)
                            -> meta_dataset.csv
  walkforward.py            the purged, embargoed 26-week walk-forward -> scores/*.csv
  run_book.sh               ONE pipeline book run (8 slots, $3,333.33 per-pos cap)
  run_study.py              select | cells | shuffle | ablate | m4thr | test
  analyze.py                book stats in Stage Q's convention + the day-paired test
  mde.py                    minimum detectable effect (TRAIN/VAL only — TEST sealed)
  grid_train.csv chosen_config.json cells.csv cells_shuffled.csv cells_m4b.csv
  ablation.csv  books/ logs/ scores/
study_orb_pipeline_static_lock.py   + ORB_META_RANK_COL / ORB_META_VETO_COL /
                                      ORB_META_VETO_THR (default-inert; M0 parity
                                      verified to the cent with them compiled in)
```
