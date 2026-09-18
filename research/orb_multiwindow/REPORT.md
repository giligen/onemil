# ORB multi-window (W = 5 / 15 / 30) — REPORT

**STATUS: COMPLETE. Verdict: NO GO. Nothing cleared G1 on TRAIN, so VAL was informational only
and the TEST split was never scored — no `FREEZE.md` was written and none was needed.**

Run 2026-09-18 16:25→19:05 UTC on the live node, one python process at a time, `nice -n 10`,
`data/cache.db` read-only. `orb.yaml`, `config.yaml`, the service, orders and crons untouched.
All bulk output under `research/orb_multiwindow/` (gitignored). Pre-registration: `PREREG.md`,
written before any bar was scanned.

---

## 0. Step 0 — the reproduction gate (PASSED, three times)

| | picks | P&L | key set | max |Δ| per pick |
|---|---|---|---|---|
| `research/fuckup_audit/D1_orb/book_n8_q1on.csv` (reference) | 215 | $14,428.62 | — | — |
| this tree, **modified** code, W=5 | 215 | **$14,428.62** | identical | $0.00 |

Re-run after every edit (`repro_w5_n8.sh`): after the `ORB_RANGE_MINUTES` plumbing, after the
refit hook, after the memory change. Features `analysis_results/orb_features_20260916_2053.csv`,
resim cache `D1_orb/candidates_dump.csv`, `ORB_BT_N=8`, `ORB_BT_ACCOUNT=26666.666666666664`,
`ORB_BT_RISK=375`, `ORB_SKIP_Q1=1`, everything else from `orb.yaml`.

**Feature-path half of the gate**: the modified `study_orb_features.py` at W=5 was re-run over
2026-09-08..09-16 into a side dir and compared to the production CSV row by row —
181/181 rows, identical key set, **31/31 columns byte-identical** (`w5check/`).

**Combiner parity**: `combine.py` with ONE window reproduces the pipeline's own 8-slot book
exactly — 215 picks, $14,428.62, identical keys, and the same per-veto counts the pipeline
printed (pdr 1860 / g1 197 / range-size 69 / catalyst 403). The multi-window book builder is
therefore the shipped slot+veto mechanics, not a re-implementation of them.

## 1. The code change

* `study_orb_features.py` — `RANGE_MINUTES` reads `ORB_RANGE_MINUTES` (default 5) and flows into
  the range window `[09:30, 09:30+W)`, `simulate_orb_trade(range_minutes=)`, the SPY block, and
  the `FEATURES_CODE_VERSION` stamp (`2026-09-05.entered_inclusive.W15`). The stamp's sidecar and
  the incremental-source glob moved from a hardcoded `analysis_results/` to `FEATURES_OUT_DIR`,
  so a research window can never append to — or be seeded from — the production W=5 CSV;
  `W != 5` with `ORB_FEATURES_OUT_DIR` unset is a hard refusal.
  The SPY columns keep their legacy `_5min` names but cover `[09:30, 09:30+W)`; neither is in
  `FILTER_FEATURES`, so selection is untouched by that.
* `study_orb_pipeline_static_lock.py` — the exit re-simulation rebuilds `range_high`/`range_low`
  over the same `ORB_RANGE_MINUTES` window (was a hardcoded 5 in two places). New
  `ORB_BT_REFIT_ZPARAMS` + `ORB_BT_TRAIN_START/END` force the TRAIN refit of z-params and
  quintile cutoffs — the `scripts/orb_weekly_refit.py` procedure exactly (`fit_z_params` +
  `fit_quintile_cutoffs`; `filter.threshold` and `adaptive_mults` never touched) — because the
  W=5 frozen fit is meaningless on a different range width.
* Both files now build the bar cache by popping the raw dict instead of holding both copies
  (peak memory on a 7.8 GB box shared with the live trader). Behaviour-neutral; the gate proves it.
* Every default is the production value. `ulimit -v 3,000,000` is **not** enough for
  `load_daily_bars_frame` (5.0M `daily_bars` rows) — the feature stage ran at 4,500,000,
  PLAN §1's own sanctioned ceiling for a pass-1 rebuild. Peak RSS stayed under 5.4 GB.

## 2. Population and availability, per window

The universe is IDENTICAL for all three windows (the same 13,796 `(symbol, date)` pairs from
`study_orb_broad.load_broad_universe`), so nothing below comes from a universe change.

| W | feature rows | % of universe pairs | entered (breakout fired) | days | NaN in any veto/selection feature |
|---|---|---|---|---|---|
| 5 | 13,033 | 94.5% | 7,402 (56.8%) | 427 | 0.0% |
| 15 | 12,125 | 87.9% | 5,312 (43.8%) | 428 | 0.0% |
| 30 | 11,072 | 80.3% | 3,978 (35.9%) | 428 | 0.0% |

The row loss is the detector's own precondition — a candidate needs W complete 1-min bars in
`[09:30, 09:30+W)`. It is **causal** (bars strictly before the decision instant) but it is a
real, disclosed coverage difference: a wider window drops 6–14% more symbol-days.
Every feature used by the composite or by a veto has 0% missingness in all three windows, so the
D1 availability rule (missingness table before use) is satisfied with nothing to report.

**Economics differ per window even at identical slot sizing.** The $3,333 per-position cap binds
on 100% / 95% / 88% of picks at W = 5 / 15 / 30, and the realized dollar risk per pick is
$148 / $214 / $260 (median `range_size_pct` 4.10 / 6.06 / 7.68). R-per-pick is comparable across
windows; dollars are not.

## 3. The range-size veto, re-derived per window

Same procedure as `research/orb_veto_study/DESIGN.md` — worst quintile of the ENTERED raw
candidates, required to be worst in BOTH years, threshold = that quintile's upper edge. Raw R is
the detector's own (features-CSV fixed +2R/−1R proxy); `range_low` is recovered exactly from the
range columns (`derive_range_size_veto.py`).

| W | Q1 edge | raw R by quintile, 2025 | raw R by quintile, 2026 | worst in both? | threshold used |
|---|---|---|---|---|---|
| 5 | 2.257 | −0.339 / −0.207 / −0.128 / −0.003 / −0.078 | −0.211 / −0.035 / +0.120 / +0.053 / −0.011 | yes (Q1) | **2.221** (the shipped live value; the re-derivation lands at 2.257 on the newer CSV — a 1.6% drift, the rule reproduces) |
| 15 | 2.983 | −0.245 / −0.222 / −0.050 / −0.023 / −0.013 | −0.059 / +0.013 / +0.141 / +0.026 / −0.007 | yes (Q1) | **2.983** |
| 30 | 3.539 | −0.170 / −0.136 / −0.103 / −0.013 / +0.040 | −0.175 / −0.037 / +0.069 / −0.001 / +0.001 | yes (Q1) | **3.539** |

The rule transfers in FORM to every window (the tightest opening ranges are the worst bucket in
both years at W=15 and W=30 too) but the LEVEL does not — inheriting 2.221 at W=30 would veto
almost nothing. This is why the PREREG required the re-derivation.

## 4. Picks and overlap — 8 shared slots

| book | picks | vs W=5 | shared with the W=5 book | added picks | by window |
|---|---|---|---|---|---|
| W=5 (reference, TRAIN-refit) | 218 | — | 218 | 0 | 5:218 |
| W=15 alone | 192 | −11.9% | 102 | 90 | 15:192 |
| W=30 alone | 160 | −26.6% | 81 | 79 | 30:160 |
| **5+15** | 235 | **+7.8%** | 218 | **17** | 5:218, 15:17 |
| **5+30** | 235 | **+7.8%** | 218 | **17** | 5:218, 30:17 |
| **5+15+30** | 242 | **+11.0%** | 218 | **24** | 5:218, 15:17, 30:7 |
| 5+15+30 @ 12 slots (sensitivity, not gated) | 324 | +48.6% | 218 | 106 | 5:268, 15:40, 30:16 |

Overlap skips under the pre-registered rule (symbol already ordered by an earlier window):
488 (5+15), 429 (5+30), 844 (5+15+30), 1,986 (@12 slots). **The later windows are mostly the same
names on the same days** — 53% of the W=15 book and 51% of the W=30 book are symbol-days the
5-minute book already owns. That, not the slot cap, is why a second window adds ~8% more picks
and not 30%: the 8-slot cap binds on at most 4–5 picks a day, so the frequency ceiling here is
the post-veto candidate pool, not the slots.

Cross-window family collisions kept (the pre-registered overlap rule is symbol-level, so a
same-family pair from two windows is allowed): 14 / 12 / 20 / 33. Disclosed, not corrected.

## 5. The six cells, per split — TRAIN and VAL

R/pick is `_sized_pnl / (_rp_position × max(range_size_pct,1)/100)`; a no-fill books 0 R and still
counts as a pick. `ADDED` rows = the picks that are NOT in the W=5 book — **the gated quantity.**

| book | split | tag | picks | fills | P&L | R/pick | t | MDD | worst mo | red mo | % weeks green |
|---|---|---|---|---|---|---|---|---|---|---|---|
| W5 | TRAIN | | 109 | 82 | +6,208 | +0.426 | 2.98 | −609 | −152 | 2 | 45.0 |
| W5 | VAL | | 51 | 39 | +5,403 | +0.719 | 1.89 | −597 | −146 | 1 | 33.3 |
| W15 | TRAIN | | 86 | 54 | −1,542 | −0.069 | −0.92 | −2,116 | −592 | 8 | 22.0 |
| W15 | VAL | | 57 | 38 | +3,241 | +0.355 | 2.43 | −462 | −457 | 1 | 33.3 |
| W15 | TRAIN | **ADDED** | 32 | 20 | −1,544 | **−0.220** | **−2.75** | −1,306 | −300 | 10 | 4.2 |
| W15 | VAL | ADDED | 32 | 25 | +2,454 | +0.541 | 2.32 | −420 | −254 | 2 | 35.7 |
| W30 | TRAIN | | 91 | 54 | −1,139 | −0.049 | −0.93 | −2,324 | −927 | 8 | 23.3 |
| W30 | VAL | | 34 | 20 | −193 | −0.018 | −0.28 | −622 | −304 | 4 | 17.6 |
| W30 | TRAIN | **ADDED** | 41 | 24 | −1,255 | **−0.114** | **−1.54** | −1,631 | −405 | 8 | 10.3 |
| W30 | VAL | ADDED | 20 | 14 | +58 | +0.018 | 0.17 | −514 | −239 | 3 | 25.0 |
| **5+15** | TRAIN | | 119 | 89 | +5,738 | +0.369 | 2.78 | −665 | −152 | 3 | 45.2 |
| **5+15** | VAL | | 58 | 43 | +5,872 | +0.692 | 2.05 | −621 | +208 | 0 | 38.9 |
| **5+15** | TRAIN | **ADDED** | 10 | 7 | −470 | **−0.254** | **−1.87** | −338 | −238 | 5 | 10.0 |
| **5+15** | VAL | ADDED | 7 | 4 | +469 | +0.497 | 1.07 | −232 | −208 | 2 | 16.7 |
| **5+30** | TRAIN | | 124 | 91 | +5,386 | +0.345 | 2.70 | −902 | −152 | 4 | 43.2 |
| **5+30** | VAL | | 52 | 39 | +5,403 | +0.705 | 1.89 | −597 | −146 | 1 | 33.3 |
| **5+30** | TRAIN | **ADDED** | 15 | 9 | −822 | **−0.240** | **−2.51** | −682 | −367 | 5 | 7.1 |
| **5+30** | VAL | ADDED | 1 | 0 | 0 | 0.000 | — | 0 | 0 | 0 | 0 |
| **5+15+30** | TRAIN | | 125 | 91 | +5,496 | +0.343 | 2.70 | −777 | −152 | 3 | 44.2 |
| **5+15+30** | VAL | | 58 | 43 | +5,872 | +0.692 | 2.05 | −621 | +208 | 0 | 38.9 |
| **5+15+30** | TRAIN | **ADDED** | 16 | 9 | −711 | **−0.224** | **−2.43** | −580 | −367 | 6 | 7.1 |
| **5+15+30** | VAL | ADDED | 7 | 4 | +469 | +0.497 | 1.07 | −232 | −208 | 2 | 16.7 |
| 5+15+30 @12 | TRAIN | | 166 | 119 | +6,026 | +0.266 | 2.59 | −1,122 | −420 | 4 | 39.2 |
| 5+15+30 @12 | VAL | | 77 | 59 | +6,144 | +0.546 | 2.07 | −589 | +47 | 0 | 50.0 |
| 5+15+30 @12 | TRAIN | ADDED | 57 | 37 | −181 | −0.040 | −0.36 | −909 | −380 | 8 | 23.1 |
| 5+15+30 @12 | VAL | ADDED | 26 | 20 | +740 | +0.208 | 0.86 | −545 | −265 | 3 | 23.1 |

(`cells_trainval.csv`. The frozen-parameter W=5 book — the live selection, not refit — is
215 picks / $14,428.62 / 2 red months; the TRAIN-refit reference used above is 218 / $14,065 /
4 red months, so the refit procedure is not what moves these numbers.)

### The gate, applied

* **G1 (TRAIN, ADDED picks, mean net R > 0 with t ≥ 2.0): FAILS IN EVERY CELL.** Every one of the
  six added-pick sets is NEGATIVE on TRAIN (−0.040 to −0.254 R/pick), and the three with |t| ≥ 2
  are significantly negative. Nothing advances.
* **Ship bar, independently: also fails.** +7.8% / +7.8% / +11.0% picks against a +30% bar, and
  R/pick of the added picks ≥ +0.30 on *every* split is violated on TRAIN in every cell. MDD and
  worst month of the combined books are worse than the 5-minute book's on TRAIN
  (−665/−902/−777 vs −609) as well.
* **G2/G3 never opened.** VAL is printed above for completeness only — no cell earned it, and
  the positive VAL added-pick numbers (+0.541 at W=15) are exactly the shape the TRAIN-first
  ordering exists to refuse. **TEST was never scored.**

### Power — what this test could have seen

| cell | n (TRAIN added) | mean R | sd | SE | smallest effect detectable at t=2 |
|---|---|---|---|---|---|
| W15 | 32 | −0.220 | 0.453 | 0.080 | +0.160 R |
| W30 | 41 | −0.114 | 0.473 | 0.074 | +0.148 R |
| 5+15 | 10 | −0.254 | 0.429 | 0.136 | +0.272 R |
| 5+30 | 15 | −0.240 | 0.371 | 0.096 | +0.191 R |
| 5+15+30 | 16 | −0.224 | 0.369 | 0.092 | +0.185 R |
| 5+15+30 @12 | 57 | −0.040 | 0.830 | 0.110 | +0.220 R |

The ship bar asks for +0.30 R/pick. In every cell the minimum detectable effect at t = 2 is
BELOW +0.30, so this test had the power to see the effect the ship bar requires. It did not see
it; it saw the opposite sign.

## 6. The honesty checklist

* **Independent reimplementation.** The multi-window book builder (`combine.py`) is an
  independent replay of the slot + veto mechanics that reproduces the pipeline's own 8-slot book
  trade-for-trade and to the cent. The feature and exit code are the SHIPPED modules with one
  new constant; the W=5 identity of both (feature rows, and the D1 book) is the check.
  A second, from-prose rebuild was **not** commissioned — it is required before an engine is
  armed, and nothing here is a survivor.
* **Obtainability.** Entry is the same order in every window: a stop-limit capped at
  `range_high × 1.003`. Stage Q measured that the ask exceeds that cap at the trigger on 14.4%
  of fills (10.3% against the rounded cap the engine actually sends), and that 92% of those
  still fill, at the cap, median 21 s later. Applying the strict arm (ask > cap ⇒ $0, slot spent)
  cannot rescue any cell: for 5+15's TRAIN added picks (10 picks, 7 fills, sum −2.54R), reaching
  +0.30 R/pick by deleting the flagged ~14% of fills would require the deleted fill to have been
  **−5.2 R**, and the per-trade floor is about −1 R plus slippage. Same arithmetic at W=15 alone
  (three deletions would each have to be −5.2 R). Obtainability is not the binding constraint
  here, and no quote walk was purchased.
* **Causality.** Every feature is computed from bars in `[09:30, 09:30+W)` or from daily bars
  strictly before the trade date; the universe is the same prev-day gap/volume/price screen for
  all windows (no window-specific membership); the z-params and quintile cutoffs are fit on TRAIN
  (2025) only and applied forward; the range-size thresholds are fit on both years of RAW
  candidates before any book was scored, as the V1 study did.
* **Price scale.** Unchanged from the shipped book — intraday 1-min and daily bars both from the
  Alpaca cache (`data/cache.db`), the same two tables the production ORB book uses. No
  Databento/Alpaca cross-source comparison was introduced by this study.
* **Tails** (whole-window R/pick, top 1% / top 5% removed / winners capped at +3R):
  W5 +0.371 / +0.151 / +0.282 · W15 +0.038 / −0.085 / +0.068 · W30 −0.049 / −0.111 / −0.019 ·
  5+15 +0.348 / +0.127 / +0.265 · 5+30 +0.328 / +0.106 / +0.245 · 5+15+30 +0.333 / +0.118 / +0.253.
  The added windows are tail-dependent on their own (W=15 alone goes negative ex-top-5%) and
  every combined book is tail-WORSE than the 5-minute book it is meant to improve.
* **Multiplicity.** 6 pre-registered cells + the W=5 reference in two parameterisations
  (frozen, TRAIN-refit) + 3 range-size derivations = **11 cells looked at**, 0 thresholds
  searched (the range-size levels are the V1 procedure's output, not a sweep). No permutation
  adjustment was computed because the best of the six gated statistics is negative — a
  search-adjusted p can only move it further from significance. Cumulative ORB-line cell count
  including D1 (108 reported cells) and Stage Q (8 fill arms × 2 slot counts): this stage adds 11.
* **Known leak, disclosed.** The per-window pipeline logs print a whole-window monthly table and
  a whole-window total, which include the 2026-06+ months. No TEST-split statistic was computed,
  and the gate was decided entirely on TRAIN. The three aggregate totals that passed my eye
  (W=15 $+1,208, W=30 $−1,196, 5+15 $+14,064) are recorded here for the same reason.

## 7. Verdict

**No edge was detectable in THIS universe (the 13,796 gap-up symbol-days the ORB broad screen
admits, 2025-01-02 → 2026-09-17), at THIS horizon (a 15- or 30-minute opening range with the
same 60-minute entry window), at THIS book size (8 shared slots, $10K-stage sizing, ~$3,333 per
position), over THIS window (21 months, TRAIN 2025 + VAL 2026-01..05), at THIS cost (30 bps
entry, 10 bps exit, the shipped static-lock/touchgo/winner-stack exits).** The smallest effect
the TRAIN test could have seen is +0.15 to +0.27 R per added pick; the observed effect is
−0.04 to −0.25 R.

Two reasons, both visible above and both worth keeping: (1) **the later windows are not a
different population** — about half of what they select is a symbol-day the 5-minute book
already owns, so the frequency lever delivers +8%, not +30%; (2) **what is genuinely new is
worse** — the added picks lose on TRAIN in every configuration, the 15/30-minute books are red
in 8 of 12 TRAIN months standalone, and they are more tail-dependent than the book they would
be bolted onto. The 12-slot sensitivity cell is the one that buys real frequency (+48.6%) and it
buys it at −0.040 R per added pick with the drawdown nearly doubled.

**Recommendation: close the multi-window line for ORB. Do not build the second submission
burst.** The owner's question — "multiply ORB's frequency" — is not answered by more entry
windows on this universe; D1 already showed the same thing one layer down (the selection edge is
gone by rank 9). If frequency is still the goal, the honest remaining levers are a wider
UNIVERSE (the gap/volume screen, not the range clock) or a different book, and both are new
pre-registrations.

## 8. Artefacts

```
research/orb_multiwindow/
  PREREG.md  REPORT.md
  repro_w5_n8.sh              step-0 gate (pipeline path)
  spotcheck_w5.sh + w5check/  step-0 gate (feature path), 181 rows x 31 cols identical
  run_window.sh               features -> bar walk -> selector, per W
  derive_range_size_veto.py   the V1 procedure, re-run per window
  combine.py                  multi-window slot+veto book (parity-tested against D1)
  analyze.py                  the split/R/MDD/tail table (--no-test rail)
  w5/ w15/ w30/               features, candidate dumps, ranked dumps, per-window books, logs
  book_w*.csv                 the six cells
  cells_trainval.csv          the table in §5
```
