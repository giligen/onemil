# Meta-labelling the ORB selection — pre-registered 2026-09-18 20:30 UTC (before any model is fit)

## Why this and not a time-series foundation model
Owner asked whether TSFM (TimesFM / Chronos / Moirai / TimeGPT) or another SOTA algorithm helps. Honest answer:
not on the price series. Those models are trained and benchmarked on smooth, strongly autocorrelated series
(electricity, traffic, retail) and are scored on MAE/MAPE of the LEVEL, where "tomorrow = today" is already
near-optimal for a stock and carries exactly zero trading edge. Our binding constraints this session were never
forecast accuracy: they were population (the HOD look-ahead), cost (the band table, N3), fill realism (Stage Q),
and leakage (D1's premarket-$ availability leak). None of those move with a better sequence model.

The problem we DO have is a **selection** problem with a documented shape: the raw ORB breakout has no edge
(2025 −0.18R, 2026 −0.04R), the Stage-2 selection stack IS the edge (+0.61R picks in 2025) and it decayed to
+0.06R in 2026. That is exactly the setting meta-labelling addresses (López de Prado 2018, *Advances in Financial
Machine Learning*, ch. 3): a primary rule generates events, a secondary model decides which events to take and at
what size. Our shipped vetoes (Q1, PDR, range-size, catalyst, touchgo) and the composite z-score ARE a
hand-built meta-labeller; this tests the principled version against it.

**Prior support in our own data**: `scripts/orb_weekly_refit.py` — refitting the SELECTION on a rolling 26-week
window beats frozen params ($7,588 vs $5,669, MDD −223 vs −551, red months 4 vs 6, robust across windows ≥ 20w).
Fitting the selection better has already been shown to pay here. That is the only reason this study is worth the run.

## Data
`research/fuckup_audit/D1_orb/candidates_dump.csv` — 13,033 entered-inclusive ORB candidates, 2025-01-02 →
2026-09-16, 7,402 fills + 5,631 modelled non-fills, with the 24 pre-entry features the live engine computes at
09:35. Label: the candidate's own realised R under the shipped exit spec (static lock 1.75R → +0.5R, 15:45 flat),
from the same pipeline that produces `book_n8_q1on.csv`. Costs and fills: Stage Q's measured model (capped limit,
`ask > cap ⇒ no fill`, measured per-trade NBBO mean). No feature computed after 09:35:00 may enter (availability
audit asserts every timestamp), and `pit_listings.py` excludes test tickers and non-`daily_bars` names.

## Models (declared; 13K rows × 24 features is gradient-boosting territory, not transformer territory)
- **M0 baseline**: the shipped composite z-score + quintile cutoffs + the five shipped vetoes = today's book.
- **M1 classifier**: gradient-boosted trees predicting P(R > 0), picks ranked by that probability.
- **M2 ranker**: LambdaMART / pairwise learning-to-rank on the per-day candidate list (the slot problem IS a
  ranking problem — we take the top 8 of each day, not an absolute threshold).
- **M3 regressor**: GBT on realised R, ranked by expected R.
- **M4 meta-label on the shipped rule**: keep the shipped ranking, train the model ONLY to veto (take / don't take
  the pick the current stack already selected) — the narrowest, most honest form.
Each × {with, without} the five shipped vetoes as input features = **8 cells** plus M0 = 9. No hyper-parameter
search beyond a single declared grid (depth 3/5, lr 0.05, 200/500 trees, min_child 20) selected on TRAIN only.

## Validation — the leakage rails, which are the whole point
D1's +0.415R was an availability leak, and a model finds leaks faster than a human does. Therefore:
- **Purged, embargoed walk-forward** (López de Prado ch. 7): train on a trailing window, predict the next month,
  purge any training row whose outcome window overlaps the test month, embargo 5 sessions either side. No random
  K-fold, ever.
- **Splits**: TRAIN 2025-01→2025-12, VAL 2026-01→2026-05, TEST 2026-06→2026-09 (sealed; opened once behind
  `FREEZE.md` for G2 survivors only).
- **Gates** (PLAN §1): G1 the walk-forward book on TRAIN beats M0 by ≥ +0.10R/pick at t ≥ 2; G2 VAL same sign and
  ≥ 55% weeks green AND the 8-slot book's MDD and worst month no worse than M0's; TEST once.
- **Feature-availability audit** on every column; **permutation p** across the 9 cells; **tail tests** (ex-top-5%,
  +3R cap); **ablation**: drop each feature family and report the change, so a single leaky column cannot hide.
- **Shuffled-label control**: the identical pipeline on shuffled labels must produce ≈ 0 edge. If it does not, the
  harness is broken and nothing else in the report is readable.
- Capacity and sizing unchanged (per-position cap binds on every pick); the model may reorder and veto, never resize.

## Ship bar and what a survivor costs to run live
A survivor must beat M0 on the 8-slot book in every split, with no worse MDD or worst month, and must be
re-fittable weekly by the same cron that already refits the selection. Live cost: the model scores 20–40
candidates once a day at 09:35 on 2 CPUs — milliseconds, no GPU, no new dependency beyond the boosting library.
A survivor still needs the independent rebuild from prose before it touches `orb_engine.py`.

## Deliverable
`research/meta_label/REPORT.md`: the 9-cell table per split vs M0, the shuffled-label control, the ablation, the
MDE, the cell count, and the verdict in the PLAN §1 phrasing. Runs AFTER the multi-day data stage finishes — one
heavy job at a time on this node.
