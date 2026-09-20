# ORB LONG — frequency without touching selection. Cells F1a / F1b / F2-10 / F2-15 (1,274–1,277)

PREREG committed first (c64da1e); FREEZE.md carries the hash + input md5s. **TEST (≥ 2026-06-01)
never scored.** Baseline `analysis_results/orb_bplus_book.csv` — 218 ranked picks, 165 fills,
8 shared slots, $10K stage. **Reproduction gate: my exit re-sim reproduces all 165 filled book
rows' `pnl_pct` to max |Δ| = 0.000000** (`f1.py`), so every delta below is apples-to-apples.

## 1. The no-fill split — and it kills the premise of F1a/F1b

Range = [09:30,09:35); breakout bar = first bar in [09:35,10:35) with high > range_high;
Cap-30 = range_high × 1.0030.

| class | all | TRAIN | VAL |
|---|---|---|---|
| (i) range_high **never broken** in 60 min | **53 (24.3%)** | 21 | 13 |
| (ii) broken, breakout bar **opened above Cap-30** (gap-through) | **4 (1.8%)** | **0** | 2 |
| (iii) broken, opened at/below Cap-30 — legitimate fill | 161 (73.9%) | 84 | 38 |

Both consequences were declared in PREREG §3 *before* scoring:

1. **Every `no_fill` row in the shipped book is class (i).** The BT entry model
   (`study_orb_features.py:636` → `simulate_orb_trade(entry_mode='touch')`) fills on ANY bar whose
   high exceeds range_high, at range_high × 1.003, **with no cap check**. The no-fill mass is not
   "the limit was too tight" — it is "the 5-min high was never touched in the next hour".
   **The cap is not the frequency constraint. The trigger is.**
2. The book is mildly OPTIMISTIC on the 4 class-(ii) picks. Honest-30bps baseline **B30** reprices
   them: VAL $6,387.04 → **$6,487.19** (+$100.15; MST 2026-02-13 −3.00% → $0); TRAIN unchanged.
   The optimism *cost* money — it was filling a loser.

**Independent confirmation from quotes:** `orb_multiwindow/REPORT.md` §6 (Stage Q, 9/18) measured
NBBO at the trigger — ask > cap on **14.4%** of fills, **92% of those still fill at the cap, median
21 s later** ⇒ ~1.2% truly lost. My bar-based 1.8% agrees in magnitude.

## 2. F1a — widen the cap to 60 bps on gap-through picks

After B30 the only unfilled picks are the two that opened above Cap-60 too (MST 87.2 bps;
AAOG 65.4 bps, TEST). WNW (45.1) and AAOX (33.7) already fill under B30 — Cap-60 only makes their
fill *worse*.

| split | added | net R | t | ex-top5% | fills/wk | stacked $ | stacked MDD |
|---|---|---|---|---|---|---|---|
| TRAIN | **0** | — | — | — | 1.65 → 1.65 (+0%) | 6,662.40 → 6,662.40 | −642 → −642 |
| VAL | **0** | — | — | — | 2.02 → 2.02 (+0%) | 6,487.19 → 6,482.24 (−$4.95) | −489 → −489 |

$50K stage = ×5 on every $ figure (the per-position cap binds on ~100% of picks); no sign changes.
**VERDICT: NO GO — no population.** MDE undefined at n=0; the cohort that could *ever* be affected
is 4 picks in 21 months (0 in TRAIN) — ≤ ~2 extra fills per 21 months vs a +30% fills/week bar.

## 3. F1b — passive re-arm at range_high after a gap-through

Resting limit BUY at range_high (zero entry premium), live to 10:35, fills on a later bar with
low ≤ range_high. Adds exactly what B30 leaves unfilled: **TRAIN n=0, VAL n=1.**

| split | added | net R | t | ex-top5% | fills/wk | stacked $ | stacked MDD |
|---|---|---|---|---|---|---|---|
| TRAIN | 0 | — | — | — | 1.65 → 1.65 | 6,662.40 → 6,662.40 | −642 |
| VAL | 1 | **−0.255** | n/a (n=1) | −0.255 | 2.02 → 2.07 (+2.6%) | 6,487.19 → **6,464.50** (−$22.69) | −489 → −489 |

**MDE:** no t at n=1; with the cohort's sd ≈ 0.45 R the SE is 0.45 R ⇒ smallest detectable effect
at t=2 is **+0.90 R/trade**, 6× the ship bar. A power failure, not a measured negative.

### F1b deciding table (the F52 adverse-selection rule)
TEST rows *italic* — shown for cohort completeness, used in **no** decision (PREREG §10 disclosure).

| pick | split | gap bps | book (cap-30, optimistic) | B30 | F1a @cap-60 | F1b re-arm @range_high |
|---|---|---|---|---|---|---|
| MST 2026-02-13 | VAL | 87.2 | −3.005% | no fill | no fill | **−0.681%** |
| WNW 2026-05-21 | VAL | 45.1 | −1.523% | −1.523% | −1.671% | (−2.609%) |
| *AAOG 2026-07-09* | *TEST* | *65.4* | *+0.035%* | *+0.035%* | *no fill* | *(+2.185%)* |
| *AAOX 2026-07-09* | *TEST* | *33.7* | *−0.256%* | *−0.256%* | *−0.293%* | *(−4.753%)* |

VAL: the one re-armed fill (−0.681%) BEAT its gap-through counterfactual (−3.005%) — **no adverse
selection detected**, the chase-guard-style limit got the better price. n=1 supports nothing.
**VERDICT: NO GO — underpowered; fails criteria 1, 2, 3, 5, 7.**

## 4. F2-10 (W=10) — run fresh here; and F2-15, already run 9/18

W=10 built with the shipped plumbing (`ORB_RANGE_MINUTES`), full feature regen into
`research/orb_multiwindow/w10/`, TRAIN-refit z-params/cutoffs, `combine.py` for the 8 shared slots
+ the shipped veto chain, symbol-level overlap, no refill. **New: the range-size veto does NOT
transfer to W=10** — worst quintile Q2 in 2025, Q1 in 2026, so under the V1 rule it is not adopted
at W=10 (it did transfer at W=15/30). Combined book 232 picks = 169 fills, $13,349.

| cell | split | added picks | fills | added R/pick | t | MDE@t2 | book picks vs W5 |
|---|---|---|---|---|---|---|---|
| **5+10 (F2-10)** | TRAIN | 8 | 6 | **−0.329** | **−2.67** | +0.246 R | +7.3% |
| **5+10 (F2-10)** | VAL | 3 | 1 | **−0.119** | −1.00 | +0.238 R | +5.9% |
| 5+15 (F2-15, 9/18) | TRAIN | 10 | 7 | **−0.254** | −1.87 | +0.272 R | +7.8% |
| 5+15 (F2-15, 9/18) | VAL | 7 | 4 | +0.497 | 1.07 | — | |
| W15 alone (9/18) | TRAIN | 32 | 20 | **−0.220** | −2.75 | +0.160 R | |

Stacked 5+10: TRAIN $6,208 → $5,706, MDD −609 → −663 (1.09×); VAL $5,403 → $5,320, MDD −597 →
−681 (1.14×). Ex-top-5% R/pick 0.151 → 0.108 (worse). ×5 at $50K stage, same signs.
**Structural finding that generalises to any W** (9/18): 53% of the W=15 book and 51% of the W=30
book are symbol-days the 5-min book already owns, so a second window buys **+6 to +11% more
picks, not +30%** — the ceiling is the post-veto candidate pool, not the slots.
**F2-10 and F2-15: NO GO.** Power was adequate (MDE ≤ +0.27 R vs a +0.30 R bar) and the sign was
negative on TRAIN in every cell.

## 5. Pass bar, line by line

| # | criterion | F1a | F1b | F2-10 | F2-15 |
|---|---|---|---|---|---|
| 1 | added ≥ +0.10 R on TRAIN and VAL | FAIL (n=0) | FAIL (VAL −0.255) | FAIL (−0.329 / −0.119) | FAIL (−0.254 TRAIN) |
| 2 | day-clustered t ≥ 2 on VAL | FAIL | FAIL (n=1) | FAIL (−1.00) | FAIL (1.07) |
| 3 | ex-top-5% ≥ 0 both splits | n/a | FAIL | FAIL (book ex-top5 falls) | FAIL |
| 4 | TRAIN halves same-signed | n/a | n/a (n=0) | n/a (n=8, both halves −) | n/a |
| 5 | stacked $ up in BOTH splits | FAIL (−$4.95) | FAIL (−$22.69) | FAIL (−$502 / −$83) | FAIL |
| 6 | stacked MDD ≤ 1.25× baseline | pass | pass | pass (1.09× / 1.14×) | pass at 5+15 |
| 7 | fills/week up ≥ 30% | FAIL (+0%) | FAIL (+2.6% VAL) | FAIL (+7.3%) | FAIL (+7.8%) |

## 6. The ONE caveat per cell that alone could explain it

* **F1a** — class (ii) is off 1-min bar OPENS, not quotes; Stage-Q's quote number (14.4%) is ~8×
  my 1.8%, so n=0 may be an artefact of using bars. An NBBO walk would re-open it.
* **F1b** — n=1; an anecdote with SE 0.45 R.
* **F2-10** — composite TRAIN-refit then scored on TRAIN (in-sample for the selector), and the
  veto non-transfer leaves W=10 one veto short of the shipped book: like-for-unlike.
* **F2-15** — same refit caveat; its positive VAL (+0.497 R) is the shape TRAIN-first refuses.

## 7. Mid-run changes, disclosed

1. PREREG §3 (the B30 baseline) was written after reading the entry code but **before any scoring**.
2. Four TEST rows appeared in a diagnostic print of the 4-pick class-(ii) cohort before I had
   partitioned it; excluded from every number in §2/§3 and from the pass bar.
3. F2-15 was found to be an already-executed cell (9/18). I report its result rather than re-run
   it; the ledger counts it once. F2-10 was run fresh, and its range-size veto was re-derived
   (and, per the V1 rule, not adopted) rather than inherited.

## 8. What this pass establishes

The binding constraint on ORB long frequency is **not** the 30-bps cap and **not** the 8 slots.
24.3% of ranked picks never touch the 5-min high in the next hour, and a second, wider opening
range re-selects mostly the same symbol-days at worse R. Future frequency work must move the
TRIGGER or the POOL — both selection changes, each needing its own pre-registration.
Program cell count through this pass: **1,277**.
