# PREREG — ORB gate REMOVAL, scored on GREEN WEEKS against each cell's own count-matched null

Written **before any cell was scored**. 2026-09-19. Stage 2 of the gate programme; stage 1 is
`research/orb_frequency/` (PREREG.md + REPORT.md, commit `a12bc6e`), which built the separation
map and named this study as its next step.

**Nothing here ships.** A survivor needs the owner's word. This file is committed before the
first cell is scored; the commit hash is recorded in `FREEZE.md`.

---

## 0. What stage 1 established, and what is therefore NOT re-opened

Carried forward as settled, not re-tested:

- **Slots are 8.** Stage 1 §5 measured N ∈ {3,8,12,16}: 3→8 was the whole green-week gain
  (+11.4pp TRAIN / +13.6pp VAL); 8→12→16 buys 0 to −1.9pp green and 2–3× the drawdown, and at
  the live ~$66K account 90–99% of positions become account-capped. **This stage runs N=8 only
  and does not re-open slots.**
- **Two gates are wrong-side in BOTH years** (stage 1 §2, separation at each gate's own position
  in the live cascade): the **Q1 quintile filter** (−0.040 in 2025 / −0.086 in 2026, pooled
  −0.063, t −1.96) and the **range-size veto ≤ 2.221%** (−0.261 / −0.156, pooled −0.231,
  t −0.76). Same shape as the two bull-flag gates already dropped, including the sizer
  diagnostic: range-size cuts names the risk-parity sizer already puts **$60** median dollar
  risk on against **$135** for the names it keeps.
- **The composite ranking does not order R.** Quintiles post-Q1: Q5 −0.069 / Q4 −0.054 /
  Q3 −0.058 / Q2 −0.065. Rank bands: 1–3 **−0.123** (the worst), 7–8 **+0.022** (the best). The
  ranking premise is inverted; the book's separation (t 3.27) comes entirely from the four
  post-ranking vetoes.
- **catalyst OFF is the standout on the owner's metric**: picks 2.0→5.3/wk TRAIN, 2.4→8.1 VAL;
  green weeks 34.0→47.2% and 31.8→54.5%; flat weeks 37.7→7.5% and 22.7→4.5%; **and P&L up on
  both splits under both fill models** ($5,673→$6,515 TRAIN, $3,939→$4,287 VAL). It failed
  stage 1's rule on exactly one clause on one split: worst TRAIN week −$895 against a 1.5×
  allowance of −$628.
- **Three knobs are suspected dead** and are confirmed-and-reported here, not searched:
  `filter.threshold` (inert — Q1's cut at composite 0.1059 sits far above the threshold
  0.0121), `filter.prev_day_range_veto` at 11.0 (~97% redundant with G1's `pdr ≥ 9.226` leg),
  touchgo **Rule D** (0 of 162 fills).

### 0a. The honesty problem, stated before the first number

Stage 1's green-week MDE₈₀ was **±25.8pp on TRAIN and ±39.3pp on VAL**, and a 2,000-draw
permutation that shuffled each book's own P&L across its own weeks — holding picks-per-week
fixed — put **every** book's observed green-week share **inside** its own band. The shipped book
sat *below* its own permuted mean on both splits. The pre-registered reading of that result:

> **Green weeks in this book may be bought with pick COUNT and nothing else. No gate has yet
> demonstrated week-level timing skill.**

This stage therefore makes the count-matched null a **first-class, pre-committed clause of the
decision rule** (§4 clause N), not a footnote. Every cell is reported beside the permutation
band **for its own pick count**, and the report must state plainly whether **any** cell clears
its own null. **If none does, the pre-registered conclusion is that ORB's week shape is a
function of how many picks it takes and nothing else** — which is itself a decision-grade
finding and will be written as the recommendation, not buried.

### 0b. This rule is NOT blind, and that is disclosed here

I have read stage 1's REPORT.md, so I know catalyst-OFF's TRAIN worst week is −$895 and that
stage 1's flat 1.5× worst-week rail is what blocked it. A rail chosen now could be rail-shopping.
Two protections, both pre-committed:

1. The worst-week rail below is **scaled by pick count on a stated mechanism** (a book taking
   k× the picks per week has ≈√k× the weekly sigma under independence), not chosen at a level.
2. **Both rails are reported for every cell** — the pick-scaled one that decides, and stage 1's
   flat 1.5× — so a reader can apply either. Any cell that passes only the scaled rail is
   labelled as such in the report's cell table.

---

## 1. Instrument (frozen)

- **Engine**: the shipped `study_orb_pipeline_static_lock.py` replayed off a candidate dump with
  `ORB_BT_RESIM_CACHE` (selector-only; exit physics are the dump's). No private
  re-implementation of the pipeline. Every constant — z-params, quintile cutoffs, adaptive
  mults, veto thresholds — comes from `orb.yaml` as it stands today. **Nothing is refit.**
  Knobs move ONLY through documented env overrides (or, for G6/G8, through the row-exact
  derivation of §2a, which is validated against the pipeline). `orb.yaml`, `config.yaml`,
  production caches, orders, the service and the crons are never written; every artifact is
  under `research/orb_gates2/`.
- **Reproduction gate — PASSED BEFORE THIS FILE WAS WRITTEN.** As-is dump, `ORB_BT_N=8`,
  `ORB_BT_ACCOUNT=26666.666666666664`, `ORB_BT_RISK=375`, `ORB_SKIP_Q1=1` reproduces
  `research/fuckup_audit/D1_orb/book_n8_q1on.csv` with `pandas.DataFrame.equals() → True`:
  **215 picks / $14,428.616990972434** (`repro.sh`, `repro_n8_q1on.csv`, `repro.log`).
  No cell number is quoted before this gate passes.
- **PRIMARY fill model**: Stage Q's **measured** capped-limit arm
  (`research/fuckup_audit/Q_fill/dump_measured.csv`) — the elected stop-limit rests as a bid at
  the cap and fills at `min(ask, cap)` the first time the walked per-trade SIP NBBO ask reaches
  it before the 10:35 time stop; **ask > cap ⇒ no fill**, $0 and a spent slot. Measured
  per-trade NBBO, never the band constant. **Secondary bracket**: the as-is dump
  (`research/fuckup_audit/D1_orb/candidates_dump.csv`, every elected order fills at the cap).
  Every headline is given in both; the ranking uses the measured arm.
- **Entered-inclusive**: a modelled non-fill is `entered=0`, `pnl_pct=0`, **R = 0, and still a
  pick that burned a slot**. Non-fill picks stay in the book.
- **Population**: 13,033 entered-inclusive candidates over 427 trading days / 90 market weeks,
  2025-01-02 → 2026-09-16.
- **Sizing**: `per_pos_cap = account / N = $3,333.33`, `ORB_BT_RISK=375` (the D1 convention),
  held invariant so that only the gate set moves.
- **R** = `pnl_pct / max(range_size_pct, 1.0)` from the candidate row — invariant to the
  quintile mult, the per-position cap and the account size.
- **Splits**: TRAIN 2025 (53 wk) · VAL 2026-01..05 (22 wk) · **TEST 2026-06+ — SEALED**, see
  `FREEZE.md`. TEST is opened **once**, **after** the recommendation is committed, for exactly
  the baseline and the single recommended cell (or, if the recommendation is the null finding,
  for the baseline and the highest-green-week cell, labelled as a sanity check that selected
  nothing).
- **Scoring code** is `research/orb_frequency/score.py` (the owner's metric, already written and
  used by stage 1) imported unchanged; this stage adds only the permutation null, the rails and
  the availability audit.

---

## 2. The declared cells — exactly 10, no more

All at N=8, all other knobs at their `orb.yaml` values.

| cell | definition | env / derivation |
|---|---|---|
| **G0** | shipped B+ — every gate on (**baseline**) | `{}` |
| **G1** | Q1 quintile filter OFF | `ORB_SKIP_Q1=0` |
| **G2** | range-size veto OFF | `ORB_RANGE_SIZE_VETO=0` |
| **G3** | catalyst veto OFF | `ORB_CATALYST_VETO=0` |
| **G4** | **G1 + G2** — the two wrong-side gates together | `ORB_SKIP_Q1=0`, `ORB_RANGE_SIZE_VETO=0` |
| **G5** | **G4 + G3** — everything the separation map flags | `ORB_SKIP_Q1=0`, `ORB_RANGE_SIZE_VETO=0`, `ORB_CATALYST_VETO=0` |
| **G6** | **catalyst PARTIAL** — `min_cohort: 2 → 1` (§2a) | derived from G3, row-exact |
| **G7** | PDR veto removed (the redundancy test) | `ORB_PDR_VETO=0` |
| **G8** | **G4 + G6** — the two wrong-side gates off, catalyst softened | derived from G5, row-exact |
| **G9** | **G4 + G7** — the full documented-defect cleanup: both wrong-side gates and the redundant PDR removed, both real gates (G1 fingerprint, catalyst) KEPT | `ORB_SKIP_Q1=0`, `ORB_RANGE_SIZE_VETO=0`, `ORB_PDR_VETO=0` |

**G8 and G9 are the two combined points permitted by the brief, declared here by name before
scoring.** No other combination will be run, and no cell will be added after scoring begins.

### 2a. G6 — the exact partial catalyst rule, declared before running

**The rule: `filter.catalyst_veto.min_cohort: 2 → 1`. One variant, no other change.**

Why this one, and why it is the only partial declared:

- It is the mitigation stage 1 §8 named ("a *partial* catalyst rule … rather than the
  all-or-nothing switch"), and it is the **only knob the shipped rule exposes**
  (`orb.yaml::filter.catalyst_veto.min_cohort`, `trading/orb_catalyst_veto.DEFAULT_MIN_COHORT`).
  A "restrict the veto to the worst sub-bucket" variant would require a new threshold **fitted
  on this data** and new live code; `min_cohort` is a **one-line config diff that ships today**.
- `min_cohort` has exactly one other sensible integer. **There is no value to tune**, so this
  cell cannot be a search.
- Semantics under the shipped `underlying_anchor`: a common stock anchors its own complex, so at
  `min_cohort=1` every newsless pick with an **identifiable** anchor is complex-confirmed and
  survives; the veto is retained **only** for newsless picks with **no identifiable underlying
  at all** (index/commodity wrappers and symbols with no usable name). It is therefore a strict
  intermediate between G0 and G3 and is guaranteed to be so by construction.

**Derivation, and the exactness proof that licenses it.** The catalyst veto is applied
POST-selection with **NO refill** (a vetoed pick's slot stays empty — the refill form re-tested
toxic, MDD +42%, 2026-07-18). Therefore the catalyst-ON book is *exactly* the catalyst-OFF book
minus the vetoed rows, with identical sizing on every surviving row. G6 is built by applying
`trading.orb_catalyst_veto.catalyst_veto_applies(..., min_cohort=1)` — the **shipped shared
helper**, with the shipped `underlying_anchor`, the shipped class map, cohort counts over the
full day's candidate universe, and the same tri-state raw-news source
(`data/research/orb_news_catalyst_*.csv`, absent pair → `None` → fail-open) — to the rows of G3.

**Pre-committed validation gate**: the same derivation at `min_cohort=2` **must reproduce G0
row-for-row and to the cent**, and the same derivation applied to G5 at `min_cohort=2` must
reproduce G4. If either check fails, G6 and G8 are reported as **NOT MEASURED** and are dropped
from the study; they are not repaired after seeing their scores.

---

## 3. The metric — the owner's, fixed 2026-09-19, in order

**PRIMARY: % of GREEN WEEKS over EVERY market week in the split.** A market week is an ISO week
(`%G-W%V`) containing at least one of the population's 427 trading days. **Weeks with no pick
are flat and stay in the denominator.** green = summed `_sized_pnl` > 0 · red = < 0 · flat = 0.

Then, in order: **longest red-week streak · worst week ($) · % green months · MDD ($).**

**Total P&L is TERTIARY** — a floor condition only (§4 clause 4), never a rank.

**Flat-week share is printed beside green-week share on every row** — converting flat weeks to
small-green weeks is the win condition.

**Monsters are explicitly permitted.** Top-1 / top-5 / top-10 P&L share and ex-top-1% /
ex-top-5% R/pick are reported as **diagnostics only** and **never penalise a cell**.

---

## 4. The pre-committed decision rule

Written before any cell was scored. A cell reaches the **LIVE-EXPLORATION bar** iff ALL of:

1. **Green-week % ≥ G0's** on TRAIN **and** VAL, strictly greater on at least one.
2. **Flat-week % strictly lower than G0's** on TRAIN **and** VAL.
3. **Longest red-week streak ≤ G0's + 2** on TRAIN and VAL. *(+2, not stage 1's +1: a book
   taking k× the picks mechanically converts flat weeks into red ones as well as green ones, and
   the owner ranks the streak above the worst week but below the green share. Declared here, in
   advance, with its reason.)*
4. **Total P&L ≥ 0** on TRAIN **and** VAL. *(The owner permits near-breakeven; not negative.)*
5. **Worst week ≥ `sqrt(picks_per_wk_cell / picks_per_wk_G0) × 1.5 × G0's worst week`** on
   TRAIN and VAL — the pick-scaled rail of §0b, mechanism stated there. **Stage 1's flat
   `1.5 ×` rail is reported beside it for every cell**, and any cell passing only the scaled
   rail is flagged in the table.

And separately, the **CLAIM bar** (PLAN §1 G1/G2, reported per cell, never merged into the
rule): **t ≥ 2.0 on TRAIN mean R/pick**, and **VAL same sign**.

And the **NULL clause (N)** — the honesty clause, pre-committed:

> **N.** A cell's green-week share must exceed the **95th percentile of its OWN count-matched
> permutation null** (2,000 draws, the cell's own pick-level P&L shuffled across its own picks
> with each week's pick count held fixed) on **at least one** split, and be **≥ that null's
> mean** on **both**.

**Selection.** Among cells passing 1–5 **and** N, the recommendation is the highest pooled
(TRAIN+VAL week-weighted) green-week share; ties within 2pp break on lower flat-week share, then
on higher pooled total R.

**If cells pass 1–5 but NONE passes N**, the recommendation is written in these words:
**"no cell demonstrated week-timing skill; ORB's week shape is a function of its pick count, and
the only lever on green weeks is how many picks the book takes."** The best 1–5 survivor is then
reported as an **exploration candidate only**, explicitly not a claim, with its config diff and
stop rule — and the report says so in the recommendation sentence itself.

**If no cell passes 1–5 either**, the recommendation is "every ORB gate earns its keep on this
rule" — a valid pre-committed outcome.

---

## 5. Mandatory reporting, whatever the outcome

- **The count-matched null band beside every cell**, on both splits, with the observed value.
- **Cell table ranked on green weeks**, carrying flat-week %, red streak, worst week, MDD, $, the
  null band, and both rails' pass/fail.
- **Availability audit on every gating field** used by any cell — `composite_z`, the quintile,
  `range_size_pct`, `prev_day_range_pct`, `return_volatility_20d`, the anchor, and the raw news
  pair — missingness per split, and what the cascade does with a missing value (fail-open or
  veto). A field whose coverage was built from another stage's key set is a look-ahead even when
  the value is causal (D1's standing rule).
- **The three dead knobs confirmed** (`filter.threshold`, PDR@11.0 vs G1's 9.226 leg, touchgo
  Rule D) with the evidence, as the brief asks.
- **Tails**: ex-top-1% and ex-top-5% R/pick per cell; top-1/5/10 P&L share. Diagnostics only.
- **Multiplicity**: the full cell count of this stage **and** the programme total including
  stage 1's 45, with the expected largest |t| under a pure null. Every best point is a maximum
  over a grid, not a discovery, in those words.
- **MDE₈₀** on the green-week share per split, unpaired.
- **Both fill models** on every headline.
- The **phrasing rule**: never "no edge exists" — always the universe / horizon / book size /
  window / cost, with the smallest effect the test could have seen.

## 6. Cell count declared in advance

| block | pipeline runs | scored cells (cell × split) |
|---|---|---|
| reproduction gate | 1 | 0 |
| G0–G5, G7, G9 × {measured, as-is} | 16 | 32 |
| G6, G8 (row-exact derivations) × {measured, as-is} | 0 | 8 |
| TEST reveal (2 cells × 2 fill models) | 0 | 4 |
| **total** | **17** | **44** |

Plus 2,000-draw permutation nulls per cell × split, and the descriptive availability /
dead-knob / tail tables. **Programme total including stage 1: ~62 pipeline runs.** Under a pure
null the expected largest |t| over that many cells is ≈ 2.9–3.1.

## 7. What this stage will NOT do

- Not ship, not flip a flag, not touch `orb.yaml`, `config.yaml`, production caches, orders, the
  service or the crons.
- Not re-open slots (settled at 8).
- Not re-tune a veto threshold, a z-param, a quintile cutoff or an adaptive mult. `adaptive_mults`
  are never refit — standing rule.
- Not add a cell after scoring begins, and not repair G6/G8 if their validation gate fails.
- Not quote a dollar figure as a forecast. The ORB pipeline at stage sizing is a relative tool.
