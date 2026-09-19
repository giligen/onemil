# PREREG — ORB gate separation + frequency frontier on GREEN WEEKS

Written **before any cell was scored**. 2026-09-19. Mirrors `research/bf_frequency/PREREG.md`
(the study that found two broken bull-flag gates) applied to ORB, with the objective function
the owner fixed mid-BF-stage: *"We must score on green weeks yes. Monster is ok if green weeks
dominate even if near breakeven."*

**Nothing here ships.** A survivor needs its own pre-registration and the owner's word.

Motivation (`research/green_weeks/REPORT.md`): ORB's flat weeks are a **PICK SHORTAGE, not an
exit failure** — 13 of 19 flat TRAIN weeks had NO pick at all, 6 had only non-fills, and the
flat-week share is identical across all 17 exit cells. Nobody has ever measured ORB's gates
the way `bf_frequency` measured the bull flag's.

---

## 0. The structural ceiling — stated before measuring

The entered-inclusive candidate population (`research/fuckup_audit/D1_orb/candidates_dump.csv`,
the D1 dump on `analysis_results/orb_features_20260916_2053.csv`) holds **13,033 candidate
symbol-days over 427 trading days / 90 market weeks**, 2025-01-02 → 2026-09-16:
7,402 modelled fills + 5,631 modelled non-fills.

- **144.8 candidates per week** raw (median 128, p10 73, p90 222).
- The shipped B+ book at 8 slots takes **215 picks = 2.4 per week**.
- The **slot ceiling** is `N × trading days`, not the candidate count: at 8 slots × 5 days
  = **40 picks/week**; at 3 slots = 15/week. The book is nowhere near either — D1 measured a
  maximum of **4 picks on the busiest day at N=8** and 5 at N=12. **The binding constraint on
  ORB frequency is the GATES, not the slots.** That is the hypothesis this stage tests.
- The **raw ORB breakout is edgeless** (`orb_veto_study/REPORT.md`: 2025 −0.18R, 2026 −0.04R),
  so R/pick must fall as gates come off, and the frontier **must bend down**. The deliverable
  is WHERE it bends on the *green-week* axis, which is not the same place it bends on P&L.

**Arithmetic stated in advance**: 90 weeks with candidates, and 215 picks at 8 slots. Even
turning every gate off cannot exceed 13,033 picks / 90 weeks; the reachable band for a book
that still ranks by composite and dedups by family is **2 – 25 picks/week**.

---

## 1. Instrument (frozen before running)

- **Engine**: the shipped `study_orb_pipeline_static_lock.py` via `ORB_BT_RESIM_CACHE`
  (selector-only replay; exit physics are the dump's). No private re-implementation.
  `orb.yaml` supplies every frozen constant (z-params, quintile cutoffs, mults, veto
  thresholds); knobs move ONLY through the documented env overrides.
  Production config, `orb.yaml`, caches, orders, services and crons are never written.
  Every output goes under `research/orb_frequency/`.
- **Reproduction gate — PASSED before this file was written**: as-is dump, `ORB_BT_N=8`,
  `ORB_BT_ACCOUNT=26666.666666666664`, `ORB_SKIP_Q1=1`, `ORB_BT_RISK=375` reproduces
  `research/fuckup_audit/D1_orb/book_n8_q1on.csv` **byte-identically**:
  **215 picks / $14,428.616990972434** (`repro_n8_q1on.csv`, `pandas.DataFrame.equals` True).
- **PRIMARY fill model**: Stage Q's **measured** arm,
  `research/fuckup_audit/Q_fill/dump_measured.csv` — the elected stop-limit rests as a bid at
  the cap, fills at the cap the first time the real NBBO ask reaches it before the 10:35 time
  stop (walked SIP quotes), $0 and a spent slot if it never does. Same 13,033 keys, same
  physics, only the dollars move. **Secondary bracket**: the as-is dump (every elected order
  fills at the cap). Both are reported on the recommendation; the frontier is ranked on the
  measured arm.
- **Sizing normalization (primary, isolates selection)**: `per_pos_cap = account / N =
  $3,333.33` for every slot count, `ORB_BT_RISK=375` — the D1 convention, so changing N
  changes only the number of positions, never their size. The **live** convention
  (`account_budget_usd` fixed, `per_pos_cap = budget / N`) is reported separately in Part 3
  for affordability, never used to rank.
- **Exit physics**: the shipped winner stack in the dump (static lock 1.75R→0.5R, ATR
  stop-floor k=0.25, 40%@+3R scale-out, touchgo M/D, 15:45 force close). Touchgo variants
  require a full bar-walk and are produced as separate dumps (`dump_tg_off.csv`,
  `dump_tgM_off.csv`, `dump_tgD_off.csv`) — declared here, launched before scoring.

### R is defined once
`R = pnl_pct / max(range_size_pct, MIN_STOP_PCT)` — the trade's percent return divided by its
own stop distance, taken from the **candidate row**, never from the sized row, so R is
invariant to the quintile mult, the per-position cap and the account size. A modelled
non-fill (`entered=0`, `pnl_pct=0`) is **R = 0 and still a pick** — the entered-inclusive
convention. This definition is checked against Q_fill's published R/pick before use.

---

## 2. Splits — TEST IS SEALED

| split | window | market weeks | role |
|---|---|---|---|
| TRAIN | 2025-01-01 → 2025-12-31 | ~52 | ladders, frontier construction |
| VAL | 2026-01-01 → 2026-05-31 | ~22 | confirmation, the decision rule |
| **TEST** | **2026-06-01 → 2026-09-16** | **~16** | **SEALED — see FREEZE.md** |

TEST is scored **once**, for exactly **two** cells (shipped B+ baseline and the single
recommended point), and only **after** the recommendation is committed to REPORT.md.
`score.py` refuses to print TEST without `--reveal-test`; FREEZE.md records the commit hash
of the sealed recommendation. TEST holds ~16 weeks and ~50 picks; it is a sanity check, not a
decision instrument, and will be labelled as such.

No cross-split state exists: the pipeline's ranking, dedup and vetoes are all per-day, so
running the whole window once and partitioning afterwards is identical to running each split.

---

## 3. The metric — the owner's, in order

**PRIMARY: % of GREEN WEEKS over EVERY market week in the split.**

- A **market week** is an ISO week (`%G-W%V`) containing at least one of the 427 trading days
  in the candidate population. Weeks with no pick are **in the denominator** — for ORB that
  IS the problem.
- **green** = the week's summed `_sized_pnl` > 0 · **red** = < 0 · **flat** = exactly 0
  (no pick at all, or every pick a modelled non-fill).
- **Flat-week share is reported beside green-week share on every single row.**

Then, in order: **longest red-week streak · worst week ($) · % green months · MDD ($).**
**Total P&L is TERTIARY** and is reported as a floor condition only (§6.4), never as a rank.

**Tail concentration is a DIAGNOSTIC, not a penalty** (owner, explicit): top-1 / top-5 /
top-10 pick share of total P&L is printed on every frontier point and never enters the
decision rule. Ex-top-1% and ex-top-5% R/pick are reported for honesty (PLAN §1 item 5).

---

## 4. PART 1 — the separation map (descriptive; no rule is drawn from it)

For **every** gate, at **its own position in the live cascade**, on the population that
actually reaches it: `sep = mean R(kept) − mean R(rejected)`, with n kept, n rejected, and a
Welch t on the pooled separation, reported for **2025 / 2026 / pooled**.

Measured gates, in cascade order:

| # | gate | knob |
|---|---|---|
| S0 | composite ≥ 0.012082 (frozen threshold) | `ORB_BT_THRESHOLD` |
| S1 | Q1 quintile filter | `ORB_SKIP_Q1` |
| S2 | quintile cutoffs themselves (Q5>Q4>Q3>Q2 rank order) | descriptive, per-quintile R |
| S3 | family / super-group dedup | descriptive (dropped rows) |
| S4 | top-K slot cut (rank ≤ N vs rank > N) | `ORB_BT_N` |
| S5 | PDR veto (`prev_day_range_pct` ≤ 11.0) | `ORB_PDR_VETO*` |
| S6 | G1 fingerprint (rv20 ≥ 7.106 ∧ pdr ≥ 9.226) | `ORB_G1_VETO` |
| S7 | range-size veto (`range_size_pct` ≤ 2.221) | `ORB_RANGE_SIZE_VETO*` |
| S8 | catalyst veto (newsless-and-alone) | `ORB_CATALYST_VETO` |
| S9 | touchgo Rule M (`bb_close_pos < 0.5`) | full walk, `ORB_TOUCHGO_RULE_M_ENABLED=0` |
| S10 | touchgo Rule D (`b1_revert ≥ 0.75R`) | full walk, `ORB_TOUCHGO_RULE_D_ENABLED=0` |
| S11 | **whole stack: picked vs rejected** | — |

**Not measurable at this stage, and it will be said so in those words:**
- the **universe screens** (`prev_volume ≥ 500K`, the $3–30 price band, the 15K RTH-9:35
  range-computability floor) are applied **upstream in `study_orb_broad.py`**, so the dump has
  **zero rejected rows** on them. Measuring them needs a features rebuild (~days of bars),
  which is out of scope; the cascade counts are reported and the gates are logged as
  **unmeasured**, exactly as `bf_decay` had left BF's ADV20 gate unmeasured.
- the **spread gate (300 bps)** is a live entry-time rule with **no BT counterpart at all**.

**BF's diagnostic is mandatory here too**: median position risk (`_rp_position × range_size%`)
and median `_rp_position` on kept vs rejected at every gate — BF's broken gate hid behind the
sizer already shrinking the names it was cutting.

**Pre-committed removal criterion**: a gate becomes a REMOVAL CANDIDATE iff its separation is
**≤ 0 or |t| < 1 in BOTH years**. Separation alone never removes anything — the candidate must
then pass §6's survival rule on its own ladder rung.

---

## 5. PART 2 — the frequency frontier

### 5a. Single-gate ladders (one at a time, everything else shipped, N=8)

| ladder | rungs |
|---|---|
| Q1 filter | on (shipped) · off |
| PDR veto | 11.0 (shipped) · 8.0 · 6.0 · off |
| G1 fingerprint | on (shipped) · off |
| range-size veto | 2.221 (shipped) · 1.5 · 1.0 · off |
| catalyst veto | on (shipped) · off |
| composite threshold | 0.012082 (shipped) · −0.5 · −99 (everything) |

**12 distinct runs** (the shipped rung is shared by all six ladders), each on the measured and
the as-is dump = **24 runs**.

### 5b. Declared combined points (provenance written next to each, so it is a curve not a search)

| point | definition |
|---|---|
| **F0** | shipped B+ — every gate on (the baseline) |
| **F1** | F0 − range-size veto |
| **F2** | F0 − range-size veto, PDR 11.0 → 8.0 |
| **F3** | F0 − range-size veto − G1 fingerprint |
| **F4** | F0 − range-size − G1 − PDR (catalyst + Q1 + threshold kept) |
| **F5** | F4 − catalyst veto (only threshold + Q1 + rank/dedup left) |
| **F6** | **the ceiling** — threshold −99, Q1 off, every veto off (rank + dedup + slots only) |

### 5c. Reported per cell, per split
picks/week · fills · fill% · **green-week % · flat-week % · red-week %** · longest red streak ·
worst week · green-month % · MDD · total $ · total R · R/pick · WR · ex-top-1% R/pick ·
ex-top-5% R/pick · top-1/5/10 P&L share · t · MDE₈₀.

---

## 6. PART 3 — slots, on the green-week metric

`N ∈ {3, 8, 12, 16}` × **{F0 and the best two frontier points by green weeks}** = 12 runs.
D1 measured slots on total P&L and found the edge gone by rank 9; **it has never been measured
on green weeks**, and converting a flat week to a green one is the owner's stated preference
even where the marginal pick is edgeless.

At each cell: **buying-power bind rate** — the share of picks whose uncapped size
`risk / stop%` exceeds `per_pos_cap`, under (i) the invariant $3,333.33 cap and (ii) the LIVE
convention at the ~$66K account (`per_pos_cap = 66,000 / N`) and at ORB's current
`26,666.67 / 8`; plus the max picks on one day and the peak concurrent notional.

---

## 7. The pre-committed decision rule (written before any cell was scored)

A frontier point **survives** iff ALL of:

1. **Green-week % ≥ shipped B+'s green-week %** on TRAIN **and** on VAL, and **strictly
   greater on at least one**.
2. **Flat-week % strictly lower than shipped B+'s** on TRAIN **and** on VAL. *(This is the
   whole point: frequency is only worth having if it removes empty weeks.)*
3. **Longest red-week streak ≤ shipped B+'s + 1** on TRAIN and on VAL.
4. **Total P&L ≥ 0** on TRAIN and on VAL (a floor, not a rank — the owner permits ordinary
   weeks near breakeven).
5. **Worst week no worse than 1.5 × shipped B+'s worst week** on TRAIN and on VAL.

**Among survivors, the recommendation is the highest green-week % (TRAIN+VAL pooled weeks).**
Ties within 2pp break on lower flat-week %, then on higher pooled total R.
**If no point survives, the recommendation is "every ORB gate earns its keep and the flat
weeks are irreducible at this universe"** — a valid, pre-committed, and useful outcome.

Reported **beside** the recommendation, never merged into it:
- **Claim bar** (PLAN §1): G1 = t ≥ 2 on TRAIN; G2 = VAL same sign.
- **MDE₈₀ on the green-week share**, per split, both **unpaired** (two-proportion normal
  approximation, the honest bound) and **paired** (McNemar on discordant weeks, since the books
  nest and share weeks). **Stated up front**: at ~52 TRAIN and ~22 VAL weeks and ~2.4
  picks/week, ORB's resolution is **worse than BF's ±9.5pp** — the unpaired VAL MDE is
  expected above ±25pp, i.e. VAL alone can only refute a catastrophe. This is a limit of the
  instrument and will be stated in the report's first paragraph, not buried.
- **Multiplicity**: the full cell count (§8) with the expected largest |t| under a pure null.
  **Any single best point is a maximum over the grid, not a discovery**, in those words.

---

## 8. Cell count and multiplicity (declared before running)

| block | runs | scored (run × split) |
|---|---|---|
| Part 1 separation map | 0 (descriptive) + 3 full walks | ~36 descriptive |
| 5a ladders (12 × 2 dumps) | 24 | 48 |
| 5b frontier (7 × 2 dumps, F0 shared with 5a) | 12 | 24 |
| Part 3 slots (4 N × 3 configs, N=8 shared) | 9 | 18 |
| **total** | **≤ 45 runs + 3 walks** | **≤ 90 + 36 descriptive** |

With ~45 cells on two splits the expected largest |t| under a pure null is ≈ 2.8–3.0. No
per-cell p-value is treated as evidence on its own; only §7 selects. A permutation view
(week-label shuffle within split, 2,000 draws) brackets the green-week share a random book of
the same pick count would produce, and is reported for the recommendation.

---

## 9. What this stage will NOT do

- Not ship, not flip a flag, not touch `orb.yaml` / `config.yaml` / production caches / orders
  / services / crons.
- Not re-tune a veto threshold, a z-param, a quintile cutoff or an adaptive mult. Ladder rungs
  are **declared** values, not fitted ones. (`adaptive_mults` are never refit — standing rule.)
- Not claim "no edge exists" — only "no gate relaxation raised the green-week share detectably
  in THIS population, over THIS window, at THIS book size, at THIS cost", with the MDE stated.
- Not quote a cell's P&L as a forecast. The ORB pipeline at stage sizing is a relative tool.

## 10. Artifacts

`PREREG.md` (this file) · `FREEZE.md` (the TEST seal) · `separation.py` → `separation.csv` ·
`run_grid.py` → `grid.csv` · `score.py` (metrics, TEST-sealed) · `frontier.csv` · `slots.csv` ·
`walk_touchgo.sh` + `dump_tg*.csv` · `REPORT.md`.
