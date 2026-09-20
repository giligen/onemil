# frames10 — PREREG (written and committed BEFORE any cell was scored)

Pass 10 of the frame programme. Queue set by `hod_frames/FRAMES.md` rows **F31 · F32 · F33**, written
by pass 9. Node rails for the whole pass: one python process at a time, `nice -n 10`,
`ulimit -v 3000000`, checkpointed walks. `cache.db`, `bars_sip.db`, the Databento stores,
`daily_bars` and `trades.db` opened **read-only**. Nothing outside
`research/mature_method/frames10/` is written **except** F33's declared code deliverable
(`trading/ramp_pool.py`, its tests, two one-line additions to the ramp checkers, one section of
`docs/scaling_plan_2026.md`). `config.yaml`, `orb.yaml`, the systemd unit, the crons and every order
are untouched. **TEST is sealed** (`FREEZE.md`): TRAIN = 2025, VAL = 2026-01..05,
TEST = 2026-06-01+ and is never opened.

---

# F31 — THE R-UNIT AUDIT

`R = move% / stop%`. Every baseline, gate separation and pond verdict in 1,090 cells was computed in
R. Pass 9 §2.4 showed the same names, clock and exit read **+0.32 R** at a 1.5 % stop and **+0.005 R**
at a 6 % stop. This frame re-expresses the programme's headline objects in **% of entry price**
beside R, with each object's stop-width distribution, and asks which conclusions flip.

## §1.1 The objects, named BEFORE any price number is computed (8)

| id | object | source | R reading on record |
|---|---|---|---|
| O1 | HOD-break B2 base book (the shipped detector + exit) | `common6.base_book()` | gross −0.039 / +0.083, net −0.107 / +0.013 |
| O2 | HOD's pond bound — a random universe name at HOD's clock (arm u) | `frames9/w9_u.csv` | −0.059 / +0.009 (G3) |
| O3 | HOD's selection margin (signal − its own matched controls) | `frames9` F29 base | G3 +0.183 / +0.245 |
| O4 | F29's wrapper margin (S1) and its stock complement | `frames9/cells29.csv` | +0.382 vs +0.082 |
| O5 | ORB book_G3_meas (the live ORB book) | `research/orb_gates2/book_G3_meas.csv` | ≈ +0.4 R / pick |
| O6 | BF P1 (the live BF book) | `research/bf_frequency/runs/P1.csv` | ≈ +0.65 R / trade |
| O7 | the F23 mirror's best cell (S-nm3, range-so-far ≤ 3 %) | `frames7/p23.csv` | gross +0.187 / +0.128 |
| O8 | the pass-9 stop-width buckets (SUPP B) | `frames9/w9_u.csv` | +0.316 … +0.005 across buckets |

## §1.2 The arithmetic (declared)

For every object with a per-trade stop width, the price-denominated reading is **per trade**:
`move% = rr × r_pct` and `net% = net_R × r_pct`, where `r_pct` is that trade's own stop as a per cent
of its entry. Objects that carry a realized % move directly (ORB `pnl_pct`, BF `pnl_pct`) use it and
their price-consistent R is recomputed as `pnl_pct / stop%` — printed beside the book's own R
definition (`pnl_pct/range_size_pct` for ORB; `pnl/$2,000` for BF) so a sizing-driven gap is visible.
Paired margins (O3, O4) convert pair by pair, because a control carries the signal's own `r_pct`.
Stop-width distributions are reported as median and IQR of `r_pct`.

## §1.3 The pre-committed flip rule

A conclusion **FLIPS** if either:
* **(a) SIGN** — the price-denominated reading has the opposite sign to the R reading **in the same
  era** (TRAIN, VAL, or a declared half), or
* **(b) RANK** — the cross-object ordering by R differs from the ordering by % of price, for the
  ranking `O1 vs O5 vs O6` (the three books) and for `O3 vs O4` (the two margins).

Flips are **counted and listed whatever they read**. The audit is run on the eight objects named in
§1.1, chosen before any price number was computed; no object may be added after.

## §1.4 The specific question the frame must answer in one sentence

*Are ORB's and BF's per-trade edges LARGER than HOD-break's in % of price, or merely measured against
tighter stops?* The answer is written as one of: **PRICE** (the edge survives the unit change and the
books differ in dollars per share), **DENOMINATOR** (the ranking is an artefact of stop width), or
**BOTH** with the decomposition stated.

**Cells: 0 new** (a re-expression of existing walks). 8 objects scored.

---

# F32 — THE WRAPPER OBJECT, BOTH SIDES

F29 named the one era-stable attribute in 1,090 cells: 2× / inverse wrappers, +0.382 R of margin,
same-signed in H1, H2 and VAL — and decomposed it into a pick that earns +0.124 / +0.104 R gross
against a matched non-signal wrapper that **decays** −0.269 / −0.252 R. Two declared books follow,
one per side.

## §2.1 (L) — LONG admission `wrapper`, pre-registered

* **Population**: `P_sig` = `common6.base_book()`'s admitted signal set (7,027 rows, the shipped B2
  cascade at the shipped $20 floor, TRAIN+VAL), restricted to `asset_class == 'wrapper'`
  (`trading/orb_asset_class`: lev-family sets → offline map → asset-name regex; static at the open,
  no lookahead).
* **Exit**: **G3**, the bare stop ridden to the 15:55 flat — F25's best geometry —
  from `frames8/w_sig.csv` (`rr_G3`, `why_G3`, `xm_G3`). X0 (the shipped +2 R cap) is a declared
  secondary.
* **Book**: `common4.book_ranked(s, 12, 4)` — the shipped slot rule, first-come, causal freeing —
  run on the **wrapper-only** stream, so wrapper signals that lost a slot to a common in B2 are now
  eligible.
* **Cost**: the programme's model — `net = rr − half − half × exit_ratio(why)`,
  `half = 0.5 × sp_pct / r_pct`, `sp_pct` measured where the NBBO fetch covers it and imputed
  otherwise; the imputed share is printed per cell.
* **BAR (pre-committed)**: positive weekly $ **AND** green weeks ≥ 50 % on **both** splits at
  ≥ 10 trades/wk, day-clustered t ≥ +2, TRAIN halves same-signed, **AND — binding —
  ex-top-5 % net POSITIVE on BOTH splits.** A negative ex-top-5 % on either split KILLS the object,
  as `hod_fresh` C1 and pass 9's SUPP A were killed.

## §2.2 (S) — SHORT the non-moving wrapper

* **Universe**: the point-in-time day panel (`research/bf_zero/universe.csv`), TRAIN+VAL, restricted
  to `asset_class == 'wrapper'` **with a resolvable single-stock underlying**
  (`orb_asset_class.underlying_anchor` validated against the class map — index/commodity wrappers
  have no underlying and are excluded, counted), `close ≥ $5`, `adv20 ≥ 100K`, `prev_close > 0`.
* **Clocks**: T ∈ {**10:30**, **11:00**, **12:00**} ET. The decision is the CLOSE of the bar at T;
  the entry is the **next** bar's open under a floor, exactly `frames7/c7.walk_short` (reused
  verbatim, not re-written): floor = `close(T) × (1 − 60 bps)`; a bar that opens below the floor is
  **NO FILL, $0, never a loss**.
* **Admission**: range-so-far (09:30 → T) ≤ **X %** of the 09:30 open, X ∈ {**2**, **3**}; and the
  **underlying within ±1 % of flat**, measured as `anchor close(T) / anchor open(09:30) − 1`.
* **Stop / target**: stop at the **session high so far** (09:30 → T inclusive), i.e.
  `r_pct = (high_so_far / entry − 1) × 100`; a candidate with `r_pct < 1.0` is skipped (the mirror of
  the long book's `r_min`). Cover at **−2 R**; flat at **15:55**; stop fills at
  `max(stop, that bar's open) × (1 + slip)`.
* **Borrow**: `shortable AND easy_to_borrow`
  (`research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv`); unknown names are **excluded** from the
  headline cell and counted. Today's flags on 2025–26 tape: survivorship, stated, not corrected.
* **Reg SHO 201**: an entry bar opening ≥ 10 % below the prior close voids the fill model →
  **excluded**, counted.
* **Cost**: the short pays **1.8 ×** the long's per-trade cost (F2 §3, F23):
  `cost_R = 1.8 × (half + half × exit_ratio(why))` with `half = 0.5 × sp_pct / r_pct` and `sp_pct`
  from the programme's price-band × hour-band imputation. Booked per cell and printed; gross beside
  net. **Note in advance**: `r_pct` here is 1–3 % by construction, so the cost in R will be LARGE —
  that is F31's point, not a defect.
* **BAR (pre-committed)**: positive weekly $ AND green ≥ 50 % on both splits at ≥ 10 trades/wk,
  clustered t ≥ +2, halves same-signed. Ex-top-5 % is reported as a diagnostic (the binding tail kill
  applies to (L) only, per the queue). A `book_ranked(12, 4)` realism cut is a declared secondary.

## §2.3 THE MECHANISM, AND THE TWO TESTS THAT CAN REFUTE IT

The claim is **decay of a levered, daily-reset product on a quiet day**. It is falsifiable:

* **(i) MONOTONICITY IN LEVERAGE.** Both the wrapper's own **decay** (its open → 15:55 return in %,
  on qualifying non-moving days) and the short's **gross R** must increase from 1× (inverse) → 2× →
  3×, pooled, **and the ordering must reproduce on VAL alone**. Leverage is parsed from the fund
  name (`\b([123])(\.\d+)?X\b`, UltraPro = 3, Ultra/UltraShort = 2, plain Inverse/Short = 1);
  unparseable names are reported and excluded from this test.
  *Declared secondary, theory-consistent*: the variance-drag coefficient `k = L(L−1)/2`
  (2× long → 1, −1× → 1, 3× long → 3, −2× → 3, −3× → 6); the same monotonicity is reported in k.
* **(ii) MAXIMUM WHEN THE UNDERLYING IS FLAT.** The short's gross must be **largest** in the
  |underlying move| ≤ 1 % bucket and **fall** across (1–3 %, ≥ 3 %). A wrapper on a trending
  underlying compounds, it does not decay.

**If either test fails, the verdict is written as: the mechanism is NOT daily-rebalancing decay, and
the short is a mean-reversion bet with no named buyer.** That sentence is pre-committed here so it
cannot be softened afterwards.

**Cells declared (14):** (L) G3 + X0 = **2**; (S) 3 clocks × 2 range thresholds = **6**; mechanism
(i) 3 leverage levels × {decay, gross} = **2 declared decompositions**; mechanism (ii) 3 underlying
buckets = **1 declared decomposition**; `book_ranked` realism cut on the best (S) cell = **1**;
the (L) X0 secondary and the (S) unknown-borrow sensitivity = **2**. Every scored object is counted
in the report's multiplicity line.

---

# F33 — BUILD THE POOLED GATE

Implement `trading/ramp_pool.py` exactly per `frames9/REPORT.md` §3.2. No research claim; an
infrastructure deliverable with a pre-committed invariant.

* `pooled_z(trades_by_book, sds)` → `(z_bar, se, n, per_book_n)` with a **day-clustered** SE (one
  cluster per session across all books).
* `trading/ramp_bt_band.py` gains `BOOK_SD` (frozen per-book BT SDs: ORB 1.694, BF 1.939,
  HOD-dry 1.260) and `pooled_band()` bootstrapping the SAME pooled statistic from the reference
  books, returning the existing BELOW-p5 / BELOW-p10 / IN-BAND / ABOVE-p90 classification.
* **THE INVARIANT, asserted in a test**: the dry stream counts toward **n and the band ONLY**; it can
  never enter a P&L clause, and **a pooled gate can never turn a negative-P&L book into ADVANCE**
  (the above-water rule is inviolable — `project_orb_ramp_above_water_rule`).
* One **advisory** line in each of `scripts/orb_ramp_check.py` and `scripts/bf_ramp_check.py`,
  printed beside the per-book band line. **The per-book verdict stays the decision** until the owner
  approves switching; the pooled reading changes no verdict in this pass.
* Every failure mode logs (CLAUDE.md fallback rule): a missing SD → excluded at WARNING; < 10 pooled
  trades → `NO-DATA`, blocks nothing; an uncomputable live R → excluded at ERROR.
* **Historical validation**: replay the pooled statistic over the dry run's sessions + ORB's stage
  trades and print what it would have read. Tests first, ~90 % coverage on the new module, full suite
  green.

**Cells: 0.** Code, tests, one docs section.

---

## Rails for the whole pass (unchanged from passes 6–9)

Reproduction gates asserted in code before any number is read; **both TRAIN halves** beside VAL on
every cell; **day-clustered t** on every cell; **count-matched permutation null** (2,000 draws, weekly
pick count fixed) on every book cell's green weeks; **cost booked per cell** from its own exit mix
with the imputed share printed; the **pass-6 control rule** (a control is never a name that itself
produced an admitted signal that session, never the signal's own symbol); an **availability audit**
on every field with an 80 % floor; **tail** (ex-top-5 %) beside every headline; **multiplicity**
counted against the programme total (1,090 + 14 = **1,104**); **TEST never opened**.

**Programme cell count after this pass: 1,090 + 14 = 1,104.**
