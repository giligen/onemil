# frames10 — F31 THE R-UNIT AUDIT · F32 THE WRAPPER OBJECT, BOTH SIDES · F33 THE POOLED RAMP GATE — REPORT (2026-09-20)

Pass 10 of the frame programme. Cells exactly as declared in `PREREG.md`, committed (`edacb52`)
before any cell was scored. Artifacts: `s31.py` → `cells31.csv` · `w32.py` → `panel32.csv`,
`p32.csv` (**86,295 walked wrapper shorts** over 347 sessions), `w32.log` · `s32.py` →
`cells32_long.csv`, `cells32_short.csv`, `s32.log` · `r33.py` → `r33.log`, `dry_pool_replay.csv`,
`replay_dry.log` · the F33 deliverable `trading/ramp_pool.py` + `tests/test_ramp_pool.py` (64 tests,
**98 %** coverage) + `trading/ramp_bt_band.py` (the HOD reference, `sd_of`, `pool_reference_r`) +
one advisory line in each ramp checker + `docs/scaling_plan_2026.md` Gate 2.

One python process at a time, `nice -n 10`, `ulimit -v 3000000`, the walk checkpointed per session;
`cache.db`, `bars_sip.db`, the Databento stores, `daily_bars` and `trades.db` opened **read-only**.
No config, `orb.yaml`, systemd unit, cron or order was written. **TEST was never opened**
(`FREEZE.md`).

---

## 0. THE THREE SENTENCES

1. **F31 — the unit is honest, and the ranking survives it.** The three books rank
   **BF > ORB > HOD** identically in R and in % of entry price, in both eras. ORB earns
   **+0.93 % of price per pick** on TRAIN *and* VAL against HOD's −0.12 % / +0.34 % gross — so
   *"ORB has an edge and HOD does not" is a fact about the markets, not about the denominators.*
   **Exactly ONE flip in 28 re-expressed rows**, and it is a near-zero object (the $20 pond bound
   under the cap on VAL: +0.0085 R vs −0.0030 % of price). What the audit DOES confirm is pass 9's
   §2.4: in price the stop-width buckets are +0.39 / +0.03 / +0.02 / +0.05 %, so the 0.316 → 0.005 R
   gradient was **entirely the denominator** — and the same arithmetic exposes a live-relevant gap:
   **BF's ramp-band R (`pnl/$2,000`, +1.65 TRAIN) is 1.8× its price-consistent R
   (`pnl_pct/stop%`, +0.91)**, because the BT book's sizing is in it.
2. **F32 — the wrapper object dies on both sides, each by its own pre-committed rule.** The LONG is
   positive-net on both splits at 18–23 trades a week (+0.028 / +0.052 R, +$53 / +$119 a week) and is
   **KILLED by the binding ex-top-5 % clause (−0.301 / −0.211)**, with clustered t +0.34 / +0.58,
   green weeks below their own null on both splits, and TRAIN halves opposite-signed. The SHORT never
   reaches an edge question: **only 8 of 226 levered wrapper names (3.5 % of rows) are
   shortable-and-easy-to-borrow**, and **both mechanism tests FAIL** — the short's gross is
   **LOWEST** when the underlying is flat (+0.289 R vs +0.799 / +0.733 when it moves) and it does NOT
   rise with leverage (1× +0.565 R, 2× +0.038 R). **The mechanism is NOT daily-rebalancing decay:
   the "wrapper decay" lives on days the underlying MOVED (−2.5 % / −3.5 %) and is +0.39 % on quiet
   days.** As pre-registered: *the short is a mean-reversion bet with no named buyer.*
3. **F33 — built, tested, and it reads the way the arithmetic said it would.** On the only streams
   that exist, ORB's live stage (n=5) has a BT band **1.140 z-units wide** — wide enough that its
   −0.463 mean R reads IN-BAND. Pool it with the HOD dry run's 31 simulated trades and the band is
   **0.508 z-units on n=36**, pooled z **−0.131 ± 0.220 → IN-BAND**. The pooled gate **would have
   held nothing** the per-book gates advanced — which is the acceptance criterion, not a
   disappointment.

**Verdict: F31 AUDIT COMPLETE (one flip, ranking unchanged) · F32 NO SHIP, BOTH SIDES DEAD ·
F33 CODE SHIPPED ADVISORY-ONLY.** `hod_break` stays `enabled: true, dry_run: true`;
`config.yaml trading.enabled` and `orb.yaml` are exactly as the owner set them; Monday's 12:30 UTC
boot is unchanged. No config file was touched.

---

## 0b. Reproduction gates — asserted in code before any number was read

| id | gate | result |
|---|---|---|
| **G-B2** | `common6.base_book()` — 1,622 TRAIN / **−$17,346**; 706 VAL / **+$893** | **MATCH** |
| **G-F29** | the paired margin under G3 rebuilt from `f29.margin_frame()`: +0.1829 / +0.2452 | **MATCH** |
| **G-ORB** | `orb_gates2/book_G3_meas.csv` — 282 TRAIN / 177 VAL picks | **MATCH** |
| **G-BF** | `bf_frequency/runs/P1.csv` — 56 trades / **$139,113.67** | **MATCH to the cent** |
| **G-AVAIL (L)** | every admitted signal has a walked G3/X0 bracket | **7,027 / 7,027 = 100 %** |

---

# F31 — THE R-UNIT AUDIT (8 declared objects, 28 scored rows, 0 new cells)

## 1.1 The table — R beside % of entry price, with each object's stop width

| # | object | split | n | **R** | **% of price** | stop % (median [p25–p75]) |
|---|---|---|---|---|---|---|
| **O1** | HOD B2 **gross** | TRAIN | 1,622 | −0.0390 | **−0.125 %** | 3.17 [1.71–4.92] |
| | | VAL | 706 | +0.0827 | **+0.343 %** | 3.72 [2.04–5.33] |
| O1 | HOD B2 **net** | TRAIN | 1,622 | −0.1069 | −0.368 % | 3.17 |
| | | VAL | 706 | +0.0127 | +0.072 % | 3.72 |
| **O5** | **ORB book_G3** (live) | TRAIN | 210 | +0.2348 | **+0.931 %** | 4.19 [3.51–5.52] |
| | | VAL | 140 | +0.1887 | **+0.919 %** | 4.12 [3.49–5.40] |
| **O6** | **BF P1** (live) | TRAIN | 34 | +0.9117 *(book R +1.651)* | **+3.629 %** | 3.63 [3.10–4.55] |
| | | VAL | 15 | +0.5555 *(book R +0.893)* | **+2.395 %** | 3.84 [3.23–4.48] |
| **O2** | pond bound G3 (arm u) | TRAIN | 14,706 | −0.0034 | −0.049 % | 4.29 |
| | | VAL | 7,922 | +0.0722 | +0.220 % | 3.90 |
| O2 | pond bound X0 | TRAIN | 14,706 | −0.0594 | −0.245 % | 4.29 |
| | | **VAL** | 7,922 | **+0.0085** | **−0.003 %** | 3.90 |
| **O3** | HOD selection margin G3 | TRAIN | 1,605 | +0.1829 | +0.615 % | 3.16 |
| | | VAL | 705 | +0.2452 | +1.092 % | 3.72 |
| **O4** | margin, **wrapper** | TRAIN | 621 | +0.3936 | **+1.263 %** | 2.61 |
| | | VAL | 276 | +0.3566 | **+1.777 %** | 3.53 |
| O4 | margin, stock | TRAIN | 976 | +0.0414 | +0.190 % | 3.55 |
| | | VAL | 427 | +0.1745 | +0.654 % | 3.78 |
| **O7** | mirror S-nm3 (short, gross) | TRAIN | 9,329 | +0.1869 | +0.628 % | 4.21 |
| | | VAL | 4,150 | +0.1284 | +0.421 % | 4.44 |
| **O8** | stop < 1.5 % bucket (G3, arm u) | TR+VA | 810 | **+0.3162** | **+0.388 %** | 1.26 |
| O8 | stop 1.5–3 % | TR+VA | 5,307 | +0.0240 | +0.035 % | 2.51 |
| O8 | stop 3–6 % | TR+VA | 10,954 | +0.0104 | +0.023 % | 4.24 |
| O8 | stop ≥ 6 % | TR+VA | 5,557 | +0.0045 | +0.049 % | 7.72 |

The one flipped row is marked in bold in the R and % columns (O2 / X0 / VAL).

## 1.2 The flip list (the rule was committed in PREREG §1.3 before any price number existed)

* **(a) SIGN — 1 of 28 rows.** `O2 pond bound X0 / VAL`: **+0.0085 R vs −0.0030 % of price.** An
  object whose price reading is three thousandths of a per cent; the flip is real by the rule and
  meaningless by size. It is reported because the rule was pre-committed, not because it matters.
* **(b) RANK — 0 of 4 orderings.** Books: `BF > ORB > HOD` by R **and** by % of price, on TRAIN and
  on VAL. Margins: `wrapper > pooled` by R **and** by % of price, on both splits.

## 1.3 The answer to the question the frame was set

> **PRICE.** ORB's edge is larger than HOD-break's *in dollars per share*, not merely against a
> tighter stop — and the stop widths run the other way, so R if anything FLATTERS HOD: ORB stops at
> **4.1–4.2 %** of price against HOD's **3.2–3.7 %**. Per pick ORB takes **+0.93 %** of price in both
> eras; HOD's whole book takes **−0.12 % / +0.34 %** gross and **−0.37 % / +0.07 %** net. BF takes
> **+3.6 % / +2.4 %** per trade at a 3.6–3.8 % stop. Nine passes of R-denominated verdicts about
> *these three books* are not an artefact of the unit.

## 1.4 What the audit does change, and it is live-relevant

1. **The §2.4 stop-width gradient is confirmed as pure denominator.** 0.3162 → 0.0045 R across the
   buckets becomes **+0.39 % → +0.05 %** of price, with the tight bucket still highest but the other
   three flat within 0.03 pp. The standing rule from pass 9 (never compare baselines without the stop
   width as a per cent of price) is right, and the correct statement is stronger: *below a ~1.5 % stop
   the R unit multiplies a ~0.4 % drift into +0.32 R.*
2. **BF's live BT-band R is a sizing-scaled number.** `pnl/$2,000` reads **+1.651 / +0.893** where the
   price-consistent `pnl_pct/stop%` reads **+0.912 / +0.556** — a 1.8× gap that is the book's own
   share sizing, not price. This is NOT a defect of the gate (`trading/ramp_bt_band` compares live
   `pnl/risk_per_trade` with BT `pnl/$2,000` — the same definition on both sides), and F33 keeps it
   consistent by standardising each book by **its own** SD in **its own** R definition. It IS a trap
   for any cross-book comparison that reads those numbers side by side — including pass 9's F30 SD
   table, superseded in §3.3.
3. **The wrapper margin is biggest in price too** (+1.26 % / +1.78 %), so F32's object was worth
   pre-registering rather than dismissing as a unit artefact. It then died on its own terms.

**Rails.** Every object carries its own per-trade stop width; paired objects (O3, O4) convert pair by
pair; the two reproduction gates (B2, F29 margin) are asserted in `s31.py` and raise on mismatch;
objects were named in PREREG §1.1 before any price number was computed; **0 new cells**.

---

# F32 — THE WRAPPER OBJECT, BOTH SIDES (14 declared cells)

## 2.1 (L) — the LONG: positive on both splits, killed by the clause written to kill it

The shipped B2 cascade at the shipped $20 floor, restricted to `asset_class == 'wrapper'` (38.6 % of
the 7,027 admitted signals), bare-stop exit, `run_book(12, 4)`:

| cell | split | n | /wk | gross | cost | **net** | green (null p95) | **wk $** | **ex-top-5 %** | H1 / H2 | clust t |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **L wrapper G3** | TRAIN | 982 | 18.5 | +0.0988 | 0.070 | **+0.0284** | 39.6 (52.8) | **+$53** | **−0.301** | −0.038 / +0.086 | +0.34 |
| | VAL | 531 | 23.1 | +0.1265 | 0.075 | **+0.0517** | 52.2 (60.9) | **+$119** | **−0.211** | — | +0.58 |
| L stock G3 | TRAIN | 1,183 | 22.3 | −0.0265 | 0.074 | −0.1007 | 35.8 (39.7) | −$225 | −0.354 | −0.092 / −0.108 | −1.97 |
| | VAL | 572 | 24.9 | +0.0320 | 0.075 | −0.0430 | 39.1 (47.8) | −$107 | −0.280 | — | −0.60 |
| L wrapper X0 | TRAIN | 1,083 | 20.4 | −0.0048 | 0.064 | −0.0686 | 41.5 (45.3) | −$140 | −0.179 | −0.177 / +0.023 | −1.31 |
| | VAL | 594 | 25.8 | +0.1345 | 0.068 | +0.0666 | 43.5 (65.2) | +$172 | −0.035 | — | +0.98 |
| L stock X0 | TRAIN | 1,276 | 24.1 | −0.0528 | 0.070 | −0.1225 | 28.3 (35.8) | −$295 | −0.234 | −0.131 / −0.115 | −3.12 |
| | VAL | 650 | 28.3 | −0.0551 | 0.072 | −0.1267 | 30.4 (34.8) | −$358 | −0.239 | — | −2.64 |

**`L wrapper G3` is the THIRD object in 1,104 cells that is positive-net on both splits at ≥ 10
trades a week** (after `hod_fresh` C1 and pass 9's SUPP A — which is this object, post-hoc). It fails
the pre-committed bar on every other axis:

* **the binding tail kill — ex-top-5 % net −0.301 (TRAIN) / −0.211 (VAL)**: remove five trades in a
  hundred and it is deeply negative on BOTH splits. PREREG §2.1 declared this the KILL, and it kills;
* clustered t **+0.34 / +0.58** (needs ≥ +2);
* green weeks **39.6 / 52.2** against their own count-matched null p95 of **52.8 / 60.9** — below the
  null on both splits, i.e. the week shape is pick COUNT, not skill;
* TRAIN halves **−0.038 / +0.086** — opposite-signed.

**The mechanism reading is unchanged from pass 9 and is why the tail kills it**: the wrapper pick
earns +0.099 / +0.127 R gross against a 0.070 / 0.075 R cost. The +0.38 R "margin" was always the
matched control's decay; the pick itself is a thin, tail-carried +0.03 R. **VERDICT: NO SHIP.**

## 2.2 (S) — the SHORT: killed twice, by borrow and by its own mechanism

**86,295 shorts walked** over 347 sessions on 418 wrapper names with a resolvable single-stock
underlying (28,773 wrapper-days; `c7.walk_short` reused verbatim, so the fill model is F23's).

### The borrow rail — the whole answer for the executable book

| cohort | rows | **borrowable** | names | borrowable names |
|---|---|---|---|---|
| levered \|L\| >= 2 | 77,364 | **3.5 %** | 226 | **8** |
| inverse / 1x | 2,502 | 4.7 % | 21 | 2 |
| leverage unparsed (index/income wrappers) | 6,429 | 26.3 % | 171 | 93 |
| TRAIN | 45,675 | 6.8 % | | |
| VAL | 40,620 | **3.6 %** | | |

**Borrow flags are known for 100 % of the names; 5.3 % of rows are shortable-and-easy-to-borrow —
and the borrowable residue is almost entirely index ETFs, not the levered single-stock wrappers F29
named.** After the fill, `r_min`, Reg SHO 201 (8.7 % excluded) and borrow screens, 86,295 rows become
**2,986**, and the six executable cells run at **0.4–1.7 trades a week with VAL under 5 trades** —
below the declared floor, so every one reads `NO-DATA (a split is missing)`. Best executable cell:
`S 11:00 rng <= 2 %`, TRAIN n=66, gross +0.387 R, **booked cost 0.269 R**, net +0.118, MDE 0.553.

**Cost is the second rail and F31 explains it.** Stops here are 1–3 % of price by construction, so
the same spread that costs 0.07 R on HOD's 3.2 % stops costs **0.24–0.34 R** at 1.8× short pricing.
A cell that needs +0.30 R gross to break even is not a book.

### The two mechanism tests — both FAIL, on the DIAGNOSTIC (no-borrow) population

The executable population cannot test a mechanism, so the tests ran on the same walk without the
borrow screen, at range <= 3 % with all three clocks pooled (2,289 rows, 532 underlying-flat),
labelled NOT EXECUTABLE in every row.

**(i) Monotonicity in leverage — FAIL.**

| \|L\| | wrapper-days | decay (open→15:55) | VAL decay | trades | short gross R | VAL gross | all-days decay |
|---|---|---|---|---|---|---|---|
| 1 (inverse) | 55 | −0.455 % | −1.950 % | 69 | **+0.565** | +1.086 | −0.722 % |
| 2 | 61 | −1.354 % | −2.813 % | 75 | **+0.038** | +0.624 | +0.439 % |
| 3 | — | n = 3, under the floor | | | | | |

The decay is larger on 2x than on 1x (the one limb pointing the right way, and 3x has no population),
but **the short's gross moves the OPPOSITE way — 1x earns fifteen times what 2x earns**, on TRAIN and
on VAL. The variance-drag secondary is worse: k = 1 reads decay −1.47 % / gross +0.438 R, k = 3 reads
**+2.53 % / −0.423 R** — the wrong sign on both.

**(ii) Maximum when the underlying is flat — FAIL, and decisively.**

| \|underlying move\| | n | gross R | TRAIN | VAL | net R | wrapper decay |
|---|---|---|---|---|---|---|
| **<= 1 % (flat)** | 532 | **+0.289** | +0.250 | +0.516 | +0.020 | **+0.388 %** |
| 1–3 % | 265 | **+0.799** | +0.842 | +0.701 | +0.565 | −2.543 % |
| >= 3 % | 46 | +0.733 | +0.799 | +0.621 | +0.525 | −3.519 % |

**The prediction was that the flat bucket is the maximum. It is the MINIMUM, on both splits.** The
decay column says why: on flat-underlying quiet days the wrapper drifts **UP** (+0.39 %); the
−2.5 % / −3.5 % "decay" belongs to days the underlying moved.

### The pre-committed sentence

> **The mechanism is NOT daily-rebalancing decay.** A levered wrapper on a quiet day with a quiet
> underlying does not bleed — it drifts up. What the short actually monetises is the reversal of a
> wrapper whose underlying has already moved 1–3 %, which is a **mean-reversion bet with no named
> buyer**, and it is not executable anyway: 8 borrowable names out of 226.

Declared secondary, the realism cut (`run_book(12, 4)` on the best diagnostic cell): TRAIN n=18 at
0.3 trades a week, VAL n=5 — far below the floor, reported for completeness.

**VERDICT: NO SHIP, BOTH SIDES. The wrapper line is CLOSED** — it was the one era-stable attribute in
1,090 cells, it has now been pre-registered from both sides, and both sides failed their own
pre-committed rules.

## 2.3 Rails (F32)

* **Reproduction**: G-B2 and the 100 % walk-availability gate asserted in `s32.py`.
* **Both TRAIN halves + VAL** on every cell; **day-clustered t**; **2,000-draw count-matched null**
  on every book cell's green weeks; **rank-trimmed ex-top-5 %** beside every headline; **MDE** per
  cell.
* **Cost booked per cell** from its own exit mix (long: the programme's model; short: x1.8, F2 §3).
* **Availability, and it demotes a cell as designed**: the underlying-move field resolves on
  **58.6 %** of short rows (TRAIN 53.7 / VAL 64.0) — **below the declared 80 % floor**, so every
  underlying-conditioned number in §2.2 is a **diagnostic**, stated here and not quietly used. Cause:
  the anchor's 1-minute bars are missing for sessions the caches never covered.
* **Borrow**: today's flags on 2025–26 tape — survivorship, stated, not corrected. It cuts the
  population 29x, so a more generous historical borrow book could raise n; nothing raises a failed
  mechanism.
* **Causality**: every field is computable at or before the decision bar (asset class and leverage are
  static at the open; range-so-far and the session high use closed bars only; the underlying's move
  uses its last close at or before T).
* **Construction note, stated rather than buried**: `underlying_anchor` resolves a token for some
  index/income wrappers, so 171 of 418 names carry an unparseable leverage and are index-like. They
  are 26.3 % of the borrowable residue and are excluded from mechanism test (i) as declared.
* **Multiplicity**: 14 declared; **26 scored rows** (8 long + 18 short, including the 6 diagnostic
  cells and the realism cut) plus the two mechanism decompositions; counted, none selected after the
  fact. **TEST never opened.**

---

# F33 — THE POOLED RAMP GATE (built; 0 cells)

## 3.1 What shipped

* **`trading/ramp_pool.py`** — `Trade`, `PooledStat`, `pooled_z()` (each book standardised by its own
  BT SD, **day-clustered SE across all books**), `pooled_band()` (bootstraps the SAME pooled
  statistic from the reference books), `classify_pooled()`, `pooled_line()`, `apply_pooled_gate()`,
  `pooled_demote()`, and the loaders (`load_live_trades`, `load_dry_trades`, `append_dry_trades`,
  `reference_sds_and_r`, `reading`, `advisory_line`).
* **`trading/ramp_bt_band.py`** — the HOD-dry reference (`hod_frames6/book6.csv`, net R, TRAIN+VAL),
  `load_hod_bt_r`, `sd_of`, `pool_reference_r`, `BOOK_SD_FALLBACK`.
* **One line in each checker**: `print(ramp_pool.advisory_line())` after the per-book band line.
  Both checkers were smoke-run; both print and both keep their own verdict.
* **`docs/scaling_plan_2026.md` Gate 2** — a "Pooled reading (ADVISORY)" block with the statistic,
  the constraints and the intended ADVANCE/DEMOTE forms, explicitly not yet in force.
* **`tests/test_ramp_pool.py`** — **64 tests, 98 % coverage** of the new module.

## 3.2 The invariant, and how it is enforced

`apply_pooled_gate(book_verdict, pooled_status)` can **only downgrade**. Tests assert it over the
full cross-product of verdicts x statuses, plus the two cases that matter:
`test_a_losing_book_with_a_hot_pool_still_does_not_advance` (a dry stream of 30 winners reads
ABOVE-p90 and the under-water book still reads HOLD) and `test_dry_trades_are_never_in_a_pnl_clause`
(`live_n == 0` when the pool is all paper). The above-water rule is untouched by anything here.

## 3.3 A defect found while building it, and fixed

Pass 9 froze the per-book SDs as **ORB 1.694 / BF 1.939 / HOD 1.260**. Only HOD's reproduces: the
other two were measured on the *walked* populations of pass 7, not on the R definition the ramp gate
actually compares. From the reference books the band is built on, the SDs are **ORB 1.431 (n=462) /
BF 3.151 (n=56) / HOD 1.260 (n=2,328)**. `ramp_pool` therefore **computes the SD from the same
distribution as the band** (`sd_of(pool_reference_r(book))`); `BOOK_SD_FALLBACK` exists only for an
unreadable reference and logs at WARNING. A band built on one distribution with an SD from another is
the defect class this house keeps shipping; it is now impossible here by construction and asserted
(`test_sd_matches_the_distribution_the_band_is_built_from`).

## 3.4 The historical replay (`r33.log`)

The dry stream was reconstructed by running `scripts/hod_break_eod_check.py` on each dry session and
parsing its own EXECUTABLE would-be book: **31 simulated trades over 4 trading sessions**
(9/14 11 trades −4.6R, 9/16 7 trades −4.2R, 9/17 7 trades −4.5R, 9/18 6 trades **+9.4R**; 9/15 the
engine logged 42 errors and no signals, 9/19 none). Mean dry R **−0.125**.

| window | stream | n | mean R / pooled z | band | status | **band width (p5..p90)** |
|---|---|---|---|---|---|---|
| **PREVIOUS stage** | ORB alone (since 2026-08-17) | 5 | **−0.463 R** | [p5 −0.62, p10 −0.50, p90 +1.01] | IN-BAND | **1.140** |
| | BF alone (since 2026-09-07) | 0 | — | — | NO-DATA | — |
| | live books only | 5 | — | — | **NO-DATA** (n < 10) | — |
| | **live + HOD-dry** | **36** | **z −0.131 ± 0.220** | [p5 −0.31, p10 −0.25, p90 +0.20] | **IN-BAND** | **0.508** |
| **CURRENT stage** (opens 2026-09-21) | ORB / BF | 0 / 0 | — | — | NO-DATA | — |
| | live + HOD-dry | 31 | z −0.100 ± 0.300 | [p5 −0.34, p10 −0.28, p90 +0.18] | IN-BAND | 0.527 |

**Three readings.**

1. **The dry stream more than halves the band** — 1.140 → 0.508 z-units — after **four sessions**.
   That is F30's calendar arithmetic in the only currency that matters here: a per-book gate at n=5
   cannot distinguish −0.46 R from its backtest; the pooled one at n=36 can see a third of a
   standard deviation.
2. **The pooled gate would have HELD NOTHING.** At every window the pooled status is IN-BAND or
   NO-DATA, so `apply_pooled_gate('ADVANCE', status)` returns ADVANCE. **That is the acceptance
   criterion of PREREG §F33** (a pooled gate that blocked a stage the book's own realized P&L
   justified would be a failure of the gate), and it passes.
3. **The reading is 86 % paper and the line says so.** `dry 31 of 36` is printed every time, so no
   reader can mistake precision for profit. Both books are PAUSED and both stages open tomorrow, so
   the live half is 5 trades; this replay is an instrument check, not a verdict on either book.

## 3.5 What is NOT done, deliberately

* `scripts/hod_break_eod_check.py` does **not** yet append to `data/hod_dry_pool.csv`. The producer
  is one call to `ramp_pool.append_dry_trades(day, taken)` inside the EOD check's dry-book block, and
  it is a write inside a script the owner's cron runs daily — **it needs the owner's approval, not
  mine.** Until then the checkers' pooled line prints the live books only (and says so at WARNING).
  `r33.py` shows the whole path working end to end on the replay file.
* The pooled gate is **advisory**. `verdict()` in neither checker calls `apply_pooled_gate`. Switching
  it on is a one-line change in each checker plus the owner's word.

---

## 4. The adequacy review (RUNBOOK step 10)

* **Did we test what the books actually ARE?** F31: yes by construction — it re-expresses walks that
  already exist, and the two live books are read from the exact reference CSVs their ramp gates use.
  F32 (L): yes — the shipped cascade, the shipped floor, the shipped slot rule, one admission rule
  added. F32 (S): the fill, stop, cover and cost are F23's walker reused verbatim; the population is
  new and is the one pass 9 named. F33: the module is tested against the real reference files, and
  both checkers were run.
* **Is the cost and fill model right?** The short's 1.8x cost is F2's measured multiplier applied to
  the programme's imputed spread — and at a 1–3 % stop it dominates the cell (0.24–0.34 R). That is
  not a modelling artefact: it is what a 1.5 % stop costs, and F31 is the proof.
* **Does any caveat in our own report explain the headline?** For (S), two do and they are named: the
  58.6 % underlying-move availability demotes §2.2's conditioned numbers to diagnostics, and the
  borrow screen uses today's flags. Neither rescues the frame — the mechanism test that failed
  hardest (flat is the MINIMUM, not the maximum, on 532 vs 265 vs 46 rows with both splits agreeing)
  is the one least sensitive to both caveats. For (L), no caveat: it is the shipped stack, 100 %
  availability, and the kill is a pre-committed clause on its own numbers.
* **What is the MDE?** F32 (L): 0.116–0.202 R. F32 (S): 0.325–2.24 R on the executable cells — the
  honest phrasing is that the executable short was never testable: *no edge was detectable in THIS
  population (borrowable levered wrappers), at THIS horizon, at THIS book size, over 2025-01 →
  2026-05, at a measured 0.24–0.34 R cost, with a smallest detectable effect of 0.33–2.24 R — because
  borrow removes 96.5 % of the population before the question is asked.* F31 has no MDE: it is an
  arithmetic re-expression. F33's replay is n=36 and is an instrument check, not a test.
* **Multiplicity**: 14 declared cells this pass; 28 re-expressed rows in F31 (no new cells) and 26
  scored rows in F32, all printed. **Programme total 1,090 + 14 = 1,104.**

**Verdicts: F31 COMPLETE (1 flip of 28, ranking unchanged, answer = PRICE) · F32 NO SHIP BOTH SIDES,
the wrapper line CLOSED · F33 SHIPPED ADVISORY-ONLY, one owner-approved line away from feeding the
band.**

---

## 5. Test and suite status

* `tests/test_ramp_pool.py` — **64 tests**, `trading/ramp_pool.py` at **98 %** coverage.
* Full suite run in four parts (a single run OOMs at 3 GB): **1,488 + 1,098 + 1,008 + 473 =
  4,067 passed, 0 failed**, 5 skipped.
* **Two pre-existing failures were found and FIXED** (both verified failing on a clean stash before
  any of this pass's code existed, so neither was caused here):
  1. `tests/test_bf_decision_parity.py::test_the_booting_config_is_what_the_harness_reads` demanded
     `min_daily_volume == 0`, a snapshot that went stale on the 9/19 ADV-gate revert. It now reads
     `config.yaml` independently and requires the harness to AGREE with it — the invariant is the
     repoint, not the value.
  2. `tests/test_book_option.py::test_load_prev_day_reads_daily_bars` inserted "yesterday" from the
     box's **UTC** date while the loader's "today" is the **ET** date. Between 20:00 and 24:00 ET the
     two differ and the test's row IS the loader's today, which it excludes by design. The test now
     derives yesterday from the ET date. (It was failing at the moment this pass ran: 03:19 UTC =
     23:19 ET.)
