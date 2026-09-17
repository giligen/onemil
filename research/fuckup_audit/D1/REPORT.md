# Stage D1 — H4 (the feature/selection program) on `candidates4.csv`

Executed 2026-09-16 per `research/fuckup_audit/PLAN.md` §3 H4 / §4 row D and the D0 hand-off (`LOG.md`, Stage D0
"NEXT"). Pre-registration: `D1/PREREG.md`, written 23:06 UTC before any D1 run. Everything written is under
`research/fuckup_audit/D1/`; everything outside it was read only (`data/cache.db`, `etf_1min.db`, `D/pm_bars.db` via
`file:...?mode=ro`). No config, service, cache or order was touched.

---

## A. One page

**No cell clears the pre-registered bar, and the headline number of the stage is a defect, not a book.** The
declared pipeline (53 causal features, incl. the premarket dollar volume and news presence D0 was missing) produced
what looked like the best result this program has ever seen — `F6 {}` regressor S2 on VAL: **+0.415 R/trade, t 5.10,
86% of weeks green, +8.4 R/week, 20 trades/week, positive in all 8 predicted months, VAL decile rho +0.90**. It is a
**look-ahead in data availability**, and the stage's own mandatory checks caught it before TEST was opened.

| question | answer |
|---|---|
| Does any cell clear **G2 AND the reversed-tape gate AND a transparent baseline that also clears**? | **No.** 10 of 48 cells clear G1; **all 10 are also profitable on the Nagel reversed tape** (up to +0.177 R, t 4.60); the transparent baseline clears G1 in **0 of 16** of its cells (PLAN §3 H4 makes that a hard ship requirement). After the contaminated features are removed, **G1 = 0 of 48** and the reversed tape goes negative. |
| Top features and their direction | `f_log_pm_dollar` is importance rank **1.0 (sd 0.00)** across all 12 monthly refits for `F11(F6)` and rank 2.0 for `F6 {}`, permutation importance **0.049–0.060 R** — an order of magnitude above every other feature; `f_news_intraday` rank **1.25** for `F8 N=30`, `f_news_pre` rank 3.3 for `F6`. Direction: "premarket dollars known and large" = good. **That direction is the artefact.** In the clean re-run the top features are day context (`f_iwm_prev_ret`, `f_prev_day_range_pct`, `f_spy_gap`, `f_consol_bars`) and nothing reaches G1. |
| The two-leg ORB rule (`news AND PM$ > $5.82M`), by family and time band | **Negative or flat everywhere.** TRAIN combo-minus-rest: F6 −0.012 · F8 N=30 −0.059 (t −2.93) · F8 N=5 −0.057 · F11 −0.023 · F14 +0.002. VAL +0.034 / +0.003 / −0.041 / +0.034 / −0.085. TEST −0.098 / −0.070 / −0.038 / −0.072. **0 of 20 family×band decisions reach ADOPT.** D0b's ≥10:00 finding generalises to the whole day. The one consistently positive bucket is **`news_only`** (news WITHOUT the PM$ leg): TRAIN +0.078..+0.093 (t 2.0–2.8) on F6/F11/F8 N=5, TEST +0.091/+0.136/+0.194 — the opposite of the shipped ORB gate's logic. It was not the pre-registered rule and is reported, not adopted. |

**The defect, named exactly.** `f_log_pm_dollar` is causal in VALUE (04:00–09:29 aggregates) but its
**availability** is not: `D/pm_bars.db` was backfilled over the key set of `D/table.csv` — (day, symbol) pairs that
produced an F6/F8 signal with entry **≥ 10:00** in `candidates3`. For a signal at 09:35, "premarket dollars are
known" therefore means "this symbol-day *also* signalled later in the day": information from the future of that
trade. The fallback (candidates4's own `pm_dollar_vol`, present only where `bars_sip.db` holds pre-09:30 bars) has
the same character.

| cohort (primary target, all four families) | n | mean net R |
|---|---|---|
| rows before 10:00 on symbol-days that DO signal later that day | 10,579 | **+0.387** |
| rows before 10:00 on symbol-days that do NOT (pm known / pm missing) | 3,823 / 6,905 | **−0.456 / −0.336** |
| premarket source `db:alpaca` | 53,179 | **+0.180** |
| premarket source `db:sip_store` | 32,448 | −0.183 |
| premarket source `db:none` — genuinely no premarket trades, a CAUSAL category | 11,236 | **+0.024** |
| no premarket source at all (`f_pm_missing = 1`) | 8,323 | **−0.295** |

Missing rate is **33.1%** before 10:00 and **1.5%** after. The `missing` bucket runs −0.28 to −0.46 R at t −6 to −16
**in every family and every split** — an availability indicator worth ~0.8 R per trade, four times the largest
honest effect this program has ever measured. `db:none` (+0.024 R) shows the mechanism is NOT "names without
premarket trading are bad"; it is which keys got backfilled.

**The clean re-run (the honest answer).** Same pipeline, same splits, same book, six premarket/news columns dropped
(47 features, `D1_TAG=_clean D1_DROP=pmnews`):

| | contaminated (the declared run) | clean |
|---|---|---|
| G1 passes (of 48) | 10 | **0** (best TRAIN t 1.65) |
| best VAL cell | F6 reg S2 **+0.415 R, t 5.10** | F6 reg S1 **+0.211 R, t 3.10** — against FCFS (no selection) **+0.163, t 2.46** |
| reversed tape on those cells | **positive**, +0.012..+0.177 (t up to 4.60) | **negative**, −0.03..−0.15 |
| VAL cells with the top 5% of trades removed | still positive, +0.086..+0.170 | **all negative**, −0.02..−0.12 |
| what selection is worth over FCFS on VAL | +0.25 (F6) / +0.21 (F8) / +0.23 (F11) R | **+0.05 (F6) / +0.06 (F8) / +0.03 (F11) / −0.01 (F14) R** |

**Power / phrasing (PLAN §1).** In THIS universe (the point-in-time ≥5%-range day population with the causal
`range_so_far_pct ≥ 5` floor), for THESE four families, ALL DAY, under THIS book (12/day, 4 concurrent), THIS
corrected cost contract (c) and THIS 47-feature causal set, **no walk-forward feature model produced a selection
that clears G1 on the predicted TRAIN months**; on VAL the best honest selection adds **+0.05 R/trade** over taking
every candidate first-come. The smallest per-trade effect the VAL tests could have seen at 80% power is
**0.08–0.20 R** (the MDE column; ≈1.5–3.8 R/week at 4 slots). Effects below that are invisible here and are NOT
excluded. What IS excluded at this power is the +0.2..+0.4 R/trade the live BF/ORB selection stacks historically
carried.

**Two methodological findings worth more than the null.**

1. **The reversed-tape gate fired correctly, for a reason Nagel did not anticipate.** When the real tape is strongly
   positive, a genuine two-sided cross-sectional signal would ALSO make the flipped tape profitable, so "rev > 0" is
   not by itself proof of inversion. What it does prove — and did here — is that the model found a **deterministic
   cohort split** in the target: on the flipped tape it simply selects the other side of the same split. That is the
   signature of a leak, and it was visible on VAL, before TEST.
2. **The tail test does not catch cohort leaks.** Every contaminated survivor passed the top-5% removal and the +3R
   cap. Only the explicit missingness audit found it. Add to the check list: *audit the availability of every feature
   whose coverage is under 100%, per split AND per time-of-day band; coverage built from another stage's key set is a
   look-ahead even when the feature's value is causal.*

---

## B. What was run

| step | script | output |
|---|---|---|
| 1 | `d1_table.py` | `table.csv` — 110,030 rows (F6 {} 12,018 · F8 N=30 54,067 · F14 N=15 2,517 · F11(F6) 11,851 · declared extra F8 N=5 29,577), 420 days, 70,633 symbol-days; read from `C/pop_c.csv` with `usecols`+`chunksize` |
| 2 | `d1_features.py` | `feat.csv` — 53 features, all causal at the signal minute; targets p and s under contract (c) |
| 3 | `d1_parity.py` | `d1_parity.csv` — **20/20 cells, max \|dn\| = 0, max \|d meanR\| = 0.0** vs `C/score5_results.csv` |
| 4 | `d1_model.py` | `preds.csv`, `importance.csv`, `baseline_univariate.csv` (11.5 min) |
| 4b | `d1_model.py` with `D1_TAG=_clean D1_DROP=pmnews` | `preds_clean.csv` — the ablation |
| 5 | `d1_eval.py trainpred val` (both tags) | `cells_*.csv`, `calib_*.csv`, `booked_*.csv` |
| 6 | `d1_perm.py` | `perm_val.csv`, `perm_val_summary.csv` |
| 7 | `d1_orb.py` | `d1_orb.md`, `d1_orb_cells.csv`, `d1_orb_booked.csv` |
| 8 | `d1_tables.py` | `results.md` — every table |

**Population** (PREREG §0.1): `next_entry` present, fill ≥ $5, `entry_m ≤ 841`, the variant's `r_pct ≥ 1`,
`range_so_far_pct ≥ 5`, ALL DAY. 109,507 rows carry the primary target, 110,030 the secondary. The $5 floor is on
the FILL (Stage A/C's convention; in candidates4 `price` IS the level) — the two readings differ on **51 of 110,030
rows**.

**Parity.** The D1 target, recomputed from candidates4's raw columns, reproduces Stage C's scorer exactly on all 20
family × split × target cells. The D1 target IS Stage C's target.

**Coverage.** premarket 92.3% (`D/pm_bars.db` 88.3% + candidates4 fallback 4.0%); news 95.9%
(`D/news_presence.csv` ∪ `E/news_presence_e.csv`; the E pull finished 23:25 UTC and is split-even — TRAIN 0.955 ·
VAL 0.964 · TEST 0.962). Both coverages are the subject of §A.

---

## C. The declared run (contaminated) — gates

Every cell is in `results.md`. G1 is scored on the predicted TRAIN months 2025-10..12 only — a weak, 13-week gate,
declared as such in PREREG §0.8.

**G1: 10 of 48.** `F6 {}` clf S1 (+0.182, t 2.68) · clf S2 (+0.224, t 2.92) · reg S1 (+0.156, t 2.12) · reg S2
(+0.189, t 2.27) on target p; the same four on target s (+0.117 t 2.28 · +0.223 t 3.62 · +0.141 t 2.70 · +0.191
t 3.14); `F8 N=30` clf S1 p (+0.113, t 2.10); `F11(F6)` reg S2 s (+0.186, t 3.15).

**G2 arithmetic: 31 of 48**, including all ten G1 passes. Best `F6 {}` reg S2 p: +0.415 R, t 5.10, 86% weeks green,
8.39 R/week, 20.2 trades/week.

**Reversed-tape gate — all 10 G1 passes FAIL** (mean net R > 0 on the flipped tape): F6 p clf S1 +0.035 · clf S2
+0.012 · reg S1 +0.084 · reg S2 +0.146 (t 3.00) · F8 p clf S1 +0.074 (t 2.15) · F6 s clf S1 +0.045 · clf S2 +0.112 ·
reg S1 +0.107 · reg S2 +0.146 · F11 s reg S2 **+0.177 (t 4.60)**.

**Transparent baseline: 0 of 16 cells clear G1** (best +0.142, t 1.65) → no ship candidate even before the reversed
tape.

**Shuffled-target twin.** The baseline's VAL numbers survive shuffling almost unchanged (`F6 {}` base S1 real +0.172
vs shuffled **+0.168**; base S2 +0.180 vs +0.168) — the baseline books 514–518 trades against FCFS's 536, i.e. it
barely selects. Its "G2 pass" is the family's VAL tape, not selection.

**Search-adjusted permutation p on VAL = 0.000** (500 within-day draws; observed max weekly R 8.39 vs null mean 1.82,
p95 2.90, max 4.28). Conditional on the fits, therefore understated — and it is measuring the leak. A significant p
is not evidence of a tradable effect when the feature carrying it is a coverage indicator.

**Cost decomposition** (checked first, because the target's cost term is a deterministic function of two features):
mean cost charged is flat across every selection — all candidates 0.017–0.023 R, selected books 0.016–0.033 R. The
advantage is entirely GROSS (F6 p reg S2 gross +0.442 vs all-candidate +0.085). The cost term is NOT the leak.

## D. The clean re-run — the honest numbers

47 features, everything else identical.

**G1: 0 of 48** (best TRAIN t 1.65). **G2 arithmetic 22 of 48**; best `F6 {}` reg S1 +0.211 (t 3.10, 77% green) and
reg S2 +0.210 (t 3.05, 82% green) against `F6 {}` FCFS +0.163 (t 2.46, 68% green). All 22 die under the top-5%
removal (−0.02 to −0.12 R). The reversed tape is negative on the F6 cells (−0.03 to −0.15).

Per family, VAL, primary target: F6 FCFS +0.163 / best selected +0.211 · F8 N=30 FCFS −0.011 / +0.061 · F14 FCFS
+0.050 / +0.046 · F11(F6) FCFS +0.078 / +0.106.

## E. The two-leg ORB rule (PREREG §0.7 — 5 declared cells)

`d1_orb.md`. Both legs known on 90.7% of rows. Bucket shares: neither 63.3% · news_only 12.5% · pm_only 10.0% ·
missing 9.3% · combo 4.9%. **No family, no band, no split reaches the pre-registered ADOPT rule.** The `missing`
bucket's −0.28..−0.46 R is §A's defect. **Disclosed deviation**: `d1_orb.py` computed TRAIN, VAL and TEST in one
pass, so this cell's TEST column was read before the model cells' TEST rule was frozen; its decision rule was
pre-registered and fails on TRAIN alone, and no model cell depends on it.

## F. The TEST rule, frozen here (written before `d1_eval.py test` ran)

Nothing satisfies PREREG §0.9's six conditions — the clean run has no G1 pass, and the contaminated run's G1 passes
all fail the reversed-tape gate while its baseline fails G1. Per §0.9 the stage's answer is **NO CANDIDATE**, and
TEST is read **once**, descriptively, in a single invocation of `d1_eval.py test` for **both** pipelines, over all 48
cells of each. The headline cells were fixed before the read: per family, the highest-VAL **S2** cell (S2 is the only
live-implementable rule — S1 needs the whole day's candidates) on the primary target —

* clean: F6 reg S2 · F8 N=30 clf S2 · F14 reg S2 · F11(F6) reg S2
* contaminated: F6 reg S2 · F8 N=30 reg S2 · F14 reg S2 · F11(F6) clf S2

No selection rule, feature set, family, exit or gate was changed after this paragraph was written.

## G. TEST — read once, after §F

Both pipelines, one invocation each, all 48 cells each (`cells_test.csv`, `cells_test_clean.csv`; full tables in
`results.md`). Descriptive, per §0.9 — there is no candidate to confirm.

**The four frozen headline cells (primary target, S2, the live-implementable rule):**

| pipeline | cell | VAL meanR | **TEST meanR** | TEST t | TEST MDE | TEST weeks green | TEST, top 5% removed |
|---|---|---|---|---|---|---|---|
| contaminated | F6 {} reg S2 | +0.415 | **+0.139** | 1.82 | 0.214 | 0.64 | **−0.085** |
| contaminated | F8 N=30 reg S2 | +0.138 | **+0.001** | 0.03 | 0.111 | 0.43 | −0.095 |
| contaminated | F14 reg S2 | +0.054 | **+0.047** | 0.50 | 0.262 | 0.64 | −0.140 |
| contaminated | F11(F6) clf S2 | +0.305 | **+0.109** | 1.48 | 0.206 | 0.57 | −0.061 |
| **clean** | F6 {} reg S2 | +0.210 | **+0.006** | 0.08 | 0.218 | 0.43 | −0.223 |
| **clean** | F8 N=30 clf S2 | +0.061 | **−0.057** | −1.14 | 0.140 | 0.29 | −0.132 |
| **clean** | F14 reg S2 | +0.043 | **+0.074** | 0.81 | 0.257 | 0.64 | −0.109 |
| **clean** | F11(F6) reg S2 | +0.106 | **−0.047** | −0.68 | 0.194 | 0.36 | −0.212 |

**Cells positive on TEST: 25 of 48 contaminated, 13 of 48 clean** (VAL: 31 and 22 respectively). The unselected
FCFS book is negative on TEST for all four families (F6 −0.083 · F8 −0.045 · F14 −0.046 · F11 −0.065) — the TEST tape
is a different tape from VAL's, which is where half of the VAL "edge" came from.

**The D0 signature repeats.** On TEST the contaminated reversed tape is the better book: `F6 {}` clf S1 rev
**+0.264 R, t 4.62, 93% of weeks green**; `F11(F6)` clf S1 rev **+0.243, t 4.53, 93% green**; `F8 N=30` clf S1 rev
+0.209, t 5.24. Every contaminated cell's reversed twin is positive on TEST. The clean pipeline's reversed twins are
positive too but much weaker (+0.03..+0.16), and its real twins are ~0 — which is what a pipeline with nothing to
find looks like.

Per-month, the clean headline cells (sum of net R, booked, 2025-10 → 2026-09):

```
month     F6 {}   F8 N=30   F14     F11(F6)
2025-10    +8.7    -16.1    +4.7     -13.3
2025-11   +10.2     -1.5    -0.8      -2.6
2025-12    +8.9     +2.4    +3.6     -14.9
2026-01   +30.5     -0.0    -4.2      +9.3
2026-02    +3.0     -1.6    +3.1      +9.7
2026-03   +14.7     +5.6    -8.9      -1.5
2026-04   +16.9     +8.3    +6.6     +21.7
2026-05   +33.6     +4.4   +13.1      +7.3
2026-06   -26.6     +2.5    +8.7     -11.1
2026-07   -10.1     -7.5    -3.8      -8.3
2026-08   +31.5     -3.4    +5.8      +9.4
2026-09    +7.1     +0.2    -0.3      -1.7
```

Only `F6 {}` is positive in 10 of 12 predicted months, and its two losing months (−26.6 and −10.1 R) are both in
TEST. At 4 slots and $100 risk that series is ≈ +$1,000/month with a −$2,700 month — inside its own MDE, and it is
the SAME book as FCFS plus 0.05 R of selection.

## H. What it means

1. **H4 is not rescued by candidates4's new features.** With the availability leak removed, a walk-forward booster
   on 47 causal features — signal-bar shape, level history, cum dollars, VWAP distance, prev-day range, day context,
   intraday index state, breadth — clears G1 in **0 of 48** cells and adds **+0.05 R/trade** over no selection on
   VAL, which does not survive TEST. Stage C's conclusion (the raw families are flat and the grid holds no signal of
   the size it can detect) is unchanged by adding a selection layer.
2. **The premarket/news pair is not the missing leg.** Its apparent power here was coverage, not content: the
   pre-registered two-leg ORB rule (`news AND PM$ > $5.82M`) is negative or flat in every family, band and split,
   and the genuinely causal "no premarket trading at all" category is ~0 (+0.024 R). The one direction worth a
   future look is `news_only` — news WITHOUT the premarket-dollar leg — positive on TRAIN (+0.078..+0.093, t 2.0–2.8)
   and on TEST (+0.091/+0.136/+0.194) for F6/F11/F8 N=5. It is not the pre-registered rule and it is NOT adopted
   here; it needs its own pre-registration on a key set whose news coverage is built causally (E's pull is the right
   source; D's is not).
3. **The smallest effect visible.** VAL MDEs on the headline clean cells: F6 0.192 R · F8 N=30 0.117 · F14 0.199 ·
   F11 0.168 (≈2.2–3.8 R/week at 4 slots). TEST MDEs: 0.218 / 0.140 / 0.257 / 0.194. An edge smaller than that is
   not excluded by anything in this stage.
4. **For the rest of the program**: the availability audit of §A must be run on every feature in Stage E and
   Stage F before their numbers are quoted — E's news file (`news_presence_e.csv`) is keyed to the U1∪U2 causal
   universes and is safe by construction; `D/pm_bars.db` is NOT safe for any population other than D0's own, and
   candidates4's `pm_dollar_vol` column (52.9% empty) must never be used as a feature on a mixed-time population.
   If premarket dollars are wanted as a feature, they must be backfilled over the FULL key set of whatever
   population is being modelled, with `src='none'` recorded as a real zero.

## I. Cell count (PLAN §1)

**Declared in PREREG: 53** — 48 model cells (4 families × 3 models × 2 selections × 2 stop/exit targets) + 5
univariate two-leg cells.

**Actually evaluated**: 960 cell rows (2 pipelines × 3 periods × 160 rows, where 160 = 48 cells × 3 tapes + 16
control rows), 254 two-leg per-trade rows + 59 two-leg booked rows, ~80 diagnostic rows (cost decomposition 56,
premarket-source 15, missingness ~9), and a 500-draw permutation null over the 48 VAL cells. **Total looked at:
1,353 cell-instances**, on top of Stage C's 1,305 and Stage D0's 42+18.

**Deviations from PREREG, all disclosed**: (i) the intraday index state is read at the signal minute, not the entry
minute (stricter, PREREG §0.3); (ii) permutation importance was computed for the primary target on the real tape
only (declared reduction); (iii) `d1_orb.py` read its own TEST column in the same pass as TRAIN/VAL (§E); (iv) the
clean ablation (`D1_TAG=_clean`) was NOT pre-registered — it was added after the VAL tables exposed the availability
leak, and it is reported as a diagnostic re-run of the declared 48 cells, not as a new search; its TEST read happened
in the same single invocation as the declared pipeline's.

**Artefacts for a Stage-F independent check** (no candidate survived, so none is required, but the trade-level books
exist): `booked_{trainpred,val,test}{,_clean}.csv` carry every booked trade (day, entry minute, exit minute, symbol,
net R, exit reason) for all real-tape cells; `feat.csv` carries the features and both targets; `d1_parity.csv` is the
tie back to Stage C's scorer.
