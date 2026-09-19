# hod_frames2 — F5 retest · F6 absorption · F9 signal-minute cohort fields — REPORT (2026-09-19)

Pass 2 of the frame programme. Cells exactly as declared in `PREREG.md`, **committed `1f26c7a`
before any cell was scored**. Artifacts: `walk2.py` → `breaks2.csv` (**1,188,186 qualifying break
rows** over 98,203 symbol-days / 344 TRAIN+VAL sessions — the first pass in this programme to walk
EVERY break of a symbol-day, not only the first) · `score.py` → `score.log`, `cells.csv`,
`nulls.csv` · `supp.py` → `supp.log` · `supp2.log`. One python process at a time, `nice -n 10`,
`ulimit -v 3000000`; `cache.db`, `bars_sip.db` opened **read-only**. No config, `orb.yaml`, systemd
unit, cron, order or cache was written. The dry run was not touched. **TEST was never opened**
(`FREEZE.md`).

---

## VERDICT — **STAY DRY on all three frames.** 0 of 26 declared cells clear either bar.

*The three frames failed for three different reasons and two of them are corrections to this
programme's own beliefs.*

1. **F5 — the retest book is flat, and the `hod_losers` §6 lead that ranked it first was a
   different object.** §6's "+0.113 / +0.236 R for re-breaks" is reproduced here to the third
   decimal (**+0.236 TRAIN / +0.066 VAL**, `supp2.log` S6) — and it is a comparison of **175 / 69
   first-qualifying signals whose symbol-day had an EARLIER candidate break that did not qualify**,
   one trade per symbol-day in both arms. It was never a second trade. Built as an actual second
   trade, the retest reads **−0.027 R (TRAIN) / +0.000 (VAL)** at the signal level and
   **−0.066 / −0.000 gross, −0.147 / −0.085 net** as a book at 11.9 / 17.5 trades a week. The
   failure filter the mechanism rests on does nothing: of the two failure definitions, the
   price-based one (`prev_back5/15`) is the whole population (783 of 783 second breaks on TRAIN) and
   the stop-based one leaves **2.0 trades a week**.
2. **F6 — absorption is not the mechanism behind old highs, and the duration arm is the WRONG
   side.** The shelf (volume inside HOD ±0.5 % before the break, as % of ADV) is non-monotone and
   sign-flips: `>=20 %` reads **+0.211 R TRAIN / −0.180 VAL** on 72 / 39 trades. Shelf **duration**
   is wrong-side on BOTH splits (`shelf_bars >= 5`: −0.065 TRAIN / **−0.175 VAL, iid t −2.10**), and
   it is wrong-side for a legible reason — a long shelf marks a QUIET day: the `>=10 %-range-day`
   share falls from the base's 59 / 60 % to **44 / 38 %**. `hod_age_bars >= 20`, the age arm, is
   negative on both splits (−0.066 / −0.111). **Whatever `hod_fresh`'s `consol_bars >= 20` control
   is, it is neither shelf volume nor the age of the high.**
3. **F9 — the cohort-membership problem is now SOLVED at the signal minute, and it is worth
   nothing.** `rng_own` (session range so far ÷ the symbol's own 20-day median daily range) at
   `>= 1.5` puts **96.8 % (TRAIN) / 95.7 % (VAL)** of its trades on `>=10 %`-range days — pass 1's
   11:00 proxy accuracy (97.3 / 98.8 %), now available **at the break bar**, on every signal, at
   every hour — and its gross is **+0.038 / −0.097 R** against a base of −0.002 / +0.015.
   `hod_frames` §2.3 predicted exactly this and the prediction is now confirmed by a causal
   signal-minute field: **membership in the two-cohort split is cheap and carries no return.**

---

## 0. Reproduction gate — EXACT, and the independent rebuild is EXACT

| id | this pass | verdict |
|---|---|---|
| R1 `B2` TRAIN | 1,622 · 30.6/wk · −0.039 gross · −0.107 net · 32.1 % green · **−$17,346** | **MATCH** (Δ$ 0) |
| R2 `B2` VAL | 706 · 30.7 · +0.083 · +0.013 · 43.5 % · **+$893** | **MATCH** |
| **R3 independent rebuild** | a NEW all-breaks bar loop vs `pass2.py`'s first-qualifying rows: **98,203 shared symbol-days, same entry minute 100.0000 %, max \|Δstop\| 0.00e+00, max \|Δrr\| 3.55e-15** | **MATCH** |
| R3b the book from THIS pass's own rows | TRAIN 1,622 / −$17,346 · VAL 706 / +$893 — identical to R1/R2 | **MATCH** |

The pre-book first-break set is **7,027** rows on TRAIN+VAL, the same 7,027 `hod_bleed` re-simulated
from raw tape. Booked cost re-measured per cell (never the retired 0.2151 constant): **0.068 TRAIN /
0.070 VAL** on the B2 book, **0.082–0.086** on the retest cells (later minutes, wider quotes),
**0.044–0.073** on the F6/F9 cells.

## 0b. Availability audit — one field is VOID by the pass's own rail

| field | coverage | miss on winners | miss on losers | verdict |
|---|---|---|---|---|
| `shelf_share`, `shelf_bars`, `hod_age_bars`, `exp5_n`, `rng_sig`, `prev_*` | **100.0 %** | 0.0 % | 0.0 % | ok |
| `dollar_frac`, `rng_own` | 78.1 % | 23.4 % | 20.9 % | ok (inside the 5 pp rail) |
| **`add30_ratio`** | **32.8 %** | **58.8 %** | **73.2 %** | **DROP — outcome-dependent missingness (14.4 pp)** |

`add30_ratio` is NaN before 10:30 by construction, and early signals are disproportionately losers,
so the field is partly **an entry-clock filter**. `PREREG` §1 drops a field with a >5 pp gap. Its two
cells are printed below and are **VOID**; §9.2 shows the confound explicitly.

---

# F5 — THE RETEST BOOK  (7 cells)

**What the population actually looks like.** Of 4,575 TRAIN (2,452 VAL) first-break pre-book signals,
the same symbol-days carry **3,057 / 1,602 second** qualifying breaks and **17,906 / 11,071 third-or-
later** ones. Second breaks whose predecessor closed back below its level inside 5 bars: **779 / 480**
— i.e. **essentially every second break follows a predecessor that failed on the price test**
(783 of 783 within 15 bars). Second breaks whose predecessor's trade had already **stopped** by our
decision bar: **107 / 86**.

| cell | TRAIN n | /wk | gross | cost | net | grn % | **$** | VAL n | /wk | gross | net | grn % | **$** | H1/H2/VAL | same-signed + |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **F5-0 first break (= B2)** | 1,622 | 30.6 | −0.039 | .068 | −0.107 | 32.1 | −17,346 | 706 | 30.7 | +0.083 | +0.013 | 43.5 | +893 | −0.077/−0.003/+0.083 | no |
| **F5-a 2nd, prev STOPPED** | 106 | **2.0** | **+0.154** | .078 | +0.076 | 39.6 | **+801** | 86 | 3.7 | +0.014 | −0.067 | 47.8 | −580 | +0.321/−0.008/+0.014 | no |
| F5-b 2nd, prev back <5 bars | 627 | 11.8 | −0.069 | .082 | −0.151 | 30.2 | −9,443 | 402 | 17.5 | −0.005 | −0.089 | 34.8 | −3,596 | −0.007/−0.118/−0.005 | no |
| F5-c 2nd, prev back <15 bars | 631 | 11.9 | −0.066 | .082 | −0.147 | 30.2 | −9,290 | 402 | 17.5 | −0.000 | −0.085 | 39.1 | −3,410 | +0.000/−0.117/−0.000 | no |
| F5-d 2nd, stopped OR back15 | 631 | 11.9 | −0.066 | .082 | −0.147 | 30.2 | −9,290 | 402 | 17.5 | −0.000 | −0.085 | 39.1 | −3,410 | — | no |
| F5-e ANY re-break [ctrl] | 1,974 | 37.2 | −0.002 | .085 | −0.087 | 37.7 | −17,245 | 888 | 38.6 | −0.019 | −0.105 | 39.1 | −9,285 | −0.004/−0.000/−0.019 | no |
| **F5-f 3rd+ break [ctrl]** | 2,050 | 38.7 | **+0.015** | .085 | −0.070 | 34.0 | −14,345 | 924 | 40.2 | **+0.080** | −0.005 | **52.2** | −485 | **+0.012/+0.018/+0.080** | **YES** |

**Three readings.**

* **The mechanism's own filter is inert.** "The first break failed" selects 783 of 783 second breaks
  on the price definition — it is not a filter, it is a description of what a second break *is*. The
  one version that discriminates (the predecessor's trade actually stopped, and the stop is in the
  past at our decision bar) is **2.0 / 3.7 trades a week** — a fifth of the frequency floor — with
  **+0.154 / +0.014 gross**, halves **+0.321 / −0.008**, and **negative VAL dollars**. MDE on it is
  **0.381 R (TRAIN) / 0.395 R (VAL)**: the cell could not have resolved its own point estimate.
* **F5-f is the only same-signed-positive admission in the pass and it is a slot artifact.** At the
  SIGNAL level the third-or-later break reads **+0.016 TRAIN / −0.030 VAL** on 17,906 / 11,071
  signals (`supp.log` S1) — the book's +0.080 VAL gross is what the 12/4 **first-come** rule picks
  out of 481 signals a week, not a property of the admission, and its sign disagrees with the
  population it is drawn from. Its cost is 0.085 R at **96–97 % imputed** quotes (third breaks land
  at minutes the measured-NBBO set never sampled), and its net is negative on both splits. *(Declared
  caveat: `run_book` has no per-symbol cap, so F5-e/F5-f can hold more than one trade in a name on a
  day; they are controls, scored as declared.)*
* **The §6 lead was misread by the ledger, and this pass is where that is caught.** `hod_losers` §6
  put "RE-break (`n_break > 0`) vs first break: +0.113 / +0.236" at the top of the F5 queue.
  `n_break` is the index of the break among a symbol-day's CANDIDATE breaks — so `n_break > 0` on a
  `first_n0 == 1` row means *the first break that QUALIFIED was not the first break that happened*.
  One trade per symbol-day in both arms. Reproduced here: **TRAIN n 175 +0.225 vs n 4,400 −0.011
  (Δ +0.236); VAL n 69 +0.080 vs n 2,383 +0.013 (Δ +0.066)** (`supp2.log` S6) — and it says nothing
  about a second trade. **A "re-break" that is one trade and a "retest" that is a second trade are
  different books; the ledger conflated them for a pass.**

**Downstream check (pass 1's own test, run on every F5 cell).** `corr(gross R, range added after the
signal)` = **+0.31 … +0.46 (TRAIN) / +0.31 … +0.65 (VAL)** — the retest book is exactly as
outcome-entangled as the first-break book (+0.369 / +0.377). The retest does not escape §2.3; it
inherits it.

---

# F6 — ABSORPTION AT THE LEVEL  (9 cells)

The shelf is thin in this population: median **1.9 % of ADV over 2 bars**, p90 6.7 % / 5 bars.

| cell | TRAIN n | /wk | gross | net | grn % | **$** | VAL n | /wk | gross | net | grn % | **$** | ≥10 % day share T/V |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shelf >= 2 % of ADV | 1,270 | 24.0 | −0.061 | −0.130 | 32.1 | −16,571 | 608 | 26.4 | +0.051 | −0.018 | 39.1 | −1,124 | 60 / 60 % |
| shelf >= 5 % | 605 | 11.4 | −0.078 | −0.150 | 26.4 | −9,088 | 323 | 14.0 | −0.030 | −0.101 | 30.4 | −3,261 | 65 / 64 % |
| shelf >= 10 % | 228 | 4.3 | +0.029 | −0.042 | 37.7 | −959 | 119 | 5.2 | −0.126 | −0.199 | 34.8 | −2,367 | 71 / 69 % |
| **shelf >= 20 %** | 72 | 1.4 | **+0.211** | +0.143 | 39.6 | **+1,028** | 39 | 1.7 | **−0.180** | −0.251 | 26.1 | −981 | 79 / 77 % |
| F6-e shelf >= 5 % AND age >= 20 | 122 | 2.3 | −0.196 | −0.271 | 30.2 | −3,307 | 68 | 3.0 | −0.137 | −0.215 | 21.7 | −1,460 | — |
| F6-f age >= 20 alone [arm] | 240 | 4.5 | −0.086 | −0.155 | 37.7 | −3,727 | 175 | 7.6 | −0.111 | −0.184 | 43.5 | −3,220 | 67 / 63 % |
| shelf_bars >= 5 | 446 | 8.4 | −0.057 | −0.123 | 34.0 | −5,470 | 270 | 11.7 | **−0.129** | −0.198 | **17.4** | −5,344 | **44 / 38 %** |
| shelf_bars >= 10 | 112 | 2.1 | −0.139 | −0.207 | 24.5 | −2,314 | 76 | 3.3 | −0.219 | −0.288 | 39.1 | −2,192 | 38 / 24 % |
| shelf_bars >= 20 | 17 | 0.3 | +0.078 | +0.015 | 13.2 | +26 | 9 | 0.4 | −0.071 | −0.134 | 8.7 | −121 | 41 / 22 % |

* **Volume and duration point in opposite directions and neither survives.** Shelf VOLUME rises with
  the mover-day share (60 → 79 %) and its gross sign-flips between the years at every rung above
  2 %. Shelf DURATION *falls* with the mover-day share (59 → 38 %) and is negative on both splits at
  iid t −1.00 / **−2.10** — a level that has been sat on for 5+ minutes is a level on a quiet tape,
  and quiet tapes are where this book loses.
* **The `hod_fresh` control is not explained.** `consol_bars >= 20` (20 consecutive bars whose LOW
  held within 4 % of the level) was the only admission in 870 cells same-signed positive in H1, H2
  and VAL. Neither the absorption at the level (shelf) nor the age of the high (`hod_age_bars >= 20`,
  −0.086 / −0.111) reproduces it. What `consol_bars` encodes is *the low staying up*, not *volume
  changing hands at the high* — a base that holds, not supply that clears. **The F6 mechanism is
  refuted; the `consol_bars` mechanism is still unidentified.**

---

# F9 — SIGNAL-MINUTE COHORT FIELDS  (10 cells, 3 excluded by the pre-committed gate, 2 void)

## 9.1 The downstream gate, applied BEFORE scoring (PREREG §2, on the TRAIN split)

| field, top rung | split | n top | P(EOD ≥10 %) top / rest | Δ | Δ mean `rng_sig` | Δ mean `rng_after` | gross top / rest | excluded |
|---|---|---|---|---|---|---|---|---|
| `dollar_frac >= 50 %` | TRAIN | 601 | 73.5 / 60.3 % | +13.3 pp | **+2.00 pp** | **−0.85 pp** | +0.073 / −0.003 | **YES** |
| `dollar_frac >= 50 %` | VAL | 405 | 70.9 / 56.9 % | +14.0 | +3.00 | +0.76 | +0.077 / +0.006 | no |
| `exp5_n >= 6` | TRAIN | 909 | 46.0 / 62.5 % | −16.6 | −0.59 | −2.08 | +0.184 / −0.048 | no |
| `add30_ratio >= 2` | TRAIN | 255 | 53.3 / 61.1 % | −7.8 | −0.15 | −0.87 | +0.580 / +0.270 | no |
| `rng_own >= 1.5` | TRAIN | 250 | **96.8** / 59.9 % | **+36.9** | +5.67 | +0.02 | +0.038 / +0.009 | no |
| `rng_own >= 1.5` | VAL | 140 | **95.7** / 57.0 % | +38.7 | +8.13 | +2.99 | −0.097 / +0.026 | no |
| `shelf_share >= 20 %` [F6] | TRAIN | 72 | 79.2 / 58.9 % | +20.2 | +3.78 | +2.58 | +0.211 / −0.005 | no |

**`dollar_frac` — the F8 institutional-footprint field — is EXCLUDED by this pass's own
pre-committed rule** (on TRAIN its membership lift runs through the already-there channel and its
arrived-after channel is negative), and its three cells are therefore **not scored**. That is
recorded here in full because it is uncomfortable: at its top rung it is the **only field in the
pass with the same-signed gross separation on both splits** (+0.076 TRAIN / +0.071 VAL,
iid t +1.28 / +1.00, `supp.log` S2) — and its lower rungs are exactly the pattern the gate exists to
catch (`>=25 %`: TRAIN +0.118 at t **+2.57**, VAL **−0.080** at t −1.53). Its §2.3 decomposition is
*inconsistent between the years* (TRAIN: already-wide −0.083 vs arrived-after +0.229; VAL: +0.233 vs
−0.079), which is the honest reason to leave it quarantined rather than to argue it back in after
seeing the number. **It goes to pass 3 as a pre-registered cell with the gate re-specified, not into
this pass's verdict.**

## 9.2 The cells

| cell | TRAIN n | /wk | gross | net | grn % | **$** | VAL n | /wk | gross | net | grn % | **$** | H1/H2/VAL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F9-a `dollar_frac >= 10/25/50 %` | — | — | **EXCLUDED by the PREREG §2 gate — not scored** | | | | | | | | | | |
| F9-b `exp5_n >= 3` | 1,264 | 23.8 | +0.023 | −0.036 | 39.6 | −4,581 | 634 | 27.6 | +0.001 | −0.063 | 39.1 | −3,966 | −0.017/+0.061/+0.001 |
| F9-b `exp5_n >= 6` | 599 | 11.3 | +0.010 | −0.046 | 39.6 | −2,738 | 342 | 14.9 | −0.097 | −0.157 | 30.4 | −5,373 | −0.132/+0.139/−0.097 |
| F9-c `add30_ratio >= 1.0` **[VOID]** | 193 | 3.6 | **+0.250** | +0.200 | 47.2 | **+3,866** | 109 | 4.7 | −0.098 | −0.154 | 43.5 | −1,675 | +0.153/+0.321/−0.098 |
| F9-c `add30_ratio >= 2.0` **[VOID]** | 81 | 1.5 | +0.111 | +0.067 | 28.3 | +541 | 26 | 1.1 | +0.160 | +0.113 | 26.1 | +293 | −0.016/+0.235/+0.160 |
| F9-d `rng_own >= 0.5` | 1,440 | 27.2 | −0.033 | −0.100 | 34.0 | −14,338 | 681 | 29.6 | +0.061 | −0.010 | **52.2** | −650 | −0.078/+0.002/+0.061 |
| F9-d `rng_own >= 1.0` | 783 | 14.8 | −0.013 | −0.079 | 35.8 | −6,179 | 492 | 21.4 | −0.008 | −0.077 | 34.8 | −3,810 | −0.112/+0.050/−0.008 |
| F9-d `rng_own >= 1.5` | 196 | 3.7 | −0.050 | −0.114 | 28.3 | −2,239 | 132 | 5.7 | −0.049 | −0.117 | 34.8 | −1,550 | −0.063/−0.044/−0.049 |

**The two largest |t| separations in the pass both sign-flip between the years**, and both are worth
saying out loud because they are the shape every future candidate will have:

* `exp5_n >= 6` (six prior 5-minute range expansions): **TRAIN +0.232 R at iid t +4.96, VAL −0.145 R
  at t −2.11** (`supp.log` S2). That is the strongest TRAIN separation any *causal* field has
  produced in 896 cells, and it is worth −$5,373 on VAL.
* `add30_ratio >= 1.0`: **TRAIN +0.570 R at iid t +10.14**, the largest t in the programme, **VAL
  −0.094**. `supp2.log` S7 shows what it is: the field is undefined before 10:30, and on TRAIN the
  late cohort alone reads **+0.319 R against the population's −0.002**, while on VAL the same late
  cohort reads **−0.025 against +0.015**. It is the entry-minute effect — the programme's strongest
  single feature, and one that changes sign between the years — wearing a range ratio. The
  availability rail voided it before the number was looked at, which is the rail working.

**The F9 answer.** `rng_own >= 1.5` reaches the `>=10 %`-range cohort at **96.8 % / 95.7 %** — the
accuracy pass 1 could only buy at 11:00 with three-quarters of the day's signals gone — **at the
break bar, on every signal, at every hour**. Its gross is **+0.038 / −0.097**. The frame asked
whether the +0.8 R separation could be reached earlier; the answer is that the *membership* can be
reached earlier and perfectly, and the *return* is not in the membership. `hod_frames` §2.3 is
confirmed by a second, independent route.

---

## BOTH BARS, the nulls, the MDE

**Claim bar G1 — 0 of 25 scored cell-rows.** No cell has TRAIN net R > 0 with iid **and** clustered
t >= 2 at >= 10 trades/week. The best TRAIN net at >= 10/wk in the pass is **−0.036** (`exp5_n >= 3`);
the largest positive TRAIN clustered t on a scored cell is **+1.79** (`add30_ratio >= 1.0`, VOID and
at 3.6 trades a week). G2 was never evaluated; **TEST was never opened**.

**Live-exploration bar — 0 cells.** Nothing has positive dollars on both splits; the only cells with
positive TRAIN dollars (F5-a, shelf >= 20 %, both `add30` rungs, shelf_bars >= 20) run at
**0.3–3.6 trades a week** and four of the five lose money on VAL.

**Nulls — 44 cell × split bands, ALL 44 INSIDE.** Zero above, zero below. The seventh pass in a row
in which the owner's primary metric on this book is indistinguishable from pick count.

**MDE (80 % power, per trade, net) against the true 0.061–0.065 R break-even**: 0.088 / 0.132 R on
the B2 book, **0.081–0.143 / 0.119–0.182 R** on the F5 cells that trade often enough to matter,
0.099–0.184 / 0.141–0.231 R on the F6 cells that do, 0.094–0.159 / 0.132–0.200 R on F9-b/d. The
frequent cells are **powered** rejections; the six cells under 4 trades a week (F5-a, shelf >= 20 %,
shelf_bars >= 10/20, both `add30` rungs) have MDE 0.22–1.47 R and are **not** — their answer is the
frequency, which is itself the verdict.

**Multiplicity.** 26 declared decision cells (7 F5 + 9 F6 + 10 F9), of which **3 were excluded by the
pre-committed downstream gate before scoring** and **2 are void on the availability rail**; 2
reproduction rows, 1 declared downstream table, and 7 supplementary diagnostics (S1–S7) carry no
decision. **Programme cumulative: 870 + 26 = 896.** Expected largest |t| under a pure null over
26 × 2 ≈ 2.8–3.0; the largest positive TRAIN t on a scored, non-void cell is **+0.56** (F5-a).

## Known deviations, stated rather than buried

1. **`add30_ratio` is void on this pass's own availability rail** and its cells are printed only to
   show the confound. It is not part of the verdict.
2. **`rng_own` uses the symbol's 20-day median DAILY range, not its first-hour range** — declared in
   `PREREG` §1 before scoring, because the first-hour version needs ~2 M symbol-days of prior 1-min
   bars.
3. **`walk2.py` read `pop.csv` without `keep_default_na=False`**, so the ticker `NA` (39 symbol-days
   in TRAIN+VAL) was read as NaN and never walked. Verified inert: **every one of those 39 rows is
   priced under $20** and dies on the shipped price floor. The repo's standing `read_orb_csv` lesson,
   re-learned in a research script.
4. **`run_book` has no per-symbol cap**, so the two F5 controls (`ANY re-break`, `3rd+`) can hold more
   than one position in a name on a day. Declared and scored as written; the four decision cells
   (F5-a…d) admit at most one second break per symbol-day by construction.
5. Cost on the retest and third-break cells is **77–97 % imputed** (they fire at minutes the measured
   NBBO set never sampled). Every verdict above therefore also holds on **gross**, and is stated on
   gross wherever it is close.

## VERDICT — **STAY DRY.** No `HodBreakParams` change, no scan-rule change, no new admission field.

`config.yaml hod_break` stays exactly as the owner set it (`enabled: true, dry_run: true`);
`trading.enabled` and `orb.yaml` untouched. There is no SHIP-TO-DRY diff to write. For the record,
had F5 cleared, the diff would have been a **scan-rule change in `trading/hod_break.py::detect`** —
today `stale_break` retires a symbol after its first break; the retest book needs the candidate to
survive a failed break, a `prior_break` state on the candidate and an `admit_break_index` field on
`HodBreakParams`, plus the engine keeping the symbol subscribed after a stop-out. None of it is built.

**The next three frames are F10, F11 and F12, appended to `FRAMES.md`.**

*(`breaks2.csv` — 329 MB, 1,188,186 rows — is gitignored and stays local; it is regenerated
deterministically by `walk2.py`, which is checkpointed per day in `walk_state.json`.)*
