# frames15 — INSTITUTIONAL VOLUME AS A PATTERN OVER TIME — REPORT (2026-09-20)

Pass 15, run on the owner's instruction of 2026-09-20: *"the institutional volume is a big one imo —
by collecting relative hourly volume of the stocks and identifying a real interest, this is huge,
money must be there"*, with the same-day addendum *"identify the interest — might be multi-day as
well."* `PREREG.md` (23 cells + 7 named diagnostics, with the prediction and the falsifier) was
committed (`e962815`) **before any cell was scored**. Programme cell count **1,217 → 1,240**.

Artifacts: `hourly.py` -> `hourly_YYYY-MM.parquet` (2,130,789 symbol-hours) · `daily.py` ->
`daily15_{2024,2025,2026}.parquet` (4,734,044 name-days) · `profile.py` -> `hourly15.parquet` ·
`stage1.py` -> `sig15.csv` + the B2 reproduction gate · `armA.py` -> `cellsA.csv`, `availA.csv` ·
`armB_multi.py` -> `cellsB_multi.csv` · `armB_intra.py` -> `intra_YYYY-MM.csv` · `intra_ctrl.py` ->
`ctrl_YYYY-MM.csv` (200K+ control walks) · `scoreB_intra.py` -> `cellsB_intra.csv`.
One python process at a time, `nice -n 10`, `ulimit -v 3000000`, the hourly build checkpointed per
month and both intraday walks per month. `cache.db`, `bars_sip.db`, the daily panels and
`daily_bars` opened **read-only**. Nothing written outside `frames15/`. No config, `orb.yaml`,
engine, checker, systemd unit, cron or order was touched; Monday's 12:30 UTC boot and the 11:27 UTC
pre-boot suite are unaffected. **TEST was never opened** (`FREEZE.md`; the TEST months of the
intraday walk were deleted before scoring so they could not be read by accident).

---

## 0. THE SENTENCE THIS PASS WAS REQUIRED TO PRINT FIRST

**Sustained institutional volume is measurable on this tape, it is not rare, and it does not select
anything that makes money — and where it does separate, the sign is NEGATIVE.** A stock trading at
3x its own normal volume for the hour is, at the next bar, worth **−0.24 R TRAIN / −0.20 R VAL**
(day-clustered t −13.0 / −7.4) against a universe bound of −0.13 / −0.16 R on the same causal
population. A stock with three or more of its last five sessions above 1.5x its own 20-day mean
volume is worth **+0.11 % of price TRAIN / +0.02 % VAL** held one session against an unconditional
floor of **+0.04 % / +0.08 %** — i.e. it does not beat doing the same thing on a random name, and
its two splits disagree in sign on the excess. As a name-day selector on HOD-break's booked book the
same field takes the book from −$17,346 to **−$6,157** only by taking 70 % fewer trades, and its
kept-minus-rejected gross R is **−0.172 (t −1.96) / −0.044 (t −0.33)** — the picks it keeps are the
WORSE half in both splits.

**Verdict: STAY-DRY. 0 of 23 cells clears either bar. Nothing is proposed for the dry run, no
config change, no engine change.**

The one part of the owner's mechanism that DOES replicate is the ordering inside the volume itself:
**absorption beats its mirror on both splits** — high hourly volume with a small price move books
−0.120 / −0.107 R where the same volume with a large move books **−0.398 / −0.323 R** (t −13.3 /
−10.5), and on the multi-day side the single-session spike control is the worst cell in its arm.
Volume without price is not a buy signal here; volume WITH price is an actively expensive one. The
pre-registered falsifier asked for both a cleared cell and the absorption-beats-mirror ordering; the
ordering survived and the cells did not, so the frame is refuted on its own terms.

---

## 0b. Reproduction gate and the data rails — asserted in code, raising, before any cell was read

| id | gate | result |
|---|---|---|
| **R1** `B2` TRAIN | 1,622 trades · gross −0.039 · net −0.107 · 32.1 % green · **−$17,346** | **MATCH** (`stage1.py` asserts n, gross and $ and raises otherwise) |
| **R2** `B2` VAL | 706 · +0.083 · +0.013 · 43.5 % · **+$893** | **MATCH** |
| **R3** tape-vs-panel volume | the SIP tape's RTH volume / the daily panel's volume per symbol-day: median **0.868**, p5 0.658, **1.3 % below 0.5** | the thin-tape defect of `bf_zero` §6a is GONE from `bars_sip.db`; the 1.3 % are excluded by the `tape_ok` rail |
| **R4** corporate actions | raw-vs-adjusted 1-session return differing by > 1 pp | **0.35 %** of panel rows dropped (frames13 G-SCALE, same order) |
| **R5** availability | every field audited for coverage AND winner-vs-loser missingness; > 5 pp => VOID | applied, and it **voided four cells** — §1.1 |
| **R6** TEST | `FREEZE.md`, no exception taken | the intraday walk's 2026-06…09 files were **deleted** before scoring |

---

## 1. THE OBJECT — and how it differs from the volume forms that already died

    share_h(s,t)    = mean over the symbol's prior AVAILABLE sessions (rolling 20, min 3, most
                      recent prior appearance within 60 sessions) of  hour-h volume / that
                      session's RTH volume                       -- the symbol's OWN hour SHAPE
    hourmean_h(s,t) = adv20(s,t) * share_h(s,t)                   -- LEVEL from the DENSE daily panel
    hrv_h(s,t)      = hour-h volume on t / hourmean_h(s,t)        -- V1, per hour, NOT cumulative

`rv_profile` (`hod_filter_stack`, +0.28 TRAIN -> **−0.10 VAL**) is cumulative volume since the open
over a **universal** clock curve; `dollar_frac` is the same object in dollars; `bar_vol_x` is one
minute. **V1 is bar-level (one hour at a time) and its denominator is the stock's own hour shape**,
which is what the owner asked for and what had never been built. The shape/level split is deliberate
and is this pass's main measurement risk, stated in PREREG §1: the only intraday tape we own covers
mostly *candidate* days, so a LEVEL estimated from it would be biased; a SHAPE is far less exposed.
The raw-denominator sensitivity `hrv_raw` (the symbol's own prior mean hour-h volume, tape / tape,
immune to the level mismatch) is reported beside every V1 cell and **agrees in sign everywhere**
(A1 −0.701 vs −0.653; A2 −0.749 vs −0.775).

Distribution of `hrv` over 2,130,789 symbol-hours: p25 **0.48**, median **0.74**, p75 1.15,
p90 1.76, p99 4.32. Fire rates: `hrv >= 2` **7.4 %** of hours, `hrv >= 3` **2.6 %**,
`sus(2,2)` 2.24 %, `sus(2,3)` 1.08 %. Multi-day, on the dense panel: **`interest5 >= 3` fires on
5.82 % of universe symbol-days** (`>= 4` 1.86 %, `rvd >= 1.5` 12.63 %, V5 "accumulation on weakness"
3.52 %) — **a median of 270 names of 5,361 eligible every session**, so frequency was never the
binding constraint on any multi-day cell.

### 1.1 The availability audit — it VOIDED the hourly fields on arm A before their numbers were read

| field | coverage on the B2 book | miss on winners | miss on losers | verdict |
|---|---|---|---|---|
| `hrv`, `hrv_raw`, `hour_ret` | **16.5 / 16.7 / 17.6 %** | 91.4 % | 78.1 % | **VOID** (both limbs: < 80 % AND a 13.3 pp outcome gap) |
| `p_interest5`, `p_rvd` | 94.9 % | 5.1 % | 5.1 % | ok |
| `p_weak` | 98.1 % | 2.0 % | 1.9 % | ok |
| `tape_ok` | 100.0 % | 0.0 % | 0.0 % | ok |

The mechanism is not subtle and it is the `add30_ratio` trap of `hod_frames2` §0b exactly: **an
hourly field does not exist before 10:00**, and only **42.2 %** of B2's booked signals happen at or
after 10:00 (winners 38.4 %, losers 44.8 %) — the HOD break is an early-session object. A1-A4, A7
and A9's hourly limb are therefore **VOID as cells** by a rail written before the run; their numbers
are printed below as diagnostics, restricted to the >= 10:00 base on both sides of the comparison so
the comparison is at least internally valid. On arm B the same fields have full coverage, because
there the hour close IS the decision time.

---

# ARM A — the fields as a NAME-DAY SELECTOR on HOD-break's B2 book (9 cells)

Kept-minus-rejected **gross R** on the 2,328 booked trades, day-clustered two-sample t, plus the
book re-run with the filter applied at ranking time.

| cell | field | split | keep n | keep R | rej R | **delta** | clust t | MDE | halves (TRAIN) | re-booked $ | green % (null p95) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A1** | `hrv >= 2` *(VOID)* | TRAIN | 30 | −0.721 | −0.020 | **−0.701** | **−4.39** | 0.156 | −0.708 / −0.737 | −$6,608 | 13.2 (13.2) |
| | | VAL | 21 | −0.510 | +0.012 | **−0.522** | **−2.43** | 0.254 | | −$4,456 | 13.0 (21.7) |
| **A2** | `hrv >= 3` *(VOID)* | TRAIN | 12 | −0.785 | −0.036 | −0.749 | **−3.46** | 0.156 | −1.060 / −0.589 | −$2,949 | 3.8 (7.5) |
| | | VAL | 12 | −0.344 | −0.016 | −0.328 | −1.15 | 0.254 | | −$2,388 | 4.3 (13.0) |
| **A3** | `sus(2,2)` *(VOID)* | TRAIN | 3 | −0.231 | −0.048 | −0.183 | −0.26 | 0.156 | −1.089 / +0.198 | −$855 | 3.8 (7.5) |
| | | VAL | 6 | −0.075 | −0.030 | −0.045 | −0.10 | 0.254 | | −$633 | 13.0 (17.4) |
| **A4** | `sus(2,3)` *(VOID)* | TRAIN | 2 | +0.188 | −0.049 | — | — | 0.156 | −1.089 / +1.466 | −$213 | 1.9 |
| | | VAL | 3 | −0.472 | −0.026 | −0.446 | −0.93 | 0.254 | | −$324 | 4.3 (8.7) |
| **A5** | `interest5 >= 3` | TRAIN | 234 | −0.186 | −0.014 | **−0.172** | −1.96 | 0.109 | −0.210 / −0.158 | −$6,157 | 41.5 (49.1) |
| | | VAL | 96 | +0.045 | +0.089 | −0.044 | −0.33 | 0.165 | | −$2,207 | 47.8 (52.2) |
| **A6** | `rvd_1 >= 1.5` | TRAIN | 456 | −0.069 | −0.027 | −0.042 | −0.56 | 0.109 | +0.011 / −0.157 | −$9,250 | 26.4 (49.1) |
| | | VAL | 208 | −0.008 | +0.121 | −0.129 | −1.36 | 0.165 | | −$310 | 47.8 (65.2) |
| **A7** | absorption *(VOID)* | TRAIN | 4 | −0.705 | −0.045 | −0.660 | −1.69 | 0.156 | −0.586 / −1.063 | −$2,060 | 0.0 (5.7) |
| | | VAL | 2 | −1.058 | −0.023 | — | — | 0.254 | | −$837 | 0.0 (4.3) |
| **A8** | V5 weakness | TRAIN | 112 | −0.014 | −0.041 | +0.027 | +0.20 | 0.109 | +0.060 / −0.128 | −$3,828 | 30.2 (43.4) |
| | | VAL | 54 | +0.000 | +0.090 | −0.089 | −0.54 | 0.165 | | −$412 | 43.5 (56.5) |
| **A9** | V6 `sus(2,2)` OR `interest5>=3` | TRAIN | 95 | −0.183 | −0.029 | −0.155 | −1.12 | 0.156 | −0.197 / −0.158 | −$2,478 | 41.5 (47.2) |
| | | VAL | 43 | −0.057 | −0.026 | −0.032 | −0.15 | 0.254 | | **+$800** | 52.2 (60.9) |

**Reading.** Of 18 split-cells, **15 have a negative delta and 2 are undefined for want of n**; not
one is positive on both splits. The owner's interaction cell A9 is the only re-booked book positive
in any split (+$800 on VAL at 7.0 trades a week) and its green-week share, 52.2 %, sits **inside its
own count-matched null band (p95 60.9)** — that is pick count, not skill, and it is negative on
TRAIN. The two cells with real |t| (A1, A2) are VOID by the availability rail AND point the wrong
way: **a break that happens in an hour already trading at 2-3x the stock's own normal is worth
−0.7 R**. Read with F24/F41 — the only significant object on this tape is name-day selection — this
pass's contribution is that **elevated own-volume is not the name-day selector**; if anything it is
an anti-selector.

**Wrapper enrichment (pass 9's confound).** `hrv >= 2` keeps **20.0 %** wrappers against a 43.3 %
base and `hrv >= 3` keeps **8.3 %** — the hourly field strongly **DE-selects** leveraged wrappers
(their volume shape is tied to the underlying's and rarely spikes against their own history), so
none of A1/A2's negative is a wrapper artefact; it is the opposite of one. `interest5 >= 3` keeps
42.7 % against 38.3 % — a mild enrichment, far too small to carry a −0.172 R difference.

---

# ARM B (multi-day) — the owner's core hypothesis as a STANDALONE detector (8 cells)

Universe: the dense point-in-time daily panel, ex-test-tickers, ex-names-absent-from-`daily_bars`,
close >= $5, ADV$ >= $1M, corporate-action rail -> **2,304,334 symbol-days, 8,683 symbols, 429
sessions**. Entry = the decision session's **closing auction** (no quoted spread, frames13 F42),
exit = the closing auction h sessions later, **financing charged at 7.0 % APR per night**. Wrapper
share of the universe: **36.2 %**.

### 2.1 D1 — the floor this arm stands on (the number every cell must beat)

| h | TRAIN | VAL |
|---|---|---|
| 1 session | **+0.0352 %** of price (+0.018 R, t +0.51) | **+0.0798 %** (+0.040 R, t +0.83) |
| 2 sessions | +0.0683 % (t +0.73) | +0.1508 % (t +1.17) |
| 5 sessions | +0.1860 % (t +1.32) | +0.3097 % (t +1.58) |

This is frames13 F40's floor rebuilt on this pass's own universe and it reproduces its sign and
order of magnitude (+0.062 % a night unconditionally there, +0.035 % a session here on a
$5 / ADV$1M-filtered universe). **A multi-day hold is not a free option: it starts ABOVE zero, so a
cell must beat the floor, not zero.**

### 2.2 The cells

| cell | field | h | n | TRAIN % (excess over D1) | t | MDE | VAL % (excess) | t | halves | ex-top-5 % | wrap | book $ TRAIN / VAL | green % (null p95) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **B1** | `interest5 >= 3` | 1 | 111,770 | +0.1120 (**+0.0768**) | +0.44 | 0.325 | +0.0186 (**−0.0612**) | +0.13 | +0.140/+0.079 | −0.555 | 34 % | +$1,072 / +$3,086 | 50.9 (58.5) / 47.8 (60.9) |
| **B2** | `interest5 >= 3` | 2 | 111,501 | +0.2345 (+0.1662) | +1.13 | 0.419 | +0.0075 (−0.1433) | +0.04 | +0.328/+0.127 | −0.594 | 34 % | +$13,556 / +$2,897 | 60.4 (66.0) / 39.1 (60.9) |
| **B3** | `interest5 >= 3` | 5 | 110,806 | +0.2932 (+0.1072) | +1.13 | 0.614 | −0.0674 (−0.3771) | −0.29 | +0.376/+0.198 | −0.948 | 34 % | +$6,288 / −$5,123 | 50.9 (58.5) / 43.5 (52.2) |
| **B4** | `interest5 >= 4` | 5 | 36,031 | +0.2727 (+0.0867) | +0.81 | 0.776 | −0.0947 (−0.4045) | −0.34 | +0.344/+0.179 | −1.215 | 34 % | +$6,288 / −$5,123 | 50.9 / 43.5 |
| **B5** | V3xV4 abs., abs(ret5)<=3 % | 5 | 40,688 | +0.1264 (−0.0595) | +1.29 | 0.297 | +0.2165 (−0.0932) | +1.51 | +0.074/+0.177 | −0.490 | **52 %** | −$427 / −$370 | 47.2 (52.8) / 39.1 (47.8) |
| **B6** | V3xV4 abs., abs(ret5)<=3 % | 2 | 40,948 | +0.0859 (+0.0176) | +1.02 | 0.196 | +0.0790 (−0.0718) | +0.78 | +0.070/+0.102 | −0.308 | **52 %** | +$1,991 / **+$11,600** | 50.9 (58.5) / **69.6 (73.9)** |
| **B7** | V5 accumulation on weakness | 2 | 64,908 | +0.1814 (+0.1131) | +0.66 | 0.379 | +0.4598 (+0.3090) | +1.58 | **−0.032/+0.395** | −0.561 | 36 % | +$4,560 / −$3,549 | 49.1 (58.5) / 47.8 (56.5) |
| **B8** | **CONTROL** `rvd >= 3` spike | 1 | 40,295 | −0.0543 (−0.0895) | −0.45 | 0.468 | +0.0266 (−0.0533) | +0.17 | −0.085/−0.023 | −0.814 | 47 % | **−$35,541 / −$14,792** | 24.5 (41.5) / 34.8 (47.8) |

*(B3 and B4 produce the identical book: with 4 concurrent slots and a 5-session hold the ranking
never reaches an `interest5 = 3` name — the book is made of 5s and 4s in both cells.)*

**Reading, against the pre-committed bar** (positive weekly $ AND >= 50 % green on BOTH splits at
>= 10 tr/wk, clustered t >= 2, halves same-signed):

1. **No cell reaches t = 2 on either split.** The largest is B7's VAL +1.58 and B5's +1.51.
2. **Every `interest5` cell FAILS the excess test.** The raw mean is positive on TRAIN and its
   excess over the floor is positive there, but on VAL the excess is **negative in all four**
   (−0.06, −0.14, −0.38, −0.40 pp). The multi-day interest cohort does not beat a random eligible
   name held the same way.
3. **B6 is the one cell positive on both splits with same-signed halves** (+0.086 / +0.079 %,
   halves +0.070 / +0.102) and its VAL book reads **+$11,600 at 8.7 trades a week, 69.6 % green** —
   and it is killed by its own count-matched null: **69.6 % sits below the null's p95 of 73.9 %**,
   the trade count is under the 10/wk floor, and the excess over D1 is −0.072 pp on VAL. Its wrapper
   share is **52 % against a 36 % base**, so half of it is leveraged-wrapper drift.
4. **B7 fails on halves** (−0.032 TRAIN-H1 / +0.395 TRAIN-H2) — the classic one-half artefact.
5. **B8, the control, behaves exactly as the mechanism predicted.** A single-session volume spike is
   the WORST cell in the arm (−$35,541 TRAIN, 24.5 % green, an 11-week red streak). "Already
   discovered" is expensive. Second confirmation of the absorption-vs-spike ordering.
6. **Every cell is tail-carried.** Ex-top-5 % is **−0.31 % to −1.22 % of price** in all eight,
   without exception. Remove one trade in twenty and the whole arm is deeply negative — the shape
   frames13 F40 found on the overnight floor and the owner has already rejected once.

---

# ARM B (intraday) — V1/V2/V4 as a standalone 1-minute detector (6 cells)

**The universe rail, decided before any number was read.** `bars_sip.db` holds the *causal
superset*: symbol-days whose **session high reached open x 1.05** — a condition known only at the
END of the day. Scoring a detector on that membership is the `bf_zero` §6b look-ahead verbatim. So
every trade carries `gate5`: TRUE iff the session high **up to the signal hour's close** already
reached open x 1.05, i.e. membership is causal at the decision bar (and is literally HOD-break's own
admission gate). **Cells are scored on `gate5` only** — 26.9 % of the 42,224 walked signals. Entry =
the next bar's OPEN under a +0.6 % cap (a fill above the cap is a SKIP, never a touch fill); stop
2 % (R = 2 % of price); cost = the frames14 F45 measured minute-of-day NBBO median, half at each
leg, **a declared proxy, not a per-trade measurement**.

### 3.1 The placebo decomposition (bare exit, gate5 population, gross R)

| object | TRAIN | VAL | what it holds fixed |
|---|---|---|---|
| **D1** the universe bound — a non-signal name, same session, same bracket | −0.134 R (−0.267 % of price), t −5.56 | −0.162 R (−0.324 %), t −5.08 | the clock and the day |
| **D3** the same name-day at another hour (causal) | −0.125 R (−0.250 %), t −7.67 | −0.116 R (−0.231 %), t −5.37 | day + name |
| **the SIGNAL hour itself** | **−0.209 R (−0.418 %), t −12.57** | **−0.166 R (−0.332 %), t −6.89** | nothing — the book |

Read down the column: on a day that has ALREADY run 5 % above its open, buying the next bar is worth
about −0.13 R whatever you do, and **buying it right after an elevated-volume hour is worth 0.04-0.08
R LESS than buying it at a random hour of the same name-day**. The signal is not neutral; it is the
worst of the three.

### 3.2 The cells

| cell | field | split | n | gross R (% of price) | t | MDE | cost R | net R | book /wk | green % (null) | weekly $ | worst wk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **B9** | `hrv >= 3` | TRAIN | 5,637 | **−0.242** (−0.483 %) | −13.01 | 0.066 | 0.117 | −0.359 | 32.3 | **0.0** (1.9) | −$1,449 | −$2,846 |
| | | VAL | 3,386 | −0.198 (−0.396 %) | −7.38 | 0.102 | 0.117 | −0.315 | 32.7 | 4.3 (8.7) | −$1,339 | −$3,425 |
| **B10** | `sus(2,3)` | TRAIN | 2,796 | −0.102 (−0.204 %) | −4.07 | 0.082 | 0.109 | −0.211 | 20.9 | 9.4 (17.0) | −$452 | −$1,706 |
| | | VAL | 1,796 | −0.088 (−0.176 %) | −3.04 | 0.099 | 0.109 | −0.197 | 23.3 | 17.4 (21.7) | −$588 | −$1,958 |
| **B11** | absorption | TRAIN | 1,973 | −0.120 (−0.240 %) | −5.47 | 0.064 | 0.110 | −0.230 | 20.5 | 13.2 (15.1) | −$507 | −$1,577 |
| | | VAL | 1,185 | −0.107 (−0.214 %) | −2.77 | 0.104 | 0.110 | −0.217 | 22.5 | 17.4 (21.7) | −$552 | −$1,803 |
| **B12** | **CONTROL** mirror | TRAIN | 2,238 | **−0.398** (−0.797 %) | −13.33 | 0.097 | 0.126 | −0.525 | 27.3 | 0.0 (1.9) | −$1,551 | −$3,072 |
| | | VAL | 1,363 | −0.323 (−0.645 %) | −10.51 | 0.136 | 0.127 | −0.449 | 31.6 | 4.3 (8.7) | −$1,492 | −$3,554 |
| **B13** | B10 + 2R target | TRAIN | 2,796 | −0.109 (−0.217 %) | −5.27 | 0.059 | 0.109 | −0.218 | 21.2 | 9.4 (17.0) | −$495 | −$1,706 |
| | | VAL | 1,796 | −0.075 (−0.150 %) | −2.83 | 0.096 | 0.110 | −0.185 | 24.7 | 17.4 (21.7) | −$554 | −$1,966 |
| **B14** | B10 + ORB static lock | TRAIN | 2,796 | −0.101 (−0.203 %) | −4.05 | 0.082 | 0.109 | −0.210 | 21.0 | 9.4 (17.0) | −$436 | −$1,706 |
| | | VAL | 1,796 | −0.082 (−0.163 %) | −2.97 | 0.097 | 0.109 | −0.191 | 23.7 | 17.4 (21.7) | −$548 | −$2,002 |

All six are **negative on GROSS**, on both splits, at MDEs of 0.06-0.14 R — this is a measurement,
not a power failure, and the RUNBOOK's step-2 rule applies: no cost or fill model revives a
gross-negative book. The three exits (bare / +2R / lock) differ by 0.01 R: **the exit is not the
problem, the entry is.**

**The ungated diagnostic (a universe look-ahead — NOT a cell).** Without `gate5` the same cells read
−0.057 / −0.033 (B9), −0.029 / **+0.014** (B10), −0.019 / **+0.005** (B11), −0.128 / −0.093 (B12).
The two near-zero VAL readings exist **only** on a population selected with end-of-day knowledge and
are printed here solely so nobody rediscovers them later and mistakes them for an edge.

**Wrapper enrichment:** 9-12 % of the intraday cells' picks against the gated tape's own mix — the
hourly field de-selects wrappers here too, consistent with arm A.

---

## 4. THE POWER STATEMENT (the MDE per field, so the null is quantified and not asserted)

| field | where scored | MDE (80 % power, day-clustered) | the honest claim |
|---|---|---|---|
| **V1** `hrv` | arm A (VOID), arm B intraday | 0.156 / 0.254 R (A); **0.064-0.104 R** (B) | at 5.6K / 3.4K trades we would have seen >= 0.07-0.10 R; we measured **−0.24 / −0.20** |
| **V2** `sus(k,N)` | arm A (VOID), arm B intraday | 0.156 R (A, n = 3 and 2 — no power at all); 0.082 / 0.099 R (B) | on arm A the field is untestable on this book; on arm B it is −0.10 / −0.09 |
| **V3** `interest5`, `rvd_j` | arm A, arm B multi-day | 0.109 / 0.165 R (A); **0.196-0.776 % of price** (B) | the multi-day cohort's excess over the floor is inside the MDE in every cell — the honest word is **UNMEASURED at this effect size**, not "zero" |
| **V4** absorption | all three arms | 0.156 R (A); 0.196-0.297 % (B multi); 0.064 / 0.104 R (B intra) | the absorption-vs-mirror ORDERING is measured (t −13 vs −5); the absorption LEVEL is negative |
| **V5** weakness | arm A, arm B multi-day | 0.109 / 0.165 R (A); 0.379 % (B) | positive point estimates, wrong-signed halves, inside the MDE |
| **V6** the interaction | arm A | 0.156 / 0.254 R | −0.155 / −0.032; one positive book inside its own null band |

The multi-day arm is the one place where this pass is **power-limited rather than conclusive**: an
edge of, say, +0.15 % of price a session (gross, before any cost) would sit inside B1's 0.325 pp MDE
and we could not have resolved it. What we CAN say is that the cohort does not beat its own
unconditional floor, that its excess flips sign between splits, and that ex-top-5 % it is deeply
negative in every cell.

---

## 5. VERDICT AND THE SHIP DECISION

**STAY-DRY. 0 of 23 cells clears either bar.** Nothing goes to the dry run. No `HodBreakParams`
universe filter is proposed (arm A's fields are either VOID or anti-selective); no new
`HodBreakEngine` book variant is proposed (arm B is negative on gross in the intraday arm and
floor-beaten in the multi-day arm). `hod_break` stays `enabled: true, dry_run: true`; `config.yaml`
and `orb.yaml` are exactly as the owner set them.

**The honest sentence on the owner's hypothesis.** Sustained and absorbing volume DOES identify a
real population — 5.8 % of symbol-days carry three or more elevated sessions in five, and those
names are measurably different — but on this tape, at these horizons, at this book size, over
2025-01 -> 2026-05, at the measured cost, **the difference is not direction**. Elevated own-volume
selects variance and attention, exactly as frames14 F44 found for the overnight tail: the same
fields that pick the right tail pick the left tail, every cell dies ex-top-5 %, and the intraday
version is actively negative because by the time a stock has traded 3x its normal hour it has
already moved. The part of the owner's mechanism that survives is the ordering — **volume WITHOUT
price beats volume WITH price, consistently, in two independent arms** — but both sides of that
ordering are below zero on the long side, which is a statement about where the profitable side of it
might be, not about this book.

---

## 6. THE NEXT THREE FRAMES (appended to `hod_frames/FRAMES.md` behind F46-F48)

**F49 — THE MIRROR CELL AS A SHORT.** The largest |t| this pass produced is B12: a name already
>= 5 % above its open, trading >= 3x its own hour-normal, with a **large** price move in that hour,
books **−0.398 R TRAIN / −0.323 R VAL (t −13.3 / −10.5)** long, at 27-32 trades a week, on 3,601
trades, in both halves. The frame prices the other side: borrow availability and fee from the
broker's own list, locate rules, the short-sale circuit-breaker rule on a −10 % day, hard-to-borrow
exclusion, and the measured NBBO at the short's own minutes — with the pre-committed kill being any
one of (no borrow on the names that carry it) / (fee >= the edge) / (the edge concentrated in the
same top 5 % that kills every long cell). Runs first because it is the only object in 1,240 cells
with a double-digit t that has never been priced on its natural side.

**F50 — THE MULTI-DAY COHORT AS A PORTFOLIO, NOT A 4-SLOT BOOK.** Every multi-day cell here was
forced through HOD's slot rule (12/day, 4 concurrent) because that is the engine we own, and at a
5-session hold that is 3.5 trades a week — a frequency at which nothing can be resolved and at which
one tail trade is the book. The frame re-asks the same eight cells as an **equal-weight portfolio of
50-200 names rebalanced daily**, in % of NAV, with the financing and the auction legs measured, and
with the tail cap applied INSIDE the portfolio rather than to a trade list. It settles the question
this pass could not: is `interest5` worth anything at a size where the tail averages out, or is the
floor all there is?

**F51 — A DENSE HOURLY PROFILE ON A SMALL UNIVERSE.** V1/V2 were VOID on arm A at 16.5 % coverage
and their denominators everywhere rest on the symbol's *candidate-day* sessions, because the only
intraday tape we own is the causal superset. The frame buys the missing denominator for a bounded
universe — every session of ~300 names (Databento/Alpaca 1-min, about $0.0004 per symbol-day, ~$60
for two years) — rebuilds `share_h` on ALL sessions rather than candidate ones, and re-runs V1/V2/V4
on a population whose membership is unconditional. It is the only way to distinguish "the field is
null" from "our denominator was built from the wrong days", and it is cheap.

*(Order: F49, F50, F51 — F49 because it is the largest measured effect in the programme, F50 because
it needs no new data, F51 last because it spends money.)*
