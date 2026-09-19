# hod_frames3 — F11 identify `consol_bars` · F10 the footprint field · F12 adaptive vs frozen — REPORT (2026-09-19)

Pass 3 of the frame programme. Cells exactly as declared in `PREREG.md`, **committed `b4b8171`
before any cell was scored**. Artifacts: `walk3.py` → `feat3.csv` and `walk3b.py` → `feat3b.csv`
(the four F11 decomposition fields + `consol_bars` at **every candidate break bar** of all 98,203
TRAIN+VAL symbol-days, attached to `hod_frames2/breaks2.csv`'s 1,188,186 qualifying-break rows) ·
`score11.py` → `score11.log`, `cells11.csv`, `nulls11.csv` · `supp11.py` → `supp11.log` ·
`score10.py` → `score10.log` (+ `score10_imputed.log`, the pre-fetch ladder), `cells10.csv`,
`nulls10.csv` · `fetch_nbbo3.py` → `nbbo3.csv` (466 fresh SIP quote-minutes) · `score12.py` →
`score12.log`, `cells12.csv`, `f12_weeks.csv`. One python process at a time, `nice -n 10`,
`ulimit -v 3000000`; `cache.db`, `bars_sip.db`, `daily_bars` opened **read-only**. No config,
`orb.yaml`, systemd unit, cron, order or cache was written. The dry run was not touched.
**TEST was never opened** (`FREEZE.md`).

---

## VERDICT — **STAY DRY on all three.** 0 of 24 declared cells clear either bar.

*And this pass closes two open questions and pre-refutes a whole class of future pass.*

1. **F11 — `consol_bars >= 20` is retired. None of its four candidate contents carries the sign, and
   the field itself is the WRONG SIDE once the clock it implies is held fixed.** The declared
   difference test: inside the admission "the first qualifying break at or after 09:50",
   `consol_bars >= 20` reads **+0.018 R (TRAIN, clustered t +0.09) / −0.133 R (VAL, clustered t
   −1.84)**, and on C1's own SPY-up days it is **−0.135 / −0.188** (iid t −2.34 / −2.37) — wrong-side
   on **both** splits. What carries C1's three-era sign is the **SPY 09:35 day gate**: `base × spy_r5
   > 0`, with no `consol_bars` anywhere in it, reads **+0.032 / +0.165 gross, 43.4 % / 65.2 % green
   weeks, −$3,059 / +$3,439** at 15.6 / 15.9 trades a week — a BETTER VAL cell than C1 — and
   `clock590 × spy_r5 > 0` is **same-signed positive in H1, H2 and VAL** (+0.094 / +0.008 / +0.112)
   with no `consol_bars` either. The programme's "only unexplained survivor" was the day gate wearing
   a base-length filter, and the filter **dilutes** it (the gate separates **+0.388 / +0.189** on the
   base and only **+0.219 / +0.147** on the `consol_bars >= 20` admission).
2. **F10 — the quarantined field is scored, its cost is now MEASURED, and it is not era-consistent at
   any rung that trades.** With the gate re-specified as a diagnostic, `dollar_frac`'s ladder is
   **H1-2025-negative at p50, p60, p70 and p80**; the one era-consistent rung, **p90, runs at 5.7 /
   8.0 trades a week** — under the declared floor, like every other survivor this programme has
   produced. The dedicated NBBO fetch did exactly what it was for: the p80 rung's imputed-cost share
   falls **44.6 % → 3 % / 5 %**, its cost/R falls to 0.056 / 0.062 and its gross rises to **+0.065 /
   +0.096** — **and its H1 stays −0.024.** Cost was never the reason.
3. **F12 — READING (ii). The separations are NOISE, and every future frozen-rule pass on this book is
   pre-refuted.** A weekly rolling refit of the admission window loses to the frozen book on the
   owner's primary metric in **every arm and every OOS half**: green weeks **36.4 / 30.6 / 31.7 %**
   (refit, L = 20 / 26 / 34) against **38.2 / 38.8 / 39.0 %** (frozen), on 55 / 49 / 41 OOS weeks, at
   a third of the frequency (9.5–11.4 vs ~31.7 trades a week). Churn is **83 / 81 / 65 %** — 44
   distinct configurations in 55 weeks — and the band whipsaws between the **earliest** and the
   **latest** entry-minute deciles, the two opposite ends of the same axis. ORB's answer (refit the
   selection, 26 weeks, and it pays) **does not transfer**: ORB refits a selection that has an edge
   to rank; this book's separations flip sign because they are noise, and a refit tracks the noise.

---

## 0. Reproduction gate — EXACT on all four rows

| id | this pass | reference | verdict |
|---|---|---|---|
| R1 `B2` TRAIN | 1,622 · 30.6/wk · −0.039 · −0.107 · 32.1 % · **−$17,346** | identical | **MATCH** (Δ$ 0) |
| R2 `B2` VAL | 706 · 30.7 · +0.083 · +0.013 · 43.5 % · **+$893** | identical | **MATCH** |
| R2b `B2` rebuilt from THIS pass's rows | 1,622 / −$17,346 · 706 / +$893 | identical | **MATCH** |
| R3 `consol_bars >= 20` × n5 | 1,411 · 26.6 · +0.028 · −0.041 · 34.0 % · **−$5,749** · 714 · 31.0 · +0.050 · −0.023 · 47.8 % · **−$1,629** | `hod_fresh` §3 | **MATCH** |
| R4 **C1** = R3 × `spy_r5 > 0` | 731 · 13.8 · +0.100 · +0.033 · 43.4 % · **+$2,391** · 368 · 16.0 · +0.123 · +0.050 · 47.8 % · **+$1,827** | `hod_fresh` §5 | **MATCH** |

R3/R4 are reproduced from a **different population object** than `hod_fresh` used — that pass emitted
one row per (symbol-day × rung) from its own bar walk; this pass selects the first qualifying break
satisfying the rung out of `breaks2.csv`'s all-breaks stream and re-applies the cascade. Same book to
the dollar. The booked cost is re-measured per cell (0.047–0.092 R; never the retired 0.2151).

## 0b. Availability audit — all four new fields pass the 5 pp rail

| field | coverage (B2 pre-book) | miss on winners | miss on losers | gap | verdict |
|---|---|---|---|---|---|
| `hl_n20` | 67.2 % | 34.1 % | 32.0 % | 2.1 pp | ok |
| `lo_slope20` | 68.1 % | 32.9 % | 31.1 % | 1.8 pp | ok |
| `atr_ratio` | 58.4 % | 42.4 % | 41.1 % | 1.3 pp | ok |
| `atr_now` | 76.1 % | 25.5 % | 22.7 % | 2.8 pp | ok |
| `consol_bars`, `touch_n` | 100.0 % | 0.0 % | 0.0 % | 0.0 | ok |
| `dollar_frac` (F10) | 68.3 % TRAIN / 96.3 % VAL | 34.3 / 3.1 % | 29.8 / 4.1 % | 4.5 / 1.0 pp | ok |

The coverage numbers ARE the clock confound, declared in `PREREG` §1 before scoring: a 20-bar
look-back is undefined before 09:50 and a 28-bar one before 09:58. Measured structural floors:
`consol_bars >= 20` and `hl_n20` both start at **break_m 591 (09:51)**, `atr_ratio` at **599 (09:59)**.

---

# F11 — IDENTIFY `consol_bars >= 20` (10 cells)

## 1.1 The four components, each on the same cascade

Admission = KEEP-SCANNING (the first qualifying break satisfying the rung), B2 base, the shipped n5
stop, the shipped exit, 12/4 at $100 risk — the identical cascade under which `hod_fresh` measured
`consol_bars >= 20`.

| cell | TRAIN n | /wk | gross | net | grn % | **$** | VAL n | /wk | gross | net | grn % | **$** | H1 / H2 / VAL | same-signed + |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R3 `consol_bars >= 20` | 1,411 | 26.6 | +0.028 | −0.041 | 34.0 | −5,749 | 714 | 31.0 | +0.050 | −0.023 | 47.8 | −1,629 | +0.005/+0.049/+0.050 | **YES** |
| **F11-a1** `hl_n20 >= 12` | 1,615 | 30.5 | −0.042 | −0.112 | 34.0 | −18,166 | 767 | 33.3 | +0.074 | +0.001 | 47.8 | +81 | −0.084/−0.003/+0.074 | no |
| **F11-a2** `hl_n20 >= 16` | 513 | 9.7 | +0.019 | −0.057 | 45.3 | −2,907 | 294 | 12.8 | +0.008 | −0.070 | 39.1 | −2,071 | −0.010/+0.043/+0.008 | no |
| **F11-a3** `lo_slope20 > 0` | 1,697 | 32.0 | −0.056 | −0.124 | 32.1 | −21,106 | 790 | 34.3 | +0.021 | −0.052 | 39.1 | −4,081 | −0.043/−0.068/+0.021 | no |
| **F11-b1** `atr_ratio <= 0.8` | 732 | 13.8 | −0.061 | −0.139 | 34.0 | −10,162 | 486 | 21.1 | −0.031 | −0.114 | 39.1 | −5,537 | +0.003/−0.114/−0.031 | no |
| **F11-b2** `atr_ratio <= 0.6` | 151 | 2.8 | −0.074 | −0.163 | 24.5 | −2,454 | 151 | 6.6 | −0.087 | −0.179 | 34.8 | −2,706 | −0.236/+0.056/−0.087 | no |
| **F11-c1** `break_m >= 590` [clock] | 1,720 | 32.5 | −0.023 | −0.091 | 39.6 | −15,567 | 797 | 34.7 | +0.064 | −0.007 | 39.1 | −578 | −0.019/−0.028/+0.064 | no |
| **F11-c2** `break_m >= 600` [clock] | 1,660 | 31.3 | −0.062 | −0.130 | 32.1 | −21,499 | 782 | 34.0 | −0.025 | −0.097 | 39.1 | −7,589 | −0.081/−0.044/−0.025 | no |
| **F11-d1** `price >= $37.4` [filter] | 1,200 | 22.6 | −0.031 | −0.101 | 37.7 | −12,075 | 622 | 27.0 | +0.007 | −0.066 | **52.2** | −4,106 | −0.114/+0.040/+0.007 | no |
| **F11-d2** `spread/R <= 0.0932` [filter] | 1,360 | 25.7 | −0.077 | −0.124 | 30.2 | −16,817 | 635 | 27.6 | +0.041 | −0.007 | **52.2** | −476 | −0.143/−0.014/+0.041 | no |

**None of the ten is same-signed positive in H1, H2 and VAL. F11-x1 (the conditional SPY cross on the
best (a)/(b) cell) was therefore NOT scored, exactly as the PREREG declared.** By the pre-committed
rule — no (a)/(b) cell qualifies and no (c)/(d) cell qualifies — the verdict is **UNRESOLVED**, and
the difference test is the declared tie-breaking evidence.

*(`F11-d1` is the pass's one green-week reading ABOVE its count-matched null — VAL 52.2 % vs a null
mean of 36.5 % [26.1, 47.8] — on a cell that loses **$12,075 on TRAIN and $4,106 on VAL**. F7's
lesson, third appearance: a green-week ratio without the dollar path lies.)*

## 1.2 The difference test — and it is the decisive number

`consol_bars >= 20` implies `break_m >= 590` by construction, so the two admissions differ only by
the field. Inside the clock admission, kept vs rejected by `consol_bars >= 20`, on **gross**:

| population | split | n kept | n rej | kept | rej | **Δ** | iid t | **clustered t** |
|---|---|---|---|---|---|---|---|---|
| first break at/after 09:50 | TRAIN | 1,608 | 2,368 | +0.016 | −0.003 | **+0.018** | +0.43 | **+0.09** |
| first break at/after 09:50 | VAL | 1,026 | 1,023 | −0.046 | +0.087 | **−0.133** | −2.29 | **−1.84** |
| the same, **SPY-up days only** | TRAIN | 847 | 1,209 | +0.144 | +0.279 | **−0.135** | −2.34 | −0.57 |
| the same, **SPY-up days only** | VAL | 579 | 558 | +0.030 | +0.218 | **−0.188** | −2.37 | −1.77 |
| [control] base, `break_m >= 590` kept vs rej | TRAIN / VAL | 3,379 / 1,518 | 1,196 / 934 | +0.005 / −0.024 | −0.020 / +0.078 | +0.025 / −0.102 | +0.62 / −1.98 | +0.14 / −1.48 |

**The field is worth +0.018 R on TRAIN at a clustered t of 0.09 and −0.133 R on VAL — and on exactly
the days C1 trades it is the wrong side on BOTH splits.** No rule is promoted.

## 1.3 The supplementary diagnostics — what C1 actually is (`supp11.log`, no decision)

**S1 — the control C1 never had.** The SPY 09:35 gate on its own, with no `consol_bars`:

| cell | TRAIN /wk · gross · grn % · $ | VAL /wk · gross · grn % · $ | H1/H2/VAL |
|---|---|---|---|
| base (= B2) | 30.6 · −0.039 · 32.1 % · −$17,346 | 30.7 · +0.083 · 43.5 % · +$893 | −0.077/−0.003/+0.083 |
| **base × `spy_r5 > 0`** | 15.6 · **+0.032** · 43.4 % · −$3,059 | 15.9 · **+0.165** · **65.2 %** · **+$3,439** | −0.062/+0.117/+0.165 |
| `consol_bars >= 20` (R3) | 26.6 · +0.028 · 34.0 % · −$5,749 | 31.0 · +0.050 · 47.8 % · −$1,629 | +0.005/+0.049/+0.050 |
| **C1** = R3 × `spy_r5 > 0` | 13.8 · +0.100 · 43.4 % · +$2,391 | 16.0 · +0.123 · 47.8 % · +$1,827 | +0.123/+0.082/+0.123 |
| **`break_m >= 590` × `spy_r5 > 0`** | 16.9 · **+0.048** · **49.1 %** · −$1,746 | 17.7 · **+0.112** · 47.8 % · **+$1,700** | **+0.094/+0.008/+0.112** |

The three-era same-signed-positive property is **reproduced by a pure clock × day-gate cell with no
`consol_bars` in it**, and the best VAL cell in the table has neither the field nor the clock.

**S2 — what the admission mechanically does.** `consol_bars >= 20` takes the **same** break as the
base 73 % (TRAIN) / 67 % (VAL) of the time; when it takes a different one it is a **later** break, a
median of **37 / 38 minutes later**. Booked, the *later* rows are the loss: TRAIN same-break
1,005 trades −$2,158 vs later-break 406 trades **−$3,591**.

**S3 — C1's own concentration, confirming `hod_fresh` §5**: TRAIN best week **+$2,027 of +$2,391
(85 %)**, VAL **+$1,215 of +$1,827 (67 %)**; net **+0.033 → ex-top-5 % −0.071** (TRAIN) and
**+0.050 → −0.056** (VAL).

**S4 — `consol_bars` as a continuous ranker on the base first break** is non-monotone and flips:
TRAIN Q1 +0.128 / Q2 −0.091 / Q3 −0.043 / Q4 −0.144 / Q5 +0.080; VAL Q1 +0.009 / … / **Q5 −0.109**.

**S5 — the gate is diluted, not concentrated, by the field**: the SPY-up minus SPY-down separation is
**+0.388 / +0.189** on the base and **+0.219 / +0.147** on the `consol_bars >= 20` admission.

**S6 — all four components as continuous rankers sign-flip between the years**: `hl_n20` TRAIN
Q1 +0.240 → Q4 −0.207 (best at the LOW end) but VAL Q5 +0.174 (best at the HIGH end); `lo_slope20`
TRAIN Q1 **+0.331** → Q5 −0.074, VAL Q5 +0.095; `atr_ratio` TRAIN Q5 **+0.557** (the *expanding*
end, the opposite of the coil) / VAL Q5 −0.118; `break_m` TRAIN Q5 **+0.578** / VAL Q5 −0.083. This
is the F12 picture measured on the F11 fields.

## 1.4 F11 verdict

**The pre-committed rule returns UNRESOLVED; the declared tie-breaking evidence retires the
candidate.** `consol_bars >= 20` is not rising lows (a), not range compression (b), not the clock (c)
and not the price/liquidity band (d) — none of the four reproduces its signature — **and the field is
the wrong side of its own population once the clock is held fixed.** The three-era sign belongs to
the SPY 09:35 day gate, which a clock-only admission carries just as well and which
`hod_preopen_regime` §5 already measured as **+0.725 H1 / −0.041 H2** on the base. **The programme no
longer has an unexplained survivor; it has a day gate whose own book is H1-negative.** Nothing is
ported anywhere and no `HodBreakParams` change follows.

---

# F10 — THE FOOTPRINT FIELD, GATE RE-SPECIFIED (8 cells declared, 5 scored)

## 2.1 The re-specified diagnostic (reported, never an exclusion)

The already-there / arrived-after split is printed for every rung, on both splits. It is **the same
inconsistency pass 2 saw at one rung, and it holds across the whole ladder**: on TRAIN the top rung's
gross runs through the **arrived-after** channel (p80: already-wide −0.074 at 37 % WR vs arrived-after
**+0.292** at 49 % WR) and on VAL through the **already-there** channel (p80: already-wide **+0.194**
at 48 % WR vs arrived-after −0.007 at 42 %). Membership rises monotonically with the rung
(ΔP(EOD ≥ 10 %) +4.6 → +16.1 pp on TRAIN, +3.4 → +22.0 pp on VAL) exactly as `rng_own` and
`dist_open_pct` did — and, exactly as they did, the membership is not the return.

**Pass 2's exclusion is therefore withdrawn on the record and the field is scored. Withdrawing it
changed nothing: the field does not clear the selector either way.**

## 2.2 The ladder, before and after the dedicated NBBO fetch

`fetch_nbbo3.py` fetched **466** fresh Alpaca-SIP quote-minutes at the p80 rung's own signal minutes
(339/min, 1.4 min, same tooling/convention as `causal_filter/fetch_nbbo.py`). Imputed-cost share on
that rung: **44.6 % → 3 % (TRAIN) / 5 % (VAL)**.

| rung (post-fetch percentile) | TRAIN n | /wk | gross | cost | net | grn % | **$** | VAL n | /wk | gross | net | grn % | **$** | H1 / H2 / VAL | era-consistent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `>= p50` (24.4 %) | 1,163 | 21.9 | +0.005 | .061 | −0.056 | 35.8 | −6,483 | 629 | 27.3 | −0.008 | −0.073 | 26.1 | −4,576 | −0.010/+0.015/−0.008 | no |
| `>= p60` (30.3 %) | 977 | 18.4 | +0.007 | .059 | −0.052 | 30.2 | −5,084 | 556 | 24.2 | +0.028 | −0.036 | 34.8 | −1,974 | −0.019/+0.024/+0.028 | no |
| `>= p70` (37.3 %) | 772 | 14.6 | +0.030 | .058 | −0.028 | 34.0 | −2,154 | 458 | 19.9 | +0.010 | −0.052 | 34.8 | −2,381 | −0.015/+0.058/+0.010 | no |
| **`>= p80` (46.1 %)** | 558 | 10.5 | **+0.065** | .056 | **+0.009** | 37.7 | **+490** | 343 | 14.9 | **+0.096** | +0.034 | **60.9** | **+1,175** | **−0.024**/+0.121/+0.096 | no (H1) |
| **`>= p90` (64.1 %)** | 302 | **5.7** | +0.056 | .055 | +0.001 | 39.6 | +26 | 185 | **8.0** | +0.116 | +0.055 | 47.8 | +1,011 | **+0.014/+0.083/+0.116** | **YES, under the floor** |

*(Pre-fetch ladder, `score10_imputed.log`: p80 TRAIN +0.034 / VAL +0.083, p90 TRAIN −0.028 / VAL
+0.093 — the measured cost lifts the top rungs' gross by re-testing obtainability on real quotes and
does not change a single verdict.)*

**The selector's answer.** No rung is positive in H1-2025, H2-2025 and VAL at ≥ 10 trades/week.
**p90 is era-consistent and trades 5.7 / 8.0 times a week** — the seventh time in this programme that
the only three-era-consistent object sits under the frequency floor. Its MDE is **0.216 / 0.274 R**
against a 0.055–0.062 R break-even: it could not have resolved its own point estimate, and its green
weeks are inside its null on both splits. **F10-6/7/8 (the crosses with the SPY gate, with
`consol_bars >= 20` and with C1) were NOT scored, as the PREREG declared.**

**Claim bar G1: 0 of 7.** Live-exploration bar: 0. Nulls: 14 cell × split, **11 inside, 3 below, 0
above**.

---

# F12 — ADAPTIVE vs FROZEN (6 cells)

7,027 pre-book B2 signals over **75 market weeks**. The rail asserted in code (`assert trw.day.max()
< cur.day.min()`) fired once and caught a real defect — the TRAIN and VAL week lists share the
2025-12-27/2026-01-02 period, which would have let the refit see the week it trades; deduplicated
before any OOS week was booked. **No OOS week is dated after 2026-05-31; TEST was never opened.**

| cell | OOS weeks | trades | /wk | **green %** | red streak | worst $ | **total $** | MDD $ | churn % | $ after a change | $ after no change |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **F12-r20 REFIT** | 55 | 627 | 11.4 | **36.4** | 7 | −1,844 | **−2,560** | −6,726 | **83** | −97 | +182 |
| F12-f FROZEN (L=20) | 55 | 1,745 | 31.7 | **38.2** | 5 | −2,136 | −8,026 | −10,639 | 0 | — | — |
| **F12-r26 REFIT** | 49 | 537 | 11.0 | **30.6** | 8 | −1,606 | **−4,491** | −8,174 | **81** | −90 | −99 |
| F12-f FROZEN (L=26) | 49 | 1,547 | 31.6 | **38.8** | 5 | −2,136 | −5,230 | −9,228 | 0 | — | — |
| **F12-r34 REFIT** | 41 | 391 | **9.5** | **31.7** | 8 | −2,097 | **−2,013** | −5,014 | **65** | +52 | −224 |
| F12-f FROZEN (L=34) | 41 | 1,302 | 31.8 | **39.0** | 5 | −2,018 | **+163** | −6,319 | 0 | — | — |
| F12-s26 SIZING control | 49 | 1,547 | 31.6 | 36.7 | 9 | −2,584 | −3,890 | −9,482 | — | — | — |
| **F12-s34 SIZING control** | 41 | 1,302 | 31.8 | **46.3** | 4 | −1,845 | **+3,441** | −6,135 | — | — | — |

Per OOS half (the declared deviation: the first L weeks are consumed by the initial window, so **no
OOS week falls in H1-2025**; the rail is read on OOS-in-TRAIN vs OOS-in-VAL):

| arm | OOS-in-TRAIN (H2-2025) refit vs frozen | OOS-in-VAL refit vs frozen | beats frozen in both halves |
|---|---|---|---|
| L = 20 | −$1,593 / 30.3 % vs −$8,873 / 36.4 % | −$967 / 45.5 % vs **+$847** / 40.9 % | **no** |
| L = 26 | −$2,870 / 18.5 % vs −$6,077 / 37.0 % | −$1,622 / 45.5 % vs **+$847** / 40.9 % | **no** |
| L = 34 | −$260 / 15.8 % vs −$684 / 36.8 % | −$1,753 / 45.5 % vs **+$847** / 40.9 % | **no** |

Clustered t on the refit book's own net R: **−0.64 / −1.32 / −0.69**. Trades/week 11.4 / 11.0 / **9.5**
— the L = 34 arm does not even clear the frequency floor.

**The whipsaw, measured.** 44 distinct (band, cut) configurations in 55 OOS weeks at L = 20, 34 in 49
at L = 26, 25 in 41 at L = 34. The **cut** is stable in kind — `spy_r5_pct` wins the TRAIN-window
score in 28 / 55, 21 / 49 and 26 / 41 weeks, which is the same day gate F11 just identified — but the
**hour band alternates between the two opposite ends of the axis**: at L = 34 the most-picked bands
are `(577, 593)` (the first quarter-hour) in 11 weeks and `(626, 840)` (everything after 10:26) in 9.
The refit is not tracking a regime; it is chasing the sign of a feature that has none.

**The sizing control, reported so the two are not confused.** Refitting only the SIZE (per-entry-minute
tercile multipliers 1.5 / 1.0 / 0.5, ORB's shape) keeps every pick and moves the owner's metric:
L = 34 reads **46.3 % green and +$3,441 against frozen's 39.0 % and +$163** — on an identical pick
set. That is Frame 4's finding reproduced out of sample and it is **not an edge**: a sizer re-weights
an edge, it cannot create one, and F4's key-shuffled null already showed this class of result on a
book with no edge. It is reported, not recommended.

## READING (ii) — stated in the words the queue asked for

**The separations this programme has found are NOISE, and every future frozen-rule pass on this book
is pre-refuted.** A rolling refit is the strongest available test of the alternative — if entry
minute, `exp5_n`, `add30_ratio`, `rv >= 5` and the SPY gate flipped sign because the regime changed,
a 20/26/34-week refit would track them and pay. It does not: it loses on the owner's metric in every
arm and every half, at a third of the frequency, with 65–83 % churn. ORB's 2026-09-08 result does not
transfer — ORB refits the ranking of a selection stack that has a measurable edge to rank; HOD-break
has **+0.0003 / −0.045 R** of raw gross and a 0.061-0.065 R wall, and a refit on that surface is a
random walk through 28 bands and 9 cuts.

---

## BOTH BARS, the nulls, the MDE, the multiplicity

**Claim bar G1 — 0 of 24 declared cells.** No cell has TRAIN net R > 0 with iid **and** clustered
t ≥ 2 at ≥ 10 trades/week. The largest positive TRAIN clustered t on any scored cell is **+0.14**
(F11-d1) and the largest positive TRAIN net at ≥ 10/wk is **+0.009** (F10 p80). G2 was never
evaluated; **TEST was never opened.**

**Live-exploration bar — 0 cells.** The only cells with positive dollars on BOTH splits are F10 p80
(+$490 / +$1,175 at 10.5 / 14.9 per week, H1 −0.024) and F10 p90 (+$26 / +$1,011 at 5.7 / 8.0), and
neither has ≥ 50 % green weeks on TRAIN or a clustered t anywhere near 2.

**Nulls — 40 cell × split bands: 34 inside, 5 below, 1 ABOVE.** The single ABOVE is F11-d1 on VAL
(52.2 % vs 36.5 % [26.1, 47.8]) on a cell that loses money on both splits. **The eighth pass in a row
in which the owner's primary metric on this book is indistinguishable from pick count except where it
points at a loser.**

**MDE (80 % power, per trade, net) against the true 0.055–0.065 R break-even**: 0.088 / 0.132 R on the
B2 book; 0.091–0.169 / 0.133–0.224 on the F11 cells that trade ≥ 10/wk; 0.110–0.161 / 0.144–0.200 on
the F10 rungs that do. Those are **powered** rejections. F11-b2 (2.8/wk, MDE 0.314) and F10 p90
(5.7/wk, MDE 0.216) are **not** — and their frequency is itself the verdict.

**Multiplicity.** 24 declared decision cells (10 F11 + 8 F10, of which 3 were not scored because the
pre-committed selector found no eligible rung + 6 F12), 5 reproduction rows, 1 availability audit, 1
downstream diagnostic table, 3 declared difference/separation tables and 6 supplementary diagnostics
(S1–S6) that carry no decision. **Programme cumulative: 896 + 24 = 920.** Expected largest |t| under a
pure null over 24 × 2 ≈ 2.8; the largest favourable clustered t on any scored cell is **+0.14**.

## Known deviations, stated rather than buried

1. **The F10 ladder was re-scored after the NBBO fetch and the fetch covered only the p80 rung's
   rows.** Because p90 ⊂ p80 ⊂ p70 ⊂ …, the lower rungs' populations shift slightly too (the
   obtainability test now runs on real quotes for those rows) and the TRAIN percentiles move by
   ~1 pp. Both ladders are printed (`score10_imputed.log` and `score10.log`) and the verdict is
   identical on both.
2. **F12 has no OOS week in H1-2025** — structural, declared in `PREREG` §2 before the walk, and the
   halves rail is read on the OOS period's own halves.
3. **The F11 (a)/(b) fields carry a clock floor by construction** (09:51 and 09:59). That is why the
   clock rungs are scored as cells rather than mentioned as a caveat, and why the difference test is
   run inside the clock admission rather than against the base.
4. **`hl_n20`, `lo_slope20` and `atr_ratio` are 58–68 % covered** on the pre-book set. The missingness
   is the clock, not the outcome (gaps 1.3–2.8 pp, all inside the rail), but every (a)/(b) cell is
   therefore a statement about the post-09:50 book, not the whole book.
5. **F12's refit optimises TRAIN-window mean net R**, not green weeks. A refit that optimised the
   owner's metric directly was not declared and is not scored here; given that the chosen configs
   whipsaw between opposite ends of the axis there is no reason to think the objective is the binding
   constraint — but it is an untested variant and is named as one.

## VERDICT — **STAY DRY.** No `HodBreakParams` change, no scan-rule change, no new admission field.

`config.yaml hod_break` stays exactly as the owner set it (`enabled: true, dry_run: true`);
`trading.enabled` and `orb.yaml` untouched. There is no SHIP-TO-DRY diff to write. For the record,
had F12 cleared, the diff would have been a **new artefact class for this repo** — a weekly cron in
the shape of `scripts/orb_weekly_refit.py` writing `HodBreakParams` (`entry_start_et`, `entry_end_et`
and one feature cut) into `config.yaml` every Sunday, plus a `hod_break_refit_check.py` beside
`bf_ramp_check.py`. It is not built and, on this evidence, must not be.

**The next three frames are F13, F14 and F15, appended to `FRAMES.md`.**

*(`feat3.csv` / `feat3b.csv` — 131 MB and 62 MB — are gitignored and stay local; both are regenerated
deterministically by `walk3.py` / `walk3b.py`, checkpointed per day.)*
