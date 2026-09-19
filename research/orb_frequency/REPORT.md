# ORB — the gate separation map and the frequency frontier, ranked on GREEN WEEKS

2026-09-19. Pre-registration `PREREG.md` (written and committed **before any cell was
scored**, commit `53a5529`). TEST seal `FREEZE.md`. **Nothing ships from this stage.**
A survivor needs its own pre-registration and the owner's word.

The method is `research/bf_frequency/` applied to ORB — the study that found two broken
bull-flag gates (`min_daily_volume`, cutting 41% of the field with negative separation;
the conviction gate, cutting 57% with none). The trigger is
`research/green_weeks/REPORT.md`: **ORB's flat weeks are a PICK SHORTAGE, not an exit
failure** — 13 of 19 flat TRAIN weeks had no pick at all, 6 had only non-fills, and the
flat-week share is identical across all 17 exit cells. Nobody had ever measured ORB's
gates this way.

---

## 0. Read this before any number below

**This instrument cannot resolve a small green-week difference.** The green-week share is a
proportion over 53 TRAIN weeks and 22 VAL weeks. The unpaired MDE80 is **+-25.8pp on TRAIN
and +-39.3pp on VAL**. At BF's frequency the same figure was +-9.5pp; ORB, at 2.4 picks per
week, is **three to four times worse**. VAL alone can only refute a catastrophe.

**And the green-week share carries no timing information at all.** A 2,000-draw permutation
that shuffles each book's own P&L across its own weeks, holding picks-per-week fixed:

| book | split | observed green wk % | permuted mean | permuted p5-p95 |
|---|---|---|---|---|
| shipped B+ | TRAIN | **34.0** | 37.5 | 32.1 - 43.4 |
| shipped B+ | VAL | **31.8** | 36.4 | 27.3 - 45.5 |
| catalyst veto OFF | TRAIN | **47.2** | 50.4 | 43.4 - 56.6 |
| catalyst veto OFF | VAL | **54.5** | 52.5 | 40.9 - 63.6 |

Every observed value sits **inside** its permutation band, and the shipped book sits *below*
its own mean on both splits. **Green weeks in this book are bought with pick COUNT and
nothing else.** No gate confers week-level timing skill. That is the same conclusion
`bf_frequency` reached, and it is the frame for everything that follows: the only lever on
green weeks is how many weeks the book trades in.

---

## 1. Instrument and the reproduction gate

- **Engine**: the shipped `study_orb_pipeline_static_lock.py`, replayed off a candidate
  dump with `ORB_BT_RESIM_CACHE` (selector-only; exit physics are the dump's). Every
  constant — z-params, quintile cutoffs, adaptive mults, veto thresholds — comes from
  `orb.yaml` as it stands today. **Nothing was refit.** Knobs moved only through documented
  env overrides. Production config, `orb.yaml`, caches, orders, services and crons were
  never written; every artifact is under `research/orb_frequency/`.
- **Reproduction gate — PASSED before the pre-registration was written**: the as-is dump at
  `N=8`, `account = 3333.33 x 8`, `risk = 375`, Q1 on reproduces
  `research/fuckup_audit/D1_orb/book_n8_q1on.csv` **byte-identically** —
  **215 picks / $14,428.616990972434** (`repro_n8_q1on.csv`, `DataFrame.equals` -> True).
- **PRIMARY fill model**: Stage Q's **measured** arm
  (`research/fuckup_audit/Q_fill/dump_measured.csv`) — the elected stop-limit rests as a bid
  at the cap, fills at the cap the first time the walked SIP ask reaches it before the 10:35
  time stop, $0 and a spent slot if it never does. **Secondary bracket**: the as-is dump.
  Every headline below is given in both; they agree on sign everywhere.
- **Population**: 13,033 entered-inclusive candidates (7,402 modelled fills + 5,631 modelled
  non-fills) over 427 trading days / 90 market weeks, 2025-01-02 -> 2026-09-16. Non-fill
  picks burn a slot at $0 and stay in the book.
- **R** = `pnl_pct / max(range_size_pct, 1.0)` — the trade's own return over its own stop,
  taken from the candidate row, so R is invariant to the quintile mult, the per-position cap
  and the account size. (Q_fill's published R/pick is the *mult-weighted* variant; the
  numbers differ by the mult and rank identically.)
- Splits: **TRAIN 2025 (53 wk)** / **VAL 2026-01..05 (22 wk)** / **TEST 2026-06+ (§9)**.

---

## 2. PART 1 — the separation map

`separation.py` -> `separation.csv`. Kept-R minus rejected-R at each gate's **own position in
the live cascade**, per year and pooled, Welch t.

| # | gate | n kept / rej | R kept | R rej | sep 2025 (t) | sep 2026 (t) | **sep pooled (t)** | verdict |
|---|---|---|---|---|---|---|---|---|
| S0 | composite >= 0.012082 | 7,225 / 5,808 | -0.049 | -0.026 | -0.073 (-4.02) | +0.027 (1.31) | **-0.023 (-1.68)** | wrong-signed in 2025, and **inert in the book** (§3) |
| **S1** | **Q1 quintile filter** | **5,771 / 1,454** | **-0.061** | **+0.001** | **-0.040 (-1.32)** | **-0.086 (-1.50)** | **-0.063 (-1.96)** | **WRONG SIDE IN BOTH YEARS — removal candidate** |
| S2 | the quintile cutoffs themselves | — | — | — | — | — | — | **no ordering at all** (below) |
| S3 | family / super-group dedup | 5,467 / 304 | -0.062 | -0.047 | +0.127 (1.80) | -0.226 (-1.93) | -0.016 (-0.24) | sign-flips by year; a *correlation* rule, not an edge rule |
| S4 | top-8 slot cut (rank <= 8) | 2,744 / 2,723 | -0.068 | -0.056 | +0.047 (1.28) | -0.059 (-1.59) | -0.012 (-0.47) | the rank order carries no edge (below) |
| S5 | PDR veto (prev-day range > 11.0) | 871 / 1,873 | +0.087 | -0.141 | +0.279 (3.51) | +0.169 (2.62) | **+0.228 (4.43)** | **real, era-consistent — earns its keep** |
| S6 | G1 fingerprint (rv20 >= 7.106 & pdr >= 9.226) | 674 / 197 | +0.122 | -0.031 | +0.241 (1.65) | +0.044 (0.33) | +0.153 (1.53) | positive both years, weak in 2026 |
| **S7** | **range-size veto (range <= 2.221% -> cut)** | **594 / 80** | **+0.095** | **+0.325** | **-0.261 (-0.56)** | **-0.156 (-0.51)** | **-0.231 (-0.76)** | **WRONG SIDE IN BOTH YEARS — removal candidate** |
| S8 | catalyst veto (news or cohort >= 2) | 210 / 384 | +0.303 | -0.019 | +0.423 (2.44) | +0.224 (1.40) | **+0.323 (2.75)** | **real, era-consistent — earns its keep** |
| S9/S10 | touchgo Rule M / Rule D | — | — | — | — | — | — | exit rules — §7 |
| S11 | **WHOLE STACK picked vs rejected** | 210 / 12,823 | +0.303 | -0.044 | +0.483 (3.19) | +0.211 (1.42) | **+0.347 (3.27)** | intact |

**Two gates come out on the wrong side in BOTH years — exactly the shape BF's two did.**

### S1 — the Q1 filter cuts the quintile that performs *best*
The bottom composite quintile it drops returns **+0.001 R** against the **-0.061 R** it
keeps, in 2025 **and** in 2026. It removes 1,454 of 7,225 (20%) of the post-threshold field.

### S2 — the quintiles do not order anything
Mean R by quintile, post-threshold, post-Q1: **Q5 -0.069 / Q4 -0.054 / Q3 -0.058 /
Q2 -0.065** (n ~ 1,420 each). There is no monotone relation between the composite quintile
and R. The quintile is still the *primary sort key* of the day's ranking **and** the driver
of the adaptive sizing mult.

### S4 — and neither does the rank
Mean R by rank band, post-dedup: **rank 1-3 -0.123 / rank 4 -0.031 / 5-6 -0.056 /
7-8 +0.022 / 9-12 -0.112 / 13-16 -0.125 / 17+ -0.015.** The **best-ranked band is the
worst-performing one.** Whatever the shipped book's edge is, the composite ranking is not
where it comes from — it comes from the four post-ranking vetoes (S5-S8), which is why the
whole stack separates at t = 3.3 while every layer above it does not.

### The BF diagnostic — the sizer was already handling it
BF's broken gate hid behind the sizer. So does ORB's:

| gate | median $ risk kept | median $ risk rejected |
|---|---|---|
| S1 Q1 filter | $102 | **$140** |
| **S7 range-size veto** | **$135** | **$60** |
| S5 PDR veto (a real gate) | $125 | $83 |
| S8 catalyst veto (a real gate) | $137 | $133 |

The range-size veto cuts names the risk-parity sizer already puts **less than half** the
dollar risk on: a tight opening range means a tight stop, `risk/stop%` slams into the
per-position cap, and the realised dollar risk is $60 instead of $135. **The gate is
removing trades the book was going to bet half as much on anyway** — the identical mechanism
BF found on ADV20.

### What could not be measured, said plainly
- The **universe screens** (`prev_volume >= 500K`, the $3-30 price band, the 15K RTH-9:35
  range-computability floor) are applied **upstream in `study_orb_broad.py`**. The candidate
  dump has **zero rejected rows** on them, so their separation is **unmeasured here** and
  measuring it needs a features rebuild. They are logged as a defect of coverage, exactly as
  `bf_decay` had left BF's ADV20 gate unmeasured until `bf_frequency` measured it.
- The **spread gate (300 bps)** is a live entry-time rule with **no BT counterpart at all**.
  It cannot be measured on this instrument in either direction.
- The cascade reconstruction lands on **210 picks against the book's 215** (97.7%). The
  5-row gap is the pipeline's interleaved dedup/top-K loop (a duplicate family at a good
  rank does not consume a slot, promoting a lower-ranked name), which the map applies as two
  sequential steps. It does not affect any separation sign.

### The cascade, per week
```
144.8 candidates/wk raw
 ->  80.3  composite threshold
 ->  64.1  Q1 filter            (-20%)   <- wrong side in both years
 ->  60.7  family/super dedup
 ->  30.5  top-8 slot cut
 ->   9.7  PDR veto             (-68%)   <- real
 ->   7.5  G1 fingerprint       (-23%)   <- real, weak in 2026
 ->   6.6  range-size veto      (-12%)   <- wrong side in both years
 ->   2.3  catalyst veto        (-65%)   <- real
```
**The pick shortage is two gates: the prev-day-range pair (PDR & G1) takes 30.5/wk down to
7.5, and the catalyst veto takes 6.6 down to 2.3. Together they are 92% of the cut. Both
have POSITIVE, era-consistent separation.** That is the whole tension of this study: the
gates that cause the flat weeks are the gates that work.

---

## 3. Two structural facts the ladders exposed

**(a) The composite threshold is INERT.** `L_thr05` (threshold -> -0.5) and `L_thrall`
(threshold off) are **byte-identical to the shipped book** — 105 TRAIN picks, $5,673. Q1 is
`composite < 0.1059`, far above the threshold `0.0121`, so once Q1 has run the threshold
never binds on anything that could win a slot. D1 said this about the pool; it is also true
of the book. **`filter.threshold` is a dead knob at 8 slots.**

**(b) The PDR veto is almost entirely REDUNDANT with the G1 veto.** PDR at 11.0 -> 8.0 -> 6.0
-> OFF are **all three identical** (117 TRAIN picks, $4,805). G1 keeps only `pdr >= 9.226`,
and it runs *after* PDR, so relaxing PDR below 9.226 adds nothing. The whole PDR ladder is
worth **+0.23 picks/week and -$868 on TRAIN**. The prev-day-range rule that actually binds is
**G1's 9.226 leg**, not PDR's 11.0. Two shipped vetoes, one effective threshold.

The same collapse hits the range-size ladder: 2.221 -> 1.5 -> 1.0 -> off are all identical
(108 TRAIN picks), because every selected pick it rejects has `range_size_pct <= 1.0`, i.e.
it sits on the `MIN_STOP_PCT` floor.

---

## 4. PART 2 — the frontier, ranked on green weeks

Measured fill model, N = 8. `grid_meas.csv` / `grid_asis.csv`. TR = TRAIN (53 wk),
VA = VAL (22 wk). **Flat-week share is beside green-week share on every row, as asked.**

| cell | TR pk/wk | VA pk/wk | **TR green%** | **VA green%** | TR flat% | VA flat% | TR streak | VA streak | TR $ | VA $ | TR R/pk | VA R/pk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **shipped B+ (F0)** | 1.98 | 2.41 | **34.0** | **31.8** | 37.7 | 22.7 | 3 | 3 | 5,673 | 3,939 | 0.410 | 0.469 |
| Q1 filter OFF | 2.43 | 2.77 | 35.8 | 40.9 | 28.3 | 22.7 | 3 | 2 | 5,312 | 8,245 | 0.31 | 0.94 |
| PDR 11->8 / 11->6 / OFF (identical) | 2.21 | 2.77 | 34.0 | 31.8 | 30.2 | 22.7 | 4 | 3 | 4,805 | 3,641 | 0.31 | 0.37 |
| G1 fingerprint OFF | 2.45 | 3.41 | 35.8 | 36.4 | 26.4 | 9.1 | 5 | 4 | 4,791 | 3,450 | 0.29 | 0.26 |
| range-size 2.221->1.5 / ->1.0 / OFF (identical) | 2.04 | 2.45 | 34.0 | 31.8 | 37.7 | 22.7 | 3 | 3 | 5,568 | 3,873 | 0.38 | 0.44 |
| **catalyst veto OFF** | **5.32** | **8.05** | **47.2** | **54.5** | **7.5** | **4.5** | 4 | 2 | **6,515** | **4,287** | 0.175 | 0.149 |
| composite threshold -0.5 / OFF | 1.98 | 2.41 | 34.0 | 31.8 | 37.7 | 22.7 | 3 | 3 | 5,673 | 3,939 | 0.410 | 0.469 |
| F1 = -range-size | 2.04 | 2.45 | 34.0 | 31.8 | 37.7 | 22.7 | 3 | 3 | 5,568 | 3,873 | 0.38 | 0.44 |
| F2 = -range-size, PDR 8 | 2.26 | 2.82 | 34.0 | 31.8 | 30.2 | 22.7 | 4 | 3 | 4,700 | 3,576 | 0.29 | 0.35 |
| F3 = -range-size -G1 | 2.64 | 3.59 | 37.7 | 36.4 | 22.6 | 9.1 | 5 | 4 | 4,643 | 3,726 | 0.25 | 0.30 |
| **F4 = -range-size -G1 -PDR** | 9.26 | 11.18 | **41.5** | **31.8** | **1.9** | **4.5** | 4 | 3 | 1,888 | 3,845 | 0.025 | 0.095 |
| F5 = F4 - catalyst | 28.06 | 31.68 | 39.6 | 54.5 | 0.0 | 0.0 | **10** | 3 | **-5,555** | 2,219 | -0.07 | 0.01 |
| **F6 = CEILING (every gate off)** | 37.19 | 37.09 | 35.8 | 63.6 | 0.0 | 0.0 | 8 | 2 | **-7,268** | 7,990 | -0.06 | 0.06 |

### The structural ceiling, explicitly
With **every** gate off (F6), the book takes **3,387 picks over 90 weeks = 37.6/week**
against a hard slot ceiling of `8 slots x 427 days / 90 weeks = 38.0/week`. **F6 is at 99%
of the slot ceiling: with the gates open the book is slot-bound, not candidate-bound**, and
there are 144.8 raw candidates a week behind it. This is ORB's analogue of BF's 44/month raw
ceiling — and like BF's, **the frontier bends before it**: F6 loses **-$7,268 on TRAIN** and
its flat weeks are replaced by red ones (35.8% green, 8-week red streak), because the raw
ORB breakout is edgeless (`orb_veto_study`: 2025 -0.18R, 2026 -0.04R). Flat weeks are not
free to remove; past a point they convert to red, not green.

### Ranked on the owner's metric (pooled TRAIN+VAL green-week %)

| rank | cell | pooled green% | TR/VA flat% | survives PREREG §7? | failed rule |
|---|---|---|---|---|---|
| 1 | **catalyst veto OFF** | **49.3** | 7.5 / 4.5 | **no** | 5 (worst week) |
| 2 | F6 ceiling | 44.0 | 0 / 0 | no | 3, 4, 5 |
| 2 | F5 | 44.0 | 0 / 0 | no | 3, 4, 5 |
| 4 | **F4** | **38.7** | 1.9 / 4.5 | **YES — the only survivor** | — |
| 5 | F3 | 37.3 | 22.6 / 9.1 | no | 3 (red streak) |
| 5 | Q1 filter OFF | 37.3 | 28.3 / 22.7 | no | 2 (flat week) |
| 7 | G1 OFF | 36.0 | 26.4 / 9.1 | no | 3 |
| 8 | everything else (PDR·, range-size·, threshold·) | 33.3 | >= 30.2 / 22.7 | no | 1, 2 |
| — | shipped B+ baseline | 33.3 | 37.7 / 22.7 | — | — |

**The pre-committed rule selects F4** (drop the range-size veto, the G1 fingerprint and the
PDR veto; keep Q1, the ranking and the catalyst veto). And **F4 does not clear the claim
bar**: PLAN §1 G1 asks for t >= 2 on TRAIN and F4's is **t = 0.57** (R/pick +0.025,
**ex-top-5% R/pick -0.153**). Its green-week gain over the baseline is **+7.5pp TRAIN and
0.0pp VAL**, against an MDE80 of +-25.8pp and +-39.3pp. **F4 is a maximum over a 45-cell
grid, not a discovery**, and it costs two thirds of TRAIN's dollars ($5,673 -> $1,888) and
triples the drawdown (-$560 -> -$1,469) to buy it.

### The cell that actually matters, and why the rule blocked it
**Dropping the catalyst veto is the only single change that moves every axis the owner named
in the right direction — and it raises total P&L on both splits under both fill models.**

| | TRAIN green% | VAL green% | TRAIN flat% | VAL flat% | TRAIN $ | VAL $ |
|---|---|---|---|---|---|---|
| shipped B+, measured | 34.0 | 31.8 | 37.7 | 22.7 | 5,673 | 3,939 |
| catalyst OFF, measured | **47.2** | **54.5** | **7.5** | **4.5** | **6,515** | **4,287** |
| shipped B+, as-is | 37.7 | 31.8 | 35.8 | 22.7 | 6,662 | 6,386 |
| catalyst OFF, as-is | **45.3** | **63.6** | **7.5** | **4.5** | **8,307** | **7,293** |

It fails the pre-committed rule on **one** clause and **one** split: worst TRAIN week
**-$895** against the allowance of 1.5 x (-$418) = **-$628**. TRAIN MDD also goes
-$560 -> -$2,639 and green months 75% -> 58%. VAL passes every clause.

So the honest sentence is: **the catalyst veto buys R/pick (+0.32 R, t 2.75, both years) and
sells green weeks, flat weeks and total dollars.** Which of those the book should want is
precisely the question the owner answered on 9/19 — green weeks — and the pre-committed rail
that blocked it is a drawdown rail, not an edge rail. **It is not this stage's
recommendation because the rule said no, and the rule was written first.** It is the
hypothesis the next stage must pre-register.

### Post-hoc combinations — reported, NOT eligible
Declared after the separation map was read, so they are labelled post-hoc and cannot be
recommended:

| post-hoc cell | TR green% | VA green% | TR flat% | VA flat% | TR $ | VA $ | TR R/pk | VA R/pk |
|---|---|---|---|---|---|---|---|---|
| Q1 off + range-size off (the two wrong-side gates) | 35.8 | 40.9 | 28.3 | 22.7 | 5,207 | 8,180 | 0.290 | 0.905 |
| · as-is bracket | 39.6 | 40.9 | 26.4 | 22.7 | 6,196 | **10,628** | 0.337 | 1.221 |
| catalyst off + Q1 off + range-size off | 47.2 | **63.6** | 1.9 | 4.5 | 7,059 | 9,531 | 0.131 | 0.311 |
| · as-is bracket | 41.5 | 63.6 | 1.9 | 4.5 | 8,603 | **12,537** | 0.155 | 0.424 |

Dropping only the two wrong-side gates is nearly free on TRAIN and large on VAL — but VAL is
a 22-week, 62-pick window whose top-5 picks are 103% of its P&L. **Not a finding. A
hypothesis for a pre-registered test.**

### Tail concentration — diagnostic only, never a penalty (owner's rule)
Top-5 picks as a share of split P&L: shipped B+ **54.8% TRAIN / 118.2% VAL**; catalyst OFF
**59.5% / 111.7%**; F4 **181.2% / 121.2%**. Every ORB book at this size is tail-carried —
removing gates does not reduce the concentration, it adds losers underneath it. Reported
because it was asked for; it did not enter the ranking.

---

## 5. PART 3 — slots, re-read on green weeks

`slots.py` -> `slots.csv`. Measured fill model. Per-position size is held at $3,333.33 for
every N (the D1 convention), so only the slot count moves.

| config | N | TR pk/wk | **TR green%** | **VA green%** | TR flat% | VA flat% | TR streak | TR worst wk | TR MDD | TR $ | VA $ | TR R/pk | max picks in a day | BP bind @ live $66K | capital used |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped B+ | **3** | 0.89 | **22.6** | **18.2** | **58.5** | **50.0** | 2 | -357 | -617 | 2,904 | 1,938 | 0.401 | 2 | 0% | $6,667 |
| shipped B+ | **8** | 1.98 | **34.0** | **31.8** | 37.7 | 22.7 | 3 | -418 | -560 | 5,673 | 3,939 | 0.410 | 4 | 64.7% | $13,333 |
| shipped B+ | 12 | 2.43 | 34.0 | 31.8 | 32.1 | 22.7 | 3 | -532 | -1,031 | 5,730 | 4,184 | 0.315 | 5 | 91.4% | $16,667 |
| shipped B+ | 16 | 2.62 | 32.1 | 36.4 | 30.2 | 18.2 | 4 | -532 | -1,535 | 5,098 | 3,551 | 0.261 | 6 | 99.3% | $20,000 |
| catalyst OFF | 3 | 2.49 | 39.6 | 45.5 | 18.9 | 4.5 | 5 | -824 | -1,825 | 2,907 | 2,619 | 0.143 | 3 | 0% | $10,000 |
| **catalyst OFF** | **8** | 5.32 | **47.2** | **54.5** | 7.5 | 4.5 | 4 | -895 | -2,639 | **6,515** | **4,287** | 0.175 | 8 | 61.5% | $26,667 |
| catalyst OFF | 12 | 6.57 | **49.1** | 54.5 | 5.7 | **0.0** | **7** | -1,009 | -3,862 | 5,522 | 3,731 | 0.119 | 12 | 90.4% | $40,000 |
| catalyst OFF | 16 | 7.38 | 47.2 | 54.5 | 5.7 | 0.0 | 8 | -1,192 | -4,321 | 4,035 | 3,115 | 0.081 | 14 | 98.8% | $46,667 |
| F4 | 3 | 4.32 | 26.4 | 31.8 | 5.7 | 9.1 | 8 | -435 | -1,383 | 303 | 372 | -0.001 | 3 | 13.4% | $10,000 |
| F4 | 8 | 9.26 | 41.5 | 31.8 | 1.9 | 4.5 | 4 | -543 | -1,469 | 1,888 | 3,845 | 0.025 | 7 | 71.2% | $23,333 |
| F4 | 12 | 11.70 | 39.6 | 36.4 | 1.9 | 0.0 | 4 | -592 | -2,410 | 2,097 | 3,757 | 0.011 | 9 | 93.3% | $30,000 |
| F4 | 16 | 12.66 | 35.8 | 36.4 | 1.9 | 0.0 | 4 | -863 | -2,847 | 1,624 | 3,200 | 0.006 | 11 | 99.6% | $36,667 |

**The finding: the biggest green-week move in this entire study has already been made.**
Going 3 -> 8 slots on the shipped gate set is **+11.4pp green TRAIN / +13.6pp VAL** and
**-20.8pp / -27.3pp flat** — the pre-9/18 3-slot book left **58.5% of TRAIN weeks and 50% of
VAL weeks completely empty**. Everything past 8 is flat-to-negative on green weeks and
monotonically worse on drawdown: 8 -> 12 -> 16 buys 0 / -1.9pp green on TRAIN while the MDD
goes -$560 -> -$1,031 -> -$1,535. **D1's "edge gone by rank 9" holds on the green-week metric
too** — more slots do not convert flat weeks to green, because at the shipped gate set there
are no candidates left to fill them (§2: 2.3 post-veto picks/week against 8 slots x 5 days).

**Slots are not ORB's lever. Gates are.** The one place slots still bite is *with* a gate
removed: catalyst-OFF at 12 slots reaches 49.1% / 54.5% green and **zero flat VAL weeks** —
at a 7-week TRAIN red streak and a -$3,862 drawdown.

**Buying power.** At the invariant $3,333.33 cap the per-position ceiling binds on **100% of
picks at every N** — D1's finding, reconfirmed: `sizing.risk_per_trade_usd: 375` is an inert
knob and the real sizing lever is `account_budget_usd / max_concurrent`. Against the **live
~$66K account** the cap binds on **0% at N=3, ~62-71% at N=8, ~90-93% at N=12 and ~99% at
N=16** — above 8 slots the account itself becomes the sizer. Capital actually deployed
(max picks in one day x per-position cap) never exceeds **$26.7K at the shipped gate set and
at catalyst-OFF/8 slots**, and reaches $40-47K only at 12-16 slots with the catalyst veto
off. ORB's current `account_budget_usd: 26,666.67` across 8 is exactly the capital the
catalyst-OFF 8-slot book would need.

---

## 6. Multiplicity

**45 pipeline runs, 90 scored cells on TRAIN+VAL, plus ~36 descriptive separation cells and
3 exit walks.** Under a pure null the expected largest |t| over 45 cells is ~ 2.8-3.0.
**Every "best" point in this report is a maximum over that grid, not a discovery.** No
per-cell p-value was treated as evidence on its own; only the §7 rule of `PREREG.md`
selected, and its survivor fails the claim bar. Zero thresholds, z-params, quintile cutoffs
or adaptive mults were fitted anywhere in this stage.

---

## 7. Touchgo Rule M and Rule D

Rules M and D are **exit** rules: they change the P&L of fills that already happened and
**cannot** change the number of picks, so they cannot move the flat-week share.
`research/green_weeks/REPORT.md` established this empirically — ORB's flat-week share is
identical across all 17 exit cells it tested. Their counterfactual needs a full bar-walk
(`walk_touchgo.sh`, three ~35-minute walks under the node's memory rail); results are
appended in §9a when the walks land. **They are not part of the recommendation either way**,
and no frontier point in §4 depends on them.

---

## 8. THE RECOMMENDATION

**Two ORB gates sit on the wrong side in both years and are defects to be fixed on their own
terms. But on this stage's pre-committed rule NOTHING in the declared grid earns a ship —
and the honest reading is that ORB's flat weeks are NOT irreducible: they are bought back by
dropping the catalyst veto, at a drawdown cost the pre-committed rail refused.**

In priority order:

1. **The rule's survivor is F4** (drop range-size + G1 + PDR). Reported as the pre-committed
   outcome, and **it should not ship**: t = 0.57 on TRAIN, ex-top-5% R/pick -0.153,
   +7.5pp / 0.0pp green weeks inside a +-26pp / +-39pp MDE, at 67% of TRAIN's dollars.
2. **The single most informative cell is `catalyst veto OFF`**: 34.0 -> 47.2% green TRAIN and
   31.8 -> 54.5% green VAL, flat 37.7 -> 7.5% and 22.7 -> 4.5%, **with total P&L up on both
   splits under both fill models**. It fails one pre-committed clause on one split (worst
   TRAIN week -$895 vs -$628 allowed; TRAIN MDD -$560 -> -$2,639). **This is the next
   pre-registration**, and it needs a drawdown-shaped mitigation declared in advance — the
   obvious one being a *partial* catalyst rule (veto newsless-and-alone picks only below a
   named liquidity or range floor) rather than the all-or-nothing switch.
3. **The two wrong-side gates — the Q1 quintile filter and the range-size veto — should be
   re-examined on their own terms**, not for frequency (range-size is worth 0.06 picks/week,
   Q1 0.45) but because **they cut the wrong names**: Q1 drops the quintile with the *best*
   mean R in both years, and the range-size veto cuts trades the sizer already bets half as
   much on. Neither is a frequency lever; both are defects.
4. **Slots are settled.** 3 -> 8 was the right call and 8 is the right number on this gate
   set. Do not go to 12 or 16: no green weeks, 2-3x the drawdown, and at the live $66K
   account 90-99% of positions become account-capped.
5. **Two knobs are dead and should be documented as such**: `filter.threshold` is inert at 8
   slots (Q1 sits far above it), and `filter.prev_day_range_veto` at 11.0 is ~97% redundant
   with the G1 veto's 9.226 leg. Carrying both as separate shipped rules is exactly the
   "accidental rule" the machine-rules doctrine forbids.

**The frame the owner should hear first**: the permutation test says ORB's green-week share
is fully explained by its pick count. There is no week-timing skill to find in this book.
**Green weeks can only be bought with picks; ORB's gates are what withhold the picks; and the
two gates that withhold the most (the prev-day-range pair and the catalyst veto) are the two
with the strongest era-consistent edge.** That is a genuine trade-off, not a bug — and it is
why "more green weeks" and "more R per pick" cannot both be maximised on this universe at
this book size.

---

## 9. TEST — the sealed split

Sealed per `FREEZE.md`; opened once, after §8 was committed at `a12bc6e`, for exactly the
two cells the pre-registration named: the shipped B+ baseline and the rule's survivor.
**TEST is 16 weeks and 57 baseline picks; its green-week MDE80 is +-38.6pp. It cannot
select anything and it did not.**

| cell | fill model | picks/wk | **green%** | **flat%** | red streak | worst wk | MDD | $ | R/pick | ex-top5% R | t |
|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped B+ (F0) | measured | 3.56 | **18.8** | 31.2 | 2 | -315 | -667 | **+685** | +0.062 | -0.104 | 0.49 |
| shipped B+ (F0) | as-is | 3.56 | 37.5 | 18.8 | 2 | -315 | -515 | +1,380 | +0.166 | +0.005 | 1.26 |
| F4 (the survivor) | measured | 10.81 | **37.5** | **0.0** | 3 | -698 | -1,225 | **-569** | -0.057 | -0.198 | -0.96 |
| F4 (the survivor) | as-is | 10.81 | 43.8 | 0.0 | 3 | -698 | -1,173 | +101 | -0.026 | -0.164 | -0.42 |

**TEST reproduces the study's central trade-off without ambiguity**: F4 **doubles the
green-week share (18.8 -> 37.5%) and removes every flat week (31.2 -> 0.0%)** — and **turns
the book negative** (+$685 -> -$569 measured; +$1,380 -> +$101 as-is). It is the §4 result
again on a window nobody looked at.

Two further honest observations on this split, neither of which changes §8:
* The **shipped** book itself is close to edgeless on TEST under the measured fill model
  (R/pick +0.062, ex-top-5% **-0.104**, t 0.49). The 2026-06+ era is the weakest of the
  three for ORB in every configuration tested here.
* F4's TEST result is **worse than its VAL result**, which is the ordinary decay pattern for
  a cell selected as the maximum of a 45-cell grid. The recommendation in §8 — do not ship
  F4 — stands, and TEST reinforces it rather than having chosen it.

### 9a. Touchgo exit walks

_(appended when the walks land)_

---

## 10. Caveats

1. **Power.** 53 TRAIN and 22 VAL weeks at 2.4 picks/week. Unpaired green-week MDE80
   +-25.8pp / +-39.3pp. Nothing in §4 outside the catalyst row exceeds it.
2. **Relative tool, never a forecast.** The ORB pipeline at $10K-stage sizing is a relative
   instrument (CLAUDE.md standing rule). No dollar figure here is a projection.
3. **Both fill models are simulations.** Stage Q's measured arm walked the real SIP quote
   path for the 1,040 flagged orders and matched the live account's entry microstructure to
   a couple of points, but it is still a model. As-is and measured bracket every headline.
4. **The universe screens and the live spread gate are unmeasured** (§2). The separation map
   is complete only for the gates that live *inside* the candidate population.
5. **Exit rules are out of scope** beyond §7; `green_weeks/REPORT.md` already refuted the
   exit thesis for ORB's flat weeks.
6. **2026-09 is a partial month** and ORB has been paused (`strategy.enabled: false`) since
   9/14, so its 2026-09 picks are BT-only.
7. **The post-hoc combinations in §4 are post-hoc.** They were formed after reading the
   separation map and are reported as hypotheses, never as results.
8. **No independent rebuild of the pipeline was made** — the instrument IS the shipped
   pipeline, byte-reproducing an already-independently-checked book (D1 -> Stage Q). The
   separation map's cascade reconstruction is the one piece of new code that re-derives
   shipped logic, and it agrees with the book to 210/215 picks (§2).

## 11. Artifacts

```
research/orb_frequency/
  PREREG.md FREEZE.md REPORT.md
  repro.sh repro_n8_q1on.csv        the reproduction gate (byte-identical to D1)
  separation.py separation.csv      PART 1
  run_grid.py                       every cell (45 runs)
  score.py                          the owner's metric, TEST-sealed
  analyse.py grid_meas.csv grid_asis.csv survival.csv
  slots.py slots.csv                PART 3
  walk_touchgo.sh dump_tg*.csv      the exit walks (§7)
  book_*.csv monthly_*.csv log_*.txt
```
