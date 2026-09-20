# frames9 — F28 HOD'S DETECTOR ON A POSITIVE POND · F29 THE POND MAP · F30 THE POOLED RAMP STATISTIC — REPORT (2026-09-20)

Pass 9 of the frame programme. Cells exactly as declared in `PREREG.md`, committed (`7dd8a4b`)
before any cell was scored. Artifacts: `c9.py` (the ponds, the gates, the pond-restricted control
pools) · `w9.py` → `sig9.csv`, `pool_{b,u}_{ORB,BF}.csv`, `w9_{sig,b,u}.csv` (**49,223 walked
brackets** over 343 sessions), `w9.log` · `s9.py` → `cells28_books.csv`, `cells28_margin.csv`,
`s9.log` · `f29.py` → `cells29.csv`, `f29.log` · `supp9.py` → `supp_a_books.csv`, `supp9.log`.
One python process at a time, `nice -n 10`, `ulimit -v 3000000`, the walk checkpointed per session;
`cache.db`, `bars_sip.db`, the Databento stores and `daily_bars` opened **read-only**. No config,
`orb.yaml`, systemd unit, cron, order or cache was written. **TEST was never opened** (`FREEZE.md`).

---

## 0. THE SENTENCE THIS PASS WAS REQUIRED TO PRINT FIRST

**The transplant fails, and it fails for a reason that retires the phrase "a positive pond" from this
programme: measured at HOD-break's own clock under HOD-break's own bracket, ORB's and BF's universes
are NOT +0.070 / +0.053 R. They are −0.088 … +0.139 R depending on era and geometry, and in PRICE
they are +0.045 % — a drift statistically indistinguishable from zero.** Pass 7's positive pond
numbers were a property of the CLOCK and the R UNIT those books are measured in, not of the names
they contain. There was never a positive pond to put the detector on.

| the pre-registered prediction | what the transplant reads |
|---|---|
| (a) the selection margin transfers — within ±0.08 R of HOD's own on TRAIN **and** same-signed positive on VAL, on each pond | **PARTIAL and only where n is smallest.** 2 of 18 (pond × rung × geometry) combinations satisfy both limbs — ORB $20 X0 (+0.086 / +0.223) and UNION $20 X0 (+0.087 / +0.064) — and **every one of those point estimates is inside its own MDE** (0.22–0.31 R). At every relaxed rung the TRAIN margin collapses to **+0.000 … +0.035 R**. On BF's pond the TRAIN margin transfers under G3 at all three rungs (+0.180 / +0.130 / +0.133) and **VAL is negative at all three** (−0.100 / −0.089 / −0.072). |
| (b) the absolute book turns positive and clears the live-exploration bar | **NO, on all 36 book cells.** Best: `ORB/$20/G3` — TRAIN net −0.1045 at 3.7 tr/wk (green 30.2 %), VAL net +0.1968 at 5.2 tr/wk (green 43.5 %). Highest-frequency positive-VAL cell: `ORB/$5/X0` — TRAIN −0.1241 / green 39.6 %, VAL +0.0267 / green 56.5 % at 13.2 / 16.7 tr/wk, **TRAIN halves −0.164 / −0.096**. No cell is positive on both splits, none reaches a clustered t ≥ 2 in the right direction, and **every cell's green weeks are at or below its own count-matched null p95**. |

**Verdict, all three frames: STAY-DRY · NO SHIP · NO CONFIG CHANGE · NOTHING RE-OPENED.**
`hod_break` stays `enabled: true, dry_run: true`; `config.yaml trading.enabled` and `orb.yaml` are
exactly as the owner set them; Monday's 12:30 UTC boot is unchanged.

---

## 0b. Reproduction gates — four, all asserted in code before a number was read

| id | gate | result |
|---|---|---|
| **G-B2** | `common6.base_book()` — TRAIN 1,622 / 30.6 wk / gross −0.039 / net −0.107 / **−$17,346**; VAL 706 / 30.7 / +0.083 / +0.013 / **+$893** | **MATCH** |
| **G-ORB** | `orb_gates2/book_G3_meas.csv` — 282 TRAIN / 177 VAL picks | **MATCH** |
| **G-BF** | `bf_frequency/runs/P1.csv` — 56 trades / **$139,113.67** | **MATCH to the cent** |
| **G-PARITY** | the new pond walker vs `frames8/w_sig.csv` on every signal they share (429 signals, `rr_X0` and `rr_G3`) | **max abs Δ = 0.00e+00** |

---

# F28 — HOD'S DETECTOR ON A POSITIVE POND (12 declared cells)

## 1.1 C0 — the overlap diagnostic (reported first; it does NOT kill the frame)

| rung | pond | pond signals | on a (day, symbol) that pond's own book takes | share |
|---|---|---|---|---|
| $20 | ORB | 317 | 22 | **6.9 %** |
| $20 | BF | 136 | 0 | **0.0 %** |
| $10 | ORB | 736 | 46 | 6.2 % |
| $10 | BF | 754 | 4 | 0.5 % |
| $5 | ORB | 1,264 | 64 | 5.1 % |
| $5 | BF | 1,663 | 7 | 0.4 % |

The HOD rule does **not** re-label trades the pond's own book already takes (ORB's book has 618 picks,
BF's 55). The frame is not a tautology and was allowed to proceed.

## 1.2 The band overlap and what it does to n (measured before PREREG was committed)

HOD's shipped floor is `next_open >= $20`; ORB's universe band caps the OPEN at $30. The two rules
can only meet in **$20–$30**, and the overlap costs **95.5 %** of HOD's admitted signals:

| rung | HOD admitted (TRAIN+VAL) | ORB pond | BF pond | UNION | median fill price | imputed NBBO (ORB/BF/UNION) |
|---|---|---|---|---|---|---|
| **$20** | 7,027 | **317** (195/122) | **136** (90/46) | **429** (268/161) | $25.0 | 59 / 91 / 67 % |
| **$10** | 11,045 | 736 (437/299) | 754 (481/273) | 1,431 (882/549) | $15.8 | 82 / 98 / 90 % |
| **$5** | 15,897 | 1,264 (808/456) | 1,663 (1,106/557) | 2,782 (1,809/973) | $10.3 | 90 / 99 / 95 % |

**The cost of the rungs, measured per cell** (the dedicated NBBO fetch only ever covered the $20+
candidates, so everything below it is imputed and is reported as such): booked cost is
**0.063–0.070 R** at the $20 rung and **0.066–0.081 R** at $5 — i.e. relaxing the floor buys
frequency at roughly **+0.007 R** of cost per trade, at 90–99 % imputation. No relaxed rung can be a
ship number on its own; they exist to show whether the $20 result is the band or the book.

## 1.3 THE POND BOUND — the headline table (arm u: a RANDOM pond name, HOD's clock, HOD's bracket)

| pond | rung | X0 TRAIN | X0 VAL | G3 TRAIN | G3 VAL |
|---|---|---|---|---|---|
| **HOD's own** (frames8 arm d) | $20 | **−0.058** | **−0.042** | **−0.054** | **−0.049** |
| **ORB** | $20 | **−0.0877** | −0.0124 | **−0.0437** | **+0.0574** |
| ORB | $10 | −0.0712 | −0.0057 | −0.0100 | +0.0670 |
| ORB | $5 | −0.0679 | +0.0000 | −0.0190 | +0.0720 |
| **BF** | $20 | −0.0039 | **+0.1040** | +0.0359 | **+0.1385** |
| BF | $10 | −0.0284 | +0.0105 | +0.0318 | +0.1026 |
| BF | $5 | −0.0465 | +0.0235 | +0.0205 | +0.0726 |
| **UNION** | $20 | −0.0716 | +0.0044 | −0.0283 | +0.0691 |
| UNION | $10 | −0.0562 | −0.0005 | +0.0047 | +0.0779 |
| UNION | $5 | −0.0592 | +0.0087 | −0.0029 | +0.0725 |

**Neither pond is positive in all three eras under either geometry.** ORB's is *worse than HOD's own*
on TRAIN under the cap (−0.088 vs −0.058). Nothing here resembles pass 7's +0.070 / +0.024 and
+0.053 / +0.010 — and §2.4 explains why.

## 1.4 THE SELECTION MARGIN (sig − its own matched pond controls, paired, day-clustered)

| pond | rung | X0 TRAIN (t) | X0 VAL (t) | G3 TRAIN (t) | G3 VAL (t) | MDE (X0 / G3, TRAIN) |
|---|---|---|---|---|---|---|
| **HOD's own** | $20 | **+0.123** (+2.96) | **+0.240** (+4.74) | **+0.183** (+2.75) | **+0.245** (+4.24) | — |
| **ORB** | $20 | +0.086 (+0.84) | **+0.223** (+1.85) | +0.005 (+0.03) | **+0.257** (+1.87) | 0.257 / 0.347 |
| ORB | $10 | +0.025 (+0.37) | +0.131 (+1.82) | −0.011 (−0.13) | +0.172 (+1.43) | 0.167 / 0.219 |
| ORB | $5 | +0.001 (+0.02) | +0.147 (+2.28) | +0.019 (+0.28) | +0.203 (+1.55) | 0.123 / 0.165 |
| **BF** | $20 | +0.049 (+0.37) | **−0.244** (−1.58) | +0.180 (+0.83) | **−0.100** (−0.39) | 0.365 / 0.635 |
| BF | $10 | +0.027 (+0.50) | −0.045 (−0.63) | +0.130 (+1.49) | −0.089 (−1.03) | 0.164 / 0.261 |
| BF | $5 | +0.035 (+0.85) | −0.055 (−0.99) | +0.133 (+2.04) | −0.072 (−0.97) | 0.114 / 0.179 |
| **UNION** | $20 | +0.087 (+1.03) | +0.064 (+0.66) | +0.067 (+0.59) | +0.122 (+1.03) | 0.218 / 0.325 |
| UNION | $10 | +0.029 (+0.62) | +0.047 (+0.91) | +0.069 (+1.12) | +0.051 (+0.67) | 0.120 / 0.177 |
| UNION | $5 | +0.025 (+0.72) | +0.028 (+0.62) | +0.093 (+1.81) | +0.049 (+0.69) | 0.086 / 0.129 |

**Readings.**

1. **The margin does not transfer at the rate the prediction required.** HOD's own +0.183 / +0.245 R
   (G3) becomes **+0.005 / +0.257** on ORB's pond at the same floor and **+0.019 / +0.203** at $5.
   The only limb that reproduces cleanly is BF's TRAIN margin under G3 (+0.180 / +0.130 / +0.133
   across the three rungs, so it is not a small-n artefact) — and BF's VAL margin is **negative at
   every rung**. A selector that is +0.18 R in 2025 and −0.09 R in 2026 on the same pond is not a
   property of the detector.
2. **The arithmetic the frame was built on does not apply, because the pond bound it assumed is not
   there.** "A +0.18 R selector on a +0.07 R pond is +0.25 R" needed a +0.07 R pond. At HOD's clock
   the best pond bound in the table is BF's +0.139 R on VAL only, against −0.004 / +0.036 on TRAIN.
3. **Availability is clean and is not the story**: arms b and u cover **99–100 %** of signals in
   every cell (the declared floor is 80 %), median 24 controls/trade, matching distance
   |dlog| 1.23 (ORB's pond — it has only ≈ 36 members a session) and 0.51 (BF's).

## 1.5 THE ABSOLUTE BOOK — C1–C9 under both geometries ($100 risk, `run_book(12, 4)`)

The bar: **positive weekly $ AND green ≥ 50 % on BOTH splits at ≥ 10 tr/wk, clustered t ≥ +2, TRAIN
halves same-signed.** The five cells that come closest:

| cell | split | n | /wk | gross | cost | **net** | green (null p95) | **wk $** | total $ | ex-top-5 % | H1 / H2 | clust t |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **ORB/$20/G3** | TRAIN | 195 | 3.7 | −0.0372 | 0.0673 | −0.1045 | 30.2 (37.7) | −38 | −2,037 | −0.385 | −0.208 / −0.036 | −0.71 |
| | VAL | 119 | 5.2 | +0.2647 | 0.0679 | **+0.1968** | 43.5 (60.9) | **+102** | +2,342 | −0.037 | — / +0.197 | +1.29 |
| **ORB/$20/X0** | TRAIN | 195 | 3.7 | +0.0045 | 0.0626 | −0.0581 | 43.4 (45.3) | −21 | −1,132 | −0.168 | −0.099 / −0.031 | −0.57 |
| | VAL | 119 | 5.2 | +0.2245 | 0.0622 | **+0.1622** | 52.2 (65.2) | **+84** | +1,931 | **+0.066** | — / +0.162 | +1.20 |
| **ORB/$5/X0** | TRAIN | 697 | **13.2** | −0.0583 | 0.0658 | −0.1241 | 39.6 (41.5) | −163 | −8,650 | −0.235 | −0.164 / −0.096 | −2.43 |
| | VAL | 385 | **16.7** | +0.0962 | 0.0696 | **+0.0267** | **56.5** (60.9) | **+45** | +1,026 | −0.080 | — / +0.027 | +0.37 |
| **UNION/$20/G3** | TRAIN | 268 | 5.1 | +0.0126 | 0.0691 | −0.0565 | 35.8 (43.4) | −29 | −1,515 | −0.371 | −0.079 / −0.040 | −0.45 |
| | VAL | 158 | 6.9 | +0.1680 | 0.0698 | **+0.0982** | 39.1 (56.5) | **+68** | +1,551 | −0.152 | — / +0.098 | +0.76 |
| **BF/$5/G3** | TRAIN | 956 | **18.0** | +0.1133 | 0.0721 | **+0.0413** | 45.3 (54.7) | **+74** | +3,945 | −0.314 | +0.008 / +0.066 | +0.60 |
| | VAL | 464 | **20.2** | −0.0788 | 0.0790 | −0.1578 | 26.1 (39.1) | −318 | −7,320 | −0.401 | — / −0.158 | −2.21 |

**0 of 36 book cells clears the bar, and no cell is even positive on both splits.** The pattern is
the same in every row and it is the programme's oldest one: the cells with a positive VAL run at
3.7–5.2 trades a week (below the declared floor of 10); the cells at ≥ 10 trades a week have one
negative split; **every green-week share is at or below its own count-matched null p95**; and every
cell positive anywhere is **tail-dependent** (ex-top-5 % is negative in 9 of the 10 rows above, the
single exception being ORB/$20/X0's VAL at +0.066 on 119 trades).

**C10 — HOD's own baseline at the same rung, for comparison:** `HOD/$20/X0` TRAIN −0.1069 /
−$17,346 / green 32.1, VAL +0.0127 / +$893 / green 43.5 at 30.6 tr/wk; `HOD/$20/G3` TRAIN −0.0596 /
−$8,541, VAL +0.0176 / +$1,071 at 27.1 / 26.5 tr/wk. **Restricting HOD's detector to a live book's
universe improves VAL net (+0.013 → +0.197 on ORB's pond under G3) and worsens TRAIN (−0.060 →
−0.105), at one seventh of the frequency.** That is a re-slicing of the same 2025-negative /
2026-positive shape, not a new object.

## 1.6 Rails

* **Reproduction**: four gates, §0b, asserted in code (the run aborts otherwise), including a
  walker-parity gate that reproduces pass 8's walk to **0.0**.
* **Both TRAIN halves** on every cell (the H1/H2 columns and `cells28_books.csv`).
* **Day-clustered t** on every cell and every paired margin.
* **Count-matched permutation null** (2,000 draws, weekly pick count fixed) on every cell's green
  weeks — again what kills the week-shape reading.
* **Cost re-measured per cell** from its own exit mix; the imputed share printed per cell (36–99 %).
* **Tail**: rank-trimmed ex-top-5 % beside every headline.
* **Availability**: 99–100 % coverage on both control arms in every cell; nothing was demoted.
* **Causality**: arms b and u are non-signal names of the same session priced at the signal's own
  minute; a control is never a name that itself produced an admitted signal that day (the pass-6
  rule), and never the signal's own symbol.
* **Multiplicity**: 12 declared cells; **72 scored objects** (9 pond × rung cells × 2 geometries ×
  2 splits for the book, the same for the margin) plus the C0 table and C10; counted, none selected
  after the fact.
* **TEST**: never opened.

---

# F29 — THE POND MAP: WHICH ATTRIBUTE OWNS THE +0.23 R (10 declared splits, 28 levels)

Population: HOD-break's own 2,310 booked B2 trades that have a matched control (TRAIN 1,605 / VAL
705, median 24 controls each). Base margin reproduced exactly: **G3 +0.1829 (t +2.75) / +0.2452
(t +4.24); X0 +0.1228 (t +2.96) / +0.2400 (t +4.74)**.

## 2.1 THE SENTENCE

> **The detector's +0.23 R is the market paying for 2× and inverse single-stock WRAPPERS — and it is
> not paying the detector, it is charging the control. 38.8 % of the trades are wrappers and they
> carry 73.6 % of the pooled margin (+0.382 R against commons' +0.082 R), era-stable at +0.399 /
> +0.389 / +0.357 in H1 / H2 / VAL at day-clustered t +3.80. The margin is that size because the
> matched non-signal WRAPPER decays at −0.269 / −0.252 R, while the detector's own wrapper pick earns
> only +0.124 / +0.104 R gross against a 0.070 R cost.**

That is a nameable attribute, and the pre-committed standard (same sign in H1, H2 and VAL, with a
materially weaker complement) is met by exactly one of the ten splits: **S1**.

## 2.2 The ten splits (margin under G3; H1 / H2 are the TRAIN halves)

| # | split | level | n | **margin G3** | H1 | H2 | VAL | clust t | margin X0 | share |
|---|---|---|---|---|---|---|---|---|---|---|
| **S1** | asset class | **wrapper** | 897 | **+0.3822** | **+0.3986** | **+0.3891** | **+0.3566** | **+3.80** | +0.2762 | 38.8 % |
| | | stock | 1,403 | +0.0819 | +0.0071 | +0.0738 | +0.1745 | +2.02 | +0.0806 | 60.7 % |
| | | *(unknown)* | 10 | +0.8730 | +0.7146 | +1.7396 | −0.0307 | +1.73 | +0.5602 | 0.4 % |
| **S2** | price band | ≥ $60 | 618 | +0.3022 | +0.3467 | +0.2944 | +0.2778 | +4.17 | +0.1860 | 26.8 % |
| | | $30–60 | 893 | +0.1810 | +0.0868 | +0.2704 | +0.1937 | +2.62 | +0.1615 | 38.7 % |
| | | < $30 | 799 | +0.1477 | +0.1356 | +0.0657 | +0.2685 | +2.51 | +0.1342 | 34.6 % |
| **S3** | ADV$ | $25–150M | 938 | +0.2612 | +0.2380 | +0.2858 | +0.2537 | +4.46 | +0.2318 | 40.6 % |
| | | ≥ $150M | 888 | +0.1970 | +0.2171 | +0.0988 | +0.2651 | +2.47 | +0.1023 | 38.4 % |
| | | < $25M | 484 | +0.0961 | **−0.1461** | +0.1977 | +0.1847 | +1.42 | +0.1199 | 21.0 % |
| **S4** | gap at the open | flat (−2..+2 %) | 822 | +0.2187 | +0.1775 | +0.2025 | +0.2862 | +3.16 | +0.1603 | 35.6 % |
| | | gap ≤ −2 % | 599 | +0.2136 | +0.2882 | +0.1716 | +0.1633 | +1.75 | +0.1437 | 25.9 % |
| | | gap ≥ +2 % | 889 | +0.1786 | +0.0400 | +0.2252 | +0.2644 | +3.13 | +0.1670 | 38.5 % |
| **S5** | `rv_profile` | ≥ 5 | 779 | +0.1858 | +0.0364 | +0.1944 | +0.3144 | +3.60 | +0.1932 | 33.7 % |
| | | < 5 | 1,531 | +0.2101 | +0.2124 | +0.2099 | +0.2077 | +3.12 | +0.1410 | 66.3 % |
| **S6** | sibling moved (causal) | sibling | 69 | +0.5449 | +0.4709 | +0.7474 | +0.3667 | +2.09 | +0.2415 | **3.0 %** |
| | | alone | 2,241 | +0.1914 | +0.1502 | +0.1869 | +0.2417 | +3.95 | +0.1560 | 97.0 % |
| **S7** | listing age | new (< 60 sessions) | 1,038 | +0.2155 | +0.2290 | +0.0861 | +0.3855 | +2.51 | +0.1593 | 44.9 % |
| | | old | 1,272 | +0.1909 | **−0.0713** | +0.2639 | +0.2023 | +3.76 | +0.1580 | 55.1 % |
| **S8** | entry-minute band | 09:37–10:30 | 1,653 | +0.2196 | +0.1349 | +0.1997 | +0.3278 | **+4.68** | +0.1685 | 71.6 % |
| | | 10:30–11:30 | 326 | +0.2854 | +0.3647 | +0.2934 | +0.1715 | +1.68 | +0.2452 | 14.1 % |
| | | 11:30–13:00 | 234 | +0.0027 | +0.0087 | +0.1246 | **−0.1726** | +0.02 | +0.0072 | 10.1 % |
| | | 13:00–14:01 | 97 | +0.1014 | +0.1815 | +0.1970 | **−0.1649** | +0.64 | +0.0645 | 4.2 % |
| **S9** | `dist_open_pct` | 5–10 % | 2,192 | +0.2103 | +0.1705 | +0.2126 | +0.2525 | +4.11 | +0.1609 | 94.9 % |
| | | 10–20 % | 105 | +0.1289 | −0.0256 | +0.1502 | +0.1849 | +0.97 | +0.1202 | 4.5 % |
| | | ≥ 20 % | 13 | −0.6166 | −0.5185 | −0.8442 | −0.5092 | −2.64 | +0.0843 | 0.6 % |
| **S10** | `dollar_frac` | T1 low | 666 | +0.3260 | +0.0988 | +0.4971 | +0.3258 | +4.53 | +0.2418 | 28.8 % |
| | | T2 | 666 | +0.2446 | +0.4777 | +0.1012 | +0.2241 | +2.53 | +0.1503 | 28.8 % |
| | | T3 high | 667 | +0.0766 | +0.0098 | +0.1012 | +0.1055 | +1.21 | +0.0967 | 28.9 % |

**What survives the "same sign in H1, H2 and VAL with a materially weaker complement" standard**:
S1 cleanly (wrapper +0.399 / +0.389 / +0.357 vs stock +0.007 / +0.074 / +0.175); S2's ≥ $60 band and
S3's $25–150M band partially — and both are largely the wrapper cut wearing a different label, since
wrappers cluster in the mid-ADV$, higher-price region. **S6 is the largest single number in the pass
(+0.545) and is 3.0 % of trades** — 69 trades, MDE 0.698: reported, not believed. **S10's T1 and
S8's late bands sign-flip between the TRAIN halves or between TRAIN and VAL** and are rejected by
the standing rule.

## 2.3 SUPP A — the absolute book behind the split (POST-HOC, declared as such)

A margin is a difference; the owner is paid an absolute. Cutting HOD's own booked B2 set by S1:

| cut | geom | split | n | /wk | signal gross | matched control | **net** | green (null p95) | wk $ | H1 / H2 | ex-top-5 % | clust t |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **wrapper** | G3 | TRAIN | 599 | 11.3 | +0.1101 | −0.2693 | **+0.0403** | 43.4 (50.9) | **+45** | +0.039 / +0.042 | −0.297 | +0.35 |
| | | VAL | 274 | 11.9 | +0.0994 | −0.2523 | **+0.0247** | 52.2 (56.5) | **+29** | — / +0.025 | −0.199 | +0.26 |
| wrapper | X0 | TRAIN | 621 | 11.7 | −0.0175 | −0.2445 | −0.0815 | 37.7 (45.3) | −96 | −0.153 / −0.018 | −0.194 | −1.23 |
| | | VAL | 276 | 12.0 | +0.1490 | −0.2379 | +0.0803 | 60.9 (65.2) | +96 | — / +0.080 | −0.021 | +1.02 |
| stock | G3 | TRAIN | 951 | 17.9 | −0.0493 | −0.1056 | −0.1235 | 32.1 (39.6) | −222 | −0.139 / −0.109 | −0.360 | −2.17 |
| | | VAL | 409 | 17.8 | +0.0623 | −0.1158 | −0.0123 | 43.5 (52.2) | −22 | — / −0.012 | −0.230 | −0.17 |

**`wrapper/G3` is same-signed positive in H1, H2 and VAL at 11.3–11.9 trades a week — the second
object in 1,090 cells with that profile** (the first was `hod_fresh` C1). It fails the bar on every
other axis, exactly as C1 did: **clustered t +0.35 / +0.26**, **green weeks BELOW their own
count-matched null p95 on both splits** (43.4 vs 50.9; 52.2 vs 56.5), and **ex-top-5 % net −0.297 /
−0.199** — remove five trades in a hundred and it is deeply negative. It is also a **post-hoc cut of
28 levels** and is therefore a hypothesis for a pre-registration, not a result.

## 2.4 SUPP B — WHY pass 7's ponds looked positive, and what a pond bound actually is

The pond bound is a **ratio**: what the name does between entry and the exit, divided by R. On the
22,628 arm-u brackets this pass walked, the numerator is essentially zero:

> **mean price move entry → exit = +0.045 %** (TRAIN −0.049 %, VAL +0.220 %) against a median stop of
> **4.18 %**.

So the bound's sign and size are set by the denominator and by the stop-hit rate, not by the names:

| bucket | n | bound G3 | H1 | H2 | VAL | bound X0 | eod % (G3) | stop % (G3) |
|---|---|---|---|---|---|---|---|---|
| stop < 1.5 % | 810 | **+0.3162** | +0.5092 | −0.0269 | +0.4006 | +0.0569 | 36 | 64 |
| stop 1.5–3 % | 5,307 | +0.0240 | −0.1116 | +0.0382 | +0.0857 | −0.0465 | 44 | 56 |
| stop 3–6 % | 10,954 | +0.0104 | +0.0263 | −0.0533 | +0.0716 | −0.0456 | 58 | 42 |
| stop ≥ 6 % | 5,557 | **+0.0045** | +0.0259 | −0.0156 | +0.0092 | −0.0193 | 80 | 20 |
| clock 09:37–10:30 | 17,418 | −0.0019 | −0.0145 | −0.0306 | +0.0408 | −0.0502 | 60 | 40 |
| clock 10:30–11:30 | 2,536 | +0.1030 | +0.1186 | −0.0115 | +0.1915 | +0.0260 | 53 | 47 |
| clock 11:30–13:00 | 2,091 | +0.1018 | +0.1394 | +0.0399 | +0.1296 | −0.0179 | 58 | 42 |
| clock 13:00–14:01 | 583 | +0.1385 | +0.2425 | −0.0144 | +0.2236 | +0.0657 | 73 | 27 |

**Three consequences, and they are the durable part of this pass.**

1. **"The pond is positive / negative" is not a statement about a population.** The same names, the
   same sessions and the same clock read **+0.32 R** with a 1.5 % stop and **+0.005 R** with a 6 %
   stop under the identical bare-stop exit, because R is the unit. Pass 7 compared ORB's pond
   (≈ 5.5 % stops, 09:36, static lock) with HOD's (4.2 % stops, 09:51 median, +2 R cap) and read the
   difference as a property of the names. **It is the R unit and the cap.**
2. **The cap flattens it and the uncapped exit amplifies it**, in both directions: under X0 every
   bucket is within ±0.07 R of zero; under G3 the spread across stop-width buckets is 0.31 R. An
   uncapped exit does not make a population better — it makes the population's R unit matter more.
3. **A pond bound is only comparable across books when the stop width, the clock AND the cap are
   held fixed** — the same lesson pass 8 wrote for geometry, now extended to the R unit. Standing
   rule: **never compare two books' baselines without stating the stop width as a per cent of price.**

## 2.5 Rails (F29)

Both TRAIN halves and VAL on every level; day-clustered t and MDE on every level; all 28 levels
printed whatever they read; the margin is a **paired** statistic (each trade against its own
controls), so no cross-trade matching bias enters; the causal form of S6 only (a sibling counts only
if its admitted break was strictly EARLIER in the session); availability 2,310 of 2,328 booked
trades = **99.2 %**; the two SUPP tables are labelled post-hoc and counted in the multiplicity.

---

# F30 — THE POOLED RAMP STATISTIC (specification only; nothing outside `frames9/` was modified)

## 3.1 The calendar arithmetic

Per-trade SD measured on each book's own walked population: **ORB 1.694**, **BF 1.939**, **HOD-break
dry 1.260** (B2 net, TRAIN+VAL, measured this pass). Frequencies: ORB 6.5/wk, BF-P1 0.65/wk
(2.8/month), HOD dry ≈ 27/wk. Effect standardised by each book's **own** SD, `δ_i = effect / SD_i`;
the pooled statistic's standardised effect is the frequency-weighted mean `δ̄ = Σ w_i δ_i / Σ w_i`,
and `n₈₀ = (2.80 / δ̄)²`.

| stream | trades/wk | **effect 0.2 R** → n₈₀ / weeks / years | **effect 0.3 R** → n₈₀ / weeks / years |
|---|---|---|---|
| ORB alone | 6.5 | 562 / 87 / **1.66** | 250 / 38 / 0.74 |
| **BF-P1 alone** | 0.65 | 737 / 1,133 / **21.79** | 327 / 504 / 9.69 |
| HOD-break dry alone | 27.0 | 311 / 12 / 0.22 | 138 / 5 / 0.10 |
| **ORB + BF pooled** | 7.15 | 575 / **80.5** / **1.55** | 256 / 35.8 / 0.69 |
| **ORB + BF + HOD-dry** | 34.15 | 349 / **10.2** / **0.20** | 155 / 4.5 / 0.09 |

**The two readings.** (a) Pooling the two LIVE books buys almost nothing — 1.55 years against ORB's
1.66 — because BF contributes 9 % of the trades. (b) **The only lever that changes the calendar is
the dry stream**: adding HOD-break's ≈ 27 paper trades a week takes a 0.2 R question from 80 weeks to
**10 weeks**, and it does so because HOD's per-trade SD is the *lowest* of the three (1.26 — a +2 R
cap truncates the variance as well as the tail). That is a statement about **measurement**, not about
money: HOD dry earns nothing, so it can enter a *diagnostic* and must never enter a P&L gate.

## 3.2 The estimator, specified

* **Statistic.** For every closed live trade *i* of book *b*, `z_i = R_i / SD_b`, where `R_i` is the
  book's own realized R (ORB: `pnl / total_risk`; BF: `pnl / trading.risk_per_trade`; HOD dry:
  `pnl_sim / risk_usd`) and `SD_b` is that book's **BT per-trade SD**, frozen in
  `trading/ramp_bt_band.py` beside the existing reference books (ORB 1.694 from
  `orb_gates2/book_G3_meas.csv`, BF 1.939 from `bf_frequency/runs/P1.csv`, HOD 1.260 from the B2
  reference). The pooled estimate is `z̄` with a **day-clustered** SE (one cluster per session across
  all books — they share the session, the account and the market factor, so the day is the right
  cluster and a naive SE overstates precision).
* **The same-name-day-later-minute control** (F27's only variance reducer, −14 % SD on ORB) is
  attached where the bars exist: `z_i' = (R_i − mean_k R(s_i, d_i, m_k)) / SD_b'`, ten later minutes
  on the same symbol-day priced through that book's own exit spec. Reported beside `z̄`, never in
  place of it, until it has 30 trades of its own.
* **Where it goes.** New `trading/ramp_pool.py` (ONE spec, both checkers import it — the house rule):
  `pooled_z(trades_by_book, sds)` → `(z_bar, se, n, per_book_n)`. `trading/ramp_bt_band.py` gains
  `BOOK_SD` (the frozen SDs) and a `pooled_band()` that bootstraps the SAME pooled statistic from the
  reference books, so the live pooled z has a BT band to sit in exactly as the per-book band works
  today. `scripts/orb_ramp_check.py` and `scripts/bf_ramp_check.py` each print one extra line —
  `POOLED z = … ± … on n = … (ORB …, BF …, HOD-dry …) vs pooled BT band […] → IN-BAND / BELOW-p10 / …`.
  `scripts/hod_break_eod_check.py` contributes its dry stream to the pool file and nothing else.
* **How a pooled ADVANCE / DEMOTE would read, and the pre-committed constraints.**
  * The pooled z is a **precision gate, not a P&L gate**. It can only ever *block*; it never
    substitutes for a book's own realized P&L.
  * **The above-water rule is inviolable** (`project_orb_ramp_above_water_rule`): a book whose OWN
    realized stage P&L is ≤ 0 does not advance, whatever the pool says. The pool can never lift a
    losing book on a winning sibling's evidence.
  * **ADVANCE** = the book's existing gate (P&L > 0, fills, sessions, parity clean, no rail) **AND**
    the pooled z is not BELOW-p10 of the pooled BT band. Effect: a book that cleared its own gate on
    8 trades while the portfolio is running a standard error below its backtest is HELD — the pool's
    job is to catch "this stage passed on noise".
  * **DEMOTE** = unchanged per book (−6u / 5 losers / a weekly rail) **plus** a pooled trigger:
    pooled z below the pooled band's p5 with n ≥ 30 demotes **every live book one stage**, because at
    that n the portfolio, not the book, is what has changed.
  * **HOD dry contributes to n and z̄ for the BAND check only** and is excluded from every P&L-based
    clause; the line prints its contribution separately so a reader can see how much of the precision
    is paper.
* **Replay.** The acceptance test is a replay of the ramp decisions ORB and BF actually took, on
  realized P&L only, showing that the pooled gate would not have advanced anything the per-book gates
  did not and naming any stage it would have held. **Not run in this pass** — both books are PAUSED
  (owner, 9/14), so the realized stage histories are frozen and the replay would be a replay of
  nothing. It is the first step when they resume.
* **Failure modes, each of which must log** (CLAUDE.md fallback rule): a missing `SD_b` → the book is
  excluded from the pool and the line says so at WARNING (never silently dropped); fewer than 10
  pooled trades → `NO-DATA`, which blocks nothing; a book whose live R cannot be computed (missing
  `total_risk`) → excluded with an ERROR, and the per-book gate still runs.

**Specify, do not build** — as declared. No file outside `frames9/` was touched.

---

## 4. The adequacy review (RUNBOOK step 10)

* **Did we test what the books actually ARE?** For the detector, yes: HOD's B2 cascade is the shipped
  one with only the price floor moved, and the new walker reproduces pass 8's to 0.0. For the ponds
  we tested each book's own **universe screen** (ORB's `gap ≥ 5 % ∧ prev_vol ≥ 500K ∧ open $3–30`,
  the rule live's `build_orb_universe_from_snapshots` applies; BF's name eligibility via the shipped
  `bf_universe_filter` plus its $2–30 band). We did **not** test those books' own clocks or
  geometries — deliberately, and declared in PREREG §2: this frame varies the name population and
  holds everything else at HOD's, which is the only way to attribute the +0.23 R. §2.4 is the price
  of the other choice.
* **Is the cost and fill model right?** Cost is re-measured per cell from its own exit mix and the
  imputed share is printed; at the relaxed rungs it is 90–99 % imputed and those rungs are labelled
  diagnostics for that reason. The fill convention is the engine's (the next bar's open under the
  cap) and is shared by signal and control. Controls are gross by design (F24's rule).
* **Does any caveat in our own report explain the headline?** The headline is a refutation, and the
  caveat that would rescue it is measured and does not: the $20 rung's positive-VAL cells run at
  3.7–5.2 trades a week with MDEs of 0.26–0.41 R, and the relaxed rungs that reach 13–20 trades a
  week take the TRAIN margin to zero. §2.4 shows the premise itself was wrong.
* **What is the MDE?** F28: 0.086–0.365 R on the paired margins, 0.10–0.78 R on the book cells. F29:
  0.099–0.698 R per level. The honest phrasing: *no transfer of HOD-break's name-day selection to
  ORB's or BF's universe was detectable at the pre-committed bar, in THIS population, at THIS
  1-minute horizon, at THIS book size (12/day, 4 concurrent), over 2025-01 → 2026-05, at the
  re-measured 0.063–0.081 R cost — with a smallest detectable effect of about 0.09–0.37 R, and with
  the pond bound the transplant was predicated on measured at −0.09 … +0.14 R rather than +0.07 R.*
* **Verdict**: **STAY-DRY · NO SHIP · NO CONFIG CHANGE · NOTHING RE-OPENED.**

**Cell count.** 22 declared in `PREREG.md` (F28 12 + F29 10 + F30 0); 72 scored objects in F28's
grid, 28 levels in F29, 6 post-hoc book rows and 8 bucket rows in the two SUPP tables, all named
above. **Programme total 1,068 + 22 = 1,090.**
