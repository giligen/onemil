# hod_frames4 — F13 the slot rule · F14 the day as the unit · F15 who is breaking out — REPORT (2026-09-19)

Pass 4 of the HOD-break frame programme. Cells exactly as declared in `PREREG.md`, **committed
`92db20a` before any cell was scored**. Artifacts: `build4.py` -> `sig4.csv` (7,027 pre-book signals,
344 sessions) + `build4.log` · `score13.py` -> `score13.log`, `cells13.csv`, `nulls13.csv` ·
`score14.py` -> `score14.log`, `cells14.csv`, `nulls14.csv` · `supp14.py` -> `supp14.log` ·
`fetch_finra.py` -> `short_interest.csv` (923,387 FINRA rows, settlement 2024-11-15 -> 2026-08-31) ·
`score15.py` -> `score15.log`, `cells15.csv`, `nulls15.csv`, `sig4_inst.csv` · `supp15.py` ->
`supp15.log`. One python process at a time, `nice -n 10`, `ulimit -v 3000000`; `cache.db`,
`daily_bars` and every store opened **read-only**. No config, `orb.yaml`, systemd unit, cron, order
or cache was written. The dry run was not touched. **TEST was never opened** (`FREEZE.md`).

---

## VERDICT — **STAY DRY on all three.** 0 of 38 declared cells clear either bar.

*And the pass produces one result worth more than the three verdicts: the first cell in the whole
programme to be positive in dollars on BOTH splits with a three-era sign was killed, inside the same
pass, by its own causality trace — before it reached the owner.*

1. **F13 — the slot rule is not the leak, and the reason is the availability rail.** The **oracle
   ceiling is enormous**: ranking each day's signals by realised R and taking the top 12 under 4
   concurrent gives **+0.391 / +0.565 gross, +$60,604 / +$45,416, 77.4 % / 95.7 % green weeks** — so
   the frame was not dead before it started. But a signal's only obtainable fill is the open of the
   minute after its break bar, so a **ranking can only reorder candidates arriving in the SAME
   minute**, and that touches **4.9 % (TRAIN) / 13.9 % (VAL)** of the booked set. All five causal
   rankings sit **inside the 200-draw random-tie-break band on both splits**. Two by-products: more
   slots is **strictly worse** (8 concurrent −$25,838 / −$3,987; 12 concurrent −$28,430 / −$771), and
   `hod_fresh` §3's "the slot rule destroys the edge" does **not** generalise — on the base
   population the slot rule moves gross **−0.002 -> −0.039 on TRAIN and +0.015 -> +0.083 on VAL**,
   opposite directions.
2. **F14 — with the day as the unit the D2 gate is not significant on either split, and NO 09:35
   state makes H1-2025 positive.** `spy_r5 > 0` reads **−$24.7/day (t −0.75) on TRAIN and +$66.1/day
   (t +1.43) on VAL**, and **both point estimates are below their own 80 %-power MDE** ($92 and $130
   a day on 124 / 52 days). The programme's headline "+2.00 / +2.37 clustered t" was a **separation**
   t between kept and rejected *trades*, never the gated book's own day-level t. The pass answers the
   frame's question: **H2-2025 IS made positive** by `spy_r5 > 0` (+$32.4/day, +$2,071, 48.4 % green
   days, VAL read +$66.1/day), **and H1-2025 is negative under every one of eight declared 09:35
   states** (−$52 to −$139 a day). The day gate is a two-era object, not a three-era one.
3. **F15 — the name matters, and the one cell that looked like an answer was a look-ahead.**
   `anchor_cohort >= 2` (>= 2 same-morning candidates sharing an underlying anchor — ORB's complex
   confirmation) was **same-signed positive in H1/H2/VAL (+0.014 / +0.227 / +0.380), +$3,167 /
   +$9,389, 52.8 / 60.9 % green weeks, VAL clustered t +3.30** — the first such cell in 920 cells.
   **Its causality trace kills it**: in 30.8 % of its rows the sibling broke *after* our entry
   minute, and **that sub-book is the entire edge** (+0.457 / +0.839 gross, t +3.54 / +6.43). The
   **causal** version — siblings that had ALREADY broken — is **−0.013 / +0.154 gross, −$2,284 /
   +$2,099, H1 −0.120, inside its null on both splits, net below its MDE on both.** Short interest,
   fetched for this pass after being dropped on availability twice, **carries nothing**
   (`daysToCover >= p75`: H1 −0.084). Common stock alone is **negative in H1, H2 and VAL**
   (−0.060 / −0.046 / −0.055): the book's positive VAL is entirely its leveraged wrappers.

---

## 0. Reproduction gate — EXACT, and the slot machine is asserted against the shipped rule

| id | this pass | reference | verdict |
|---|---|---|---|
| R1 `B2` TRAIN | 1,622 · 30.6/wk · −0.039 · −0.107 · 32.1 % · **−$17,346** | identical | **MATCH** (Δ$ 0) |
| R2 `B2` VAL | 706 · 30.7 · +0.083 · +0.013 · 43.5 % · **+$893** | identical | **MATCH** |
| R2b `B2` rebuilt from `breaks2.csv` | 1,622 / −$17,346 · 706 / +$893 | identical | **MATCH** |
| **slot-machine parity** | `common4.book_ranked(score=None)` vs `trading.hod_break.run_book` | 2,778 vs 2,778 rows, **identical set** | **asserted in code** |

**Declared population difference from pass 3**: `hod_frames3/nbbo3.csv` (the F10 dedicated fetch,
which did not exist when pass 3 reproduced the reference) is NOT merged — merging it moves the B2
rebuild by 17 trades (1,605 / −$18,668). With it excluded the rebuild is byte-identical to the
reference, which is why it is the primary population; `use_nbbo3=True` remains the sensitivity arm
and was never needed (no cell cleared).

---

# F13 — RANK, DON'T RACE (16 cells)

## 1.1 The availability rail, measured before the cells

A signal's only obtainable fill is the open of the minute after its break bar. A candidate from an
earlier minute is not buyable later at the price this dataset prices it at, so **any "K-minute queue"
that enters a stale candidate at its original price is an unobtainable fill** (CLAUDE.md rule 1b) and
was not scored as a book. What the rail leaves: **48.6 % (TRAIN) / 41.3 % (VAL) of signals share
their entry minute with at least one other signal** (520 / 372 contested minutes, 18–20 % of the
day's minutes) — the frame has real scope.

## 1.2 The oracle ceiling — the frame is NOT dead before it starts

| cell | TRAIN n | /wk | gross | net | grn % | **$** | VAL n | /wk | gross | net | grn % | **$** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **F13-O1** oracle top-4/day | 957 | 18.1 | +1.160 | +1.112 | 90.6 | **+106,466** | 408 | 17.7 | +1.657 | +1.613 | 95.7 | **+65,815** |
| **F13-O2** oracle top-8/day | 1,819 | 34.3 | +0.581 | +0.527 | 79.2 | +95,820 | 811 | 35.3 | +1.144 | +1.092 | 95.7 | +88,555 |
| **F13-O3** oracle top-12/day | 2,454 | 46.3 | +0.306 | +0.247 | 64.2 | +60,560 | 1,187 | 51.6 | +0.763 | +0.707 | 95.7 | +83,928 |
| **F13-O4** oracle 12/day, **4 conc** | 1,816 | 34.3 | +0.391 | +0.334 | 77.4 | **+60,604** | 896 | 39.0 | +0.565 | +0.507 | 95.7 | **+45,416** |

**The pre-committed gate passes**: the oracle at 12/day is far above the 0.061–0.065 R cost line, so
the slot rule leaves a very large amount on the table in principle. *These are bounds, not
strategies, and they are excluded from both bars by construction.*

## 1.3 The causal rankings — every one inside the tie-break's own noise

The control band: 200 seeded **random** tie-breaks within the minute.

| split | random tie-break green % | random tie-break total $ | random gross | **C0 (alphabetical)** |
|---|---|---|---|---|
| TRAIN | 33.7 [30.2, 35.9] | −$16,951 [−18,714, −15,241] | −0.0366 [−0.0474, −0.0261] | 32.1 % / −$17,346 / −0.0390 |
| VAL | 46.8 [43.5, 52.2] | +$672 [−898, +2,089] | +0.0790 [+0.0574, +0.0986] | 43.5 % / +$893 / +0.0827 |

*(The shipped alphabetical tie-break is itself slightly unlucky on green weeks — 32.1 vs a 33.7 mean
on TRAIN, 43.5 vs 46.8 on VAL — and comfortably inside the band. The alphabet costs about 1.6 / 3.3
pp of green weeks and nothing that matters.)*

| cell | TRAIN /wk · gross · net · grn % · **$** | VAL /wk · gross · net · grn % · **$** | H1/H2/VAL | swapped vs C0 | $ vs band |
|---|---|---|---|---|---|
| **F13-r1** rank `rv_profile` | 30.6 · −0.038 · −0.106 · 30.2 · −17,158 | 30.8 · +0.076 · +0.006 · 43.5 · +429 | −0.064/−0.013/+0.076 | 83 / 88 | inside / inside |
| **F13-r2** rank `dollar_frac` | 30.7 · −0.047 · −0.115 · 30.2 · −18,690 | 30.5 · +0.098 · +0.029 · 43.5 · +2,035 | −0.077/−0.018/+0.098 | 80 / 76 | inside / inside |
| **F13-r3** rank `dist_open/med_rng` | 30.6 · −0.039 · −0.106 · 35.8 · −17,273 | 30.7 · +0.089 · +0.020 · 47.8 · +1,414 | −0.065/−0.014/+0.089 | 69 / 87 | inside / inside |
| **F13-r4** rank spread/R (cheapest) | 30.5 · −0.030 · −0.097 · 35.8 · −15,711 | 30.8 · +0.080 · +0.012 · 47.8 · +869 | −0.059/−0.004/+0.080 | 73 / 92 | inside / inside |
| **F13-r5** rank composite z | 30.5 · −0.034 · −0.101 · 35.8 · −16,313 | 30.7 · +0.094 · +0.026 · 47.8 · **+1,841** | −0.056/−0.012/+0.094 | 76 / 97 | inside / inside |
| **F13-r6** SPY-state × entry minute | **VOID by the availability rail** — measured, not asserted: within a (day, minute) the score takes exactly **1** distinct value, so it cannot order simultaneous candidates | | | | |

**The most a ranking can even touch is 4.9 % of the TRAIN book and 13.9 % of the VAL book** (69–83 of
1,622 and 76–97 of 706 trades swapped). The best cell, `F13-r5`, improves TRAIN by $1,033 and VAL by
$948 — **both inside the random band, on both splits.** The pre-committed selector returns **NONE**.

## 1.4 The slot COUNT and the reserve

| cell | TRAIN /wk · gross · net · grn % · **$** | VAL /wk · gross · net · grn % · **$** | vs band |
|---|---|---|---|---|
| **F13-n8** r5 @ 8 concurrent | 44.8 · −0.041 · −0.109 · 30.2 · **−25,838** | 49.6 · +0.035 · −0.035 · 30.4 · **−3,987** | below / below |
| **F13-n12** r5 @ 12 concurrent | 46.3 · −0.047 · −0.116 · 28.3 · **−28,430** | 51.6 · +0.065 · −0.006 · 39.1 · −771 | below / inside |
| **F13-rs1** reserve 1 of 4 until 10:30 | 28.7 · −0.036 · −0.103 · 35.8 · −15,649 | 30.5 · +0.032 · −0.037 · 39.1 · −2,593 | inside / below |
| **F13-rs2** reserve 2 of 4 until 10:30 | 26.0 · −0.018 · −0.082 · **41.5** · −11,307 | 28.3 · +0.008 · −0.059 · 43.5 · −3,870 | **ABOVE** / below |

**More slots is strictly worse on both splits** — the marginal candidate the 4-slot cap currently
turns away is negative, so the book already wants fewer slots than it has, not more. `F13-rs2` is
this pass's one ABOVE reading on TRAIN dollars **and it loses $3,870 on VAL** (below the band):
F7's lesson, fourth appearance.

## 1.5 F13 verdict

**The ordering among simultaneous candidates is non-informative.** The oracle says the slot rule
leaves ~$45–60K on the table at $100 risk; the availability rail says only ~5–14 % of the booked set
is reachable by a ranking; and inside that 5–14 %, five declared causal scores are indistinguishable
from shuffling the alphabet. **The claimed cost of the slot rule does not generalise either**:
`hod_fresh` §3 measured +0.064 -> +0.007 on the fresh-high rungs, but on the base population the slot
rule takes gross **−0.002 -> −0.039 on TRAIN and +0.015 -> +0.083 on VAL** — it hurts in one year and
helps in the other, the same sign instability every other object in this programme has. No `run_book`
change follows.

---

# F14 — THE DAY AS THE UNIT (10 cells, realised as 15 scored day-gate rows + a declared 6-state search)

Day universe **344 sessions** (TRAIN 242 = H1 117 + H2 125; VAL 102), **no-trade days: 0** — the book
trades every session, so the flat-day term in the owner's metric is not in play here. Field coverage
100 % on `spy_r5_pct`, `qqq_r5_pct`, `spy_rng5_atr`. **IWM and the VIX open gap are not in this
repo's index tape (SPY/QQQ only) and were declared NOT SCORED rather than fetched.**

| cell | TRAIN days · **$/day** · t · grn days % · grn wk % · **$** | VAL days · **$/day** · t · grn days % · grn wk % · **$** | H1 / H2 / VAL $/day |
|---|---|---|---|
| **F14-d0** base | 242 · **−71.7** · −2.97 · 38.8 · 32.1 · −17,346 | 102 · **+8.8** · +0.26 · 52.0 · 43.5 · +893 | −96.6 / −48.3 / +8.8 |
| **F14-d1** `spy_r5>0` [D2] | 124 · **−24.7** · **−0.75** · 45.2 · 43.4 · −3,059 | 52 · **+66.1** · **+1.43** · 65.4 · 65.2 · +3,439 | **−85.5 / +32.4 / +66.1** |
| F14-d2a `spy_r5` T1 (most down) | 81 · −94.1 · −2.08 · 34.6 · 28.3 · −7,625 | 33 · −63.0 · −1.07 · 36.4 · 39.1 · −2,078 | −87.5 / −101.3 / −63.0 |
| F14-d2b `spy_r5` T2 | 80 · −75.5 · −1.87 · 43.8 · 39.6 · −6,040 | 26 · −0.1 · −0.00 · 46.2 · 30.4 · −2 | −105.0 / −52.5 / −0.1 |
| F14-d2c `spy_r5` T3 (most up) | 81 · −45.4 · −1.14 · 38.3 · 28.3 · −3,680 | 43 · +69.1 · +1.37 · 67.4 · 60.9 · +2,973 | −98.8 / +6.7 / +69.1 |
| F14-d3 `qqq_r5>0` | 114 · −102.3 · −3.12 · 34.2 · 32.1 · −11,667 | 52 · +42.4 · +0.88 · 57.7 · 52.2 · +2,206 | −139.1 / −68.1 / +42.4 |
| F14-d4 `spy_r5>0 AND qqq_r5>0` | 94 · −67.0 · −1.90 · 38.3 · 34.0 · −6,294 | 40 · +63.8 · +1.10 · 65.0 · 60.9 · +2,554 | −133.2 / −6.1 / +63.8 |
| F14-d5a `spy_rng5_atr` T1 (quiet) | 81 · −69.3 · −1.70 · 40.7 · 28.3 · −5,610 | 39 · +68.9 · +1.22 · 59.0 · 43.5 · +2,687 | −122.0 / +15.9 / +68.9 |
| F14-d5b `spy_rng5_atr` T2 | 80 · −103.4 · −2.66 · 36.2 · 28.3 · −8,274 | 28 · −76.0 · −1.01 · 39.3 · 21.7 · −2,129 | −102.2 / −104.2 / −76.0 |
| F14-d5c `spy_rng5_atr` T3 (wild) | 81 · −42.7 · −0.93 · 39.5 · 24.5 · −3,462 | 35 · +9.6 · +0.22 · 54.3 · 39.1 · +335 | −57.7 / −30.1 / +9.6 |
| **F14-d6** breadth at 09:35 | **VOID BY CONSTRUCTION** — 0 of 7,027 signals have `entry_m <= 576`; the **earliest entry minute in the whole population is 09:37**, because the rule needs >= 5 closed bars holding within 4 % of a high already >= 5 % above the open. The field is identically 0 on every session. | | |
| **F14-d8** d1 × F13's composite ranking | 124 · **−16.8** · −0.54 · 48.4 · 43.4 · −2,088 | 52 · **+87.2** · **+1.85** · 67.3 · **60.9** · **+4,534** | −62.3 / +25.8 / +87.2 |
| **F14-d9** ORACLE day set (`y>0`) [bound] | 94 · +290.1 · +11.43 · 100.0 · 83.0 · +27,269 | 53 · +263.5 · +9.47 · 100.0 · 95.7 · +13,968 | +284.8 / +295.0 / +263.5 |

## 2.1 The power statement the programme has never printed

| split | gated days | day mean | SE | **day-level t** | **80 %-power MDE** | point estimate vs MDE |
|---|---|---|---|---|---|---|
| TRAIN | 124 | −$24.7 | 32.8 | **−0.75** | **$92/day** | **BELOW** |
| VAL | 52 | +$66.1 | 46.4 | **+1.43** | **$130/day** | **BELOW** |

**The D2 gate's own book is not distinguishable from zero on either split with the day as the unit,
and the test never had the power to distinguish it.** The "+0.388 / +0.189 R, clustered t +2.00 /
+2.37" that has carried this object through four passes is the **separation between kept and rejected
trades** — a different estimand from the gated book's day-level mean; both are true and only the
second is what a day gate would earn. **Day-level MDE on the base is $68 / $94 a day; the whole book
is inside it on VAL (+$8.8).**

## 2.2 The frame's own question, answered both ways

**Is there a 09:35 state under which H2-2025 (the flat half) is positive?** **YES, one:**

| state | H2 days | **H2 $/day** | H2 t | H2 green days | H2 total $ | H1 $/day | VAL $/day |
|---|---|---|---|---|---|---|---|
| **`spy_r5>0`** | 64 | **+32.4** | +0.71 | 48.4 % | **+$2,071** | −85.5 | **+66.1** |
| `spy_r5` T3 (most up) | 41 | +6.7 | +0.13 | 41.5 % | +$273 | −98.8 | +69.1 |
| `qqq_r5>0` | 59 | −68.1 | −1.50 | 37.3 % | −$4,016 | −139.1 | +42.4 |
| `spy_r5>0 AND qqq_r5>0` | 49 | −6.1 | −0.13 | 42.9 % | −$301 | −133.2 | +63.8 |
| `spy_rng5_atr` T3 (wild) | 44 | −30.1 | −0.50 | 36.4 % | −$1,326 | −57.7 | +9.6 |
| breadth T3 | VOID (§F14-d6) | | | | | | |

`spy_r5 > 0` makes H2-2025 positive (+$2,071 over the half) and its VAL read is non-negative
(+$66.1/day, +$3,439), so by the pre-committed rule **the frame answers YES — at a t of +0.71.**

**And the mirror question `FRAMES.md` asked — is there a state under which H1-2025 is positive? NO**
(`supp14.log` S3, 8 declared states):

| state | H1 days | H1 $/day | H1 t | H1 green days | H1 total $ |
|---|---|---|---|---|---|
| `spy_gap>0 AND spy_r5>0` | 30 | **−52.4** | −0.73 | 46.7 % | −$1,573 |
| `spy_rng5_atr` T3 (wild) | 37 | −57.7 | −0.81 | 43.2 % | −$2,136 |
| `spy_gap>0` | 63 | −81.5 | −1.61 | 41.3 % | −$5,132 |
| `spy_r5>0` | 60 | −85.5 | −1.85 | 41.7 % | −$5,130 |
| `spy_r5` T3 | 40 | −98.8 | −1.62 | 35.0 % | −$3,953 |
| `spy_rng5_atr` T1 (quiet) | 50 | −122.0 | −2.52 | 36.0 % | −$6,102 |
| `spy_r5>0 AND qqq_r5>0` | 45 | −133.2 | −2.52 | 33.3 % | −$5,992 |
| `qqq_r5>0` | 55 | −139.1 | −2.95 | 30.9 % | −$7,652 |

**Every 09:35 state leaves H1-2025 negative, the best at −$52 a day.** The programme's failure is
H1-shaped and no day-level state known at 09:35 addresses it. The day gate is a **two-era** object.

## 2.3 F14 verdict

Pre-committed selector (day mean positive in H1, H2 and VAL; day t >= 2 on TRAIN; >= 50 % green weeks
on both splits at >= 10 tr/wk): **1 of 15 — and it is `F14-d9`, the oracle bound.** No causal day gate
passes. Nulls: 14 cell × split, **all 14 inside** the count-matched band. The best causal cell in the
pass is `F14-d8` (the day gate × F13's composite ranking) at VAL **+$4,534, 60.9 % green, day t
+1.85** — and its TRAIN is −$2,088 with H1 −$62.3/day, so it fails on the half that has failed every
object in nine passes.

---

# F15 — WHO IS BREAKING OUT (12 cells declared; 2 VOID on coverage, 2 demoted to diagnostics)

## 3.1 Availability audit — and what it decided before anything was scored

| field | coverage | miss on winners | miss on losers | gap | verdict |
|---|---|---|---|---|---|
| `days_to_cover` (FINRA) | **98.8 %** | 1.0 % | 1.3 % | 0.3 pp | ok |
| `si_ratio` (SI ÷ adv20) | 98.8 % | 1.0 % | 1.3 % | 0.3 pp | ok |
| `shares_out` (EDGAR) | **39.9 %** | 60.3 % | 59.9 % | 0.5 pp | **-> F15-b1 / b2 VOID** (PREREG < 50 % rule) |
| `venue` (Databento PIT) | 100.0 % | 0.0 % | 0.0 % | 0.0 | ok |
| `n_articles` (ORB news) | **3.3 %** | 96.7 % | 96.7 % | 0.0 | **-> F15-e1 / e2 are DIAGNOSTICS** |
| `asset_class` identified | 99.5 % | — | — | — | ok |

**Short interest is now in the repo and it is point-in-time.** `fetch_finra.py` pulled 923,387 FINRA
consolidated-short-interest rows (settlement 2024-11-15 -> 2026-08-31, 29,809 symbols) keyed on
**`usable_from` = settlementDate + 13 calendar days**; the median report age at a signal is **8
days**, p95 14. The field that was "dropped on availability" twice is fetched, joined and measured.

## 3.2 The cells

| cell | TRAIN n · /wk · gross · net · grn % · **$** | VAL n · /wk · gross · net · grn % · **$** | H1 / H2 / VAL | era-consistent |
|---|---|---|---|---|---|
| base (B2) | 1,622 · 30.6 · −0.039 · −0.107 · 32.1 · −17,346 | 706 · 30.7 · +0.083 · +0.013 · 43.5 · +893 | −0.077/−0.003/+0.083 | no |
| **F15-a1** `daysToCover >= 1.49` (med) | 1,085 · 20.5 · −0.054 · −0.126 · 30.2 · −13,661 | 585 · 25.4 · −0.031 · −0.105 · 26.1 · −6,135 | −0.095/−0.018/−0.031 | no |
| **F15-a2** `daysToCover >= 3.27` (p75) | 733 · 13.8 · +0.006 · −0.069 · 45.3 · −5,023 | 449 · 19.5 · +0.007 · −0.067 · 43.5 · −3,026 | **−0.084**/+0.078/+0.007 | no |
| **F15-a3** `SI/adv20 >= 2.99` (p75) | 682 · 12.9 · −0.015 · −0.091 · 43.4 · −6,200 | 429 · 18.7 · −0.002 · −0.077 · 34.8 · −3,300 | −0.074/+0.034/−0.002 | no |
| **F15-b1 / b2** float turnover / low float | **VOID** — `shares_out` coverage 39.9 % < 50 % | | | |
| **F15-c1** common stock only | 1,276 · 24.1 · −0.053 · −0.122 · 28.3 · −15,628 | 650 · 28.3 · −0.055 · −0.127 · 30.4 · **−8,234** | **−0.060/−0.046/−0.055** | **no (era-consistently NEGATIVE)** |
| **F15-c2** leveraged wrapper only | 1,083 · 20.4 · −0.005 · −0.069 · 41.5 · −7,434 | 594 · 25.8 · **+0.135** · +0.067 · 43.5 · **+3,958** | −0.117/+0.091/+0.135 | no |
| **F15-c3** anchor cohort >= 2 | 463 · **8.7** · +0.126 · +0.068 · **52.8** · **+3,167** | 295 · 12.8 · **+0.380** · +0.318 · **60.9** · **+9,389** | **+0.014/+0.227/+0.380** | **YES — see §3.3** |
| **F15-d1** venue NASDAQ | 1,346 · 25.4 · −0.032 · −0.101 · 35.8 · −13,566 | 614 · 26.7 · +0.009 · −0.064 · 39.1 · −3,952 | −0.068/+0.001/+0.009 | no |
| **F15-d2** venue NYSE/ARCA/AMEX | 1,062 · 20.0 · −0.024 · −0.090 · 35.8 · −9,592 | 600 · 26.1 · +0.073 · +0.006 · 47.8 · +374 | −0.069/+0.014/+0.073 | no |
| **F15-e1** premarket news present [diag] | 69 · 1.3 · −0.016 · −0.085 · 30.2 · −588 | 36 · 1.6 · +0.034 · −0.028 · 39.1 · −102 | +0.181/−0.128/+0.034 | no |
| **F15-e2** news same morning [diag] | 58 · 1.1 · −0.074 · −0.144 · 26.4 · −838 | 31 · 1.3 · +0.137 · +0.080 · 39.1 · +248 | +0.173/−0.186/+0.137 | no |

Two readings that do not depend on any survivor:

* **Short interest carries nothing on this book.** Both SI ladders are H1-2025-negative; the *higher*
  SI rung is the *less bad* one (a1 −$13,661 vs a2 −$5,023) purely by trading less, and both are
  net-negative on both splits. The forced-buying mechanism is measurable here and it is absent.
* **"Is the book edgeless everywhere, or edgeless on a mixed universe?"** — the mixture matters and
  it points the wrong way for the obvious answer. **Common stock alone is negative in H1, H2 AND
  VAL** (−0.060 / −0.046 / −0.055, −$15,628 / −$8,234); the book's entire positive VAL is its
  **leveraged wrappers** (+0.135, +$3,958) — whose own H1 is −0.117. The universe is not hiding a
  clean equity book inside a dirty wrapper book; both halves lose in H1-2025.

## 3.3 The causality trace that killed F15-c3 — the pass's real result

`anchor_cohort >= 2` was the first cell in the programme to be **same-signed positive across three
eras with positive dollars on both splits and >= 50 % green weeks on both** (VAL clustered t +3.30).
It is also, as scored, **not computable at the decision bar**: it counts every same-anchor candidate
of the WHOLE session, including siblings that break *after* our entry minute.

| decomposition (`supp15.log`) | TRAIN /wk · gross · net · grn % · **$** | VAL /wk · gross · net · grn % · **$** | H1/H2/VAL |
|---|---|---|---|
| **F15-c3 as declared** (`anchor_cohort >= 2`) | 8.7 · +0.126 · +0.068 · 52.8 · **+3,167** | 12.8 · +0.380 · +0.318 · 60.9 · **+9,389** | +0.014/+0.227/+0.380 |
| **the sibling broke LATER** (30.8 % of rows) | 3.1 · **+0.457** · +0.398 · 60.4 · **+6,484** | 5.4 · **+0.839** · +0.774 · **82.6** · **+9,677** | +0.346/+0.542/+0.839 |
| **F15-c3c CAUSAL** (`cohort_causal >= 2`, sibling ALREADY broke) | 6.4 · **−0.013** · −0.068 · 39.6 · **−2,284** | 9.7 · **+0.154** · +0.094 · 52.2 · **+2,099** | **−0.120**/+0.084/+0.154 |
| cohort == 1 (alone all session) | 27.3 · −0.083 · −0.154 · 26.4 · −22,342 | 29.7 · +0.030 · −0.043 · 39.1 · −2,908 | −0.111/−0.059/+0.030 |

**The entire edge is in the look-ahead half.** The 30.8 % of rows whose sibling had not yet broken
carry +0.457 / +0.839 R at t +3.54 / +6.43 — that is not an instrument property, it is *the day's own
continuation* read backwards: a complex that keeps producing breakouts all session is a complex that
is running, and "it will produce another one later" is the trade's own outcome wearing a hat. **This
is `hod_frames` §2.3 for the third time, now on an instrument field.**

The causal version fails on every rail: H1 −0.120 (not era-consistent), 6.4 / 9.7 trades a week
(under the floor on both splits), green weeks **inside** its count-matched null on both splits
(39.6 vs 40.6 [34.0, 47.2]; 52.2 vs 53.5 [43.5, 65.2]), net **−0.068 / +0.094 against an MDE of
0.209 / 0.242**, and the usual lottery concentration (best week = 77 % of the VAL total; net +0.094
-> ex-top-5 % **−0.014**). Its stock rows and its wrapper rows each fail separately, and crossing it
with the SPY 09:35 gate does not save it (H1 +0.023 / H2 −0.040).

**F15-c3 is DEAD, and it was killed inside the pass by its own causality trace — not by the owner.**

---

## BOTH BARS, the nulls, the MDE, the multiplicity

**Claim bar G1 — 0 of 38.** No cell has TRAIN net R > 0 with iid *and* clustered t >= 2 at >= 10
trades/week. The largest favourable TRAIN clustered t on any causal cell is **+0.73** (`F15-c3`, and
that is the look-ahead cell). G2 was never evaluated; **TEST was never opened.**

**Live-exploration bar — 0 cells.** The only cells with positive dollars on BOTH splits are `F15-c3`
(look-ahead, refuted in §3.3) and, among the bounds, the oracles.

**Nulls — 62 cell × split bands: 56 inside, 5 below, 1 ABOVE.** The single ABOVE is `F13-rs2` on
TRAIN dollars, and it loses $3,870 on VAL. The four oracle cells read **below** their own nulls on
TRAIN, which is itself informative: even a perfect ranker's green-week share is lower than random
allocation of its own P&L — the oracle's money is concentrated in weeks too.

**MDE (80 % power, per trade, net) against the 0.061–0.065 R break-even**: 0.088 / 0.132 R on the B2
book; 0.087–0.098 / 0.132–0.140 on the F13 cells; 0.095–0.126 / 0.132–0.162 on the F15 cells that
trade >= 10/wk; **0.184 / 0.219 on F15-c3 and 0.209 / 0.242 on its causal version** — not powered,
and its frequency is itself the verdict. `F15-e1/e2` at 1.1–1.6 tr/wk have MDE 0.39–0.64 R and say
nothing. **Day-level MDE: $68 / $94 a day on the base, $92 / $130 on the D2 gate.**

**Multiplicity.** 38 declared decision cells (16 F13 + 10 F14 + 12 F15), realised as 15 + 15 + 12
scored book rows after the voids; plus 4 reproduction rows, 1 slot-machine parity assertion, 1
availability audit, 1 structural-count table, 2 supplementary blocks (`supp14` S1–S4, `supp15` C1–C5)
that carry no decision, and one declared 6-state H2 search plus its 8-state H1 mirror.
**Programme cumulative: 920 + 38 = 958.** Expected largest |t| under a pure null over 38 × 2 ≈ 3.0;
the largest favourable TRAIN clustered t on a causal cell is +0.73.

## Known deviations, stated rather than buried

1. **`hod_frames3/nbbo3.csv` is not merged** (see §0). It is a strictly *better* cost measurement on
   466 quote-minutes and is excluded only so the reproduction gate is exact. No cell cleared, so the
   arm was never needed; any future pass that keeps it must re-state R2b as 1,605 / −$18,668.
2. **F13's ranking is within-minute only.** A wider decision window is the form `FRAMES.md` F13
   described, and it is **unobtainable in this dataset** — every signal is priced at its own next
   open. Building it needs a bar walk that re-prices a deferred entry: a new population, not a new
   cell. Named as the frame's live remainder, not as a result.
3. **F14-d6 is void by construction, not by data** — the book's earliest possible entry minute is
   09:37, so no candidate can have broken by 09:35. The breadth idea survives at a later clock and is
   reported as a supplementary with no decision (`supp14.log` S2: at 10:00 the broad tercile is VAL
   +$103/day and TRAIN −$19/day with H1 −$54.6).
4. **F15-b1/b2 are void on coverage** (EDGAR shares outstanding, 39.9 %) and **F15-e1/e2 are
   diagnostics on coverage** (the ORB news file covers 3.3 % of this universe). Float and news are
   therefore **not tested** on this book, not refuted. A float join with real coverage needs a
   shares-outstanding source for the whole point-in-time universe.
5. **The oracle cells are bounds and are excluded from both bars by construction** — they select on
   the realised outcome and are reported only to size what a ranking could be worth.

## VERDICT — **STAY DRY.** No `HodBreakParams` change, no `run_book` change, no new admission field.

`config.yaml hod_break` stays exactly as the owner set it (`enabled: true, dry_run: true`);
`trading.enabled` and `orb.yaml` untouched. There is no SHIP-TO-DRY diff. For the record, the diffs
that would have been written: **F13** -> the sort key in `trading/hod_break.py::run_book` (and the
engine's per-minute candidate ordering in `hod_break_engine`) changing from `(entry_m, symbol)` to
`(entry_m, -score, symbol)`; **F14** -> a `spy_r5_pct` day gate evaluated once at 09:35 in the
engine's session setup; **F15** -> an `anchor_cohort` admission plus a FINRA nightly job. None is
built and, on this evidence, none should be.

**The next three frames are F16, F17 and F18, appended to `FRAMES.md`.**
