# frames9 — PREREG (committed BEFORE any cell is scored)

Pass 9 of the frame programme. Frames **F28** (HOD's detector on a positive pond), **F29** (the pond
map), **F30** (the pooled ramp statistic — specification only). Queue as written by pass 8
(`hod_frames/FRAMES.md` §"THE QUEUE AFTER PASS 8").

Everything below is fixed before a single R value of a new cell is looked at. Population sizes (n)
were measured first and are quoted here, because a frame cannot be scoped without them; **no
outcome, no R, no P&L of any new cell has been read.**

---

## 0. Rails (identical to passes 6–8, restated so the run can abort on them)

| rail | this pass |
|---|---|
| **Reproduction gates** | (a) **B2** = `common6.base_book()` → TRAIN 1,622 / 30.6 wk / gross −0.039 / net −0.107 / **−$17,346**, VAL 706 / 30.7 / +0.083 / +0.013 / **+$893**; (b) **ORB** `research/orb_gates2/book_G3_meas.csv` → 282 TRAIN / 177 VAL picks; (c) **BF** `research/bf_frequency/runs/P1.csv` → **56 trades / $139,113.67**. All three asserted in code; the run aborts otherwise. (d) an in-run **walker parity gate**: the new walk's `rr_X0`/`rr_G3` on every signal it shares with `frames8/w_sig.csv` must agree to ≤ 1e-9. |
| **TEST** | sealed (`FREEZE.md`). `day >= 2026-06-01` is dropped at load and never scored. |
| **Both TRAIN halves** | H1 = day < 2025-07-01, H2 = day >= 2025-07-01, printed beside VAL on every cell. |
| **Day-clustered SE** | `common4.clustered_t` (one cluster per session) on every cell and every paired difference. |
| **Count-matched null** | `score.null_band` — 2,000 draws, each cell's own P&L shuffled across its own weeks with the weekly pick COUNT held fixed; green weeks reported against its own p95. |
| **Booked cost per cell** | re-measured from that cell's own exit mix (`g8.exit_ratio`), never carried. The **imputed share** of the NBBO is printed per rung — the relaxed floors are below the range the dedicated NBBO fetch covered and are expected to be ~90–99 % imputed. |
| **Pass-6 control rule** | a control is never drawn on a name that itself produced an admitted signal that session; controls are priced at the SIGNAL's own minute (arm b / arm u) so no control is drawn *before* a signal on a day known to signal. |
| **Availability audit** | per arm, per cell: the share of signals with ≥ 1 priceable control. A cell whose coverage is < 80 % is demoted to a diagnostic BEFORE its number is read. |
| **Tail** | ex-top-5 % (rank trim) beside every headline book. |
| **MDE** | two-sided 80 %-power minimum detectable effect on every rejection. |
| **Multiplicity** | declared cells counted below; programme total updated in REPORT.md. |
| **Node** | one python process, `nice -n 10`, `ulimit -v 3000000`, the walk checkpointed per session; `cache.db`, `bars_sip.db`, `trades.db`, the Databento stores and `daily_bars` opened READ-ONLY. `config.yaml`, `orb.yaml`, the systemd unit, the crons and every order are untouched. |

---

# F28 — HOD'S DETECTOR ON A POSITIVE POND

## 1. The object being transplanted

The one significant object in 1,068 HOD cells is **name-day SELECTION**: HOD's detector's name-days
beat a matched non-signal name of the same session by **+0.183 / +0.245 R** (day-clustered
t +2.75 / +4.24) under the bare-stop exit, **+0.123 / +0.240** under the shipped +2 R cap
(`frames8/REPORT.md` §1.3). The detector fishes in a **−0.16 R** pond, so the book nets ≈ 0.

## 2. THE PREDICTION AND THE FALSIFIER (pre-committed)

* **If the selection value is a property of the DETECTOR**, it transfers: on ORB's and BF's
  universes the detector's picks beat those ponds' matched non-signal names **by a similar margin**
  (declared as: the margin's point estimate is within ±0.08 R of HOD's own +0.183 (G3) / +0.123 (X0)
  on TRAIN and same-signed positive on VAL), and — because those ponds are the positive ones — the
  absolute book turns positive.
* **If it is a property of HOD's POND** (the −0.16 R non-movers it is measured against), it does not
  transfer: the margin **collapses toward the pond's own bound**, i.e. the measured margin on a pond
  whose bound is B is approximately `(HOD margin) − (B − (−0.16))`.
* **FALSIFIER**: the frame is dead if, on every pond and rung, the selection margin fails to reach
  +0.10 R on TRAIN **or** the absolute book fails the live-exploration bar. The bar is unchanged:
  **positive weekly $ AND green weeks ≥ 50 % on BOTH splits at ≥ 10 trades/week, day-clustered
  t ≥ +2, TRAIN halves same-signed.**

**Declared in advance — the measurement is NOT inherited from pass 7.** Pass 7's ponds
(+0.070 / +0.024 ORB, +0.053 / +0.010 BF) were measured at **those books' own clocks** (ORB 09:36,
BF's flag-break minute), with **those books' own stop widths** (ORB's `range_size_pct`, median
≈ 5.5 %) and geometries. This frame holds the **clock, the stop width and the bracket at HOD's**
and varies **only the name population**, exactly as the queue specifies ("the pond's own universe
bound — random name, HOD clock, HOD bracket"). If ORB's pond measured HOD's way is NOT positive,
that is itself the answer and it hands F29 its subject; it is recorded as such, not as a failure of
the frame.

## 3. The ponds (membership rules, fixed here)

Membership is evaluated on the point-in-time day panel `research/bf_zero/universe.csv` — the same
panel every control in six passes has been drawn from.

| pond | rule | symbol-days (2025-01 → 2026-09) |
|---|---|---|
| **ORB** | symbol ∈ the ORB entered-inclusive candidate names (`analysis_results/orb_features_20260918_2052.csv`, 13,163 symbol-days / **2,461 names** — the 13,033 of the queue plus the sessions since) **AND** the session passes ORB's live universe screen (`study_orb_broad.py`: prev close > 0, **gap ≥ 5 %**, **prev-day volume ≥ 500,000**, **open ∈ [$3, $30]**) | **15,718** (2,301 names, 419 sessions) |
| **BF** | symbol ∈ the names detected in `data/bull_flag_cache_causal_full_20260905.csv` (593 names) **AND** `trading/bf_universe_filter.is_bf_eligible` (→ **465 names**) **AND** open ∈ **[$2, $30]** | **49,781** (447 names, 419 sessions) |
| **UNION** | ORB ∪ BF, as a SIGNAL set. A signal in both ponds is counted once and is assigned ORB's control pool. | **64,402** (intersection 1,097) |

**The band overlap, and what it does to n.** HOD's `sigset5` floor is `next_open >= $20` and ORB's
band caps the OPEN at $30, so at the shipped floor the two rules can only meet in **$20–$30**. The
measured effect on n (counted before this file was committed, no outcome read):

| rung | HOD admitted (TRAIN+VAL) | ORB pond | BF pond | UNION | median price | imputed NBBO |
|---|---|---|---|---|---|---|
| **$20** (shipped) | 7,027 | **317** (195/122) | **136** (90/46) | **429** (268/161) | $25.0 | 59 / 91 / 67 % |
| **$10** | 11,045 | **736** (437/299) | **754** (481/273) | **1,431** (882/549) | $15.8 | 82 / 98 / 90 % |
| **$5** | 15,897 | **1,264** (808/456) | **1,663** (1,106/557) | **2,782** (1,809/973) | $10.3 | 90 / 99 / 95 % |

So the overlap costs **95.5 %** of HOD's admitted signals at the shipped floor, and the relaxed rungs
are the only ones with a book-sized rate — bought at a cost that is almost entirely **imputed**, which
is why the imputed share is printed beside every rung and why the relaxed rungs can only ever be a
diagnostic for the cost, never a ship number on their own.

## 4. Construction (fixed)

* **Signals.** `common4.load_breaks4` → `admit(all True)` (the keep-scanning first qualifying break
  per symbol-day) → `common5.sigset5(min_price = rung)` — i.e. HOD's **exact** B2 cascade with only
  the price floor moved. The detector itself (≥ 5 % above the 09:30 open, `rv_profile >= 1`, five
  closed bars within 4 % of the running HOD, no consolidation-tightness gate, spread ≤ 100 bps and
  ≤ 15 % of R, entry ≤ 14:01, obtainable under the 60-bps cap) is untouched. Pond restriction is
  applied to the symbol-day, which commutes with `admit`.
* **Geometry.** `g8.geoms` — **X0** (the shipped +2 R cap) and **G3** (the bare stop ridden to 15:55,
  F25's best). Entry at the OPEN of the bar after the break bar; stop fills at
  `min(stop, that bar's open) × (1 − 10 bps)`; flat at 15:55 at that bar's open.
* **Arm b — the matched non-signal control.** For every pond signal, the **12 nearest** non-signal
  names **of the same session drawn from that signal's own pond**, distance
  `|dlog(prev_close)| + |dlog(adv20)|`, exact match on asset class (`trading/orb_asset_class`),
  priced at the SIGNAL's own minute with the SIGNAL's own `r_pct` applied to the control bar's open.
* **Arm u — the pond's universe bound.** **12 random** names of the same session from the same pond,
  same clock, same `r_pct`. This is the "random name, HOD clock, HOD bracket" object the queue asks
  for.
* **Exclusions.** A control is never the signal's own symbol and never a name that produced an
  admitted signal (at ANY rung) that session.
* **The book.** `common4.book_ranked(s, 12, 4)` — `trading.hod_break.run_book`'s first-come 12/day,
  4 concurrent, causal slot freeing — run on the pond-restricted signal set with that geometry's own
  `exit_m`. $100 risk. Cost = entry half-spread + exit half-spread × the leg ratio (`g8.exit_ratio`).
* **Controls are GROSS** (F24's rule: a control's own spread is a different instrument's).

## 5. The declared cells (12)

| # | cell | what is scored |
|---|---|---|
| **C0** | **the OVERLAP diagnostic — reported FIRST, and it can kill the frame on its own** | the share of the pond signals that fall on a (day, symbol) the pond's own book ALREADY takes (ORB `book_G3_meas` picks; BF P1's 56 trades). If the HOD rule merely re-labels trades ORB's composite already takes, the frame is a tautology and dies here. |
| **C1–C3** | **ORB pond** × rung {$20, $10, $5} | pond bound (arm u), matched control (arm b), signal, **selection margin sig−b** (paired, day-clustered), and the absolute book under **G3** and **X0**: gross, booked cost, net, green weeks (+ null p95), weekly $, trades/wk, H1/H2/VAL, clustered t, ex-top-5 %, MDE. |
| **C4–C6** | **BF pond** × rung {$20, $10, $5} | as C1–C3. |
| **C7–C9** | **UNION** × rung {$20, $10, $5} | as C1–C3. |
| **C10** | **HOD's own baseline at the same rungs** (no pond restriction) | the comparison arm: does moving the price floor alone do what the pond does? |
| **C11** | **the cost rung table** | booked cost and imputed share per rung per pond (already scoped above; scored with the cells). |
| **C12** | **the re-measured pond bound vs pass 7's** | arm u at HOD's clock/bracket against pass 7's +0.070 / +0.053 at the books' own clocks — the number F29 needs. |

Each of C1–C9 is reported under **two geometries** (G3, X0), so the scored-object count is
9 × 2 × 2 splits = 36 book cells + 9 × 2 × 2 margin objects = **72 scored objects against 12 declared
cells**; counted, not selected after the fact.

## 6. Verdict rule (pre-committed)

**SHIP-TO-DRY** only if some (pond, rung, geometry) clears the live-exploration bar of §2 **and** C0
shows the picks are not already the pond book's own. Anything else is **STAY-DRY**, and the frame is
recorded DEAD with the margin-collapse arithmetic of §2 as the reading.

---

# F29 — THE POND MAP: which attribute owns the +0.23 R

Population: HOD's OWN booked B2 trades and their matched arm-b controls (`frames8/w_sig.csv`,
`frames8/w_b.csv` — 2,328 booked trades, 51,051 control brackets, already walked). The object is the
**paired selection margin** `m = rr_sig − mean(rr_b over that trade's own controls)` under **G3**
(where it is largest, +0.183 / +0.245) and **X0** (the shipped cap).

Each split below is a **declared partition of the margin**. For every level: n, the margin, H1, H2,
VAL, day-clustered t, and the MDE. Nothing is selected after the fact; all ten are reported whatever
they read.

| # | split | levels | why (the attribute has to be nameable and findable WITHOUT the detector) |
|---|---|---|---|
| **S1** | asset class | wrapper / common stock | F22: the bare bracket is −0.131 on wrappers vs −0.030 on commons; if the margin is a wrapper effect it is a borrow-and-decay story, not a selection story |
| **S2** | price band | next_open < $30 / $30–60 / ≥ $60 | the band is the only thing the transplant's overlap is limited by |
| **S3** | ADV$ band | < $25M / $25–150M / ≥ $150M | F22's least-negative region is the liquid end |
| **S4** | gap at the open | gap ≥ +2 % / −2..+2 % / ≤ −2 % | is the detector buying a gap-up (ORB's population) or an intraday mover? |
| **S5** | `rv_profile` | ≥ 5 / < 5 | the rv tail was +0.177 in `hod_break` and negative in `hod_fresh` |
| **S6** | sibling/anchor moved (CAUSAL) | another name sharing the `orb_asset_class.underlying_anchor` produced an admitted break EARLIER the same session / not | ORB's own catalyst-veto mechanism; the causal form only (strictly earlier `entry_m`) |
| **S7** | listing age | < 60 prior sessions in the panel / ≥ 60 | the F19 intersection split (new listings were the H1-positive / VAL-negative signature) |
| **S8** | entry-minute band | 09:37–10:30 / 10:30–11:30 / 11:30–13:00 / 13:00–14:01 | the clock is the other half of pass 7's population-vs-clock ambiguity |
| **S9** | `dist_open_pct` | 5–10 / 10–20 / ≥ 20 | how far above the open the detector is buying |
| **S10** | `dollar_frac` terciles | cum $ volume ÷ 20-day ADV$ at the signal | F10's quarantined footprint field, now as a split of the MARGIN rather than of the return |

**The deliverable is one sentence**: *"the detector's +0.23 R is the market paying for ___"* — and the
pre-committed standard for filling that blank is a level that carries the margin **in H1, H2 and VAL
with the same sign** and whose complement carries materially less. If no split does that, the honest
sentence is that the margin is not attributable to any of the ten attributes, and that is reported.

---

# F30 — THE POOLED RAMP STATISTIC (specification only)

No cells, no book, no code outside `frames9/`. Deliverable: the estimator, the calendar arithmetic
(pooled vs per book at 80 % power for a 0.2 R effect), how a pooled ADVANCE/DEMOTE would read, and
the exact changes that `trading/ramp_bt_band.py`, `scripts/orb_ramp_check.py` and
`scripts/bf_ramp_check.py` would need. **Pre-committed constraints**: (a) the pooled gate may never
advance a book whose OWN realized stage P&L is negative (`project_orb_ramp_above_water_rule`);
(b) the replay is on realized P&L only — no backtest slice enters a ramp decision. **Specify, do not
build.**

---

## Cell count

F28 12 + F29 10 + F30 0 = **22 declared cells**. Programme total **1,068 + 22 = 1,090**.
