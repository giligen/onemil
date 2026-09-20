# frames7 — F24 the PLACEBO on the LIVE books · F23 the MIRROR · F22 the BARE GEOMETRY — REPORT (2026-09-20)

Pass 7 of the frame programme, and the first one that is **about the books that are actually
running**. Cells exactly as declared in `PREREG.md`, committed (`e6e5fad`) before any cell was
scored. Artifacts: `c7.py` (the three walkers, the match pool, the stats) · `f24.py`
→ `book_{orb,bf}.csv`, `pool{b,u}_{orb,bf}.csv`, `p24.csv` (19,789 walked trades), `cells24.csv`,
`f24_score.log` · `f23.py` → `p23.csv` (204,201 walked shorts), `cells23.csv`, `f23_score.log` ·
`f22.py` → `cells22.csv`, `f22_score.log`. One python process at a time, `nice -n 10`,
`ulimit -v 3000000`; `cache.db`, `bars_sip.db`, the Databento stores and `daily_bars` opened
**read-only**. No config, `orb.yaml`, systemd unit, cron, order or cache was written; the BF and ORB
live configs were never opened for writing and Monday's boot is untouched. **TEST was never opened**
(`FREEZE.md`).

---

## 0. THE SENTENCE THIS PASS WAS REQUIRED TO PRINT FIRST

**Both live books read NO DIFFERENT under the pre-committed rule — and both read it from the
OPPOSITE side to HOD-break.** Every one of the **12 arm × split paired differences is POSITIVE**, the
signal sits at the **92.5th–100th percentile** of its own 200-draw control band in **12 of 12** cells
(≥ 99.5th in 7 of them), and the signal beats every control arm in **18 of 18** era × arm
comparisons. What is missing is not the sign — it is the **t**: no arm reaches a day-clustered
t ≥ 2 on both splits, and every point estimate is at or below its own 80 %-power MDE.

| book | signal gross R (TRAIN / VAL) | vs matched non-signal name, same clock | vs the SAME name-day, a later minute | vs the universe bound |
|---|---|---|---|---|
| **ORB B+ / G3** | **+0.216 / +0.470** | +0.153 (t +0.93) / **+0.436** (t +1.72) | +0.198 (t +1.64) / +0.324 (t +1.45) | +0.187 (t +1.06) / +0.479 (t +1.89) |
| **BF P1** | **+0.691 / +0.816** | +0.507 (t +1.34) / +0.349 (t +0.58) | +0.646 (t +1.57) / +0.622 (t +1.10) | +0.563 (t +1.55) / +0.482 (t +0.88) |

**The reading that matters for the scaling plan is the one HOD-break did not give**: on both live
books the **MINUTE** is where the value sits, not only the name-day. On HOD-break the break minute
was worth **−0.063 / +0.026 R** against a later minute of the same name-day; on ORB it is worth
**+0.198 / +0.324 R** and on BF **+0.646 / +0.622 R**. Formally NO DIFFERENT — directionally the
opposite book.

**Verdict, all three frames: NO SHIP, NO CONFIG CHANGE, NOTHING RE-OPENED.** 0 of 38 declared cells
clears its bar. ORB and BF are unchanged for Monday; `hod_break` stays `enabled: true, dry_run: true`.

---

## 0b. Reproduction gates and the two obtainability repairs

| id | gate | result |
|---|---|---|
| **G-ORB** | `book_G3_meas.csv` picks per split | **282 TRAIN / 177 VAL — MATCH** (asserted in code), sized $ +6,515.08 / +4,287.21 |
| **G-BF** | `runs/P1.csv` | **56 trades / $139,113.67 — MATCH to the cent** (asserted in code) |
| **G-HOD** | `hod_frames6/{book6,pb6,pd6}.csv` | pass 6's asserted `B2` + R3 walker parity (2.22e-16) inherited unchanged |

Two construction defects were found **by the declared rails, before the numbers they produced were
allowed into a cell**, and both are recorded here in full because each is a class this programme has
been burned by:

1. **BF's signal entry was an unobtainable fill** (rail 1b). The booked BF fill minute is the minute
   in which price ran UP THROUGH the breakout level, so entering at **that bar's open** is entering
   before the run at a price the engine could never have had. Measured size of the bias:
   **sig +1.296 / +1.754 R → +0.691 / +0.816 R** once entry moves to the next bar's open (the
   programme's convention, and the clock every control arm uses). Withdrawn version kept as
   `p24_bfvoid.csv`. Under the unobtainable fill BF's arm b/a′/u paired t were +2.88/+3.37/+3.43
   (TRAIN) and +2.66/+3.04/+2.87 (VAL) — i.e. **the defect alone would have produced a "BETTER"
   verdict on a live book.** ORB was never exposed: its signal entry was the bar AFTER the bar that
   reached the pick's own fill level from the start.
2. **The availability rail fired on BF and was obeyed.** With a strict same-minute lookup BF's arms
   covered **76.5 %** of TRAIN trades — below the declared 80 % floor, which would have demoted the
   whole BF frame to a diagnostic. The cause is a missing 1-minute bar (no print in that minute), so
   the walker now takes the **first EXISTING bar at or after the target minute, within 5 minutes**,
   which is what the engine would actually get. Applied to **both** books and every arm; coverage is
   now 90.0–100 %. The strict-minute ORB run is kept as `p24_exactmin.csv` and moves ORB's arm b by
   0.001 / 0.010 R — the repair is an availability fix, not a result.

---

# F24 — THE PLACEBO ON ORB AND BF-P1  (16 cells)

## 1.1 Coverage (the availability rail, applied before any number was read)

| book | arm | TRAIN coverage | VAL coverage | controls / trade (median) |
|---|---|---|---|---|
| ORB | sig | 204/210 = **97.1 %** | 136/140 = **97.1 %** | 1 |
| ORB | b (matched non-signal, same clock) | 189/210 = 90.0 % | 133/140 = 95.0 % | 24 / 25 |
| ORB | a′ (same name-day, later minute) | 97.1 % | 97.1 % | 10 |
| ORB | u (random universe name, same clock) | 90.0 % | 95.0 % | 19 / 20 |
| BF | sig | 32/34 = 94.1 % | 15/15 = **100 %** | 1 |
| BF | b | 31/34 = 91.2 % | 14/15 = 93.3 % | 21 / 18 |
| BF | a′ | 94.1 % | 100 % | 10 |
| BF | u | 91.2 % | 93.3 % | 19 / 22 |

All eight arms clear the declared 80 % floor. 19,789 walked trades (ORB 17,389 / BF 2,400).

**Walker parity (diagnostic, not a gate).** ORB: n = 340, corr **+0.683**, walker +0.317 vs the
book's own R +0.234. BF: n = 50, corr **+0.677**, walker +0.660 vs the book's own +0.777. The walker
is deliberately *not* the book — it enters at a bar open on both sides of every comparison, which is
why the placebo statistic is walker-internal and the book's own $ is the reproduction gate.

## 1.2 THE DECOMPOSITION — the finding of this pass

Read down each column exactly as `hod_frames6` §1.5 asked for it:

| object | ORB TRAIN | ORB VAL | BF TRAIN | BF VAL | what it holds fixed | HOD-break, for contrast |
|---|---|---|---|---|---|---|
| **the universe bound** — a random universe name at the same clock | **+0.0696** | **+0.0237** | **+0.0527** | **+0.0101** | the clock only | **−0.058 / −0.042** |
| **matched non-signal name**, same clock (price, ADV20, class, ±5 pp gap for ORB) | **+0.1094** | **+0.0662** | **+0.1467** | **+0.1339** | clock + price + ADV20 + class | **−0.161 / −0.160** |
| **the same name-day**, a random LATER minute (causal — after the signal) | **+0.0184** | **+0.1471** | **+0.0453** | **+0.1943** | day + name | **+0.014 / +0.073** |
| **the BOOKED signal minute** | **+0.2161** | **+0.4695** | **+0.6914** | **+0.8163** | nothing — the book | **−0.039 / +0.083** |

Three structural facts come out of that table, and none of them was knowable before this pass:

1. **Neither live book fishes in a negative pond.** The universe bound under ORB's own geometry is
   **+0.070 / +0.024 R** and under BF's **+0.053 / +0.010 R** — *positive*, where HOD-break's is
   **−0.16 R at its clock**. Pass 6's headline ("a real detector whose entire output is consumed by
   the negative baseline it fishes in") is a **property of the HOD-break bracket, not of this tape**.
   The difference is geometry, not skill: ORB rides a static lock with no target and BF an R-trail
   with no target, so a winner is unbounded; HOD-break caps every winner at +2 R while the stop is
   the same 1 R. F22 §3 measures that directly.
2. **Name-day selection is worth much less than the programme assumed, on both books.** Matched
   non-signal minus the universe bound is **+0.040 / +0.043 R** (ORB) and **+0.094 / +0.124 R** (BF).
   ORB's own audit sentence — *"the raw ORB breakout has no edge; the pipeline's selection is the
   edge"* — is only half right: the selection of the NAME-DAY is real but small.
3. **The MINUTE is the larger half on both live books.** Signal minus the same name-day at a later
   minute: **+0.198 / +0.324 R** (ORB), **+0.646 / +0.622 R** (BF). This is the exact reverse of
   HOD-break, where the same statistic is −0.063 / +0.026 R.

## 1.3 The cells as pre-registered

| cell | split | n | signal gross R | control mean | ctrl p5 | ctrl p95 | signal pctile | paired Δ | day-clust t | MDE |
|---|---|---|---|---|---|---|---|---|---|---|
| ORB-b | TRAIN | 4,255 | +0.2161 | +0.1094 | −0.0273 | +0.2336 | 92.5 | +0.1527 | +0.93 | 0.346 |
| ORB-b | VAL | 3,105 | +0.4695 | +0.0662 | −0.0859 | +0.2215 | **100.0** | +0.4359 | +1.72 | 0.667 |
| ORB-a′ | TRAIN | 2,040 | +0.2161 | +0.0184 | −0.0898 | +0.1098 | **100.0** | +0.1978 | +1.64 | 0.285 |
| ORB-a′ | VAL | 1,352 | +0.4695 | +0.1471 | +0.0167 | +0.2890 | **100.0** | +0.3239 | +1.45 | 0.594 |
| ORB-u | TRAIN | 3,595 | +0.2161 | +0.0696 | −0.0649 | +0.2087 | 95.5 | +0.1866 | +1.06 | 0.366 |
| ORB-u | VAL | 2,702 | +0.4695 | +0.0237 | −0.1185 | +0.1679 | **100.0** | +0.4789 | +1.89 | 0.667 |
| BF-b | TRAIN | 628 | +0.6914 | +0.1467 | −0.1695 | +0.4270 | **100.0** | +0.5074 | +1.34 | 0.982 |
| BF-b | VAL | 274 | +0.8163 | +0.1339 | −0.2739 | +0.4833 | 99.5 | +0.3487 | +0.58 | 1.587 |
| BF-a′ | TRAIN | 330 | +0.6914 | +0.0453 | −0.2313 | +0.3314 | **100.0** | +0.6461 | +1.57 | 1.068 |
| BF-a′ | VAL | 150 | +0.8163 | +0.1943 | −0.2964 | +0.7193 | 97.5 | +0.6220 | +1.10 | 1.469 |
| BF-u | TRAIN | 667 | +0.6914 | +0.0527 | −0.2208 | +0.4099 | 99.5 | +0.5632 | +1.55 | 0.957 |
| BF-u | VAL | 303 | +0.8163 | +0.0101 | −0.4426 | +0.4421 | 99.5 | +0.4815 | +0.88 | 1.511 |

**The pre-committed rule resolves both books to NO DIFFERENT**: BETTER requires the signal above p95
on **both** splits **and** a paired day-clustered t > +2 on both, for **both** arm b and arm a′.
ORB-b is at the 92.5th percentile on TRAIN and no cell reaches |t| = 2. Nothing was re-selected after
the fact.

*What the rule cannot express, and is reported because it is true*: 12 of 12 paired differences are
positive and 12 of 12 percentiles are ≥ 92.5. The arms are **not independent** (they share the same
34/210 signal trades), so this is a consistency observation, not a p-value.

## 1.4 Both halves

| object | ORB H1-25 | ORB H2-25 | ORB VAL | BF H1-25 | BF H2-25 | BF VAL |
|---|---|---|---|---|---|---|
| **signal** | **+0.3569** | **+0.1196** | **+0.4695** | **+0.6357** | **+0.7670** | **+0.8163** |
| matched non-signal, same clock | +0.2156 | +0.0371 | +0.0662 | +0.2346 | +0.0346 | +0.1339 |
| same name-day, later minute | +0.0917 | −0.0320 | +0.1471 | −0.0544 | +0.1806 | +0.1943 |
| universe bound | +0.0482 | +0.0838 | +0.0237 | +0.1629 | −0.0726 | +0.0101 |

The signal is above every control arm in **all 18** era × arm comparisons, and **positive in all three
eras on both books** — the one thing eleven HOD-break passes never produced for a single object.

## 1.5 The exit-mix mechanism (rail 7)

| book | arm | split | n | mean R | WR | stop % | **lock / trail %** | flat % | eod % |
|---|---|---|---|---|---|---|---|---|---|
| ORB | **signal** | TRAIN | 204 | +0.2161 | 48.5 | 45.1 | **23.5** | 28.4 | 2.9 |
| ORB | **signal** | VAL | 136 | +0.4695 | 47.1 | 46.3 | **11.8** | 39.7 | 2.2 |
| ORB | b | TRAIN | 4,255 | +0.1094 | 49.0 | 33.5 | 5.4 | 59.9 | 1.2 |
| ORB | b | VAL | 3,105 | +0.0662 | 47.7 | 34.2 | 5.5 | 59.6 | 0.6 |
| ORB | a′ | TRAIN | 2,040 | +0.0184 | 42.8 | 33.5 | 5.2 | 56.3 | 5.0 |
| ORB | u | VAL | 2,702 | +0.0237 | 47.3 | 31.8 | 3.7 | 63.4 | 1.1 |
| BF | **signal** | TRAIN | 33 | +0.6914 | 51.5 | 48.5 | **48.5** | 3.0 | 0.0 |
| BF | **signal** | VAL | 15 | +0.8163 | 53.3 | 40.0 | **53.3** | 6.7 | 0.0 |
| BF | b | TRAIN | 628 | +0.1467 | 49.0 | 33.1 | 12.7 | 53.2 | 1.0 |
| BF | a′ | TRAIN | 330 | +0.0453 | 44.5 | 41.5 | 23.3 | 23.6 | 11.5 |
| BF | u | VAL | 303 | +0.0101 | 47.9 | 35.0 | 6.3 | 58.4 | 0.3 |

**This is the mechanism, and it is the exact opposite of HOD-break's.** On HOD-break the break's
target rate was *identical* to a random minute's (20.3 % vs 20.3 %) while its stop rate was 21 points
higher — it bought only a worse stop. Here:

* **ORB's signal minute reaches the +1.75 R lock 4.3× as often as its matched control** (23.5 % vs
  5.4 % TRAIN; 11.8 % vs 5.5 % VAL) and its stop rate is 11 points higher (45.1 % vs 33.5 %). It buys
  a worse stop **and** a far better right tail; the lock is what pays for the stop.
* **BF's signal minute exits via the armed R-trail 3.8× as often** (48.5 % vs 12.7 % TRAIN; 53.3 % vs
  7.7 % VAL) — i.e. it reaches +2 R and arms the trail — with a stop rate 15 points higher. Same
  shape.
* **The controls sit flat at 15:45** (56–64 % of the time vs 28–40 % for ORB's signal, 3–7 % for
  BF's): a random name at a mover's clock simply does not move enough to resolve either leg.

## 1.6 Power, and the honest phrasing

MDE at 80 % power on the paired differences: **ORB 0.285–0.366 R (TRAIN), 0.594–0.667 R (VAL)**;
**BF 0.957–1.068 R (TRAIN), 1.469–1.587 R (VAL)**. ORB's VAL arm-b (+0.436) and arm-u (+0.479) are
within 35 % of their MDE; every other estimate is below its own.

> *No difference between the signal minute and its matched controls was detectable at the
> pre-committed bar in THIS candidate population, at THIS 1-minute horizon, at THIS book size (ORB 8
> slots, BF ~2.8 trades a month), over 2025-01 → 2026-05, on the gross of each book's own shipped
> exit geometry — with a smallest detectable effect of about **0.3 R (ORB TRAIN) / 0.65 R (ORB VAL) /
> 1.0 R (BF TRAIN) / 1.5 R (BF VAL)**. The direction is positive in 12 of 12 cells and in 18 of 18
> era × arm comparisons; the books are small, not flat.*

**BF's power problem is structural, not fixable by analysis**: P1 books 34 TRAIN and 15 VAL trades.
At its own frequency BF cannot produce a t ≥ 2 on an effect below ~1 R **ever**, on any split, from
this cache. That is a fact about the book's frequency, and it is the strongest argument in this pass
for judging BF on its live weekly path rather than on any further backtest slicing.

## 1.7 What each reading implies for the scaling plan (stated in PREREG before scoring)

**ORB.** The pre-registered branch that fires is *"arm a′ < sig — the minute carries edge"*
(directionally; formally NO DIFFERENT). Consequences, in the order they cost money:

* **The entry-thread fix is the higher-value of the two open items.** Being in *at the signal minute*
  is worth **+0.198 / +0.324 R** over the same name-day later. A pick lost because the order was not
  working at 09:36 is not recoverable by entering the same name at 11:00 — that alternative is worth
  **+0.018 / +0.147 R**, at or below ORB's own cost.
* **The 50 bps cap buffer is CHEAP relative to what it buys, and the decomposition prices it.** At
  the book's median `range_size_pct` (≈ 5.5 %), 50 bps of extra price is **≈ 0.09 R**. The signal
  minute is worth ≈ 0.20–0.32 R more than the alternative entry. Paying up to 0.09 R to convert a
  no-fill into a signal-minute fill is a favourable trade at these point estimates — *provided* the
  buffer only converts no-fills and does not degrade fills that would have happened anyway. It does
  not resolve whether 50 is the right number; it says the sign of the trade is right.
* **What it does NOT license.** The name-day layer (the composite, the quintiles, the vetoes) is
  worth **+0.040 / +0.043 R** against the universe — real but small, and already the most-refit part
  of the stack. On this evidence the next ORB rule should **not** be another name-day filter.

**BF.** Same branch: the breakout *trigger* is load-bearing (**+0.646 / +0.622 R** over a later entry
on the same name-day), so the pre-registered conclusion "further entry-rule work is the wrong place
to spend" is **not** supported — the opposite is. Consequences:

* **The reshaping levers are the risk, not the entry rules.** P1 ships a 50 %-at-+2R partial. The
  exit mix says **48.5 % / 53.3 % of BF's signal trades exit via the armed R-trail** — i.e. the trade
  reached +2 R — against 7.7–12.7 % of controls. The partial monetises exactly the leg that carries
  the signal's advantage over its control. That is not an argument against the partial (the
  consistency case for it stands on its own), but it is an argument for watching the **post-+2R** leg
  specifically in the live ramp, because that is where the measured edge lives.
* **The P1 name-day gates (price ≤ $20, pole ≥ 5 %, the VWAP gate) are worth +0.094 / +0.124 R** — the
  larger name-day number of the two books, and consistent with those gates being selected on
  era-consistency rather than on a single year.
* **Do not size BF off any further backtest slice.** §1.6: the book cannot resolve an effect below
  ~1 R on 34/15 trades. The ramp on **realized** P&L (`docs/bf_p1_ramp.md`) is the right instrument
  and this pass is a reason to keep it, not to accelerate it.

---

# F23 — THE MIRROR: SHORT THE MATCHED NON-MOVER  (10 cells)

**204,201 shorts walked** over pass 6's own arm-b keys (every (day, symbol, entry_m, ctrl) pair),
four clocks each. Construction rails, all measured rather than assumed:

* **Fill**: the mirror of the long's capped BUY — a **floored sell-limit** set from the close of the
  decision bar (`c[e−1] × (1 − 60 bps)`); the bar opens at or above the floor → fill at that open;
  the bar **gaps down through the floor → NO FILL**, $0, never a loss. **1.8 %** of rows are no-fills.
  *(The first implementation of this frame inverted the limit — it required the bar to trade DOWN to
  a resting sell — which produced a **93.2 % no-fill rate** and scored the short only on bars that had
  already fallen 60 bps intraminute. That arm is void and is kept as `p23_badfloor.csv`; it read
  +0.188 / +0.166 gross on 6.8 % of the population and would have been a false positive.)*
* **Borrow**: `shortable` AND `easy_to_borrow`, flags known for **98.8 %** of control names,
  **BORROWABLE = 64.6 %** — within 3 pp of pass 1's 62 %. Today's flags on 2025–26 tape: survivorship,
  stated, not corrected.
* **Reg SHO 201**: entry bar opening ≥ 10 % below the prior close → the fill model is void →
  **excluded, 5.18 %** of rows.
* **Cost**: **1.8 × 0.063 = 0.113 R** booked on every short. Gross printed beside net.

| cell | split | n | /wk | gross R | **net R** | $ @ $100/R | green % | worst wk | clust t | null p95 green | MDE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **S-base** arm-b clock | TRAIN | 21,857 | 412 | +0.1259 | **+0.0125** | +27,232 | 54.7 | −15,127 | +0.33 | 66.0 | 0.020 |
| **S-base** | VAL | 9,582 | 436 | +0.0941 | **−0.0193** | −18,523 | 45.5 | −13,967 | −0.51 | 45.5 | 0.029 |
| S-nm2 range-so-far ≤ 2 % | TRAIN | 4,598 | 87 | +0.2004 | +0.0870 | +40,002 | 54.7 | −4,724 | +1.30 | 86.8 | 0.040 |
| S-nm2 | VAL | 2,017 | 92 | +0.1128 | −0.0006 | −120 | 50.0 | −4,800 | −0.01 | 59.1 | 0.058 |
| **S-nm3 range-so-far ≤ 3 %** | TRAIN | 9,329 | 176 | **+0.1869** | **+0.0735** | **+68,601** | 62.3 | −9,396 | **+1.40** | 88.7 | 0.030 |
| **S-nm3** | VAL | 4,150 | 189 | **+0.1284** | **+0.0150** | **+6,243** | 54.5 | −7,415 | +0.29 | 68.2 | 0.041 |
| S-nm4 range-so-far ≤ 4 % | TRAIN | 13,484 | 254 | +0.1652 | +0.0518 | +69,823 | 62.3 | −11,896 | +1.11 | 84.9 | 0.025 |
| S-nm4 | VAL | 5,956 | 271 | +0.1227 | +0.0093 | +5,523 | 45.5 | −10,278 | +0.20 | 68.2 | 0.036 |
| S-c1000 entry 10:00 | TRAIN | 21,994 | 415 | +0.1004 | −0.0130 | −28,566 | 52.8 | −20,462 | −0.34 | 47.2 | 0.021 |
| S-c1000 | VAL | 9,658 | 439 | +0.0607 | −0.0527 | −50,928 | 40.9 | −17,541 | −1.27 | 22.7 | 0.030 |
| S-c1100 entry 11:00 | TRAIN | 22,004 | 415 | +0.0612 | −0.0522 | −114,856 | 37.7 | −21,226 | −1.36 | 18.9 | 0.018 |
| S-c1100 | VAL | 9,708 | 441 | +0.0472 | −0.0662 | −64,223 | 36.4 | −19,048 | −1.77 | 13.6 | 0.026 |
| S-c1200 entry 12:00 | TRAIN | 21,918 | 414 | +0.0112 | −0.1022 | −223,903 | 18.9 | −18,831 | **−2.90** | 3.8 | 0.017 |
| S-c1200 | VAL | 9,684 | 440 | −0.0208 | −0.1342 | −129,946 | 13.6 | −16,944 | **−4.28** | 0.0 | 0.023 |
| S-wrap wrappers only | TRAIN | 2,677 | 51 | +0.3189 | +0.2055 | +55,021 | 45.3 | −2,444 | +1.61 | 94.3 | 0.064 |
| S-wrap | VAL | 1,037 | 47 | +0.0881 | −0.0253 | −2,620 | 50.0 | −2,616 | −0.37 | 54.5 | 0.085 |
| S-stock commons only | TRAIN | 19,180 | 362 | +0.0989 | −0.0145 | −27,789 | 49.1 | −19,083 | −0.45 | 47.2 | 0.021 |
| S-stock | VAL | 8,545 | 388 | +0.0948 | −0.0186 | −15,903 | 45.5 | −13,344 | −0.48 | 45.5 | 0.031 |
| S-nb *(diagnostic, no borrow screen)* | TRAIN | 33,117 | 625 | +0.1536 | +0.0402 | +133,151 | 58.5 | −17,517 | +1.32 | 88.7 | 0.017 |
| S-nb | VAL | 14,433 | 656 | +0.1353 | +0.0219 | +31,675 | 50.0 | −14,400 | +0.71 | 81.8 | 0.025 |

**0 of 10 cells clears the pre-committed live-exploration bar.** Halves (H1 / H2 / VAL net R):
S-base +0.022 / +0.003 / **−0.019** (fails); **S-nm3 +0.0785 / +0.0686 / +0.0150 — the only
borrowable cell same-signed positive in all three eras**; S-nm4 +0.059 / +0.045 / +0.009; S-wrap
+0.271 / +0.108 / **−0.025** (the H1-positive-VAL-negative wrapper signature F19 warned about, for
the third time); S-stock negative in all three.

**What the frame actually found.** The long's −0.16 R *does* partly invert: the short's **gross is
positive on every non-mover cell and on both splits** (+0.09 … +0.32 R). **The 1.8× cost eats it.**
At the mover's own clock the base short nets **+0.013 / −0.019 R** — a coin flip around zero. The one
cell that survives cost on both splits is the **tighter non-mover definition** (session range so far
≤ 3 % at the clock): **+0.0735 / +0.0150 R net, +$68.6K / +$6.2K at $100 a trade** — and it fails on
every other axis the bar asks about: day-clustered t **+1.40 / +0.29**, VAL net **below its own MDE**
(0.0150 vs 0.041), and **green weeks INSIDE its own count-matched null on both splits** (62.3 vs a
p95 of 88.7; 54.5 vs 68.2), i.e. the week shape is pick count, not skill — the identical finding
`orb_gates2` reported for ORB.

**The clock ladder is the pass's clean mechanism, and it is monotone and decisive.** Net R at a fixed
clock: **10:00 −0.013 / −0.053 → 11:00 −0.052 / −0.066 → 12:00 −0.102 / −0.134**, with clustered t
−2.90 / **−4.28** at 12:00 and green weeks 18.9 % / 13.6 % against null p95 of 3.8 / 0.0. **The later
the short is entered, the worse it is, on both splits, monotonically.** The exit mix says why:
S-base exits **58.3 % flat at 15:55**, 29.4 % stopped, only 12.1 % on the −2 R target. A short with a
2:1 target and a shrinking session is a trade that mostly does not resolve, and what it collects
while not resolving is the market's upward drift. The non-mover ladder works for the same reason in
reverse: a name that has not moved by the clock has more residual session volatility per unit of
stop.

**Capacity, stated because the /wk column is not a book.** These are **population** rates (≈ 25
controls per booked HOD trade), not a tradeable book: 176 trades a week at 4 concurrent slots is
impossible. A slot-rationed version would take a fraction of S-nm3 and would need its own
pre-registration. No engine work is named, because the bar was not cleared.

---

# F22 — THE BARE GEOMETRY  (12 marginal cells + a 72-bucket screen)

**339,225 detector-free control trades** (arm b 51,051 + arm d 288,174), r_pct join 100 %, ADV$ join
100 %, class stock 60.8 % / wrapper 38.9 % / unknown 0.3 %. No admission rule anywhere in the
construction.

**The population mean is −0.0693 R** (H1 −0.0831 / H2 −0.0645 / VAL −0.0600).

| cell | n | H1 | H2 | VAL | all | null p95 | min tr/wk | clust t |
|---|---|---|---|---|---|---|---|---|
| minute 09:37–10:30 | 96,877 | −0.1486 | −0.1446 | −0.1280 | **−0.1407** | −0.0585 | 29.7 | −7.10 |
| minute 10:30–11:30 | 74,686 | −0.0859 | −0.0462 | −0.0540 | −0.0620 | −0.0587 | 29.7 | −2.88 |
| minute 11:30–13:00 | 101,236 | −0.0415 | −0.0207 | −0.0457 | −0.0353 | −0.0581 | 29.7 | −1.83 |
| minute 13:00–14:01 | 66,426 | −0.0520 | −0.0337 | **+0.0153** | −0.0254 | −0.0585 | 29.7 | −1.54 |
| r_pct < 1.5 % | 59,503 | −0.0724 | −0.0029 | −0.0575 | −0.0433 | −0.0585 | 4.6 | −1.58 |
| r_pct 1.5–3 % | 95,316 | −0.1317 | −0.0665 | −0.0779 | −0.0943 | −0.0580 | 7.6 | −3.48 |
| r_pct >= 3 % | 184,406 | −0.0556 | −0.0833 | −0.0531 | −0.0648 | −0.0583 | 14.5 | −4.93 |
| ADV$ < $25M | 87,337 | −0.0763 | −0.0989 | −0.1030 | −0.0937 | −0.0585 | 13.5 | −5.20 |
| ADV$ $25–150M | 149,959 | −0.0987 | −0.0728 | −0.0765 | −0.0830 | −0.0583 | 22.0 | −4.83 |
| ADV$ >= $150M | 101,929 | −0.0653 | −0.0077 | −0.0077 | −0.0283 | −0.0584 | 16.0 | −1.22 |
| class **wrapper** | 131,869 | −0.1754 | −0.1055 | −0.1154 | **−0.1313** | −0.0586 | 11.2 | −6.33 |
| class **stock** | 206,310 | −0.0265 | −0.0385 | −0.0241 | −0.0300 | −0.0582 | 18.3 | −1.68 |

**All 12 marginal cells are negative in every era. 0 of 84 cells is a FINDING.** Of the 72-bucket
cross-map, **3 are positive in all three eras** and **none** clears the >= 10 book-sized
opportunities/week condition:

| bucket | n | H1 | H2 | VAL | all | null p95 | min tr/wk | clust t |
|---|---|---|---|---|---|---|---|---|
| 10:30–11:30 · r_pct < 1.5 % · ADV$ >= $150M · **stock** | 4,892 | +0.068 | +0.172 | +0.030 | **+0.090** | −0.048 | 2.4 | +1.39 |
| 11:30–13:00 · r_pct < 1.5 % · ADV$ >= $150M · **stock** | 6,570 | +0.090 | +0.135 | +0.017 | **+0.080** | −0.051 | 2.4 | +1.35 |
| 13:00–14:01 · r_pct 1.5–3 % · ADV$ $25–150M · stock | 3,992 | +0.006 | +0.019 | +0.009 | +0.012 | −0.046 | 2.3 | +0.34 |

*Read this correctly.* "Above the null p95" here means **less negative than a random draw from a
population whose own mean is −0.069** — 33 of 84 cells clear that, which is why the permutation is
not, on its own, evidence of a positive cell. The three buckets above are positive in all three eras,
but their clustered t is about 1.4 and their rate is **2.4 opportunities a week** against the declared
floor of 10. **The gradient is nonetheless real and interpretable**: the bare bracket is worst
**early** (−0.141 at 09:37–10:30), worst on **wrappers** (−0.131 vs −0.030 for commons), and worst at
**intermediate stop distances** (−0.094 at 1.5–3 %). It is least bad — and the only place it turns
positive at all — on a **liquid common (>= $150M ADV$) with a TIGHT stop (< 1.5 %) entered midday**.

**The conclusion the frame was built to force, and its limit.** On the HOD-break geometry — a hard
+2 R target, a 1 R stop, flat at 15:55 — the instrument itself is negative in every entry-minute,
stop-distance, ADV$ and class band, so **no admission rule on that geometry can do better than select
the least-negative region, which is what 1,012 cells found.** But §1.2 of this same pass shows the
sentence does **not** generalise to the tape: under ORB's static-lock geometry the same universe
prints **+0.070 / +0.024 R** and under BF's R-trail **+0.053 / +0.010 R**. The negative instrument is
the **capped +2 R target**, not the market. That is one concrete, testable mechanism — a 2:1 bracket
truncates the right tail on exactly the names whose right tail is the payoff — and it is the first
time in seven passes that a HOD-break finding has a clean counterexample rather than a caveat.

---

## 4. Rails

* **Reproduction**: ORB G3 picks per split and BF P1's $139,113.67 both **asserted in code** (the run
  aborts otherwise); HOD's `B2` + R3 inherited from pass 6.
* **Both TRAIN halves** printed beside VAL on every cell in all three frames.
* **Day-clustered t** on every cell; iid t not quoted where the cluster is the right unit.
* **Count-matched permutation null**: F23's green weeks (2,000 draws, pick count fixed) — *every*
  cell inside or below its band; F22's 2,000-draw label permutation across all 84 cells. F24 is
  scored against its own 200-draw control-book bands (the placebo's native null).
* **Availability audit**: BF's arms failed the declared 80 % floor under a strict-minute lookup and
  the frame was repaired, not the rail (§0b.2); F23 reports the 1.8 % no-fill, 5.18 % Reg SHO 201 and
  64.6 % borrowable shares; F22 reports 100 % r_pct and ADV$ joins.
* **Booked cost per cell**: F23 at 1.8 x 0.063 = 0.113 R, gross beside net. F24 is a **gross**
  comparison by design (a control's own spread is a different instrument's and would confound it) —
  ORB's and BF's own measured costs are unchanged and live in their own reports.
* **MDE at 80 % power** on every rejection; §1.6 states it for both books.
* **Obtainability (rail 1b)**: the BF unobtainable-fill defect and the F23 inverted-limit defect were
  both found, both quantified, and both are recorded above with the false-positive they would have
  produced.
* **Causality**: pass 6's control rule applied from the start — arm a′ draws only minutes **after**
  the signal; arm b and arm u are non-signal names and cannot leak.
* **TEST**: never opened.

**Cell count.** 38 declared in `PREREG.md` (F24 16 + F23 10 + F22 12), plus the F22 72-bucket
cross-map screen = **110 scored objects**. Programme total **1,012 + 38 = 1,050** (the 72-bucket
screen counted as declared multiplicity, not as 72 independent claims).

---

## 5. The adequacy review (RUNBOOK step 10)

* **Did we test what the books actually ARE?** Partly. Each book's shipped exit geometry is
  reproduced by a walker written from prose, and the reproduction gates hold to the cent — but the
  walker enters at a bar open on both sides of every comparison, so it is a *relative* instrument.
  ORB's own no-fill picks (slot spent, R = 0) are in the reproduction gate and out of the walker
  comparison; BF's exhaustion rule and vol-confirmed trail guard are not modelled.
* **Is the cost and fill model right?** F24 is gross, so cost does not enter; the fill convention is
  the engine's and is shared by signal and control. F23's fill model was **wrong on the first pass
  and was fixed** — the correct mirror of a capped buy is a floored sell that fills at the open when
  the open is above the floor, not a resting limit the tape must come down to.
* **Does any caveat in our own report explain the headline?** Yes, and both are named: the BF
  headline under the unobtainable fill was +1.30 / +1.75 R and would have read BETTER; the F23
  headline under the inverted limit was +0.19 / +0.17 gross on 6.8 % of the population.
* **What is the MDE?** ORB 0.29–0.67 R, BF 0.96–1.59 R, F23 0.017–0.356 R, F22 (population) about
  0.004 R. BF's is larger than any plausible effect — that book cannot be resolved by backtest at its
  own frequency.
* **Verdict**: **NO SHIP · NO CONFIG CHANGE · NOTHING RE-OPENED.** ORB stays paused as the owner set
  it; BF P1 boots Monday exactly as configured; `hod_break` stays `enabled: true, dry_run: true`.
