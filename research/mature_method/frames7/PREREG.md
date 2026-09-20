# PREREG — frames7 · F24 the placebo on the LIVE books · F23 the mirror · F22 the bare geometry

**Committed BEFORE any cell in this pass was scored.** Queue and specification: `hod_frames/FRAMES.md`
(the pass-6 queue, rows F22/F23/F24). Run order is **F24 → F23 → F22** (the owner's order: the live
books first, because ORB and BF-P1 are the only two books that can cost or make money on Monday).

Method: `research/mature_method/RUNBOOK.md`. The pass-6 machine being ported is
`hod_frames6/{common6,build20,walk20,score20}.py`, and the **control-causality rule it discovered is
binding here from the start**:

> **No control may be drawn before a signal on a day that is known to signal.** A control minute
> earlier than the signal on a signalling name-day is selected using the knowledge that the day would
> later produce a signal; pass 6 measured that defect at **+0.98 / +1.06 R** and it was 100 % of that
> pass's pre-registered "WORSE" headline. Controls are therefore drawn **after** the signal, or on a
> **matched non-signal name**, and never before.

Stores are opened **read-only**. `config.yaml`, `orb.yaml`, orders, the systemd unit and the crons are
never touched. One python process at a time, `nice -n 10`, `ulimit -v 3000000`. **TEST is never
opened** (`FREEZE.md`): TRAIN = 2025, VAL = 2026-01..05, TEST = 2026-06+ for all three books.

---

## 0. Reproduction gates (asserted in code — the run aborts otherwise)

| id | book | reference | source |
|---|---|---|---|
| **G-ORB** | ORB B+ / G3 (catalyst veto OFF), 8 slots, measured fill | **282 TRAIN picks / 177 VAL picks**, sized $ +6,515.08 / +4,287.21 | `research/orb_gates2/book_G3_meas.csv`, REPORT §2 |
| **G-BF** | BF P1 as it boots Monday | **56 trades / $139,113.67 to the cent** | `research/bf_frequency/runs/P1.csv`, REPORT §1 |
| **G-HOD** | HOD-break `B2` (F22/F23 population) | 1,622 TRAIN / 706 VAL booked, 288,174 control trades | `hod_frames6/{book6,pb6,pd6}.csv` (pass 6 asserted `B2` and the R3 walker parity to 2.22e-16) |

---

# F24 — PORT THE PLACEBO TO ORB AND BF-P1  (8 scored arm-cells per book = 16)

**Question.** Neither live book has ever been given a control. ORB's own audit says the raw breakout
is −0.18 R (2025) / −0.04 R (2026) and "the pipeline's selection is the edge"; BF's raw detector is
−0.01 / −0.07 R. Is the *edge* in **which name-day** the book picks, or in **which minute** it enters?

**The instrument.** Three walkers, written from prose in `c7.py`, each reproducing its book's shipped
exit spec, and each applied **identically to the signal and to every control**:

* `walk_orb` — stop at `r_pct` below entry, static lock (a closed bar's high at +1.75 R moves the stop
  to +0.5 R forever), no target, flat 15:45. `r_pct` = the pick's own `range_size_pct` (clipped at 1),
  which is exactly the denominator `orb_gates2` uses for R.
* `walk_bf` — hard stop at `r_pct` below entry; the R-trail arms at +2 R and rides 1 R below the
  running **closed-bar** high (`trading/bf_trail`: check-then-ratchet, the stop a bar produces is live
  from the next bar); flat 15:45. `r_pct` = (entry − stop_loss)/entry of the booked trade.
  *(The 50 %-at-+2R partial is available in the walker but is OFF in the primary cells, because the
  reproduction gate `P1.csv` has `partial_taken == 0` on all 56 trades — regen-7's own exits.)*
* Entry convention, both books and every arm: **the open of the entry bar**, exits evaluated from the
  next bar on, stops filled at the worse of (level, that bar's open) minus one 10 bps slip.

**R geometry is held fixed exactly as pass 6 held it**: a control's stop is `E × (1 − r_pct/100)` with
the **matched booked trade's own** `r_pct`, so the control trades the same %-risk bracket at its own
price.

**The four arms** (`sig` + three causal controls):

| arm | definition | what it holds fixed | causality |
|---|---|---|---|
| **sig** | the booked trade, walker-priced at its own signal minute | nothing — the book | — |
| **b** | the 25 nearest **NON-SIGNAL** names of the same session, entered at the **same clock**; matched on \|Δlog prev_close\| + \|Δlog adv20\|, exact on asset class (stock/wrapper/unknown), and for ORB inside a ±5 pp **gap band** | clock + price + ADV20 + asset class (+ gap for ORB) | a non-signal name cannot leak the signal |
| **a′** | the **same symbol-day** at 10 random minutes **strictly after** the signal minute (ORB: any later minute is a non-signal minute by construction; BF: after the booked entry) | day + name | **after** the signal — the pass-6 rule |
| **u** | 25 **random** universe names of the same session at the same clock, no matching at all | clock only | the universe bound |

Non-signal membership: ORB = not in the 13,033-row entered-inclusive candidate dump
(`Q_fill/dump_measured.csv`) for that date; BF = not in the 886-detection honest cache
(`data/bull_flag_cache_causal_full_20260905.csv`) for that date.

**Scored cells: 4 arms × 2 splits × 2 books = 16**, of which 12 are comparisons (arms b/a′/u) and 4
are the signal rows. Reported per cell: n, gross R, the 200-draw control-book band (one control per
booked trade per draw; mean, p5, p95), the signal's percentile in it, the **paired** difference
(signal minus the mean of its own controls) with a **day-clustered t**, the MDE at 80 % power, and the
**exit mix** (stop / lock-or-trail / flat / eod rates). Both TRAIN halves printed beside VAL.

**Availability rail, declared before any number is read:** an arm whose matched coverage is below
**80 %** of the booked trades on either split is demoted to a **diagnostic**, not a cell.

**Decision rule, pre-committed.** Per book:

* **BETTER** — the signal's gross R is above p95 of the arm's draw band on **both** splits **and** the
  paired day-clustered t > +2 on both, for **both** arm b and arm a′.
* **WORSE** — below p5 on both splits and clustered t < −2 on both, for both arm b and arm a′.
* **NO DIFFERENT** — anything else, including (explicitly) **arms b and a′ disagreeing**: no arm is
  selected after the fact.

The reading is stated for the **entry timing** (arm a′ — the same name-day, a later minute) and for
the **name-day selection** (arm b — a matched non-signal name at the same clock) separately, and the
decomposition table is printed exactly as pass 6's §1.5 printed it: universe bound / matched
non-signal / same-name-day-other-minute / the signal minute.

**Stated in advance — what each reading implies for the scaling plan** (so the implication cannot be
chosen after the number):

* **ORB.** If arm a′ ≈ sig (the minute is worth nothing) the ORB edge is the 09:35 *ranking*, and the
  next dollar belongs to the selector and to **not losing picks**: the 50 bps cap buffer and the
  entry-thread fix are then buying back picks the book has already earned, and widening the buffer is
  cheap insurance. If arm a′ < sig (the minute carries edge) the cap buffer is the opposite trade —
  a worse fill on the signal minute is worth more than a good fill later — and the thread fix matters
  more than the buffer.
* **BF.** If arm a′ ≈ sig, the breakout *trigger* is not where BF's money is and further entry-rule
  work (tighter pole, VWAP distance, retest rules) is the wrong place to spend; the P1 gates that pick
  the name-day (price ≤ $20, pole ≥ 5 %, the VWAP gate, the tier stack) are. If arm a′ < sig, the
  entry rules are load-bearing and the partial/trail reshaping is the risk.

**Known deviations, declared before scoring** (they apply to *both* arms and therefore cancel in the
comparison, which is why the comparison is walker-internal and the book's own $ is the reproduction
gate, never the placebo statistic):

1. The walker enters at a **bar open**; live ORB fills intrabar at the stop-limit cap and live BF at
   the breakout level. A parity diagnostic (correlation and mean difference of walker R vs the book's
   own R on the booked picks) is printed and is **not** a gate.
2. The ORB signal minute is recovered from the tape as the first bar at/after 09:36 whose high reaches
   the pick's own fill level; picks whose level is never reached on the tape are dropped and counted.
3. `walk_bf` implements the hard stop + R-trail + flat; the exhaustion rule and the vol-confirmed
   trail guard are not modelled (they fire on 10 of 56 booked trades and have no control-side
   analogue).
4. ORB no-fill picks (R = 0, slot spent) are part of the *book* gate but not of the walker comparison,
   which is over **entered** picks only.

---

# F23 — THE MIRROR: SHORT THE MATCHED NON-MOVER  (10 cells)

**Question.** Pass 6 measured the matched non-signal name at a mover's clock at **−0.161 / −0.160 R**
on 51,051 trades — the largest, most era-stable negative in 1,012 cells, and a **population**, not a
rule. Does the short side of that same object clear its (higher) cost?

**Population.** `hod_frames6/pb6.csv`'s own keys — every (day, symbol, entry_m, ctrl) pair of pass 6's
arm b, re-walked as a SHORT. No new selection is introduced.

**The short is simulated properly, never a sign flip** (`c7.walk_short`):

* entry at the next bar's open **under a floor**: the engine rests a sell-limit at `o[e] × (1 − 60 bps)`;
  a bar whose low never reaches the floor is **NO FILL** (dropped, counted), never a loss;
* stop **above** at `E × (1 + r_pct/100)` — the same % distance as the long, filled at the worse of
  (stop, that bar's open) **plus** one slip;
* cover target at **−2 R** (a resting buy-limit, filled on a bar CLOSE at or below it);
* flat 15:55; `r_pct` = the matched booked trade's own, exactly as in arm b.

**Borrow.** `easy_to_borrow` **and** `shortable` from today's Alpaca asset file
(`data/research/alpaca_assets_all_20260905.csv` + the live asset endpoint if the flags are absent).
These are **today's** flags applied to 2025–2026 tape: stated as survivorship, not corrected.
Pass 1 measured ~62 % borrowable on these names; the borrowable share is reported per cell and a cell
is scored on the borrowable subset only.

**Reg SHO 201.** A short-sale circuit breaker is in force for a name whose tape is **down ≥ 10 %** from
the prior close; from that point a short may only be entered on an up-bid and our fill model is **void**.
Trades whose entry bar is ≥ 10 % below prev close are **counted and excluded**; the excluded share is
reported per cell.

**Cost.** Short cost is booked at **1.8 × the programme's measured long cost** (0.063 R) = **0.113 R per
trade**, charged to every cell; gross is printed beside net.

**The 10 declared cells:**

| # | cell | definition |
|---|---|---|
| 1 | `S-base` | the full arm-b population, borrowable, ex-201 |
| 2 | `S-nm2` | non-mover ladder: session range-so-far at the clock ≤ **2 %** |
| 3 | `S-nm3` | ≤ **3 %** |
| 4 | `S-nm4` | ≤ **4 %** |
| 5 | `S-c1000` | clock ladder: entry at **10:00** on the same matched names |
| 6 | `S-c1100` | entry at **11:00** |
| 7 | `S-c1200` | entry at **12:00** |
| 8 | `S-wrap` | wrappers only (the complex that supplies a third of the recent stream) |
| 9 | `S-stock` | commons only |
| 10 | `S-nb` | **diagnostic** — no borrow screen (the bound if every name were borrowable) |

Every cell is reported on **both TRAIN halves and VAL**, with day-clustered t, a count-matched
permutation null (2,000 draws, pick count held fixed), trades/week, gross and net R and dollars at
$100 risk.

**Live-exploration bar, pre-committed** (RUNBOOK step 10): positive weekly dollars **AND** ≥ 50 % green
weeks on **both** splits at **≥ 10 trades/week**, day-clustered t ≥ 2, and the two TRAIN halves
same-signed. A cell that clears it is named as a **build candidate only** — a `side` field, an inverted
OCO sell bracket, the borrow check and 201 handling, exactly as pass 1 listed — and ships nothing.

---

# F22 — THE BARE GEOMETRY  (12 cells + a counted 72-bucket cross-map)

**Question.** On pass 6's **288,174 detector-free control trades** (`pb6.csv` arm b + `pd6.csv` arm d),
with **no admission rule anywhere in the construction**, is any region of the *instrument itself*
positive? This is the floor under every rule the programme has written.

**Construction.** Each control row carries its own `rr` from pass 6's walker, its entry minute, and —
by the join on its matched booked trade — the `r_pct` it traded. ADV$ and asset class come from the
PIT panel and the shipped offline class map. No new bar is read.

**The 12 declared marginal cells** (each scored on both TRAIN halves and VAL):

| group | cells |
|---|---|
| entry-minute band | 09:37–10:30 · 10:30–11:30 · 11:30–13:00 · 13:00–14:01 (4) |
| stop-distance band (`r_pct`) | < 1.5 % · 1.5–3 % · ≥ 3 % (3) |
| ADV$ band | < $25M · $25–150M · ≥ $150M (3) |
| asset class | wrapper · common (2) |

**The cross-map** is the full 4 × 3 × 3 × 2 = **72 buckets**; it is declared here as a screen, its
multiplicity is counted into the programme total, and a bucket is reported as a finding **only** if it
is (i) positive in **both** TRAIN halves **and** VAL, (ii) ≥ 10 trades/week at the book's own slot
count, and (iii) above the **95th percentile of a 2,000-draw permutation** that shuffles the bucket
label across the whole control population. All three, or it is noise.

**Pre-committed interpretation.** If the bare instrument is negative in every cell, then no admission
rule on this tape can do better than *select the least-negative region* — which is what 1,012 cells
found — and that sentence goes on the ledger. If a cell is positive and clears the three conditions,
the detector must be re-run **on that geometry** before any claim, and the era-consistency rail is
applied to the CONTROL population first.

---

## Cell count

Programme total after pass 6: **1,012**. This pass declares **16 (F24) + 10 (F23) + 12 (F22) = 38**
scored cells, plus the F22 72-bucket cross-map screen counted as a screen. Closing count is stated in
`REPORT.md §0` and appended to `hod_frames/FRAMES.md`.

## Rails, every cell

Reproduction gates above; both TRAIN halves beside VAL; day-clustered SE; count-matched permutation
null wherever a *rule* is scored (F23, F22's cross-map); booked cost stated per cell; availability
audit on every arm before its number is read; MDE at 80 % power on every non-rejection; the pass-6
control-causality rule; TEST sealed (`FREEZE.md`). Verdicts in PLAN §1 phrasing — *"no effect was
detectable in THIS universe, at THIS horizon, at THIS book size, over THIS window, at THIS cost, with
a smallest detectable effect of X"* — never "no edge exists".
