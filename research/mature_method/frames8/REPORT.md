# frames8 — F25 THE GEOMETRY TRANSPLANT · F26 the ORB cap priced · F27 the power floor — REPORT (2026-09-20)

Pass 8 of the frame programme, and the first one whose headline frame **predicted a sign before it
was run**. Cells exactly as declared in `PREREG.md`, committed (`2d968b2`) before any cell was
scored. Artifacts: `g8.py` (the vectorised multi-exit walker + the cost model) · `parity8.py`,
`parity8b.py` → the five reproduction gates · `f25_walk.py` → `w_{sig,a,b,d}.csv` (**369,419 walked
brackets**), `walk25.log` · `score25.py` → `cells25.csv`, `decomp25.csv`, `score25.log` · `f26.py`
→ `f26_converted.csv`, `cells26.csv` · `f27.py` → `cells27.csv`. One python process at a time,
`nice -n 10`, `ulimit -v 3000000`, the walk checkpointed per session; `cache.db`, `bars_sip.db`,
`trades.db`, the Databento stores and `daily_bars` opened **read-only**. No config, `orb.yaml`,
systemd unit, cron, order or cache was written. **TEST was never opened** (`FREEZE.md`).

---

## 0. THE SENTENCE THIS PASS WAS REQUIRED TO PRINT FIRST

**The prediction is REFUTED on BOTH limbs and the pre-registered falsifier FIRES. F25 is dead, and
with it the explanation pass 7 gave for seven passes of zeros.**

| the pre-registered prediction | what the transplant actually reads |
|---|---|
| (i) HOD-break's **control population** (the universe bound) turns from −0.058 / −0.042 R to **≥ 0 in all three eras** under ORB's or BF's geometry | **NO.** Under ORB's static lock it is **−0.0522 / −0.0470**, under BF's R-trail **−0.0522 / −0.0421**, under a bare stop ridden to 15:55 **−0.0536 / −0.0488** — negative in **H1, H2 and VAL under every one of the 8 geometries**, and the whole move is **+0.005 R**. |
| (ii) the HOD **signal minute** turns from ≈ 0 against its own later-minute control to **> 0 on both splits** | **NO.** signal − (same name-day, a later minute) is **−0.064 / +0.029 R** under the shipped cap and **−0.066 / +0.048 (lock) · −0.064 / +0.041 (trail) · −0.050 / +0.039 (bare stop) · −0.054 / +0.049 (+6 R)** — **negative on TRAIN in 8 of 8 geometries**, |t| < 2 in 16 of 16 cells. The falsifier's exact wording ("still ≤ its own later-minute control") holds on TRAIN everywhere. |

**Therefore F22's sentence — *"the negative instrument is the capped +2 R target, not the market"* —
is WITHDRAWN.** It was generalised from a comparison across two different POPULATIONS (HOD-break's
universe at HOD-break's clocks vs ORB's gap-ups at 09:35 and BF's flags), and when the geometry alone
is varied on one population the baseline does not move. **What is negative is the population and the
clock, not the exit.** That correction is the real deliverable of this pass, and it was only
obtainable because the frame was written to be falsifiable.

**Verdict, all three frames: STAY-DRY · NO SHIP · NO CONFIG CHANGE.** 0 of 18 declared cells clears
the live-exploration bar. `hod_break` stays `enabled: true, dry_run: true`; `config.yaml` and
`orb.yaml` are untouched and Monday's boot is unchanged. F26 returns one supported *answer* (the
50-bps cap that ships is right; 100 and 150 are not) and no change, because 50 bps is already the
shipped value.

---

## 0b. Reproduction gates — five, all asserted in code before a number was read

| id | gate | result |
|---|---|---|
| **G-B2** | `B2` = 1,622 / 30.6 wk / gross −0.039 / net −0.107 / **−$17,346** (TRAIN) and 706 / 30.7 / +0.083 / +0.013 / **+$893** (VAL) | **MATCH** |
| **P1** | the transplant walker's **X0** (+2 R cap) vs `hod_frames6/book6.csv::rr`, all 2,328 booked trades | **max abs Δ = 1.07e-14** |
| **P2** | vectorised **G1** vs `frames7/c7.walk_orb` (the prose-written ORB walker), 2,328 trades | **1.72e-14** |
| **P3** | vectorised **G2** vs `frames7/c7.walk_bf`, 2,328 trades | **1.07e-14** |
| **P4** | **G1 vs the SHIPPED `study_orb_pipeline_static_lock.simulate_static_lock`** on 50 real ORB trades, touchgo off | **27 of 27 stop/lock legs identical in reason AND price, 0 deviations**; 23 force-close legs are a **declared convention deviation** (ORB exits at the last ≤ 15:45 bar's CLOSE with 10 bps slip, this programme at the force-close bar's OPEN with none — common-mode across every frames8 cell) |
| **P5** | **G2's stop path vs `trading/bf_trail.arm_and_ratchet`** fed the same closed bars, 400 real trades | **max abs Δ = 0.0** |

The two live exit specs are therefore transplanted *exactly*, not approximately, and the shipped
+2 R cell reproduces the honest book to machine epsilon.

---

# F25 — THE GEOMETRY TRANSPLANT (14 declared cells)

## 1.1 The booked book under each geometry — B2's trade set held FIXED

**Trades/wk is 30.6 / 30.7 in every row by construction** (the exit is the only thing that moves;
the re-booking honesty check is §1.2). $100 risk. Cost is **re-measured per cell from that cell's own
exit mix** — the 0.065 R was not carried, and it rises exactly as predicted as free target fills are
converted into paid stop/force-close fills.

| cell | split | gross | **cost** | **net** | exit mix tgt/lock/trail/stop/eod | green % (null p95) | red streak | worst wk $ | **wk $** | total $ | ex-top-5 % | H1 / H2 | clust t | MDE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **X0** shipped +2 R | TRAIN | −0.0390 | 0.0680 | **−0.1069** | 20/0/0/52/28 | 32.1 (37.7) | 5 | −2,231 | **−327** | −17,346 | −0.218 | −0.143 / −0.073 | −2.99 | 0.088 |
| | VAL | +0.0827 | 0.0701 | **+0.0127** | 21/0/0/46/33 | 43.5 (60.9) | 4 | −957 | **+39** | +893 | −0.093 | — / +0.013 | +0.26 | 0.132 |
| **G1** ORB static lock | TRAIN | −0.0041 | 0.0735 | **−0.0776** | 0/12/0/50/38 | 34.0 (41.5) | 4 | −2,244 | **−238** | −12,588 | −0.344 | −0.101 / −0.055 | −1.45 | 0.112 |
| | VAL | +0.0891 | 0.0760 | **+0.0131** | 0/10/0/44/45 | 47.8 (56.5) | 3 | −1,508 | **+40** | +925 | −0.194 | — / +0.013 | +0.23 | 0.146 |
| **G2** BF R-trail | TRAIN | −0.0248 | 0.0750 | **−0.0998** | 0/0/18/51/31 | 32.1 (39.6) | 6 | −2,339 | **−306** | −16,191 | −0.284 | −0.130 / −0.071 | −2.43 | 0.094 |
| | VAL | +0.0848 | 0.0780 | **+0.0067** | 0/0/18/46/36 | 52.2 (56.5) | 3 | −1,036 | **+21** | +475 | −0.153 | — / +0.007 | +0.14 | 0.136 |
| **G2p** G2 + 50 % @ +2 R | TRAIN | −0.0244 | 0.0753 | **−0.0997** | 0/0/18/51/31 | 34.0 (39.6) | 6 | −2,318 | **−305** | −16,168 | −0.247 | −0.134 / −0.067 | −2.64 | 0.089 |
| | VAL | +0.0961 | 0.0785 | **+0.0175** | 0/0/18/46/36 | 56.5 (60.9) | 3 | −1,015 | **+54** | +1,238 | −0.115 | — / +0.018 | +0.36 | 0.133 |
| **G3** bare stop, no target | TRAIN | **+0.0142** | 0.0721 | **−0.0579** | 0/0/0/55/45 | 35.8 (45.3) | 4 | −2,198 | **−177** | −9,392 | −0.339 | −0.079 / −0.038 | −0.95 | 0.121 |
| | VAL | +0.0786 | 0.0746 | **+0.0040** | 0/0/0/48/52 | 47.8 (56.5) | 6 | −1,576 | **+12** | +281 | −0.214 | — / +0.004 | +0.07 | 0.155 |
| **G5** target +3 R | TRAIN | −0.0212 | 0.0701 | −0.0913 | 11/0/0/54/36 | 35.8 (41.5) | 4 | −2,131 | −279 | −14,805 | −0.255 | −0.128 / −0.057 | −2.19 | 0.098 |
| | VAL | **+0.1040** | 0.0724 | **+0.0316** | 10/0/0/47/43 | 52.2 (60.9) | 3 | −1,346 | **+97** | +2,228 | −0.126 | — / +0.032 | +0.60 | 0.147 |
| **G6** target +4 R | TRAIN | −0.0094 | 0.0709 | −0.0803 | 6/0/0/54/40 | 37.7 (43.4) | 5 | −2,198 | −246 | −13,031 | −0.296 | −0.116 / −0.047 | −1.74 | 0.105 |
| | VAL | +0.0888 | 0.0737 | +0.0151 | 4/0/0/48/48 | **60.9 (60.9)** | 3 | −1,576 | +46 | +1,067 | −0.190 | — / +0.015 | +0.28 | 0.150 |
| **G7** target +6 R | TRAIN | +0.0010 | 0.0717 | −0.0707 | 2/0/0/55/43 | 34.0 (43.4) | 5 | −2,198 | −216 | −11,469 | −0.334 | −0.115 / −0.029 | −1.37 | 0.112 |
| | VAL | +0.0904 | 0.0744 | +0.0160 | 1/0/0/48/51 | 52.2 (60.9) | 6 | −1,576 | +49 | +1,130 | −0.206 | — / +0.016 | +0.28 | 0.156 |
| **G1b** lock + 40 % @ +3 R | TRAIN | −0.0112 | 0.0726 | −0.0838 | 0/12/0/50/38 | 34.0 (41.5) | 4 | −2,244 | −257 | −13,594 | −0.308 | −0.112 / −0.057 | −1.81 | 0.101 |
| | VAL | +0.1018 | 0.0751 | +0.0267 | 0/10/0/44/45 | 52.2 (60.9) | 3 | −1,399 | +82 | +1,886 | −0.160 | — / +0.027 | +0.49 | 0.142 |
| **G2v** trail + prev-bar vol guard | TRAIN | −0.0147 | 0.0744 | −0.0892 | 0/0/15/51/34 | 30.2 (41.5) | 5 | −2,325 | −273 | −14,461 | −0.291 | −0.128 / −0.052 | −2.06 | 0.098 |
| | VAL | +0.0730 | 0.0772 | −0.0042 | 0/0/14/46/40 | 39.1 (56.5) | 3 | −1,093 | −13 | −294 | −0.174 | — / −0.004 | −0.08 | 0.138 |
| **G2pl** trail on **plan-R** | TRAIN | −0.0120 | 0.0750 | −0.0870 | 0/0/19/51/31 | 35.8 (39.6) | 4 | −2,110 | −266 | −14,114 | −0.269 | −0.113 / −0.063 | −2.19 | 0.093 |
| | VAL | +0.0810 | 0.0781 | +0.0029 | 0/0/19/45/37 | 52.2 (56.5) | 3 | −1,187 | +9 | +207 | −0.163 | — / +0.003 | +0.06 | 0.136 |

**`G1a` (ORB's ATR stop floor) is a DIAGNOSTIC, not a cell — the availability rail fired.** `atr14`
is present for 994 of 1,622 TRAIN and 353 of 706 VAL booked trades = **61.3 % / 50.0 %, below the
declared 80 % floor**, decided before the number was read. For completeness the number is *worse*
than everything above (net **−0.0839 / −0.1321**, green 18.9 / 21.7 %): ORB's floor tightens the stop
toward the entry, and on a book whose stop rate is already 50 % that is the wrong direction.

**Readings.**

1. **Removing the cap moves gross a lot and net a little.** TRAIN gross goes −0.039 (+2 R) → +0.014
   (no target at all), a **+0.053 R** swing — real, monotone across the ladder (+2 → +3 → +4 → +6 →
   ∞ gives −0.039, −0.021, −0.009, +0.001, +0.014). **The cost eats 0.004–0.007 R of it** (0.0680 →
   0.0721, because the target leg is a free resting limit and a stop or a 15:55 market exit is not),
   and the remaining gain leaves TRAIN net at **−0.058** — still deeply negative, and negative in
   **both** halves.
2. **The whole of the gain is the top 5 % of trades.** X0's ex-top-5 % net is −0.218; G3's is
   **−0.339**. Strip the tail and the uncapped exits are *worse than the cap*, because what the cap
   removed was precisely the right tail and nothing else. This is the frame's own subject matter, and
   it resolves against it: the tail exists, it is thin, and it does not pay for the 55 % stop rate.
3. **Week shape never separates from its own count-matched null.** Every cell's green-week share is
   **below** its 2,000-draw permutation p95 (the one tie is G6 VAL at 60.9 vs p95 60.9). Week shape
   here is pick count, not exit design — the third book in a row (`orb_gates2`, F23, now F25) where
   that is the answer.
4. **The BF partial and the ORB scale-out are the two best sub-arms, and both are VAL-only.** G2p
   VAL +0.0175 / 56.5 % green and G1b VAL +0.0267 / 52.2 % green against TRAIN nets of −0.0997 and
   −0.0838. Halves are opposite-signed: rejected by the standing rule, not by a new one.
5. **BF's volume guard is actively harmful here** (G2v VAL −0.0042 against G2's +0.0067): on this
   population the low-volume trail exits it skips go on to be *worse* exits, not better ones.

## 1.2 RB — the re-booking honesty check (the declared approximation, measured)

An uncapped exit holds a 4-concurrent slot longer, so the fixed-set table above is optimistic on
COUNT. `trading.hod_break.run_book(12, 4)` re-run on all 7,027 admitted signals with each geometry's
own `exit_m`:

| cell | TRAIN n (Δ) | TRAIN net | TRAIN $ | VAL n (Δ) | VAL net | VAL $ |
|---|---|---|---|---|---|---|
| RB_X0 | 1,622 (—) | −0.1069 | −17,346 | 706 (—) | +0.0127 | +893 |
| RB_G1 | **1,500 (−7.5 %)** | −0.0904 | −13,562 | 636 (−9.9 %) | **−0.0021** | −134 |
| RB_G2 | 1,557 (−4.0 %) | −0.0964 | −15,003 | 675 (−4.4 %) | **−0.0166** | −1,124 |
| RB_G3 | **1,434 (−11.6 %)** | −0.0596 | −8,541 | 609 (−13.7 %) | +0.0176 | +1,071 |

**The slot cost is real and it takes back most of the VAL improvement**: G1 and G2 go from a small
positive VAL net to a small negative one once the book actually has to hold the slot. Only G3 keeps
its sign, and its TRAIN net is unchanged at −0.060.

## 1.3 THE DECOMPOSITION under each geometry (gross R) — the falsifier's table

| geometry | split | **signal** | same name-day, LATER minute (a′) | matched non-signal name, same clock (b) | **the universe bound (d)** | d H1 / H2 | **sig − a′** (t) | sig − b (t) | sig − d (t) |
|---|---|---|---|---|---|---|---|---|---|
| **X0** +2 R (shipped) | TRAIN | −0.0390 | +0.0230 | −0.1614 | **−0.0579** | −0.069 / −0.048 | **−0.0638** (−1.85) | +0.1228 (+2.96) | +0.0187 (+0.46) |
| | VAL | +0.0827 | +0.0538 | −0.1596 | **−0.0423** | — / −0.042 | **+0.0287** (+0.65) | +0.2400 (+4.74) | +0.1244 (+2.63) |
| **G1** ORB lock | TRAIN | −0.0041 | +0.0599 | −0.1625 | **−0.0522** | −0.065 / −0.040 | **−0.0660** (−1.48) | +0.1589 (+2.74) | +0.0478 (+0.82) |
| | VAL | +0.0891 | +0.0435 | −0.1700 | **−0.0470** | — / −0.047 | **+0.0481** (+0.93) | +0.2565 (+4.55) | +0.1355 (+2.47) |
| **G2** BF trail | TRAIN | −0.0248 | +0.0367 | −0.1574 | **−0.0522** | −0.064 / −0.041 | **−0.0636** (−1.66) | +0.1341 (+2.87) | +0.0280 (+0.60) |
| | VAL | +0.0848 | +0.0446 | −0.1608 | **−0.0421** | — / −0.042 | **+0.0410** (+0.90) | +0.2436 (+4.73) | +0.1263 (+2.62) |
| **G3** bare stop | TRAIN | +0.0142 | +0.0623 | −0.1688 | **−0.0536** | −0.067 / −0.041 | **−0.0499** (−1.07) | +0.1829 (+2.75) | +0.0671 (+0.99) |
| | VAL | +0.0786 | +0.0428 | −0.1707 | **−0.0488** | — / −0.049 | **+0.0388** (+0.81) | +0.2452 (+4.24) | +0.1268 (+2.31) |
| **G7** +6 R | TRAIN | +0.0010 | +0.0527 | −0.1704 | **−0.0559** | −0.069 / −0.044 | **−0.0535** (−1.27) | +0.1699 (+3.01) | +0.0554 (+0.97) |
| | VAL | +0.0904 | +0.0441 | −0.1702 | **−0.0482** | — / −0.048 | **+0.0493** (+1.04) | +0.2566 (+4.51) | +0.1380 (+2.56) |

*(G2p, G5 and G6 sit inside this envelope and are in `decomp25.csv`. MDE on the paired sig−a′
difference: 0.080–0.097 TRAIN, 0.121–0.135 VAL — the point estimates are inside it everywhere.)*

Three things come out of this table and they are the pass's substance:

1. **The falsifier fires.** sig − a′ is **negative on TRAIN under every one of the eight
   geometries** and its VAL positive never reaches t = 1.2. The break minute is worth nothing under
   an uncapped exit for exactly the same reason it was worth nothing under the cap.
2. **The universe bound does not move.** −0.052 … −0.056 R across every geometry, **negative in H1,
   H2 and VAL in all of them**. The pre-registered rail — *"the new baseline must be ≥ 0 in all three
   eras before any detector goes on top"* — is failed by 8 of 8 geometries, so no detector was put on
   top and none of the 1,050 earlier cells is re-opened.
3. **The one real object gets BIGGER, and it is still not enough.** The name-day selection
   (sig − b) rises from **+0.123 / +0.240** under the cap to **+0.183 / +0.245** under a bare stop,
   day-clustered t +2.75 / +4.24 — an uncapped exit does monetise the detector's selection better.
   But the pond is −0.17 R, so +0.18 R of selection lands the book at +0.014 R gross against a
   **0.072 R** cost. *The detector is real, the baseline is what kills it, and the baseline is not
   the exit.*

## 1.4 Why the prediction failed — the mechanism, stated so the next frame does not repeat it

Pass 7's inference was: *the same tape reads −0.16 R capped and +0.07 R uncapped, therefore the cap
is the negative instrument.* It compared **HOD-break's universe at HOD-break's clocks under a cap**
with **ORB's gap-up universe at 09:35 and BF's flag universe under a trail**. Holding the population
fixed and varying only the geometry — this pass — the baseline moves **+0.005 R**, not +0.23 R.
**The +0.23 R lives in the population and the clock, not in the exit.** ORB's and BF's controls are
positive because a gap-up at 09:35 and a flag-breakout name are populations with upward intraday
drift at those clocks; HOD-break's matched non-mover at 10:30 is not, under any exit.

The exit mix says the same thing from the other side. Removing the cap moves HOD's exits from
20 % target / 52 % stop / 28 % eod to **0 / 55 / 45** — it converts target fills into *force-close*
fills, not into locks or trails. The ORB lock arms on only **12 %** of these trades (ORB's own signal
minute arms it on 23.5 %) and the BF trail on **18 %** (BF's own: 48.5 %). **HOD-break's trades do
not reach +1.75 R or +2 R often enough for an uncapped exit to have anything to ride.** That is the
mechanical reason the transplant could not work, and it is measurable in advance on any future book:
*before transplanting an uncapped exit, check the rate at which the book reaches the arm level.*

## 1.5 Rails

* **Reproduction**: five gates, §0b, every one asserted in code (the run aborts otherwise).
* **Both TRAIN halves** on every cell; the H1/H2 columns are in the table and in `cells25.csv`.
* **Day-clustered t** on every cell and every paired difference.
* **Count-matched permutation null** (2,000 draws, pick count fixed) on every cell's green weeks —
  and it is what kills the week-shape reading (§1.1.3).
* **Cost re-measured per cell** from its own exit mix, with the declared per-reason ratios; the rise
  from 0.0680 to 0.0721–0.0785 is the arithmetic proof it was re-measured and not carried.
* **Tail**: rank-trimmed ex-top-5 % beside every headline (the +2 R point mass ties quantile trims).
* **Availability**: `G1a` failed the 80 % floor (61.3 % / 50.0 %) and was demoted to a diagnostic
  before its number was read; every other arm covers 100 % of the B2 set.
* **Causality**: arm a′ draws minutes strictly AFTER the signal (pass 6's rule, inherited); arms b
  and d are non-signal names and cannot leak.
* **Multiplicity**: 12 geometry cells × 2 splits + 4 RB cells × 2 + 8 geometries × 3 arms × 2 splits
  of decomposition = **88 scored objects** against 14 declared cells; counted, and no cell was
  selected after the fact.
* **TEST**: never opened.

---

# F26 — THE ORB MINUTE PRICED AS A LIVE DECISION (4 cells)

**Population**: the honest ORB book's fills with a measured NBBO ask at the trigger instant —
**7,202 rows, of which 1,612 are TEST and were dropped unopened**. Of the remaining TRAIN+VAL rows,
**5,023 (89.9 %) are already marketable at the 30-bps cap** and *a wider cap cannot touch them*
(asserted in code, not assumed: every already-marketable ask is also inside the 150-bps cap). The
whole question therefore touches **10.1 %** of fills. Converted picks are re-simulated paying the
**actual ask**, so the wider cap's price is charged, not assumed; their alternative is the Stage-Q
**measured** treatment at the tighter cap (the order rests, and fills late or never).

| rung | era | n converted | **R of the converted picks** | R under the measured 30-bps treatment | **gain** | clust t | $ at $375 risk |
|---|---|---|---|---|---|---|---|
| **50 bps** | H1-25 | 88 | −0.047 | −0.200 | **+0.154** | +1.88 | +5,069 |
| | H2-25 | 120 | −0.120 | −0.176 | **+0.056** | +0.82 | +2,516 |
| | VAL | 90 | +0.269 | +0.171 | **+0.098** | +1.14 | +3,311 |
| | **all** | **298** | +0.019 | −0.079 | **+0.098** | **+2.17** | **+10,897** |
| **100 bps** | H1-25 | 58 | +0.012 | −0.118 | +0.130 | +1.25 | +2,817 |
| | H2-25 | 63 | −0.110 | −0.059 | **−0.051** | −0.63 | **−1,198** |
| | VAL | 71 | +0.255 | +0.077 | +0.179 | +1.36 | +4,760 |
| | all | 192 | +0.062 | −0.027 | +0.089 | +1.35 | +6,379 |
| **150 bps** | H1-25 | 14 | −0.040 | −0.229 | +0.189 | +0.74 | +993 |
| | H2-25 | 23 | −0.361 | −0.282 | **−0.079** | −0.64 | −681 |
| | VAL | 15 | −0.297 | −0.280 | **−0.017** | −0.25 | −96 |
| | all | 52 | **−0.256** | −0.263 | +0.011 | +0.12 | +217 |

**Was 50 bps right? YES, and it is the last rung that is.** The 50-bps conversion is the only rung
whose gain is **positive in all three eras** (+0.154 / +0.056 / +0.098), pooled **+0.098 R per
converted pick at day-clustered t +2.17**, worth **+$10,897** over 298 converted picks at the shipped
$375 risk. **100 bps is negative in H2-2025** (−0.051 R, −$1,198) and **150 bps buys outright bad
picks** (converted R −0.256) for a gain of +0.011 R at t +0.12. The **pre-committed kill rule** —
a negative converted-subset R at any rung kills every wider rung — also fires: the 50-bps converted
subset's own R is negative on TRAIN (−0.089), which kills 100 and 150 independently of their own
numbers. Rule and evidence agree.

**The discriminator: chase guard, NOT dip-buy.** If a wider cap were letting broken setups in, the
converted picks would underperform the picks that filled anyway. They do not:

| split | already marketable at 30 bps | converted (any rung) | still resting past 150 bps |
|---|---|---|---|
| TRAIN | n 3,336, R **−0.133** | n 366, R **−0.092** | n 17, R −0.172 |
| VAL | n 1,687, R **+0.024** | n 176, R **+0.215** | n 8, R −0.089 |

Converted picks are **better** than the already-marketable ones on both splits. This is the opposite
sign to the halt-resume result in memory `project_passive_entry_adverse_selection` (fills −2.03 R vs
non-fills +0.63 R) and confirms Stage Q's reading on the subset that actually decides it: on ORB the
cap is a **chase guard**, a fill is *a price on the same setup*, and the order that never fills is
the one that was going to be worst (the still-resting cohort is the worst of the three on both
splits).

**Consequence: none.** 50 bps is what `orb.yaml` already carries for Monday. No config change is
proposed, no file was written, and the widening beyond 50 is refused on the evidence and by the
pre-committed rule.

---

# F27 — THE POWER FLOOR (infrastructure; no cells, no book, no rule)

## 3.1 The arithmetic, on each book's OWN measured effect

Per-trade SD from pass 7's walked populations; n at 80 % power = (2.80 × SD / effect)²; calendar time
at the shipped frequency (ORB ≈ 6.5 trades/wk, BF-P1 ≈ 0.65/wk = 2.8/month).

| book | split | statistic | effect | per-trade SD | **n at 80 % power** | **weeks** | years |
|---|---|---|---|---|---|---|---|
| ORB | TRAIN | raw R | +0.216 | 1.694 | 482 | 74 | 1.4 |
| ORB | VAL | raw R | +0.470 | 2.732 | 265 | 41 | 0.8 |
| ORB | TRAIN | minus matched non-signal (b) | +0.153 | 1.698 | **970** | 149 | 2.9 |
| ORB | TRAIN | **minus same name-day later minute (a′)** | +0.198 | **1.453** | **423** | **65** | **1.3** |
| ORB | VAL | minus a′ | +0.324 | 2.474 | 457 | 70 | 1.4 |
| BF | TRAIN | raw R | +0.691 | 1.939 | 62 | **95** | 1.8 |
| BF | VAL | raw R | +0.816 | 2.127 | 53 | **82** | 1.6 |
| BF | TRAIN | minus b | +0.507 | 1.983 | 120 | 184 | 3.5 |
| BF | VAL | minus b | +0.349 | 2.120 | 290 | **446** | **8.6** |
| **POOL** | TRAIN | raw R | +0.282 | 1.733 | 296 | **41** | 0.8 |
| **POOL** | VAL | raw R | +0.504 | 2.674 | 221 | **31** | 0.6 |

**And the honest version, because each book's own point estimate is itself inside its MDE.** At a
plausible true effect rather than the measured one:

| book (SD) | effect +0.30 R | **+0.20 R** | +0.10 R |
|---|---|---|---|
| ORB raw (1.694) at 6.5/wk | 250 tr, 38 wk | **562 tr, 87 wk (1.7 yr)** | 2,250 tr, 6.7 yr |
| ORB minus a′ (1.453) at 6.5/wk | 184 tr, 28 wk | **414 tr, 64 wk (1.2 yr)** | 1,655 tr, 4.9 yr |
| **BF raw (1.939) at 0.65/wk** | 328 tr, 9.7 yr | **737 tr, 1,134 wk = 21.8 yr** | 2,948 tr, 87 yr |
| POOL raw (1.733) at 7.15/wk | 262 tr, 37 wk | **589 tr, 82 wk (1.6 yr)** | 2,355 tr, 6.3 yr |

**Two conclusions, one of them a correction to the frame's own premise.**

1. **BF-P1 cannot be resolved by measurement at its own frequency, ever.** At 2.8 trades a month a
   0.2 R effect needs **22 years**. Every further BF backtest slice and every "BF is up/down this
   month" reading is noise; the ramp on **realized P&L** with a bounded downside is the only honest
   instrument, exactly as `docs/bf_p1_ramp.md` already has it. ORB resolves a 0.2 R effect in
   **~1.7 years** and a 0.3 R effect in **~9 months**.
2. **The control-differenced estimator is NOT the power lever pass 7 assumed it was — and this frame
   measured it rather than asserting it.** Differencing against the matched non-signal name **does
   not reduce variance at all** (ORB TRAIN SD 1.694 → 1.698; BF 1.939 → 1.983): a 2:1 bracket on a
   non-mover is flat at the close 56–64 % of the time, so it carries almost none of the signal
   trade's day factor and subtracting it adds noise while shrinking the effect. **The only control
   that reduces variance is the SAME name-day at a later minute** (ORB SD 1.694 → **1.453**, −14 %,
   n80 482 → 423, −12 %) — and 12 % is not a gate-changing gain. **The lever that does work is
   POOLING**: ORB + BF as one portfolio statistic resolves a 0.2 R effect in **82 weeks** against
   BF's 1,134.

## 3.2 The estimator, SPECIFIED (nothing outside `frames8/` was modified)

Specified anyway, because the a′ control is cheap and the diagnostic value is real even at −12 % n:

* **Statistic.** For a live trade on (day *d*, symbol *s*, entry minute *m*) with R unit *R*:
  `Δ = R_trade − mean over k of R(s, d, m_k)` where `m_k` are **10 minutes drawn uniformly from
  (m, last_entry_minute] on the SAME symbol-day**, each priced through **the same exit spec the live
  trade used** (`trading/bf_trail.arm_and_ratchet` for BF, the static-lock walk for ORB), entered at
  the open of `m_k` with the same stop distance as a % of price. Same name, same day, same geometry —
  the only variable is the minute.
* **Where.** New helper `control_diff(trade, bars)` in `scripts/report_common.py`; called by
  `scripts/orb_eod_check.py` and the BF EOD check; surfaced by `scripts/orb_ramp_check.py` and
  `scripts/bf_ramp_check.py` as a **reported diagnostic beside realized P&L, never as a gate** until
  it has 30+ trades of its own history (its own power floor, §3.1).
* **Data.** Everything is already on the node: the trade's fill/stop/exit from `data/trades.db`; the
  symbol's own 1-minute bars for that session from `data/cache.db` (the engine already has them —
  no control symbols, no extra REST pull, which is the practical advantage of a′ over b).
* **Failure modes, each of which must log (CLAUDE.md fallback rule).** A missing minute → take the
  first existing bar at or after it within 5 minutes (pass 7's availability repair); fewer than 8 of
  10 controls priceable → report the raw number and flag; a control minute that is itself a second
  signal that day → excluded; the control is **gross** (a control's own spread is a different
  instrument's — F24's rule), so cost stays on the live side and is reported separately.
* **What it buys.** −12 % on the trade count needed, an interpretable per-trade "was the minute worth
  anything" number for the daily brief, and a check that a bad live week is the market rather than
  the book. **What it does not buy: a faster ramp.** The ramp gate stays on realized P&L.

---

## 4. The adequacy review (RUNBOOK step 10)

* **Did we test what the books actually ARE?** Yes, more tightly than any previous pass: the two
  transplanted exits are gated against the SHIPPED code (`simulate_static_lock` price-and-reason
  identical on 50 real ORB trades; `bf_trail.arm_and_ratchet` stop path identical to 0.0 on 400), and
  the shipped +2 R cell reproduces the honest book to 1e-14. The declared deviation is the
  force-close fill convention (open, no slip, vs ORB's last-bar close with 10 bps), common-mode.
* **Is the cost and fill model right?** The cost was **re-measured per cell** rather than carried, and
  it moved in the predicted direction and magnitude. The fill convention is the engine's and is
  shared by signal and control. F26 charges the actual ask on every converted pick.
* **Does any caveat in our own report explain the headline?** The headline is a refutation, and the
  caveat that *would* have rescued it is measured and does not: the re-booking check (§1.2) makes the
  uncapped cells worse, not better, and the tail check (§1.1.2) makes them worse than the cap once
  5 % of trades are removed. The one caveat that survives is stated: `G1a` is a diagnostic because
  ATR coverage is 61 %, and it points the same way.
* **What is the MDE?** F25: 0.088–0.156 R on the booked cells, 0.080–0.135 R on the paired sig−a′
  differences — every point estimate is inside it, so the honest phrasing is *no difference between
  HOD-break's break minute and a later minute of the same name-day was detectable under ANY of these
  eight exit geometries, in THIS universe, at THIS 1-minute horizon, at THIS book size (12/day, 4
  concurrent), over 2025-01 → 2026-05, at the re-measured 0.068–0.079 R cost — with a smallest
  detectable effect of about 0.09 R (TRAIN) / 0.13 R (VAL)*. F26: 0.20–0.50 R per rung on the
  converted subsets. F27 is arithmetic and has no MDE of its own.
* **Verdict**: **STAY-DRY · NO SHIP · NO CONFIG CHANGE · NOTHING RE-OPENED.** `hod_break` remains
  `enabled: true, dry_run: true`; `config.yaml trading.enabled` and `orb.yaml` are as the owner set
  them; ORB's 50-bps cap is confirmed as the right shipped value and is not changed because it is
  already the shipped value.

**Cell count.** 18 declared in `PREREG.md` (F25 14 + F26 4 + F27 0); 88 scored objects in F25's
supplementary grid and 3 in F26's discriminator, all named above. **Programme total 1,050 + 18 =
1,068.**
