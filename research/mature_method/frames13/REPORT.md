# frames13 — F40 THE OVERNIGHT FLOOR · F41 THE OVERNIGHT CONTROL AS THE DENOMINATOR · F42 THE CLOSING AUCTION AS AN EXIT LEG — REPORT (2026-09-20)

Pass 13. `PREREG.md` committed (`f0ddaaf`) with 30 cells, their bars, their predictions and their
falsifiers **before any cell was scored**. TEST never opened (`FREEZE.md`, no exception taken).
Programme cell count **1,161 -> 1,191**.

Artifacts: `_year.py` -> `on_2016..2026.parquet` (17,422,216 name-nights) · `_merge.py` ->
`col_*.npy` · `_prep.py` -> `symbols.csv`, `spy_daily.csv` · `f40.py score` -> `cells40.csv`,
`a13_by_year.csv` · `openspread.py` -> `openspread.csv` (500 sampled name-nights, 0 errors) ·
`f41.py bfwalk` -> `p41_bf.csv`, `f41.py score` -> `cells41.csv` · `f42.py walk` -> `p42.csv`,
`f42.py score` -> `cells42.csv`.

One python process at a time, `nice -n 10`, `ulimit -v 3000000`, the panel scan checkpointed per
year and both walks per session. `cache.db`, the multiday price panels, `daily_bars` and the frames7
artifacts opened **read-only**. Nothing written outside `frames13/`. No `config.yaml`, `orb.yaml`,
checker, systemd unit, cron or order was touched; Monday's 12:30 UTC boot and the 11:27 UTC pre-boot
suite are unaffected.

---

## 0. THE THREE SENTENCES

1. **F40 — the overnight floor is REAL, POSITIVE and era-stable back to 2016 at +0.062 % of price a
   night, and it is entirely the top 5 % of nights.** Over 16,551,885 eligible name-nights
   (2016-02 -> 2026-05, corporate actions removed) the unconditional close-to-next-open return net of
   Reg-T margin is **+0.0616 % of price (t +40.9)**, positive in PRE (+0.064), 2025-H1 (+0.026),
   2025-H2 (+0.102) and 2026 (+0.023). **Ex-top-5 % it is −0.168 %, and it is negative ex-top-5 % in
   every cell, every band and every era.** The declared liquid cell (common, >= $20, ADV$ >= 10M)
   reads **+0.031 % net**, is **negative in 2025-H1 (−0.055 %)**, and dies twice over on execution:
   the engine's own legs (15:55 -> 09:31 marketable) cost a **measured 0.368 % of price** round trip,
   and the opening minute's own NBBO quotes **0.91 % of price** on those names. **0 of 16 cells
   clears the bar.**
2. **F41 — ORB resolves R-POS and BF resolves R-ZERO, and the difference between them is the whole
   finding.** Under **ORB's own** exit spec, clock and universe, a matched non-signal name books
   **+0.109 R TRAIN / +0.066 R VAL** (+0.43 % / +0.26 % of price) against ORB's **+0.216 / +0.470** —
   **51 % / 14 % of the headline is the exit geometry, not the pick**. Under **BF's own** spec
   (re-walked WITH the shipped 50 % @ +2 R partial, which pass 7 omitted) the same control books
   **+0.147 / +0.149 R** against BF's **+0.621 / +0.722** — **24 % / 21 %**, a hair inside the
   pre-committed 25 % line. The universe bound is +0.070 / +0.024 (ORB) and +0.056 / +0.015 (BF).
   **The ramp band is NOT affected** (the control is common to both of its sides and cancels); the
   **expected-P&L level and the attribution ARE**, and §2.5 says exactly what should change.
3. **F42 — the closing auction as an exit leg is NOT an improvement on any of the three books, and
   its declared falsifier fires on all three.** The mechanical saving is real and small
   (**+0.0185 R = +0.084 % of price** per force-closed HOD trade, reproducing pass 12's 0.064-0.068 pp
   to the hundredth). The drift that would have to pay for it is **negative at the median on every
   book and every split** (ORB −0.081 / −0.111 %, HOD −0.034 / −0.059 %), so the net change is
   **−0.006 / −0.044 R per ORB trade** and −0.005 / −0.001 R per HOD trade. **BF force-closes 1 trade
   in 48 — the leg does not exist for that book.** No engine diff is proposed.

**Verdict: STAY-DRY / NO CHANGE on all three. 0 of 30 cells clears its bar.** The one live-relevant
deliverable is F41's attribution, written for the owner in §2.5.

---

## 0b. Reproduction gates and rails — asserted in code, raising, before any cell was read

| id | gate | result |
|---|---|---|
| **G-ORB** | `research/orb_gates2/book_G3_meas.csv` picks per split | **282 TRAIN / 177 VAL — MATCH** |
| **G-BF** | `research/bf_frequency/runs/P1.csv` | **56 trades / $139,113.67 — MATCH to the cent** |
| **G-HOD-B2** | `hod_frames6/book6.csv` | **1,622 TRAIN / 706 VAL — MATCH** |
| **G-BFWALK** | this pass's BF re-walk must reproduce pass 7 with the partial OFF | arm `sig` **+0.7304 vs +0.7304 (d 0.00000)**, arm `u` **d 0.00000**, arm `b` d −0.00130 (one control row) |
| **G-PANEL** | the raw and adjusted price panels are row-identical (DATA.md §2) | asserted per year in `_year.py`, **never violated** |
| **G-SCALE** | corporate-action rail: rows where the raw and adjusted overnight return differ by > 1 pp | **41,952 of 16,593,837 = 0.25 % dropped**; the unconditional mean moves +0.0768 % -> +0.0757 %, i.e. the whole map is insensitive to the rail |
| **G-CLOSE** | official daily close joined to the three books' keys | **100.0 % of 2,717 keys** |
| **G-AVAIL** | F41 arm coverage (inherited, frames7 §1.1) | 90.0-100 % on all eight arms, above the 80 % rail |

**Not applicable this pass, and said so rather than skipped:** the count-matched permutation null is a
green-week instrument and **no cell here makes a green-week claim** — F40 is a floor map, F41 is a
re-reading of existing books, F42 is a within-book exit swap. Day-clustered SE is used wherever a
claim is made (F41's paired differences, F42's delta).

---

# F40 — THE OVERNIGHT FLOOR (16 cells)

## 1.1 The population
`research/multiday/data/prices_by_year/{raw,all}` — 18,221,178 daily rows x 2 adjustments,
11,823 symbols, 2016-01-04 -> 2026-09-18. After the causal membership rule (next session's open for
the SAME symbol, `adv20` from the 20 sessions strictly before `t`, close >= $1, test tickers and
non-universe names out, TEST cut at 2026-06-01) and the corporate-action rail:
**16,551,885 name-nights over 2,596 sessions**, 2016-02-02 -> 2026-05-29.

The return is the ADJUSTED panel's `open[t+1] / close[t] − 1` in % of price — the economically correct
overnight return through splits and dividends, cross-checked against the raw panel row by row.

## 1.2 The 16 scored cells (net of margin at APR 7.0 %, in % of entry price)

| cell | n | **net %** | t | PRE (16-24) | 2025H1 | 2025H2 | 2026 |
|---|---|---|---|---|---|---|---|
| **A1** whole panel | 16,551,885 | **+0.0616** | +40.9 | +0.0637 | +0.0264 | +0.1020 | +0.0229 |
| A2 price < $5 | 1,821,886 | **+0.2173** | +46.9 | +0.2323 | +0.1727 | +0.2294 | +0.0783 |
| A3 $5-20 | 4,287,515 | +0.0663 | +12.4 | +0.0693 | +0.0417 | +0.1022 | +0.0064 |
| A4 $20-50 | 6,085,770 | +0.0353 | +56.0 | +0.0348 | +0.0068 | +0.0775 | +0.0207 |
| A5 $50-200 | 3,846,901 | +0.0281 | +43.8 | +0.0294 | **−0.0341** | +0.0770 | +0.0140 |
| A6 >= $200 | 509,813 | +0.0320 | +17.0 | +0.0357 | **−0.0530** | +0.0711 | +0.0291 |
| A7 ADV$ < $1M | 5,795,704 | +0.0791 | +52.9 | +0.0809 | +0.0854 | +0.0886 | +0.0414 |
| A8 $1-10M | 4,777,586 | +0.0677 | +14.2 | +0.0718 | +0.0152 | +0.1117 | +0.0137 |
| A9 $10-100M | 4,005,235 | +0.0423 | +38.2 | +0.0445 | **−0.0338** | +0.1143 | +0.0039 |
| A10 >= $100M | 1,973,360 | +0.0344 | +22.2 | +0.0346 | **−0.0287** | +0.1016 | +0.0190 |
| A11 wrapper | 743,878 | +0.0515 | +12.3 | +0.0385 | +0.0150 | +0.1361 | +0.1082 |
| A12 common | 8,351,577 | +0.0749 | +26.2 | +0.0782 | +0.0125 | +0.1278 | +0.0242 |
| **A13** common, >= $20, ADV$ >= $10M | 3,310,743 | **+0.0306** | +37.1 | +0.0346 | **−0.0547** | +0.0737 | +0.0046 |
| **A14** A13 on **leg (b)** (measured) | 3,310,743 | **−0.3372** | −410.0 | −0.3332 | −0.4224 | −0.2940 | −0.3631 |
| **A15** A13 **ex-top-5 %** | 3,145,205 | **−0.1385** | −211.2 | −0.1279 | −0.2661 | −0.0958 | −0.2134 |
| A16 A13, 2016-2024 only | 2,757,848 | +0.0346 | +39.6 | +0.0346 | — | — | — |
| *d1 A1 ex-top-5 %* | 15,724,324 | *−0.1678* | −494.1 | −0.1567 | −0.2523 | −0.1407 | −0.2498 |
| *d2 A2 ex-top-5 %* | 1,730,794 | *−0.2783* | −152.2 | −0.2484 | −0.3715 | −0.3176 | −0.4661 |
| *d3 A7 ex-top-5 %* | 5,505,920 | *−0.1846* | −307.4 | −0.1790 | −0.2327 | −0.1557 | −0.2290 |
| *d4 A12 ex-top-5 %* | 7,934,001 | *−0.1863* | −346.4 | −0.1736 | −0.3072 | −0.1656 | −0.2759 |

(d1-d4 are diagnostics added to the run, not scored cells — the cap rule applied to the broad cells so
the ex-top-5 % statement is measured rather than asserted. The cap is applied INSIDE each era as well
as pooled.)

## 1.3 The gap-risk distribution (the frame's first deliverable, in full)

| cell | p1 | p5 | p50 | p95 | p99 | <= −5 % | <= −10 % | worst name-night |
|---|---|---|---|---|---|---|---|---|
| A1 whole panel | −4.65 | −1.91 | 0.00 | +2.03 | +5.04 | 0.86 % | 0.17 % | **−99.97 % AQB 2017-01-06** |
| A2 < $5 | −7.89 | −3.64 | 0.00 | +4.31 | +9.80 | 2.71 % | 0.57 % | −90.02 % MGN 2026-03-25 |
| A7 ADV$ < $1M | −5.00 | −2.12 | 0.00 | +2.31 | +5.66 | 1.00 % | 0.17 % | −99.96 % BBB 2017-06-02 |
| A11 wrapper | −7.35 | −3.37 | 0.00 | +3.50 | +7.75 | 2.37 % | 0.44 % | −95.02 % NRGU 2025-02-19 |
| **A13** | **−3.87** | **−1.65** | **+0.05** | **+1.68** | **+3.90** | **0.60 %** | **0.13 %** | **−99.97 % AQB 2017-01-06** |

Read against A13's own **+0.045 % gross**: the p5 night is **37x the edge against you** and 0.13 % of
nights lose a tenth of the position. *(The −99.97 % rows are a handful of corporate actions the
1 pp raw-vs-adjusted rail did not catch because BOTH panels carry the same figure — they are ~1e-5 of
the population and removing them does not move any cell; they are named rather than quietly clipped.)*

## 1.4 The two legs and the margin, all MEASURED

**Leg (a), the auction leg** pays no quoted spread — a cross is a single price — but the frame
forbade assuming the open is free, so the opening minute's own liquidity was measured: **500 sampled
name-nights, stratified over the 5 x 4 price x ADV$ cells, Alpaca SIP quotes, 0 errors.**

| instant | whole stratified sample (median % of price) | the A13-like rows (n = 143) |
|---|---|---|
| **09:30-09:31 NBBO (the opening minute)** | **0.696 %** | **0.908 %** |
| **round-trip marketable (15:55 buy + 09:31 sell)** | **0.360 %** | **0.368 %** |

So: leg (a) is priced at ratio 0 as the cost model says, **and** the quoted spread in the minute after
the cross is **twenty times A13's whole overnight premium**. If the cross does not fill you — an
opening cross is exactly where that happens — you pay a cost an order of magnitude larger than the
thing you came for. That is the honest content of *"opening auctions are illiquid"* for this book.

**Margin** at Reg-T 2x (half the position financed), `APR/360 x 0.5 x calendar_nights`:

| APR | A13 net | mean charge | median charge |
|---|---|---|---|
| 6.0 % | +0.0326 % | 0.0121 % | 0.0083 % |
| **7.0 %** | **+0.0306 %** | **0.0141 %** | **0.0097 %** |
| 8.0 % | +0.0285 % | 0.0162 % | 0.0111 % |

The margin charge is **31 % of A13's gross premium**. The APR is an ASSUMPTION — it must be read off
the broker statement before anything is built on this map.

## 1.5 The map rows (diagnostics on A13)

| weekday of the CLOSE | n | gross % | net % |
|---|---|---|---|
| Mon | 617,154 | +0.087 | +0.078 |
| Tue | 683,218 | +0.084 | +0.074 |
| **Wed** | 678,141 | **−0.020** | **−0.030** |
| Thu | 665,545 | +0.064 | +0.053 |
| Fri (the weekend night) | 666,685 | +0.012 | −0.018 |

| SPY regime | n | gross % | net % |
|---|---|---|---|
| A clean bull | 2,282,240 | +0.047 | +0.033 |
| B volatile | 444,702 | +0.025 | +0.010 |
| **C1 true defensive** | 287,635 | **−0.011** | **−0.025** |
| **C2 shallow dip** | 295,260 | **+0.108** | **+0.094** |

By calendar year (A13, net at APR 7 %, and ex-top-5 %):
2016 **−0.001 / −0.118** · 2017 +0.044 / −0.051 · 2018 +0.025 / −0.096 · 2019 +0.042 / −0.073 ·
2020 +0.082 / −0.185 · 2021 +0.091 / −0.066 · **2022 −0.067 / −0.249** · 2023 +0.009 / −0.136 ·
2024 +0.075 / −0.089 · 2025 +0.013 / −0.178 · 2026 +0.005 / −0.213.
**Two of eleven years are negative even on the level, and eleven of eleven are negative ex-top-5 %.**
Worst months: March (−0.096 net), February (−0.064). Best: November (+0.137).

## 1.6 Prediction verdict (PREREG §1.6)

| clause | stated before | measured | verdict |
|---|---|---|---|
| P40.1 the unconditional panel return is positive, 0..+0.10 %, era-stable to 2016 | yes | **+0.0616 %, positive in all four eras, 10 of 11 calendar years** | **CONFIRMED** |
| P40.2 the premium is larger in low price / low ADV$ | yes | **< $5 +0.217 % vs >= $200 +0.032 %; ADV$ < $1M +0.079 % vs >= $100M +0.034 %** | **CONFIRMED** |
| P40.2b … and those cells go negative ex-top-5 % | yes | **< $5 −0.278 %, ADV$ < $1M −0.185 %** | **CONFIRMED** |
| P40.3 A13 is positive gross but FAILS the bar | fails | **fails on era stability (2025-H1 −0.055 %), on ex-top-5 % (−0.139 %), and on leg (b) (−0.337 %)** | **CONFIRMED** |
| falsifier of P40.3 (a candidate) | — | **did not fire** | — |
| falsifier of the frame (A1 negative on the auction leg in 2025+2026) | — | **did not fire: +0.026 / +0.102 / +0.023** | not refuted |

**The frame is NOT refuted and no cell clears the bar.** Pass 12's incidental +0.037..+0.044 R is
confirmed as a real, measurable, unconditional overnight risk premium — and mapped properly it is a
**lottery premium**: the whole of it lives in the top 5 % of nights, in every band, in every era. That
is what "compensation for bearing gap risk" looks like when you actually measure it, and it is exactly
the shape this owner has already rejected once.

---

# F41 — THE OVERNIGHT CONTROL AS THE DENOMINATOR FOR THE LIVE BOOKS (8 cells)

## 2.1 What was rebuilt, and what was not
**ORB: no new walk.** Pass 7's `frames7/p24.csv` already walked ORB's arms under **ORB's own exit
spec** (`c7.walk_orb`: stop at `range_size_pct` below entry, static lock arms at +1.75 R and moves the
stop to +0.5 R forever, flat 15:45), at **ORB's own clock** (the pick's own break minute + 1) and on
**ORB's own universe** (matched on price, ADV20, asset class, +/-5 pp gap). Reused verbatim.

**BF: one small re-walk, because pass 7's BF control was NOT under BF's own spec.** Pass 7 called
`walk_bf(partial=False)`; the shipped P1 book carries the **50 % @ +2 R profit partial**. BF's four
arms were re-walked with `partial=True, partial_r=2.0, partial_frac=0.5, stop->breakeven` — same bars,
same pools, same seed, 2,400 walked trades — and the partial-OFF version was re-walked alongside as a
parity gate, which reproduces pass 7 exactly (G-BFWALK above).

## 2.2 The 8 cells — the control's own return FIRST, as the frame pre-commits

| cell | book | arm | split | n sig | n ctrl | **control R** | **control % of price** | headline R | **share C/H** | paired d | day-clust t | MDE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **B1** | ORB | b matched non-signal | TRAIN | 204 | 4,255 | **+0.1094** | **+0.428 %** | +0.2161 | **51 %** | +0.1527 | +0.93 | 0.346 |
| **B2** | ORB | b | VAL | 136 | 3,105 | **+0.0662** | **+0.255 %** | +0.4695 | **14 %** | +0.4359 | +1.72 | 0.667 |
| **B3** | ORB | u universe bound | TRAIN | 204 | 3,595 | **+0.0696** | **+0.288 %** | +0.2161 | **32 %** | +0.1866 | +1.06 | 0.366 |
| **B4** | ORB | u | VAL | 136 | 2,702 | **+0.0237** | **+0.064 %** | +0.4695 | **5 %** | +0.4789 | +1.89 | 0.667 |
| **B5** | BF | b | TRAIN | 33 | 629 | **+0.1474** | **+0.405 %** | +0.6205 | **24 %** | +0.4373 | +1.23 | 0.898 |
| **B6** | BF | b | VAL | 15 | 274 | **+0.1492** | **+0.623 %** | +0.7215 | **21 %** | +0.3228 | +0.56 | 1.498 |
| **B7** | BF | u | TRAIN | 33 | 667 | **+0.0564** | **+0.062 %** | +0.6205 | **9 %** | +0.4907 | +1.45 | 0.869 |
| **B8** | BF | u | VAL | 15 | 303 | **+0.0151** | **−0.148 %** | +0.7215 | **2 %** | +0.4679 | +0.90 | 1.409 |

The signal in % of price for scale: ORB **+0.995 % / +2.027 %**, BF **+2.556 % / +2.601 %** (median
stop 4.14 / 4.13 % and 3.73 / 3.84 % of price respectively — F31's unit, printed beside R everywhere).
The BF headline moves from pass 7's +0.6914 / +0.8163 to **+0.6205 / +0.7215** purely because the
shipped partial is now in the walk.

## 2.3 The three readings, per book (the pre-committed rule, §2.3)

* **ORB — R-POS, on both arms.** The matched non-signal control is **51 % of the TRAIN headline** and
  14 % of VAL; the universe bound is 32 % / 5 %. Half of ORB's TRAIN R per pick is what a *matched
  name that did not signal* earns under ORB's static lock at ORB's clock.
* **BF — R-ZERO, on both arms, and only just.** The matched non-signal control is **24 % / 21 %**
  against a 25 % threshold. Called R-ZERO by the pre-committed rule and reported as **borderline** —
  a 5 % change in either number flips it, and the VAL cell rests on 15 booked trades.
* **Neither book is R-NEG.** No control is negative on both splits, so no book's edge is understated.

## 2.4 Both halves, and what the partial did

| arm | ORB H1-25 | ORB H2-25 | ORB VAL | BF H1-25 | BF H2-25 | BF VAL |
|---|---|---|---|---|---|---|
| **signal** | **+0.357** | **+0.120** | **+0.470** | **+0.563** | **+0.699** | **+0.722** |
| b matched non-signal | +0.216 | +0.037 | +0.066 | +0.240 | +0.030 | +0.149 |
| a' same name-day, later minute | +0.092 | −0.032 | +0.147 | −0.006 | +0.248 | +0.238 |
| u universe bound | +0.048 | +0.084 | +0.024 | +0.174 | −0.077 | +0.015 |

**In ORB's H2-2025 half the matched control (+0.037) is 31 % of the signal (+0.120)** — the half where
ORB's own edge was smallest is also where the control claims the largest share. The signal is still
above every control arm in all six era x book comparisons.

**PREREG P41.2** said the partial would lower BOTH the BF signal and the BF control, lowering the
signal by more. Measured: signal **−0.071 TRAIN / −0.095 VAL**, control **+0.003 / +0.015** (it
*rises*). The clause is **half FALSIFIED** — the partial caps the right tail, and the control has
almost no right tail to cap, so it only gains the breakeven stop. The conclusion the clause was for —
**BF's C/H share RISES once the shipped exit is in the control** — is CONFIRMED (21 % -> 24 % TRAIN,
16 % -> 21 % VAL on arm b).

## 2.5 What should change, and what should NOT — written for the owner, applied by nobody

**1. The ramp band must NOT be rebuilt on the difference.** `trading/ramp_bt_band.py` compares the
LIVE book's R/trade with the BT reference's R/trade. Both sides are the same book under the same exit
geometry, and live bears that geometry's unconditional return exactly as the backtest does, so the
control **cancels inside the gate**. Rebuilding the band on `book − control` would subtract a term
from one side only and would make the gate wrong in the direction that matters (it would pass live
books that are under-performing). **No change to `ramp_bt_band.py`, `scaling_plan_2026.md` or either
ramp checker is recommended.** This was pre-committed in PREREG §2.4 before the numbers existed.

**2. The ATTRIBUTION sentences in ORB's own documents are over-claimed and should be amended.**
CLAUDE.md and `research/orb_veto_study/REPORT.md` both say *"the raw ORB breakout has no edge; the
pipeline's selection is the edge."* Under ORB's own geometry that is only partly true: **+0.109 R of
ORB's +0.216 TRAIN R per pick is earned by a matched name that did not break out at all.** The
selection is worth **+0.153 R TRAIN (t +0.93) / +0.436 VAL (t +1.72)** against MDEs of 0.35 / 0.67 —
positive in all four cells, significant in none. The honest sentence is: *ORB's static lock on a
gap-up name at 09:35 is worth about a tenth of an R by itself, and the pick adds another tenth to a
half on top of it, inside the noise.*

**3. The gate maps are UNAFFECTED, and this is worth stating because it is the reassuring half.**
`research/orb_gates2` and `research/bf_frequency` measure *kept-minus-rejected* R inside each book's
own picks. Any control return common to both sides of that subtraction cancels exactly, so **every
number in both gate maps stands as published** — including BF's "whole P1 stack picked vs rejected
+0.659 R, t 3.13". A positive control damages LEVEL claims, never DIFFERENCE claims.

**4. The forward-looking consequence for sizing.** If a future regime removes the unconditional
gap-up-continuation return that the control measures, ORB loses roughly **half its TRAIN R per pick
and a seventh of its VAL R** before the pick quality is touched at all. The stage ladder is already
gated on realized P&L (the above-water rule), so nothing needs to change today — but a stage that
advances on 8 trades is advancing on a number of which up to half is market, not skill, and the
pooled-precision reading in `trading/ramp_pool.py` is the right place to say so if the owner wants it
encoded.

## 2.6 Prediction verdict (PREREG §2.5)

| clause | stated before | measured | verdict |
|---|---|---|---|
| P41.1 both live books resolve R-POS on >= 1 split | both | **ORB yes (both arms); BF no — R-ZERO at 24 %/21 %** | **HALF CONFIRMED** |
| P41.2 the partial lowers signal and control, signal by more | both down | **signal down, control UP; the share rises as intended** | **HALF FALSIFIED** |
| falsifier: either control <= 0 on both splits => R-NEG | — | **did not fire** | — |

---

# F42 — THE CLOSING AUCTION AS AN EXIT LEG (6 cells)

## 3.1 The re-simulation
Each book was re-walked twice on the same bars: once with its shipped force close (ORB 15:45,
BF 15:45, HOD-break 15:55) and once with **no flat at all** — stop, lock, trail and partial live
through 16:00, and whatever is still open sold in the **closing auction at the official daily close**
(joined from `daily_bars`, 100 % of 2,717 keys). A position that the shipped rule flattens at 15:45
can therefore be stopped at 15:52 under MOC, and 1.3-4.7 % of force-closed trades are.

## 3.2 The 6 cells

| cell | book | split | n | n force-closed | fc share | **d R per fc trade** | **d R per trade** | d % of price | day-clust t | resolved before 16:00 | **d $/week at live size** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **C1** | ORB | TRAIN | 204 | 64 | 31.4 % | **−0.0188** | **−0.0059** | −0.060 | −0.33 | 4.7 % | −$9.23 |
| **C2** | ORB | VAL | 136 | 57 | **41.9 %** | **−0.1040** | **−0.0436** | −0.193 | −2.05 | 1.8 % | **−$105.82** |
| **C3** | BF | TRAIN | 33 | **0** | **0.0 %** | — | +0.0021 | +0.009 | +1.01 | — | +$0.47 |
| **C4** | BF | VAL | 15 | 1 | 6.7 % | −0.1120 | −0.0075 | −0.033 | −1.03 | 0.0 % | −$1.40 |
| **C5** | HOD | TRAIN | 1,622 | 453 | 27.9 % | **−0.0170** | **−0.0047** | −0.010 | −2.29 | 1.3 % | −$14.78 |
| **C6** | HOD | VAL | 706 | 233 | 33.0 % | **−0.0023** | **−0.0007** | −0.010 | −0.22 | 2.1 % | −$2.39 |

## 3.3 (official close − force-close print), on each book's OWN names, in % of entry price

| book | split | n fc | **median** | p5 | p95 | mean | mechanical spread saving |
|---|---|---|---|---|---|---|---|
| ORB | TRAIN | 64 | **−0.081** | −2.887 | +2.619 | −0.165 | not measured per trade |
| ORB | VAL | 57 | **−0.111** | −2.761 | +1.342 | −0.394 | not measured per trade |
| BF | VAL | 1 | −0.497 | — | — | −0.497 | — |
| HOD | TRAIN | 453 | **−0.034** | −0.921 | +0.899 | −0.029 | **+0.0185 R = +0.084 % of price** |
| HOD | VAL | 233 | **−0.059** | −1.067 | +1.006 | −0.040 | **+0.0186 R = +0.088 % of price** |

The mechanical saving reproduces pass 12's measurement (0.064-0.068 pp of price) on the book's own
names to within a hundredth of a percentage point — **it is the one repeatable part, and it is real.**
It is also **smaller than the drift the swap buys**: on HOD it wins +0.084 % of price and gives back
−0.034 % of median drift plus the tail of the p5/p95 spread, netting −0.010 % per trade. On ORB the
last 15 minutes are where its names give back the most.

## 3.4 The pre-registered falsifier (PREREG §3.3) — it fires on all three

| book | H1-25 | H2-25 | VAL | drift median TRAIN | drift median VAL | verdict |
|---|---|---|---|---|---|---|
| ORB | +0.0295 | **−0.0302** | **−0.0436** | **−0.081 %** | **−0.111 %** | **NOT IMPROVED** |
| BF | +0.0036 | +0.0000 | **−0.0075** | n/a (0 fc trades) | **−0.497 %** | **NOT IMPROVED** |
| HOD | +0.0004 | **−0.0096** | **−0.0007** | **−0.034 %** | **−0.059 %** | **NOT IMPROVED** |

Pass 12's rule — *"a book whose entire improvement is the mean rather than the median is reported as
NOT improved"* — does not even get to apply: **there is no improvement in the mean either.**

## 3.5 Prediction verdict (PREREG §3.5)

| clause | stated before | measured | verdict |
|---|---|---|---|
| P42.1 the mechanical saving is ~0.06-0.07 % of price per force-closed trade | 0.06-0.07 | **+0.084 / +0.088 % on HOD's own names** | **CONFIRMED** (slightly larger) |
| P42.2 the drift fails its falsifier on >= 2 of 3 books | >= 2 | **3 of 3** | **CONFIRMED** |
| P42.3 ORB is the book most affected (it flats earliest) | ORB | **ORB force-closes 31 % / 42 % of its trades and takes the biggest hit, −$106/week at VAL** | **CONFIRMED** |

## 3.6 What would have changed, and does not
Had a book cleared, the engine diff would have been: `TimeInForce.CLS` (unused in this repo —
`data_sources/alpaca_client.py` carries `TimeInForce.DAY` only) on the force-close order, submitted
before the 15:50 imbalance publication, with the stop left live until the cross prints, plus the
reconcile path for a stop that fills between the submit and the cross. **None of that is proposed, no
engine file is touched, and no independent rebuild is requested** — the falsifier fired first.

---

## 4. The adequacy review (RUNBOOK step 10)

* **Did we test what the objects actually ARE?** F40: the whole PIT daily panel back to 2016, not a
  mover screen and not one detector's matched controls — 16.5M name-nights, both auction legs, both
  adjustments, the margin charge, and a MEASURED opening-minute spread rather than an assumed free
  cross. F41: each book's own exit spec, and the one place pass 7 got it wrong (BF's missing partial)
  was found and fixed before scoring, with a parity gate proving the re-walk reproduces pass 7 without
  it. F42: the stop kept live to 16:00 rather than assuming the auction inherits the position.
* **Is the cost and fill model right for these venues?** The auction legs pay no quoted spread
  (ratio 0) — and F40 shows why that is not the same as free: the quoted spread in the minute after
  the cross is 0.91 % of price on the liquid cell. Leg (b) is charged the MEASURED round trip
  (0.368 %), not the band. F41's differences are cost-invariant by construction (the control carries
  the same spread as the signal). F42's delta is a within-trade difference on the same fill.
* **Does any caveat in our own report explain the headline?** **Yes, once, and it was pre-committed.**
  F40's positive floor is entirely the top 5 % of nights — the cap rule (F35) was written into the bar
  before the map was built, and it is what disqualifies every cell. F41's BF reading sits 1 pp inside
  its own threshold and is reported as borderline rather than as a clean R-ZERO.
* **What is the MDE?** F40: **0.0007-0.005 % of price** per cell (n = 0.5M-16.5M) — the floor is
  measured to a ten-thousandth of a percent, so a null here is a real null, not a power failure.
  F41: **0.35-0.37 R (ORB TRAIN), 0.67 (ORB VAL), 0.87-0.90 (BF TRAIN), 1.41-1.50 (BF VAL)** — the
  paired differences (+0.15 to +0.49) are all below their own MDE; the CONTROL levels, however, are
  measured on 274-4,255 walked trades and are not MDE-limited. F42: the delta is a paired within-trade
  quantity; day-clustered t reaches −2.29 (HOD TRAIN) and −2.05 (ORB VAL) on the NEGATIVE side.
* **Multiplicity.** 30 cells declared in `PREREG.md` before scoring, 30 scored and printed, plus 4
  labelled diagnostics (d1-d4) added to make an ex-top-5 % claim measured rather than asserted. None
  selected after the fact. **Programme cell count 1,161 + 30 = 1,191.**

## 5. What this pass settles for the programme

1. **The overnight risk premium exists, is era-stable over eleven years, and is a lottery.**
   +0.062 % of price a night unconditionally; **negative ex-top-5 % in every cell, band and era.**
   Any future object that wants to hold overnight must show its edge SURVIVING the cap, and must
   price the opening minute at the measured 0.7-0.9 % of price rather than at zero.
2. **An exit geometry's own unconditional return is a first-class quantity and must be printed beside
   any book-level R.** ORB's static lock at 09:35 on a gap-up name is worth **+0.109 R TRAIN /
   +0.066 R VAL to a name that never signalled**; BF's trail-plus-partial is worth **+0.147 / +0.149**.
   Neither number changes the ramp band (it cancels) and neither changes a gate map (differences
   cancel) — both change what fraction of the headline should be called skill.
3. **The closing auction is worth its spread and no more, and on every book that force-closes, the
   15:45/15:55 -> 16:00 drift is larger and negative.** Pass 12 measured the saving; pass 13 tried to
   collect it on three real books and it does not survive contact with their own names. The programme
   can stop proposing it.

---

## 6. VERDICT

**F40 STAY-DRY (floor mapped, 0 of 16 cells clears) · F41 NO CHANGE TO ANY CHECKER, ONE ATTRIBUTION
AMENDMENT RECOMMENDED (ORB R-POS, BF R-ZERO borderline) · F42 NOT IMPROVED ON ANY BOOK, NO ENGINE
DIFF.** `trading/`, `config.yaml`, `orb.yaml`, `docs/scaling_plan_2026.md`, `trading/ramp_bt_band.py`,
the systemd unit, the crons and every order are exactly as the owner left them. `hod_break` stays
`enabled: true, dry_run: true`.

Phrased as CLAUDE.md requires: **no overnight cell with a positive, era-stable, cap-surviving,
execution-net return was detectable in THIS universe (the whole PIT daily panel, 11,823 symbols),
at THIS horizon (one night, close to next open), at THESE two execution legs (both auctions; 15:55 ->
09:31 marketable at a measured 0.368 % of price), over 2016-02 -> 2026-05, at a smallest detectable
effect of 0.0007-0.005 % of price** — and the reason is not power, it is the cap: the premium is the
top 5 % of nights everywhere it exists.
