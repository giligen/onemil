# frames14 — F45 THE COST MODEL AT THE CLOCKS THE BOOKS TRADE · F43 THE GEOMETRY FLOOR MAP · F44 IS THE OVERNIGHT TAIL SELECTABLE — REPORT (2026-09-20)

Pass 14. `PREREG.md` committed (`e160d2e`) with **26 cells**, their bars, their predictions and their
falsifiers **before any cell was scored**. TEST never opened (`FREEZE.md`, no exception taken).
Programme cell count **1,191 → 1,217**.

Artifacts: `f45_fetch.py` → `f45_minutes.csv` (2,940 SIP name-minutes, 14 clocks) · `f45_bf.py` →
`f45_bf.csv` (1,008 BF entry/exit minutes) · `f45.py` → `f45.log`, `f45_minute_table.csv` ·
`f43_walk.py` → `w43.csv` (288,174 keys × 5 bar geometries × 3 stop widths) · `f43.py join` →
`x67.csv` · `f43.py score` → `f43_score.log`, `cells43.csv` · `f44_build.py` → `feat_*.parquet` ·
`f44.py` → `f44.log`, `cells44.csv`.

One python process at a time, `nice -n 10`, `ulimit -v 3000000`, the geometry walk checkpointed per
session. `cache.db`, `daily_bars`, the multiday price panels and the frames6–13 artifacts opened
**read-only**. Nothing written outside `frames14/` **except the one CLAUDE.md sentence F41 §2.5 asked
for** (§5). No `config.yaml`, `orb.yaml`, checker, systemd unit, cron or order was touched; Monday's
12:30 UTC boot and the 11:27 UTC pre-boot suite are unaffected. `hod_break` stays `enabled: true,
dry_run: true`; BF and ORB stay paused exactly as the owner left them.

---

## 0. FIRST, BECAUSE IT MOVES A LIVE BOOK'S NUMBER BY MORE THAN 10 %

**BULL FLAG'S HONEST NUMBER MOVES −50 % AND ITS VAL SPLIT GOES NEGATIVE. ORB AND HOD-BREAK DO NOT
MOVE.** The shipped BF Stage-2 charges a **flat 50 bps** entry slip, spread-blind, and 30 bps on
stop-type exits only. Measured at BF's OWN detection and exit minutes (Alpaca SIP, 96.4 % coverage of
the 56 P1 trades, 96.9 % of the 896 raw regen-7 detections, 92.9 % with BOTH legs priced):

| BF P1 book, $2K normalisation | booked | **measured** | Δ | R booked | **R measured** | % of price booked → measured |
|---|---|---|---|---|---|---|
| H1-2025 (19 tr) | +$70,800 | **+$54,748** | −22.7 % | +0.756 | **+0.513** | +2.79 → +1.95 |
| H2-2025 (15 tr) | +$41,481 | **+$22,553** | −45.6 % | +1.109 | **+0.686** | +4.69 → +3.00 |
| **VAL 2026 (22 tr)** | **+$26,833** | **−$8,045** | **−130 %** | **+0.376** | **−0.115** | +1.73 → +0.14 |
| ALL (56 tr) | +$139,114 | **+$69,255** | **−50.2 %** | +0.701 | **+0.313** | +2.88 → +1.52 |

The conservative arm (minute MEAN rather than minute MEDIAN) reads **−51.7 %** and VAL **−$9,520 /
−0.130 R**. Both arms agree in sign, size and split pattern. **The entry leg alone** — the
unambiguous half, where a marketable buy at the breakout level pays the ask and nothing in the book
already charges it beyond the flat 50 bps — is worth **−$20,048 (−14.4 %)**; the exit leg **−$44,059**
and the un-charged partial leg **−$5,752**. This was predicted in PREREG §1.4 (P45.5): BF's picks
have a **median half-spread of 91.8 bps** at their own entry minute against the 50 bps charged
(1.84×), and their exit minute quotes **89.1 bps** against 30 bps on a stop and **nothing** on every
other exit.

It changes **no gate**: every P1-vs-alternative comparison is a same-cost diff, and
`entry_cost_audit/REPORT.md` §3 had already flagged the BF ramp band as ~0.2 R too high for exactly
this reason (bias toward false DEMOTION, never toward passing an under-performing book). BF is
paused. **No config change is proposed.** What it changes is the sentence: not "BF P1 is a +0.70 R
book" but **"+0.31 R, and its 2026 half is flat to negative once both legs pay a price the market
showed."**

---

## 0b. Reproduction gates — asserted in code, raising, before any cell was read

| id | gate | result |
|---|---|---|
| **G-IMPUTE** | `hod_break/score.py::build_impute` rebuilt from `breaks.csv` × `causal_filter/nbbo.csv` | **15,493 rows, 23 cells, global median 0.345 %** — `score.py` prints 0.345 % — **MATCH**; fit window **entry_m 577 (09:37) → 841 (14:01)**, exactly as F45 alleged |
| **G-X1** | the vectorised +2R bracket must reproduce F34's gross (−0.136 % TRAIN, s = 2 %) | **−0.1352 %** — **MATCH** (net differs by construction: F34 charged the imputed cost, this pass the measured one) |
| **G-TWIN** | the vectorised ORB-lock / BF-trail / BF-partial walkers vs `frames7/c7.py`'s python loops, 250 keys × 3 stops, tag and exit-minute equality asserted | **max \|diff\| x1 4.4e-16, x2 0.0, x3 0.0, x4 0.0** |
| **G-A13** | frames13's A13 overnight cell | **n = 3,310,743, gross +0.0447 %, net +0.0306 %, ex-top-5 % −0.1385 %** — all four **MATCH** |
| **G-SCALE** | X6/X7 daily-vs-intraday price scale (`daily_bars`, same vendor as the bars) | median ratio **0.9987**, **0.01 % of rows dropped** |
| **G-JOIN** | the X6/X7 join must not multiply rows | asserted; pd6's duplicate (day, control, minute) keys deduped, **288,174 → 288,174** |
| **G-AVAIL** | per-field availability, 80 % rail | F45 clocks **95.7–100 %**; BF legs 96.4/96.9 %; F44 fields **85–100 %** (V4/V5/V8 in 2026 at 85 %, the lowest) |

---

# F45 — THE COST MODEL AT THE CLOCKS THE BOOKS TRADE (6 cells + the table)

## 1.1 The minute-of-day NBBO table

210 symbol-days of the imputation table's **own** population (the HOD measured-NBBO set, 2,601
symbols × 412 TRAIN/VAL sessions), 14 declared clocks, Alpaca SIP, mean ask−bid over each minute —
the same statistic `S.IMPUTE` is built from, so any gap is a **clock** effect and not a population
effect. `IMPUTED` is the table's own prediction for the same names at that clock.

| clock | n | cov | **median % of price** | mean % | **imputed %** | **meas/imp** |
|---|---|---|---|---|---|---|
| **09:30** (the cross) | 210 | 100 % | **0.815** | 1.439 | 0.578 | **1.41×** |
| 09:31 | 207 | 98.6 % | 0.690 | 1.148 | 0.578 | 1.19× |
| **09:35 (ORB submit)** | 206 | 98.1 % | **0.525** | 0.866 | 0.578 | **0.91×** |
| 09:37 | 206 | 98.1 % | 0.485 | 0.778 | 0.578 | 0.84× |
| 09:40 | 209 | 99.5 % | 0.484 | 0.766 | 0.578 | 0.84× |
| 09:45 | 207 | 98.6 % | 0.454 | 0.705 | 0.383 | 1.19× |
| 10:00 | 208 | 99.0 % | 0.387 | 0.558 | 0.317 | 1.22× |
| 11:00 | 206 | 98.1 % | 0.278 | 0.403 | 0.283 | 0.98× |
| 12:00 | 205 | 97.6 % | 0.259 | 0.368 | 0.283 | 0.91× |
| **13:00** | 201 | 95.7 % | 0.245 | 0.353 | 0.357 | **0.69×** |
| 14:00 | 206 | 98.1 % | 0.270 | 0.362 | 0.357 | 0.76× |
| 15:00 | 207 | 98.6 % | 0.233 | 0.338 | 0.357 | 0.65× |
| **15:45 (ORB flat)** | 209 | 99.5 % | **0.174** | 0.317 | 0.357 | **0.49×** |
| **15:55 (HOD/BF flat)** | 208 | 99.0 % | **0.172** | 0.285 | 0.357 | **0.48×** |

**Independent confirmation, nothing refetched:** `frames13/openspread.csv` (500 stratified
name-nights, a different sample, a different builder) reads **09:30 0.786 %** vs this pass's 0.815 %,
**15:55 0.168 %** vs 0.172 %. 09:31 differs (0.508 vs 0.690) because openspread's sample is stratified
over the whole PIT panel while this one is the HOD universe.

**The shape, in one line: the quoted spread falls monotonically from 0.82 % at the opening cross to
0.17 % at 15:55 — a factor of 4.8 across the session — and the imputation table, fit on 09:37–14:01,
is ~1.4× too NARROW before 09:35 and ~2× too WIDE after 13:00.** Predictions: **P45.1 CONFIRMED**
(09:30 is 1.41× the table, 09:31 1.19×); **P45.2 CONFIRMED** (09:35 sits between 09:30 and 09:37 and
the curve is monotone through the morning); **P45.3 CONFIRMED** (15:45/15:55 are ~half the `1300+`
cell — every book that force-closes has been **over**-charged, the conservative direction); the
frame's falsifier (every clock inside ±25 %) **did not fire** — 6 of 14 clocks are outside it.

## 1.2 The six cells

| cell | question | result | verdict |
|---|---|---|---|
| **E1** | ORB @ 09:35 — is Stage P really a per-trade measurement? | **Yes, code-read**: `P_cost/fetch_spreads.py` pulls the SIP NBBO at the **fill instant** (last quote at or before the first trade above `range_high`) and at the **exit instant**, per trade, 97.3 % / 99.7 % coverage of 7,402 fills; Stage Q then walked the whole order life. **ORB is not priced by `S.IMPUTE` anywhere.** Its own 09:35 fills (n = 2,200, 30.5 % of the book) quote **26.0 bps median** against this pass's 52.5 bps on the HOD universe — ORB's 500K-prev-volume screen buys it names twice as liquid, which is why the transfer would have been wrong in both directions. | **CONFIRMED, residual 0** |
| **E2** | ORB @ the 15:45 flat | ORB's own exit instants at 15:45 (n = 1,033) quote **17.2 bps full / 8.6 bps half** against the flat **10 bps** `EXIT_SLIP_BPS` it charges → residual **−1.4 bps**, i.e. **−0.0028 R** per force-closed trade at ORB's median `r_pct` of 4.03 % of price. ORB is **over**-charged, by a thousandth of an R. | **NOT EXPOSED** |
| **E3** | BF entry/exit at its OWN minutes | 896 raw: full spread median **113.8 bps** (mean 169.7, p90 363.4) → half **54.4 bps** vs 50 charged (1.09×). 56 P1: full **186.9 bps** (mean 231.9, p90 424.8) → half **91.8 bps** (1.84×). **Stage-2's selection picks the WIDER half of its own raw population.** BF's detection clock: 15.7 % before 09:45, 26.4 % 09:45–10:00, **57.5 % 10:00–11:00**, 0.5 % after 11:00. Exit minute: full **178.2 bps** → half **89.1 bps** vs 30 charged on stops, 0 elsewhere. | measured, ≥ 96 % coverage |
| **E4** | the honest BF book under measured cost | **§0.** −50.2 % (primary) / −51.7 % (conservative); **VAL negative in both arms**. | **MOVES > 10 %** |
| **E5** | HOD dry @ 09:37–14:01 | 7 of 8 clocks inside ±25 % (0.84, 0.84, 1.19, 1.22, 0.98, 0.91, 0.76); **13:00 fails at 0.69×** — and it fails on the CONSERVATIVE side. | **CONFIRMED with one exception, direction conservative** |
| **E6** | HOD @ the 15:55 flat | measured half-spread **8.6 bps**; the contract charges `0.412 × half of a typical signal minute (0.333 %)` = **6.9 bps** → residual **+1.8 bps** per force-closed trade, and HOD force-closes 27.9 % / 33.0 % of trades → **+0.49 / +0.58 bps per trade = 0.0003 R**. | **NOT EXPOSED** |

## 1.3 What this does and does not change

* **ORB's honest reference ($6,085 / 21 mo, B+ book) stands unchanged** — its cost was always
  measured per trade at its own instants, and the only residual found (15:45) runs in its favour.
* **HOD-break's numbers stand** — its imputation is in-population and the 13:00 and 15:55 residuals
  are 3e-4 R.
* **BF's LEVEL moves by half.** No decision in the repo turns on it (see §0); the ramp band's known
  bias is already the conservative direction.
* **Standing rule for the programme:** `S.IMPUTE` may be used for signal minutes in 09:37–14:01 and
  **nowhere else**. Before 09:37 it under-charges by up to 1.4×; after 13:00 it over-charges by up to
  2×. `f45_minute_table.csv` is the measured replacement and is what F43 charged.

**Verdict F45: AUDIT COMPLETE — the table is wrong at both ends of the session, the error is
conservative at the close and anti-conservative at the open, ONE live book's level moves (BF, −50 %)
and it is the book whose cost was never measured at all.**

---

# F43 — THE GEOMETRY FLOOR MAP (12 cells)

## 2.1 The population and the walk

Pass 6's **arm-d detector-free controls** — a matched NON-signal name at a random eligible minute on
the PIT HOD universe (prev close ≥ $17, ADV20 ≥ 100K), **288,174 keys over 344 sessions, TEST cut
off, 100 % walked**. No detector anywhere in the construction (the pass-6 control rule). Seven
geometries, stop width a pure function of price, so every cell is a property of the **universe and
the clock** and never of a signal. Cost is **F45's measured surface** (14 clocks × 5 price bands,
linearly interpolated in minute): entry marketable; a **target** fill free; a **stop / flat / eod**
fill marketable at its own exit minute; **MOC** at ratio 0 (a single-price cross); **X6's** overnight
exit at the measured 09:31 quote.

## 2.2 The 12 cells — unconditional, % of ENTRY PRICE

| cell | geometry | s % | n | gross | cost | **net** | day-clust t | H1-25 | H2-25 | VAL | net R | ex-top-5 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| X1-2 | +2R bracket, flat 15:55 (HOD's) | 2 | 288,174 | −0.126 | 0.334 | **−0.460** | −11.0 | −0.480 | −0.467 | −0.430 | −0.230 | n/a (capped) |
| X2-2 | **ORB static lock**, flat 15:45 | 2 | 288,174 | −0.143 | 0.356 | **−0.499** | −10.7 | −0.508 | −0.499 | −0.490 | −0.250 | n/a (capped) |
| X3-2 | **BF R-trail**, flat 15:45 | 2 | 288,174 | −0.137 | 0.357 | **−0.495** | −11.5 | −0.509 | −0.501 | −0.472 | −0.247 | n/a (capped) |
| X4-2 | **BF R-trail + 50 % @ +2R partial** | 2 | 288,174 | −0.119 | 0.357 | **−0.476** | −11.3 | −0.492 | −0.484 | −0.450 | −0.238 | n/a (capped) |
| X5-2 | bare stop, no target, flat 15:55 | 2 | 288,174 | −0.147 | 0.350 | **−0.497** | −10.1 | −0.513 | −0.493 | −0.484 | −0.248 | **no** |
| X6-2 | hold to NEXT OPEN | 2 | 286,739 | −0.168 | 0.612 | **−0.780** | −6.6 | −1.179 | −0.588 | −0.573 | −0.390 | **no** |
| X7-2 | MOC (official close) | 2 | 286,744 | −0.223 | 0.199 | **−0.422** | −5.8 | −0.557 | −0.431 | −0.265 | −0.211 | **no** |
| X2-3 | ORB static lock | 3 | 288,174 | −0.213 | 0.347 | −0.560 | −9.9 | −0.596 | −0.560 | −0.522 | −0.187 | n/a |
| X4-3 | BF trail + partial | 3 | 288,174 | −0.203 | 0.347 | −0.550 | −10.2 | −0.591 | −0.556 | −0.499 | −0.183 | n/a |
| X5-3 | bare stop | 3 | 288,174 | −0.213 | 0.340 | −0.552 | −9.5 | −0.590 | −0.554 | −0.509 | −0.184 | no |
| X6-3 | hold to next open | 3 | 286,739 | −0.168 | 0.612 | −0.780 | −6.6 | −1.179 | −0.588 | −0.573 | −0.260 | no |
| X7-3 | MOC | 3 | 286,744 | −0.223 | 0.199 | −0.422 | −5.8 | −0.557 | −0.431 | −0.265 | −0.141 | no |

**0 of 12 clears the bar. Not one geometry is positive unconditionally — not net, and not even
GROSS, on either TRAIN half or on VAL.** The gross column is the load-bearing one: it carries no cost
assumption at all, and every geometry loses between **0.12 % and 0.22 % of price** on a name nobody
selected. P43.1 **CONFIRMED** (the gate). **P43.2 CONFIRMED.** P43.3 is moot — X6 is the *worst* net
cell, not the survivor. **The frame's falsifier did not fire.**

## 2.3 The diagnostics (declared, never promoted)

Net % of price at s = 2 %, by entry-hour band — the floor is worst at the open and least bad late, in
every geometry, which is F34's U1..U4 shape reproduced on six new geometries:

| geo | 09:37–10:30 | 10:30–11:30 | 11:30–13:00 | 13:00–14:01 |
|---|---|---|---|---|
| X1 | −0.622 | −0.467 | −0.404 | −0.375 |
| X2 (ORB lock) | −0.718 | −0.513 | −0.417 | −0.393 |
| X3 (BF trail) | −0.698 | −0.507 | −0.424 | −0.391 |
| X4 (BF trail+pp) | −0.656 | −0.483 | −0.414 | −0.386 |
| X5 (bare stop) | −0.733 | −0.512 | −0.411 | −0.381 |
| X6 (next open) | −1.140 | −0.843 | −0.668 | −0.554 |
| X7 (MOC) | −0.769 | −0.475 | −0.292 | −0.229 |

**The internal consistency check nobody asked for and which matters most:** X6's gross (−0.168 %) is
X7's gross (−0.223 %) **plus 0.055 %** — and F40 measured the unconditional overnight premium at
**+0.0616 % of price**. Two independent builds on two different populations agree on the overnight
leg to five thousandths of a percent. The premium is real; it is simply **smaller than the intraday
drift it is bolted to and an order of magnitude smaller than the opening minute's spread** (the
auction-free variant of X6 — exiting IN the cross at ratio 0 — is still net **−0.370 %**, H1 −0.771,
H2 −0.166, VAL −0.175, and ex-top-5 % −1.34 / −0.75 / −0.79).

## 2.4 MDE and what the null means

n = 286,739–288,174 with day clustering over 344 sessions; the **MDE is 0.012–0.018 % of price per
cell**, an order of magnitude below every measured level. **This is not a power failure — it is a
measurement.** The unconditional return of every exit geometry this programme has ever used is
negative on this universe, gross and net, in both TRAIN halves and in VAL.

**Verdict F43: NO GEOMETRY IS POSITIVE UNCONDITIONALLY. 0 of 12. The programme's null is
STRUCTURAL** — thirteen passes and 1,191 cells filtered signals on top of a floor that loses
0.12–0.22 % of price gross and 0.42–0.78 % net before any admission rule is applied. A signal on this
universe does not have to be good; it has to be good enough to pay for a geometry that starts half a
percent under water. That reframes every prior "no edge" verdict in the ledger: they were not failures
to find a signal, they were failures to clear a floor that was never priced.

---

# F44 — IS THE OVERNIGHT TAIL SELECTABLE AT 15:55 (8 cells + 8 short-side)

## 3.1 What was scored

frames13's declared **A13** cell (common, close ≥ $20, ADV$ ≥ $10M, corporate-action rail, TEST cut):
**3,310,743 name-nights**, feature join **100 %**. Eight conditioners, every one computable from the
PIT daily panel strictly at or before the 16:00 cross, each cut at its **top and bottom within-day
quintile** (a within-day cut is causal at 15:55 and cannot drift with the market). Tail label = the
top 5 % of `on_pct` inside each split. The count-matched permutation null on the tail LABEL is drawn
**exactly**, via its per-day hypergeometric equivalent (2,000 draws, per-day tail and selection counts
preserved) — O(days × draws) instead of O(rows × draws), and it is the same instrument frames11 used.

| cell | side | n | **tail rate** | null p50 | null p95 | mean % | ex5 H1-25 | ex5 H2-25 | ex5 VAL | **ex5 + measured execution, VAL** |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 day return | top | 661,124 | **0.0643** | 0.0502 | 0.0505 | +0.041 | −0.289 | −0.147 | −0.238 | −0.606 |
| V1 day return | bot | 661,124 | **0.0640** | 0.0502 | 0.0505 | +0.037 | −0.270 | −0.062 | −0.230 | −0.598 |
| **V2 day range** | **top** | 661,124 | **0.0940** | 0.0502 | 0.0505 | +0.045 | −0.380 | −0.125 | −0.281 | −0.649 |
| V2 day range | bot | 661,124 | 0.0229 | 0.0502 | 0.0505 | +0.019 | −0.168 | −0.067 | −0.139 | −0.507 |
| V3 close position | top | 661,024 | 0.0478 | 0.0501 | 0.0505 | +0.013 | −0.277 | −0.130 | −0.219 | −0.587 |
| V3 close position | bot | 661,024 | **0.0528** | 0.0501 | 0.0505 | +0.051 | −0.240 | −0.053 | −0.215 | −0.583 |
| V4 rv | top | 661,124 | **0.0683** | 0.0502 | 0.0505 | +0.046 | −0.345 | −0.166 | −0.274 | −0.642 |
| V4 rv | bot | 661,124 | 0.0456 | 0.0502 | 0.0505 | +0.023 | −0.252 | −0.064 | −0.215 | −0.583 |
| V5 dollar_frac | top | 239,756 | **0.0719** | 0.0498 | 0.0504 | +0.056 | −0.344 | −0.172 | −0.274 | −0.642 |
| V5 dollar_frac | bot | 239,756 | 0.0474 | 0.0498 | 0.0504 | +0.028 | −0.258 | −0.064 | −0.213 | −0.581 |
| V6 SPY day | top | 661,124 | **0.0526** | 0.0502 | 0.0505 | +0.031 | −0.277 | −0.099 | −0.209 | −0.577 |
| V6 SPY day | bot | 661,124 | 0.0502 | 0.0502 | 0.0505 | +0.033 | −0.255 | −0.098 | −0.227 | −0.595 |
| V7 gap | top | 661,124 | **0.0666** | 0.0502 | 0.0505 | +0.044 | −0.252 | −0.145 | −0.238 | −0.606 |
| V7 gap | bot | 661,124 | **0.0632** | 0.0502 | 0.0505 | +0.033 | −0.310 | −0.078 | −0.208 | −0.576 |
| V8 20-day-high proximity | top | 661,124 | 0.0315 | 0.0502 | 0.0505 | +0.014 | −0.201 | −0.117 | −0.184 | −0.552 |
| **V8 20-day-high proximity** | **bot** | 661,124 | **0.0818** | 0.0502 | 0.0505 | +0.052 | −0.322 | −0.080 | −0.246 | −0.614 |

**Tail rate above its own count-matched null p95: 10 of 16. Ex-top-5 % positive on both TRAIN halves
AND VAL net of the measured execution: 0 of 16.** P44.1 **CONFIRMED** (V2 nearly doubles the tail
rate, 9.40 % vs a 5.05 % null ceiling; V8-bottom 8.18 %; V5/V4/V7 6.7–7.2 %). P44.2 **CONFIRMED.**
P44.3 **CONFIRMED** — the fields that separate the top tail separate the bottom tail too (V1 top
6.43 % / bot 6.40 %; V7 6.66 / 6.32): they select **variance**, not direction, so the short is the
same lottery reflected and no borrow pricing was needed to kill it. **The frame's falsifier did not
fire.**

The **2016–2024 era check** says the same thing on 129K–551K rows per cell: every mean is positive
(+0.015 % to +0.074 %) and **every ex-top-5 % is negative** (−0.080 % to −0.200 %), the biggest
uncapped mean (V5-top +0.0735 %) belonging to the most negative capped one (−0.1703 %). The
wrapper/common conditioner is degenerate inside A13 and is reported as a diagnostic from frames13
(A11 wrapper +0.0515 %, A12 common +0.0749 %, both negative ex-top-5 %, d4 −0.1863 %).

**MDE: 0.002–0.005 % of price per cell** (n = 240K–661K). A null here is a real null.

**Verdict F44 — the sentence the frame asked for: the overnight tail is SELECTABLE IN RATE and
WORTHLESS IN MONEY — a 15:55 filter on the day's range nearly doubles the chance of catching a
top-5 % night (9.4 % vs a 5.05 % null ceiling) but every field that raises the tail rate raises the
LEFT tail by as much, so ex-top-5 % stays between −0.14 % and −0.38 % of price on every split of
every cell, before the 0.368 % round trip is even charged. Conditioning on volatility buys more
lottery tickets, not a better lottery.**

---

## 4. The adequacy review (RUNBOOK step 10)

* **Did we test what the objects actually ARE?** F45: the table was rebuilt from its own source and
  its fit window read off the data (577→841) rather than assumed; the new pull is on the table's own
  population so measured-vs-imputed is a clock effect; BF was priced at the minute of every one of
  its 896 raw detections and both legs of its 56 shipped trades, and reported under two statistics
  (minute median ≈ the fill instant per P_cost §3.2; minute mean) plus an entry-leg-only arm.
  F43: five geometries re-walked bar by bar from their shipped specs with a twin-parity gate against
  the pass-7 implementations, and two priced off the daily panel with a price-scale check. F44: the
  whole A13 panel, eight causal fields, a within-day cut, and the exact permutation null.
* **Is the cost and fill model right?** It is now MEASURED per clock (§1.1) rather than imputed, a
  target fill pays nothing, an auction pays nothing, and the overnight leg pays the measured 09:31
  quote — with the ratio-0 variant printed beside it because that choice is worth 0.41 pp.
* **Does any caveat in our own report explain a headline?** Two, both pre-committed and both stated.
  (i) F43's net levels depend on the cost surface, so the **gross** column is quoted as the finding —
  it is negative without any cost model at all. (ii) F45's BF re-pricing charges the exit leg a full
  half-spread where the modelled exit is already a bid-side level; that is why the **entry leg alone**
  (−14.4 %, unambiguous) is reported separately from the both-legs number (−50.2 %).
* **MDE.** F45: a measurement, not a test (coverage 95.7–100 %, 201–210 names per clock; the BF book
  is 56 trades and its VAL split 22, so the −50 % is a level restatement and not a significance
  claim). F43: **0.012–0.018 % of price**. F44: **0.002–0.005 % of price**.
* **Multiplicity.** 26 cells declared in `PREREG.md` before scoring; 26 scored and printed (F44's 8
  declared cells were each run on both sides, so 16 rows are shown and the 8 bottom-side rows are the
  short extension PREREG §3.3 declared, not new cells); plus the labelled diagnostics (hour band,
  stop width, the auction-free X6, the 2016–2024 era read). None selected after the fact.
  **Programme cell count 1,191 + 26 = 1,217.**

## 5. Code and documents touched

* Everything under `research/mature_method/frames14/`.
* **`CLAUDE.md`, one sentence, owner-approved direction (F41 §2.5):** the ORB section's *"the
  pipeline's selection is the edge"* now carries the corrected attribution — the static lock on a
  gap-up name at 09:35 earns **+0.109 R TRAIN / +0.066 R VAL (+0.43 % / +0.26 % of price) by
  itself**, the pick adds **+0.153 R (t +0.93) / +0.436 R (t +1.72)** on top, positive in all four
  cells and significant in none, and **the ramp band stays on the LEVEL** because both of its sides
  share the geometry and the control cancels inside the gate. Nothing else in CLAUDE.md, nothing in
  any config.

## 6. VERDICT

**F45 AUDIT COMPLETE — the imputation table is 1.4× too narrow at the open and ~2× too wide after
13:00; ORB (−0.003 R) and HOD-break (+0.0003 R) are NOT exposed; BULL FLAG'S HONEST NUMBER MOVES
−50.2 % AND ITS 2026 HALF GOES NEGATIVE. · F43 NO GEOMETRY IS POSITIVE UNCONDITIONALLY, 0 of 12,
gross as well as net, both halves and VAL — the programme's null is STRUCTURAL. · F44 THE OVERNIGHT
TAIL IS SELECTABLE IN RATE AND WORTHLESS IN MONEY, 0 of 16 on the only clause that counted.**
**0 of 26 cells clears a bar. No config, engine, checker, cron or order changed.**

Phrased as CLAUDE.md requires: **no exit geometry with a positive unconditional return was detectable
in THIS universe (the PIT HOD liquid universe, 288,174 detector-free keys), at THESE seven geometries
and three stop widths, over 2025-01 → 2026-05, at a smallest detectable effect of 0.012–0.018 % of
price** — and the reason is not power, it is the level: the floor is negative before any cost is
charged.
