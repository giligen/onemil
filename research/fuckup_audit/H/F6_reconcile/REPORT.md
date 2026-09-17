# F6-PDR reconciliation — implementation A vs implementation B, and the engine's own book (E)

Stage H, 2026-09-17. Everything written is under `research/fuckup_audit/H/F6_reconcile/`. Everything outside it was
read only (`data/cache.db`, `research/bf_zero/bars_sip.db` via `file:...?mode=ro`). No config, service, cache or
order was touched. One `nice -n 10` process at a time, `ulimit -v` on every run.

**TEST is read here.** That is allowed: both A (`H/F6/f6_pdr_book.md`) and B (`H/F6_rebuild/REPORT.md`) have already
read TEST for this same book, and this document reconciles their two readings rather than selecting on it. No new
selection was made on TEST; the engine-convention book E is read straight out of `trading/red_to_green.py`, which was
committed before this run, and is reported on all three splits whatever it says.

---

## 0. The answer in six lines

1. **A's TEST book is one NASDAQ test symbol.** A's booked hold TEST is **+39.00 R** (n=376, +0.104 R/trade).
   **ZVZZT** — NASDAQ's test ticker — contributes **+50.80 R of it in 4 trades**; `ZVZZT 2026-08-18` alone **+46.76 R**
   (a $24 -> $126 tape on 62,852 synthetic shares). Remove all `Z?ZZT` symbols and A's TEST is **-11.80 R,
   -0.032 R/trade** — the same sign and size as B's **-15.36 R, -0.042 R/trade**. A's "2026-08 +69.3 R" month IS that
   one print. ZVZZT has 0 rows in `daily_bars` and 0 in `intraday_bars_1min`: the live engine could never have traded
   it. It enters only through the study universe (`research/bf_zero/universe.csv`, 391 test-ticker symbol-days).
2. **The rest of the A/B gap is ONE coding defect in A.** `B/build_candidates4.py:99` sets
   `os.environ['BFZ_SLIP'] = '0.0'`, so `bf_zero/build_candidates.fam_r2g` breaks at `prev_close x (1+0)` — **A's level
   is the prior close exactly** (7,137/7,137 rows have `level == prev_close` to 1e-9) — while A's own declared
   docstring, `H/F6/ENGINE_SPEC.md` and `trading/red_to_green.py` all say **prior close x 1.003**. Only 41% of A's
   signal bars would also have cleared the declared level on the same bar. This single switch explains **308 of A's
   339 TEST-only candidates (+56.05 R booked)**, **309 of B's 395 TEST-only candidates (-0.06 R booked)** and
   **54 of the 56 shared TEST trades whose fields differ**.
3. **Bar-source precedence is a non-issue** (B's suspect 1): on the 1,993 TEST F6 candidate keys `cache.db` and
   `bars_sip.db` are **disjoint** — 1,458 cache-only, 535 sip-only, **0 in both**. Precedence cannot change a bar.
   **Prior-day OHLC source is also dead**: on all **644,580** universe symbol-days universe.csv and the Databento panel
   agree on prior high/low/close to 1e-6, with **0** PDR>=8 disagreements. **"Next bar"** and **"14:01"** are not A/B
   differences either — A uses the next EXISTING bar and applies 14:01 to the FILL bar, exactly as B does.
4. **Neither study implements the shipped rule.** `red_to_green.detect` **skips** a bar that fails the range floor and
   keeps scanning; A and B both take the FIRST level break and kill the day if that bar's floor fails. That one
   convention is worth **+0.09 R/trade on TRAIN and +0.23 R/trade on VAL** — i.e. the whole of both studies' positive
   pre-TEST result (section 6).
5. **Under the live engine's conventions (implementation E) the book is negative on every split and every exit**:
   hold TRAIN **-0.027** / VAL **-0.012** / TEST **-0.102** R per trade; 2R -0.039 / -0.061 / -0.098; partial
   -0.038 / -0.050 / -0.081. 13,541 candidates, ~30 booked trades a week. Every cell gets **more** negative with the
   top 1% or 5% removed and with winners capped at +3R.
6. **What survives the reconciliation**: A's TEST number is an artefact twice over; B's TEST number is right but is
   the number for a rule the engine does not implement; the engine's own rule loses on all three splits.

---

## 1. The join — booked HOLD trades, on (day, symbol)

`r2_join.py` -> `join_hold.csv`, `join_hold_shared.csv`, `r2_join.md`.

| split | A n | B n | A-only | B-only | both | A total R | B total R | A-only R | B-only R | both (A) | both (B) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN | 1122 | 1114 | 389 | 381 | 733 | +87.40 | +69.79 | +16.18 | +5.34 | +71.22 | +64.45 |
| VAL | 519 | 531 | 208 | 220 | 311 | +130.04 | +109.73 | +38.36 | +24.29 | +91.68 | +85.43 |
| **TEST** | **376** | **366** | **159** | **149** | **217** | **+39.00** | **-15.36** | **+49.79** | **-4.78** | **-10.79** | **-10.59** |

On TEST the 217 shared trades agree almost exactly (-10.79 vs -10.59 R). **The whole +54.4 R A-B gap on TEST lives in
the disjoint sets**, and +50.8 R of A's side of it is ZVZZT.

### Field differences on the 1,261 shared trades (all splits)

| field (A - B) | n differing | % | mean abs | p50 | min | max |
|---|---|---|---|---|---|---|
| signal minute | 394 | 31.2 | 2.38 | 0 | -140 | 0 |
| entry minute | 394 | 31.2 | 2.37 | 0 | -140 | 0 |
| exit minute | 4 | 0.3 | 0.66 | 0 | -379 | 0 |
| entry price | 381 | 30.2 | 0.048 | 0 | -4.00 | +0.94 |
| stop | 6 | 0.5 | 0.0002 | 0 | 0 | +0.13 |
| gross R | 387 | 30.7 | 0.044 | 0 | -6.25 | +1.49 |
| net R | 520 | 41.2 | 0.045 | 0 | -6.45 | +1.48 |
| exit reason | 2 | 0.2 | — | — | — | — |

`d_sig_m <= 0` on every row: A always signals at or before B, exactly as a lower break level implies. Exit-reason
cross-tab (shared): eod/eod 854, stop/stop 405, A-stop/B-eod 2, A-eod/B-stop 0. Histogram of `d_entry_m`: 867 at 0,
149 at -1, 59 at -2, 40 at -3, tail out to -140. The 520 rows with a net-R difference against 387 with a gross
difference are the cost-band edge (A keys the cost curve on the LEVEL price and the SIGNAL minute with `<=` band
edges, B on the ENTRY price and the FILL minute with `<`) — a <=0.01 R effect.

Per split, shared trades that are identical in entry, stop, exit minute and net R: TRAIN 414/733, VAL 187/311,
TEST 140/217.

### The 2R exit (A's declared secondary), same join

| split | A n | B n | A-only | B-only | both | A total R | B total R | A-only R | B-only R |
|---|---|---|---|---|---|---|---|---|---|
| TRAIN | 1217 | 1206 | 427 | 416 | 790 | +70.70 | +87.59 | -3.24 | +24.87 |
| VAL | 593 | 600 | 245 | 252 | 348 | +65.38 | +45.83 | +19.66 | +10.49 |
| TEST | 420 | 408 | 188 | 176 | 232 | +1.42 | -8.97 | +10.60 | -0.92 |

Test tickers again: A's 2R TEST +1.42 R contains +4.18 R of `Z?ZZT`; ex-test-ticker A's 2R TEST is **-0.0066 R/trade**
against B's **-0.0220**. B's 2R book has zero test-ticker trades on VAL and TEST.

### Test-ticker impact, booked hold

| impl | split | n | total R | mean R | `Z?ZZT` n | `Z?ZZT` R | ex-test total R | ex-test mean R |
|---|---|---|---|---|---|---|---|---|
| A | TRAIN | 1122 | +87.40 | +0.0779 | 7 | +2.61 | +84.79 | +0.0760 |
| A | VAL | 519 | +130.04 | +0.2505 | 2 | +1.32 | +128.71 | +0.2490 |
| **A** | **TEST** | **376** | **+39.00** | **+0.1037** | **4** | **+50.80** | **-11.80** | **-0.0317** |
| B | TRAIN | 1114 | +69.79 | +0.0626 | 2 | +0.79 | +69.00 | +0.0620 |
| B | VAL | 531 | +109.73 | +0.2066 | 0 | 0 | +109.73 | +0.2066 |
| B | TEST | 366 | -15.36 | -0.0420 | 0 | 0 | -15.36 | -0.0420 |

---

## 2. Cause table

`r5_causes.py` (734 TEST candidate-set divergences, each re-derived from the raw bars under both conventions),
`r6_stores.py`, `r7_shared.py`. The parameterised pipeline (`pipeline.py`) **reproduces the divergence in 734/734
cases** (A-ok/B-not for all 339 A-only rows, B-ok/A-not for all 395 B-only rows) and reproduces both sides' entry,
stop, exit minute and exit type **exactly on 56/56** of the shared TEST trades that differ in any field.

### 2a. Candidate sets, pre-book (`cand_join.csv`) — book-slot effects removed

| split | A candidates | B candidates | A-only | B-only | both |
|---|---|---|---|---|---|
| TRAIN | 3545 | 3816 | 718 | 989 | 2827 |
| VAL | 1994 | 2136 | 405 | 547 | 1589 |
| TEST | 1598 | 1654 | 339 | 395 | 1259 |

### 2b. Cause -> trades -> net R (TEST). "Cause" = the set of single switch flips that change the verdict.

| cause | side | candidates | of which booked | booked net R |
|---|---|---|---|---|
| **level buffer** (A `prev_close`, B `prev_close x 1.003`) | A-only | 308 | 75 | **+56.05** |
| **level buffer** | B-only | 309 | 76 | -0.06 |
| level buffer OR floor denominator (either alone flips it) | B-only | 20 | 1 | +3.15 |
| **range-floor denominator** (A `/ 09:30 open`, B `/ running low`) | B-only | 32 | 11 | -8.08 |
| **B's day-level `universe open >= 5` prefilter** | A-only | 19 | 5 | -0.49 |
| day-open prefilter OR the $5 gate's basis (A level, B entry) | A-only | 5 | 2 | -2.13 |
| level buffer OR day-open prefilter | A-only | 5 | 2 | -0.65 |
| level buffer / $5 basis / day-open | A-only | 2 | 1 | -0.12 |
| no single flip (a combination) | B-only | 34 | 9 | -1.20 |
| **bar-source precedence** | — | **0** | 0 | 0.00 |
| **prior-day OHLC source** | — | **0** | 0 | 0.00 |
| **stop window** (excl / incl the signal bar) | — | 0 candidates; **2 of 56** shared-trade field differences | | |

First-gate cross-tab (A's terminating gate x B's, 734 TEST divergences): A `cap` -> B `taken` 174 · A `floor` -> B
`taken` 217 · A `r_small` -> B `taken` 3 · A `few_bars` -> B `taken` 1 · A `taken` -> B `cap` 165 · A `taken` -> B
`no_signal` 99 · A `taken` -> B `late` 43 · A `taken` -> B `day_open5` 30 · A `taken` -> B `no_next` 1 · A `taken` ->
B `r_small` 1. The `cap` rows are the second-order effect of the level buffer: A's cap is `prev_close x 1.006`, B's is
`prev_close x 1.009`, so each side rejects fills the other accepts. The `floor` -> `taken` rows are the floor
denominator plus the level buffer moving the signal bar.

### 2c. The two stores (`store_compare_test.csv`)

1,993 TEST F6 candidate keys: **1,458 served only by `data/cache.db`, 535 only by `research/bf_zero/bars_sip.db`,
0 by both, 0 by neither.** `bars_sip.db` is the Alpaca-SIP re-fetch of keys `cache.db` lacks
(`bf_zero/refetch_thin_tape.py`, wired as `BFZ_SIP_STORE`), so the union is ONE tape and precedence is unobservable.
B's remark that "75% of booked trades come from cache.db, so this book is mostly NOT on the SIP tape" is a misreading
of the store layout: both stores are the Alpaca SIP tape.

### 2d. The remaining named suspects

- **Red open**: A takes `o[0]` of its RTH array (first bar with minute >= 570), B takes the 09:30 bar's open, the
  engine takes `o[0]` of the first RTH bar. They coincide except on the 3,040 of 140,144 prefiltered symbol-days
  (2.2%) whose tape does not start at 09:30. The daily file disagrees with the 09:30 bar about the red open on 790 of
  140,144 (0.56%) — and **neither A nor B uses the daily file for it**. Not a cause of any TEST divergence.
- **"Next bar"**: both use the next existing bar. It is not the next clock minute for 14.4% of B's candidates and
  18.7% of E's — a real obtainability question (section 3), not an A/B difference.
- **14:01**: both apply it to the FILL bar. The engine applies 14:00 to the SIGNAL bar (sections 3, 6).
- **Signal from 09:31**: both start at index 1 of the RTH array. B measured that allowing the 09:30 bar to be the
  signal never ADDS a trade (the floor has no prior bars), so the switch only removes 622 candidates.

---

## 3. Implementation E — the book under the live engine's conventions

`r8_engine_book.py` -> `e_cands.csv`, `e_trades_{hold,r2,partial}_{TRAIN,VAL,TEST}.csv`, `r8_engine_book.md`;
sensitivities `r9_e_sensitivity.py` -> `e_stats.csv`, `r9_e_sensitivity.md`.

E is read out of the shipped code, not out of either study:

| decision | live source | E |
|---|---|---|
| prior day | `daily_bars` prior row | prior row of the Databento daily panel (identical to universe.csv on all 644,580 keys) |
| precondition | `red_to_green.eligible(o[0], prior_close, pdr)` | `o[0]` of the first RTH bar < prior close, PDR >= 8 |
| level | `level_for` = prior close x (1 + `level_buffer` 0.003) | same |
| floor | `(run_hi[i-1] - run_lo[i-1]) / run_lo[i-1] >= 5%` | same — denominator is the **running low** |
| scan | `detect`: a bar failing the floor is **skipped** (`continue`); scanning continues | same |
| cut | `if m[i] > last_entry_minute (840): return None` — on the **signal** bar | same |
| stop | `run_lo[i]` — lowest low 09:30 **through** the signal bar | same |
| fill | `hod_break.entry_fill`: next printed bar's open, iff <= level x 1.006 | same |
| floors | engine `min_price` 5 on the entry, `r_ok` >= 1% of entry | same |
| exits | `hod_break.walk_exit` from the bar AFTER the fill bar | same; partial = half at +2R on a close, stop -> entry |
| book | `trading.hod_break.run_book(rows, 12, 4)` | the same function, imported |
| cost | contract (c) | `half = 0.5*(cc_bps/100)/max(r_pct,0.05)`; entry 0.25*half; stop 0.875 / eod 0.412 / target 0.875 |

Population: the 273,488 universe symbol-days with a prior panel row and PDR >= 8. Funnel: not red 125,602 · no tape
35,341 · no signal by 14:00 (`late` 51,941 + `no_signal` 113) · above the cap 35,759 · entry < $5 10,834 · R < 1% 343 ·
fewer than 2 bars 13 · no next bar 1 · **taken 13,541**.

### E, booked (12/day, 4 concurrent)

| exit | split | n | tr/wk | mean net R | t | WR | weekly R | weeks green | worst week | ex-top-1% | ex-top-5% | cap +3R | total R | max DD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hold | TRAIN | 1500 | 28.3 | **-0.0268** | -0.64 | 38.5% | -0.76 | 0.51 | -20.8 | -0.1234 | -0.2879 | -0.1252 | -40.2 | -60.8 |
| hold | VAL | 664 | 30.2 | **-0.0117** | -0.17 | 38.6% | -0.35 | 0.59 | -26.7 | -0.1349 | -0.3060 | -0.1433 | -7.8 | -37.0 |
| hold | TEST | 427 | 30.5 | **-0.1017** | -1.58 | 37.7% | -3.10 | 0.43 | -18.1 | -0.1625 | -0.3172 | -0.1562 | -43.4 | -55.5 |
| 2R | TRAIN | 1713 | 32.3 | **-0.0389** | -1.43 | 41.7% | -1.26 | 0.43 | -14.8 | -0.0591 | -0.1449 | -0.0389 | -66.6 | -80.1 |
| 2R | VAL | 761 | 34.6 | **-0.0606** | -1.46 | 42.4% | -2.10 | 0.55 | -25.5 | -0.0823 | -0.1671 | -0.0606 | -46.1 | -60.1 |
| 2R | TEST | 491 | 35.1 | **-0.0979** | -1.99 | 39.5% | -3.43 | 0.43 | -16.3 | -0.1193 | -0.2084 | -0.0979 | -48.1 | -48.7 |
| partial | TRAIN | 1532 | 28.9 | **-0.0381** | -1.18 | 42.0% | -1.10 | 0.45 | -15.0 | -0.0951 | -0.2166 | -0.0698 | -58.4 | -76.8 |
| partial | VAL | 679 | 30.9 | **-0.0501** | -0.97 | 41.5% | -1.55 | 0.55 | -24.4 | -0.1178 | -0.2428 | -0.0932 | -34.0 | -45.1 |
| partial | TEST | 434 | 31.0 | **-0.0805** | -1.44 | 40.6% | -2.50 | 0.43 | -17.6 | -0.1196 | -0.2425 | -0.0931 | -35.0 | -42.6 |

**Tail tests.** Every cell is negative BEFORE the tail is touched, and each becomes MORE negative with the top 1% or
5% removed and with winners capped at +3R (`ex1`, `ex5`, `cap3` above). There is no tail carrying a hidden edge — the
book is negative at the centre.

### Monthly net R (E)

hold: 2025-01 -11.6 (129) · 02 -16.0 (114) · 03 -11.1 (127) · 04 +7.3 (133) · 05 +19.4 (122) · 06 -29.2 (119) ·
07 +10.5 (126) · 08 +2.9 (119) · 09 -0.1 (122) · 10 -23.8 (152) · 11 +3.1 (113) · 12 +8.3 (124) · 2026-01 -20.5 (131) ·
02 +6.7 (122) · 03 -2.9 (142) · 04 +28.1 (132) · 05 -19.2 (137) · 06 -10.2 (134) · 07 -25.1 (141) · 08 -14.1 (129) ·
09 +6.0 (23). **Months green 9/21.**

2R: 2025-01 -31.7 · 02 -8.0 · 03 -17.3 · 04 -3.0 · 05 +11.5 · 06 -8.4 · 07 +15.9 · 08 -6.1 · 09 -2.9 · 10 -23.9 ·
11 -1.8 · 12 +9.2 · 2026-01 -13.2 · 02 +13.6 · 03 -20.1 · 04 -7.1 · 05 -19.3 · 06 -17.0 · 07 -22.5 · 08 -8.2 ·
09 -0.5. **Months green 4/21.**

partial: 2025-01 -27.6 · 02 -10.7 · 03 -14.5 · 04 +1.6 · 05 +15.4 · 06 -19.4 · 07 +12.7 · 08 -0.2 · 09 -1.9 ·
10 -24.2 · 11 -2.3 · 12 +12.6 · 2026-01 -13.7 · 02 +8.2 · 03 -17.1 · 04 +9.5 · 05 -20.9 · 06 -3.6 · 07 -20.3 ·
08 -11.7 · 09 +0.7. **Months green 7/21.**

### E sensitivities

| variant | exit | TRAIN | VAL | TEST |
|---|---|---|---|---|
| E (all) | hold | -0.0268 | -0.0117 | -0.1017 |
| E ex test-tickers (22 of 13,541 candidates) | hold | -0.0284 | -0.0123 | -0.1017 |
| E ex test-tickers, fill must be the next CLOCK minute | hold | -0.0004 | +0.0036 | -0.1435 |
| E (all) | 2R | -0.0389 | -0.0606 | -0.0979 |
| E ex test-tickers, next clock minute | 2R | -0.0041 | -0.0244 | -0.1062 |
| E (all) | partial | -0.0381 | -0.0501 | -0.0805 |
| E ex test-tickers, next clock minute | partial | -0.0056 | -0.0256 | -0.1188 |

- **Test tickers** move E's book by <= 0.002 R/trade. E does not depend on them; A's TEST book does.
- **Fill on the next clock minute only** — the honest live constraint, since the HOD engine's order lives ~20 s and is
  cancelled when the symbol's next bar closes: 2,536 of 13,541 candidates (18.7%) fill on a later print. Removing them
  takes TRAIN and VAL to flat and TEST further down. No exit turns positive.
- **Obtainability**: E's entry is a bar's OPEN, so every entry fill is inside its bar by construction. E's exit rule is
  byte-for-byte `hod_break.walk_exit`, the rule A and B also used, so B's audit carries over: 37.3% of stop exits book
  `min(stop, open) x 0.999`, a price BELOW the exit bar's low (conservative for a long book); target legs fill at the
  target on bars that gapped entirely above it (also conservative); no fill was ever above what its bar offered.

---

## 4. Which of A's and B's numbers were artefacts of which choice

| claim | verdict |
|---|---|
| A: "TEST hold +0.104 R/trade, 2026-08 +69.3 R, 17/21 months green" | **artefact.** +50.80 R of the +39.00 R total is `Z?ZZT`, a NASDAQ test symbol with 0 rows in the live universe tables. Ex-test-ticker: **-0.032 R/trade, -11.80 R**, i.e. B's answer. |
| A: "signal = first bar whose high reaches prev close x 1.003" | **not what A ran.** `BFZ_SLIP=0` -> the level is the prior close exactly on 7,137/7,137 rows; 59% of A's signal bars never cleared the declared level. Largest single A/B cause. |
| A: "range_so_far_pct >= 5" | **a different rule from the engine's.** A divides the pre-signal range by the **09:30 open**; `red_to_green.detect` and B divide by the **running low**. 52 TEST candidates hang on it. |
| A + B: "first level break, then test the floor (else the day is dead)" | **neither matches the shipped spec**, which skips floor-failing bars and keeps scanning. Worth +0.09 R/trade on TRAIN and +0.23 R on VAL (section 6) — i.e. all of both studies' positive pre-TEST result. |
| A: 51.8% of its fills are BELOW the break level | mechanically legal (a capped limit fills at the next open) but it means half of A's "breaks" were not confirmed by the fill. |
| B: "TEST negative on all three exits (-0.042 / -0.022 / -0.016)" | **stands**, and A agrees once test tickers are removed (hold -0.032, 2R -0.007). |
| B: suspect (1) bar-source precedence | **dead** — the two stores are disjoint on every TEST candidate key. |
| B: "75% of booked trades come from cache.db, so this book is mostly not on the SIP tape" | **wrong reading** — `bars_sip.db` is an Alpaca-SIP top-up for keys `cache.db` lacks; both ARE the SIP tape. |
| B: suspect, prior-day OHLC universe.csv vs the panel | **dead** — identical on all 644,580 keys, 0 PDR>=8 disagreements. |
| B: suspects (3) "next bar" and (4) "14:01 on fill vs signal" | **not A/B differences** — A does the same in both cases. Both ARE study-vs-engine differences. |
| B: stop window incl/excl the signal bar | real but negligible: 6 of 1,261 shared trades, 2 of 56 sampled field differences, <= 0.03 R on any split. |
| B: "TEST ends 2026-09-04 because the universe file does" | **confirmed** for A, B and E. |
| A: TRAIN +0.078 / VAL +0.251 · B: TRAIN +0.063 / VAL +0.207 | both positive pre-TEST, negative on TEST, from two independent code paths — and both are products of the non-engine scan rule. |

---

## 5. Cells looked at

No search was run here. Cells: the A/B join (2 exits x 3 splits), the cause table (7 switches x 734 TEST keys plus
7 x 56 shared trades), E (3 exits x 3 splits x 3 variants = 27), the B->E ladder (4 variants x 3 exits x 3 splits =
36). Everything is a re-scoring of one already-declared book; no rule was selected on any split here.

**Phrasing.** No edge was detectable for F6 red-to-green under the engine's own conventions on THIS universe (the
point-in-time >=5%-range day list, PDR >= 8, price >= $5), at THIS horizon (intraday, flat 15:55), at THIS book size
(12/day, 4 concurrent), over 2025-01-02..2026-09-04, at THIS cost (contract c). With 1,500 TRAIN trades and a
per-trade SD near 1 R the smallest mean effect this test could have resolved at t = 2 is about **+0.05 R/trade**;
E's TRAIN estimate is -0.027 +/- 0.042, so effects between 0 and +0.05 R remain undetectable here.

---

## 6. The B -> E ladder — which convention moves the book

`r10_ladder.py` -> `ladder_*.csv`, `ladder_stats.csv`, `r10_ladder.md`. One tape walk over the same 273,488
symbol-days, four configurations each one convention apart.

| step | what changes | pre-book candidates |
|---|---|---|
| **L0** | B's rule set, on E's plumbing (panel prior day, cache-first bars) | 7,606 |
| **L1** | L0 minus B's day-level `universe open >= 5` prefilter | 7,743 |
| **L2** | L1 with the engine's cut: 14:00 on the SIGNAL bar instead of 14:01 on the fill bar | 7,755 |
| **L3 = E** | L2 with the engine's SCAN: floor-failing bars are skipped, not fatal | **13,541** |

Mean net R per trade, booked:

| variant | hold TRAIN | hold VAL | hold TEST | 2R TRAIN | 2R VAL | 2R TEST | partial TRAIN | partial VAL | partial TEST |
|---|---|---|---|---|---|---|---|---|---|
| L0 (B's rules) | +0.0623 | +0.2065 | -0.0421 | +0.0725 | +0.0762 | -0.0221 | +0.0780 | +0.1146 | -0.0165 |
| L1 | +0.0668 | +0.2158 | -0.0461 | +0.0762 | +0.0886 | -0.0242 | +0.0815 | +0.1282 | -0.0204 |
| L2 | +0.0667 | +0.2154 | -0.0461 | +0.0761 | +0.0884 | -0.0242 | +0.0815 | +0.1279 | -0.0204 |
| **L3 = E** | **-0.0268** | **-0.0117** | **-0.1017** | **-0.0389** | **-0.0606** | **-0.0979** | **-0.0381** | **-0.0501** | **-0.0805** |

**L0 reproduces B to the third decimal on every cell** — n 1114/531/366 vs B's 1114/531/366, hold means +0.0623 /
+0.2065 / -0.0421 vs B's +0.0626 / +0.2066 / -0.0420 (the residual is the cost-band edge convention). That is a
THIRD independent implementation agreeing with B, from a different code path and a different prior-day source.

**L2 -> L3 is the whole story.** The engine's scan rule adds 5,786 candidates (+75%) and takes hold TRAIN from
+0.067 to -0.027 and VAL from +0.215 to -0.012. The trades A and B never looked at — days whose FIRST level break
came before the 5% range floor was satisfied, and which the engine would enter on the NEXT break — are decisively
worse than the ones they did look at. The declared book's positive TRAIN/VAL result is a property of a scan rule that
`trading/red_to_green.py` does not implement.

**Consequence for the ship path — this needs the owner's eyes.** `config.yaml` line 297 already carries a
`red_to_green:` block, `enabled: true` / `dry_run: true`, armed for the 2026-09-18 12:30 UTC boot, and its comment
quotes A's numbers verbatim as the justification: "17/21 months green, TRAIN +0.058 / VAL +0.110 / TEST +0.003"
(the 2R book) and "TRAIN +0.078 / VAL +0.251 / TEST +0.104, tail-heavy" (hold). Every one of those six numbers is A's,
and A's TEST figures are the ZVZZT artefact plus the missing level buffer. **Nothing is at risk right now** —
`dry_run: true` places zero orders, and BF/ORB are paused — but:

- the config's `level_buffer: 0.003`, `range_floor_pct: 5.0` and `last_entry_minute: 840` mean the LIVE engine will run
  implementation **E**, not the book those numbers describe. The dry run will emit roughly **twice** the signals the
  study booked (13,541 vs 7,606 candidates; ~30 vs ~23 booked trades a week), and its expected result is
  **-0.03 to -0.10 R per trade**, not +0.06 to +0.25.
- `H/F6/ENGINE_SPEC.md` step 5 ("dry run for 5 sessions -> live at risk_usd 100 on the owner's word") is not supported
  by any of the three implementations: A's TEST is a test ticker, B's TEST is negative, and the engine's own rule is
  negative on all three splits at every exit.
- Before `dry_run: false`, one of two things must happen: either `red_to_green.detect` is changed to the studies'
  first-break-then-floor rule and that rule is re-validated (it is **-0.042 R/trade on TEST** even then), or the book
  does not go live. The config comment should be corrected either way — it currently states numbers no implementation
  reproduces.
- Independent of this book, the study universe (`research/bf_zero/universe.csv`) contains **391 NASDAQ test-ticker
  symbol-days** (`ZVZZT` 389, `ZJZZT` 1, `ZXZZT` 1). Every study built on it should exclude `^Z[A-Z]ZZT$`; they are
  absent from `daily_bars`/`intraday_bars_1min`, so live can never trade them and a backtest that does is inflated.

---

## 7. Files

`pipeline.py` (the parameterised spec, one function, every convention a switch) · `r1_extract_A.py` ->
`a_signals_all.csv`, `a_cands.csv` · `r2_join.py` -> `join_hold.csv`, `join_hold_shared.csv`, `r2_join.md` ·
`r3_cand_join.py` -> `cand_join.csv`, `r3_cand_join.md` · `r4_prev_table.py` -> `prev_table.csv` ·
`r5_causes.py` -> `causes_test.csv`, `r5_causes.md` · `r6_stores.py` -> `store_compare_test.csv`, `r6_stores.md` ·
`r7_shared.py` -> `shared_sample_test.csv`, `r7_shared.md` · `r8_engine_book.py` -> `e_cands.csv`,
`e_trades_*.csv`, `r8_engine_book.md`, `r8.log` · `r9_e_sensitivity.py` -> `e_stats.csv`, `e_trades_extick_*.csv`,
`r9_e_sensitivity.md` · `r10_ladder.py` -> `ladder_*.csv`, `ladder_stats.csv`, `r10_ladder.md`, `r10.log`.
