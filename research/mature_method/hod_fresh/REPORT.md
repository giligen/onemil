# hod_fresh — the fresh-high admission and the re-shaped stop — REPORT (2026-09-19)

The LAST pre-registered pass on HOD-break. Cells exactly as declared in `PREREG.md`, **committed
`b1e5470` before any cell was scored**. Artifacts: `pass3.py` -> `sig3.csv` (150,998 rows, 327
sessions, TRAIN+VAL only) · `score4.py` -> `score4.log`, `cells.csv`, `nulls.csv` · `supp.py` ->
`supp.log`. One python process at a time, `nice -n 10`, `ulimit -v 3000000`; `cache.db`,
`bars_sip.db`, `daily_bars` opened **read-only**. No config, `orb.yaml`, systemd unit, cron, order or
cache was written. The dry run was not touched. **TEST was never opened** (`FREEZE.md`; `pass3.py`
never read a TEST-dated bar).

---

## VERDICT — **STAY DRY**, and the HOD-break line is **CLOSED**

*The lead was chased exactly as the anatomy asked and it did not hold. Made an **admission** instead
of a filter, the fresh high trades **15-21 times a week instead of 5.6** — the frequency problem is
solved — and its VAL gross falls from the anatomy's **+0.207 R on 219 trades to +0.055 R on 474**.
The direction is real but small and it is **not what it looked like**: the fresh-high rungs put
**78-90 % of their trades on >=10 %-range days** against the base's 64 % and the control's 55 %, i.e.
`consol_bars` is substantially a mover-day marker, which is the two-cohort diagnostic doing its job.
Not one of the four fresh rungs is same-signed positive in H1-2025, H2-2025 and VAL.*

*Re-shaping the stop did exactly what the mechanism predicted and it was **not enough**. Widening to
`entry - 1.0 x ATR20d` takes the median R from **1.72 % to 5.96 % of price and the measured cost from
0.069 to 0.043 R a trade — a 38 % cut in cost/R, the largest move the cost wall has made in the
programme** — and the book still reads gross **+0.019 / +0.022**, net **-0.024 / -0.022**. **The cost
wall moved and the gross did not move with it.***

***And the pass found that the wall was never where the programme said it was.*** *The
**+0.2151 R** every prior report quoted as the break-even is the cost on the **un-gated** population.
Inside the cost-gated book that ships it is **0.061 R (TRAIN) / 0.065 R (VAL)** — the 100 bps and
15 %-of-R gates remove the expensive trades before the book sees them. The bar this programme has
been holding books to was **3.2x the book's actual break-even** (§2). Under the correct break-even
**one cell is above water on both splits**: `C1` = old-HOD admission x last-5-bar stop x SPY's first
five minutes up — TRAIN **+0.100 gross / +0.033 net / +$2,391**, VAL **+0.123 / +0.050 / +$1,827**, at
13.8 / 16.0 trades a week, **same-signed positive in H1, H2 and VAL** (+0.123 / +0.082 / +0.123) —
the first cell in 799 to do that. **It is still a no.** Its day-clustered t is **+0.54 / +0.66**; its
green-week share sits **below its own count-matched null mean on both splits** (43.4 vs 49.2;
47.8 vs 50.0); **one week carries 85 % of the TRAIN year and 66 % of the VAL period**; and the net
goes **negative ex-top-5 % on both splits** (+0.033 -> -0.071, +0.050 -> -0.056). That is the lottery
ticket this owner has already rejected once, and no pre-committed bar is moved after the fact.*

**0 of 15 declared cells pass G1. 0 pass the pre-committed ship bar. 0 sit above their null band on
both splits.**

---

## 1. Reproduction gate — EXACT on all three rows

| id | what | this pass | reference | verdict |
|---|---|---|---|---|
| **R1** | `B0` shipped | TRAIN 1,688 · 31.8/wk · -0.027 · -0.088 · 41.5 % · **-$14,835** · VAL 820 · 35.7 · +0.016 · -0.050 · 43.5 % · **-$4,128** | identical | **MATCH** (dn 0, d$ 0) |
| **R2** | `P13` = B0 ^ `consol_bars < 8` (the lead) | TRAIN 297 · 5.6/wk · -0.038 · -0.104 · 35.8 % · **-$3,099** · VAL 219 · 9.5 · **+0.207** · +0.136 · **52.2 %** · **+$2,977** | identical | **MATCH** |
| **R3** | `pass3.py` base rung x last-5-bar stop = `B2` | TRAIN 1,622 · 30.6 · -0.039 · -0.107 · 32.1 % · **-$17,346** · VAL 706 · 30.7 · +0.083 · +0.013 · 43.5 % · **+$893** | identical | **MATCH** |

R3 is the independent-rebuild check and it is exact: a **new bar pass**, written against a different
admission loop with a vectorised exit walk, reproduces `hod_filter_stack/pass2.py`'s book to the
dollar. On the shared first-break population (4 sample sessions, 870 signals) the two passes agree on
**870 of 870 entry minutes, max |d stop| = 0.00, max |d rr| = 0.00**.

**Availability audit.** `consol_bars` **100.0 % covered**, missingness 0.0 % on winners and losers,
computed from bars **strictly before** the break bar (`hod_losers/walk.py`'s definition verbatim:
consecutive bars back from `i-1` whose LOW >= `level x 0.96`). Distribution on the base pre-book set:
p10 = 2, p25 = 3, **p50 = 9**, p75 = 27, p90 = 55. `atr20d` **95.9 % covered** (3.2 % missing on
winners vs 4.8 % on losers — inside the 5 pp rail, kept), from the **20 daily sessions strictly
before** `day`. Nothing imputed; rows without 20 prior sessions are dropped from the two ATR cells
only.

## 2. The cost wall, re-measured — and it was never +0.2151 R for a booked trade

The runbook's own instruction for this pass: *recompute cost/R for each stop, do not carry the
constant.* Doing that located the constant.

| population (B0, the shipped book's own cascade) | split | n | **cost/R** | median R % of price | gross R |
|---|---|---|---|---|---|
| pre-book, **no cost gates** | TRAIN | 7,388 | **0.1965** | 1.83 | +0.0003 |
| pre-book, **no cost gates** | VAL | 4,745 | **0.2300** | 1.77 | -0.0451 |
| + the 100 bps gate | TRAIN / VAL | 6,568 / 4,137 | 0.1458 / 0.1580 | 1.77 / 1.74 | +0.006 / -0.029 |
| + the 15 %-of-R gate | TRAIN / VAL | 3,387 / 1,889 | **0.0623 / 0.0661** | 2.02 / 1.91 | +0.004 / +0.001 |
| **the shipped pre-book set (+ obtainable)** | TRAIN / VAL | 3,166 / 1,775 | **0.0612 / 0.0650** | 1.95 / 1.85 | +0.007 / +0.001 |

**`+0.2151 R` is the cost on the population BEFORE the two cost gates.** Those gates are part of the
shipped book. A booked HOD-break trade pays **0.061-0.065 R**, and `hod_break` §6's own table says so
in plain sight — `B0` gross -0.027 and net -0.088 differ by 0.061, not by 0.215. Every report in this
programme, this pass's own PREREG included, has been holding cells to a bar **3.2x the book's actual
break-even**. The pre-committed bars below are **not** relaxed on the strength of this — a bar is
moved by a new pre-registration, never by the result it was applied to — but the correction is the
single most load-bearing number in the pass and it changes what the next one should ask.

## 3. FAMILY A — the fresh-high admission ladder (scan rule: **KEEP-SCANNING**)

Stop = the last-5-bar low (the `B2` stop). Book 12/day, 4 concurrent, $100 risk. `>=10 %` = the
two-cohort diagnostic: the share of the cell's trades on symbol-days whose daily range is >= 10 %.

| rung | TRAIN n | /wk | gross | net | grn % | **$** | VAL n | /wk | gross | net | grn % | **$** | H1 / H2 / VAL gross | **same-signed +** | >=10 % T/V |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base (= B2) | 1,622 | 30.6 | -0.039 | -0.107 | 32.1 | -17,346 | 706 | 30.7 | +0.083 | +0.013 | 43.5 | +893 | -0.077 / -0.003 / +0.083 | no | 64 / 66 % |
| **`consol_bars <= 3`** | 809 | **15.3** | +0.007 | -0.060 | 35.8 | -4,850 | 474 | **20.6** | +0.055 | -0.011 | **52.2** | -516 | +0.017 / -0.000 / +0.055 | no | **90 / 88 %** |
| `consol_bars <= 5` | 1,021 | 19.3 | -0.011 | -0.083 | 32.1 | -8,460 | 550 | 23.9 | **+0.099** | +0.029 | **56.5** | **+1,568** | -0.056 / +0.024 / +0.099 | no | 82 / 82 % |
| `consol_bars <= 8` | 1,126 | 21.2 | -0.047 | -0.122 | 32.1 | -13,714 | 599 | 26.0 | +0.069 | -0.005 | 34.8 | -299 | -0.110 / +0.004 / +0.069 | no | 80 / 80 % |
| `consol_bars <= 12` | 1,204 | 22.7 | -0.024 | -0.098 | 32.1 | -11,839 | 615 | 26.7 | +0.092 | +0.017 | 43.5 | +1,017 | -0.076 / +0.020 / +0.092 | no | 78 / 78 % |
| **`consol_bars >= 20` [CONTROL]** | 1,411 | 26.6 | **+0.028** | -0.041 | 34.0 | -5,749 | 714 | 31.0 | **+0.050** | -0.023 | 47.8 | -1,629 | **+0.005 / +0.049 / +0.050** | **YES** | **55 / 53 %** |

**Three readings, in order of how much they matter.**

1. **The frequency problem is SOLVED and the effect shrank with it.** `consol_bars < 8` as a filter on
   the first break was 5.6 / 9.5 trades a week at VAL gross **+0.207 R**; as an admission the ladder
   trades **15.3-22.7 / 20.6-26.7** a week and the VAL gross is **+0.055 ... +0.099**. At the signal
   level (`supp.log` S2) the fresh rungs are the best pre-book gross in the pass — `<=3` reads
   **+0.064 (TRAIN) / +0.040 (VAL)** against the base's -0.002 / +0.015, positive in both years — and
   the **12/4 first-come book destroys it**: 1,850 signals become 809 fills and +0.064 becomes +0.007.
   The slot rule takes the earliest, not the best.
2. **The two-cohort diagnostic says what `consol_bars` largely is.** 88-90 % of the `<=3` rung's trades
   land on >=10 %-range days against 55 % for the control. A fresh HOD is, to a first approximation,
   *a day that is running*. That is the enrichment the diagnostic was declared to catch.
3. **The CONTROL is the only rung that passes the half-consistency rail.** `consol_bars >= 20` —
   expected negative, and it is the one rung whose gross is positive in H1-2025, H2-2025 *and* VAL.
   It is not a contradiction of `hod_losers` §5 (which measured `consol_bars >= 8` as a **filter on
   B0's first break**, -0.130 / -0.226). A filter on the first break and an admission that keeps
   scanning are different books, which is why `PREREG` §1 requires the scan rule to be named. The
   control's book still loses money on both splits.

## 4. FAMILY B — the stop, on the selected rung (`consol_bars >= 20`)

| stop | med **R % of price** T/V | **cost/R** T/V | TRAIN gross / net / $ | VAL gross / net / $ | /wk T/V |
|---|---|---|---|---|---|
| **last-5-bar low** (the B2 stop) | 1.72 / 1.92 | 0.0692 / 0.0731 | +0.028 / -0.041 / **-5,749** | +0.050 / -0.023 / -1,629 | 26.6 / 31.0 |
| last-3-bar low | 1.50 / 1.64 | 0.0703 / 0.0740 | -0.042 / -0.112 / -10,554 | -0.001 / -0.075 / -4,176 | 17.8 / 24.3 |
| shipped consolidation low K5 / 4 % | 1.72 / 1.92 | 0.0692 / 0.0731 | **identical to the last-5-bar low** | | 26.6 / 31.0 |
| the breakout bar's own low | 1.68 / 1.29 | 0.0762 / 0.0733 | -0.107 / -0.183 / -842 | +0.165 / +0.092 / +193 | **0.9 / 0.9** |
| `entry - 0.5 x ATR20d` | 3.40 / 3.38 | 0.0611 / 0.0675 | +0.009 / -0.052 / -6,815 | +0.022 / -0.045 / -2,575 | 24.9 / 24.6 |
| **`entry - 1.0 x ATR20d`** | **5.96 / 6.26** | **0.0426 / 0.0435** | +0.019 / **-0.024** / **-2,545** | +0.022 / -0.022 / -1,002 | 20.3 / 20.0 |

* **The mechanism is confirmed and the mechanism is not the problem.** R widens 1.72 % -> 5.96 % of
  price and cost/R falls **0.069 -> 0.043, -38 %** — the largest movement of the cost term anywhere in
  this programme. The book's loss halves (-$5,749 -> -$2,545). **The gross does not move**: +0.028 ->
  +0.019. Widening the stop buys back cost; it does not buy an edge, because there was ~0.02 R of
  gross to defend.
* **The shipped consolidation-low stop is BYTE-IDENTICAL to the last-5-bar low on this admission**
  (`supp.log` S3: 99.99 % of rows). `consol_bars >= 20` means the last 20 lows already sit within 4 %
  of the level, so the K5 / 4 % proximity test is satisfied by construction. Not a bug — a
  consequence of the admission, and the reason B-i and B-ii5 are one cell here.
* **Tightening is strictly worse** (the 3-bar low: -$10,554 / -$4,176). The breakout bar's own low is
  killed by the shipped `min_r_pct >= 1 %` — 6,936 of 61,347 admitted rows survive it and 70 survive
  the full cascade, so the cell is 0.9 trades a week and carries no information.
* **The cascade shifts with the stop, and it is reported rather than hidden**: of 61,347 admitted
  breaks, `stop computable ^ R >= 1 %` keeps 38,650 (5-bar low), 30,164 (3-bar), 6,936 (breakout bar),
  **59,152 / 59,438 (the two ATR stops)** — a wider stop walks past `min_r_pct` and then past the
  15 %-of-R spread gate, which is the second half of why cost/R falls.

## 5. FAMILY C — the SPY 09:35 day gate, and the best cell in the pass

`spy_r5_pct` = SPY's 09:30 bar OPEN -> 09:34 bar CLOSE, known at **09:35:00**, two minutes before the
earliest possible decision. Applied to `consol_bars >= 20` x last-5-bar low.

| cell | split | n | /wk | gross | net | **grn %** | rs | worst $ | **total $** | MDD $ | gr mo % | iid t | **clustered t** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **C1 `spy_r5 > 0`** | TRAIN | 731 | **13.8** | **+0.100** | **+0.033** | 43.4 | 5 | -1,571 | **+2,391** | -4,116 | 50.0 | +0.63 | **+0.54** |
| **C1 `spy_r5 > 0`** | VAL | 368 | **16.0** | **+0.123** | **+0.050** | 47.8 | 4 | -1,067 | **+1,827** | -3,290 | 60.0 | +0.69 | **+0.66** |
| C2 `spy_r5 >= +0.2 %` | TRAIN / VAL | 95 / 64 | **1.8 / 2.8** | +0.046 / +0.033 | -0.024 / -0.040 | 11.3 / 17.4 | | | -224 / -259 | | | | |
| C3 `spy_r5 >= +0.4 %` | TRAIN / VAL | 15 / **0** | **0.3 / 0.0** | +0.533 / n/a | +0.465 / n/a | 3.8 / 0.0 | | | +698 / 0 | | | | +1.87 / n/a |

**The size ladder is empty, again** — SPY moves >= 0.4 % in its first five minutes on 0.4 % of
sessions, so `C3` is 15 trades in a year and `C2` is under the frequency floor. `C1` is the cell.

**C1's halves: H1-2025 +0.123, H2-2025 +0.082, VAL +0.123 — same-signed positive in all three.** That
is new. `hod_preopen_regime` §5 measured the same gate on the base population as **+0.725 H1 /
-0.041 H2**, i.e. a pure H1-2025 effect; on the `consol_bars >= 20` admission it is not. The gate's own
separation on this population (`supp.log` S5): TRAIN kept +0.112 vs rejected -0.108 = **+0.220, iid t
+3.70, clustered t +2.18**; VAL +0.046 vs -0.101 = **+0.147, iid t +2.04, clustered t +1.54**.

**And C1 is still a no.** Four reasons, each on its own sufficient:

1. **The cell's own t is nowhere.** Net +0.033 / +0.050 at **iid t +0.63 / +0.69** and **day-clustered
   t +0.54 / +0.66**. The gate separates; the resulting book does not separate from zero. MDE on the
   cell is **0.146 R (TRAIN) / 0.202 R (VAL)** — it could not have resolved its own point estimate.
2. **One week is the year.** Weekly dollars at $100 risk (`supp.log` S4): TRAIN best week **+$2,027**
   against a **+$2,391** total (85 %); VAL best week **+$1,215** against **+$1,827** (66 %). 23 of 53
   green weeks on TRAIN, 11 of 23 on VAL.
3. **The net is tail-carried.** TRAIN +0.033 -> ex-top-1 % +0.011 -> **ex-top-5 % -0.071**; VAL +0.050 ->
   +0.028 -> **-0.056**. Reported as a diagnostic, never a rejection reason — but the owner has already
   rejected one book whose edge vanished under a cap, and this one does.
4. **Green weeks are below its own null.** Count-matched permutation null, 2,000 draws, per-week pick
   count fixed: TRAIN 43.4 % against a null **mean of 49.2 %** [41.5, 56.6]; VAL 47.8 % against **50.0**
   [39.1, 60.9]. Inside the band on both, **below the mean on both** — the fifth independent time this
   programme has found that green weeks on this book are bought with pick count.

## 6. FAMILY D — the two declared interactions

| cell | split | n | /wk | gross | net | grn % | **$** | cost/R | H1 / H2 / VAL |
|---|---|---|---|---|---|---|---|---|---|
| D1 `x rv >= 5` | TRAIN | 179 | **3.4** | -0.022 | -0.119 | 39.6 | -2,138 | 0.098 | -0.098 / +0.035 / -0.007 |
| D1 `x rv >= 5` | VAL | 115 | **5.0** | -0.007 | -0.106 | 43.5 | -1,221 | 0.099 | |
| **D2 `C1 x spread <= 8 % of R`** | TRAIN | 369 | **7.0** | +0.023 | -0.015 | 43.4 | -540 | **0.038** | **+0.002 / +0.041 / +0.107** |
| **D2 `C1 x spread <= 8 % of R`** | VAL | 229 | 10.0 | **+0.107** | **+0.069** | 47.8 | **+1,570** | **0.038** | |

`rv >= 5` — the strongest positive gate in `hod_break` §5's map (+0.177 R, t 4.24) — is **negative on
both splits here** and falls to 3.4 / 5.0 trades a week. The cost arm cuts cost/R to 0.038 and lifts
VAL to +$1,570, and takes TRAIN below the 10/week floor at -$540. Neither is a candidate.

## 7. The fill model and the unfilled counterfactual

Capped limit at `level x 1.006`, filled at the next bar's open iff that open is at or under the cap;
`obtainable` (quoted SIP ask <= cap) enforced on every cell. On the selected admission rung, before the
cap gate: **TRAIN 6,012 of 6,876 fill (87.4 %)**, gross +0.044, and the **864 unfilled** would have
been +0.080 had the open been paid anyway; **VAL 3,972 of 4,513 (88.0 %)**, gross -0.055 against the
unfilled -0.046. Same classification as `hod_break` §4 — **neither a chase guard nor a dip-buy; the
cap is neutral-to-mildly-adverse** and both gaps sit inside their standard error.

**Standing caveat, louder here than anywhere.** The measured per-trade NBBO was collected at `B0`'s
signal minutes. The new admissions fire at other minutes, so the imputation share is **100 % (`<=3`),
95 % (`<=5`), 87 % (`<=8`), 81 % (`<=12`)** and only **26-31 % on `consol_bars >= 20`** — the control rung
is the one closest to the shipped book, which is why its spread is mostly *measured*. Every number
above is therefore quoted on **gross** as well as net; the conservative band arm (`net(b)`) is 0.02-
0.06 R worse and changes no verdict.

## 8. Both bars, the nulls, and the adequacy review

**Claim bar G1 — 0 of 15 cells pass** (TRAIN net R > 0 with iid t >= 2.0 **and** clustered t >= 2.0,
>= 10 trades/week, TRAIN gross >= +0.25 R). Best TRAIN net R in the pass is **+0.4655** and it is `C3`
at **0.3 trades a week**; the best TRAIN net at >= 10/wk is `C1`'s **+0.033**; the best TRAIN clustered
t anywhere is **+1.87** (`C3`, n = 15). **G2 was never evaluated and TEST was never opened.**

**Ship bar (the owner's) — 0 of 15 pass.** It asks gross >= +0.25 R on both splits, >= 10/wk, VAL green
weeks >= 50 %, positive weekly dollars on both splits, clustered t >= 2 on TRAIN. `C1` clears
**two of five** (>=10/wk, positive dollars both splits) and misses gross by 2.0-2.5x, VAL green weeks
by 2.2 pp and clustered t by 3x.

**Live-exploration bar.** As literally written — a positive point estimate on green weeks *and*
dollars at live size, a stated mechanism, bounded downside, resolution inside a quarter — **`C1` is
the first cell in 799 to meet it**: +$2,391 / +$1,827 at 13.8 / 16.0 trades a week, with a mechanism
(a long-only continuation book works when the tape is bid in its first five minutes and the level has
been held long enough to be a real supply shelf). It is **not recommended**, for §5's four reasons,
and above all because 85 % of the TRAIN year is one week and the edge is gone ex-top-5 %.

**Nulls.** 34 cell x split nulls (`nulls.csv`): **31 inside their band, 3 below** (the
`consol_bars >= 20` rung and its two identical stop twins, TRAIN 34.0 vs [35.8, 49.1]), **0 above, and
0 favourably outside on both splits.** The sixth pass in a row to reach that result.

**Adequacy review, answered in writing.**

1. *Did we test what the book actually IS — and the thing the anatomy actually pointed at?* Yes, and
   the reproduction gate is exact on three rows including the anatomy's own lead cell (R2, to the
   dollar). The admission was changed, not filtered, which is precisely what `hod_losers` asked for,
   and the scan rule is named on every table. Standing caveats unchanged: the universe file is the
   daily range >= 5 % screen made superset-exact by the causal +5 % floor; 1,140 delisted symbol-days
   have no consolidated tape; touch-only breaks (2.1 % of live signals) are outside the population.
2. *Is the cost model right for its venue?* It is now right in a way it was not before — §2 shows the
   programme's headline constant was the ungated population's cost, 3.2x the booked one. The residual
   weakness is the **imputation share on the new admissions** (81-100 % on the fresh rungs), which is
   why gross is quoted beside every net and why the conservative band arm is printed.
3. *Does a caveat of our own explain the headline?* **Yes, and it is the headline's defeat.** The one
   cell that reads positive on both splits is 85 %-carried by a single TRAIN week and dies ex-top-5 %;
   and the fresh-high rungs' apparent quality is substantially a >=10 %-range-day enrichment, which the
   pre-declared two-cohort diagnostic exposed rather than a reviewer.
4. **MDE.** Per trade, pre-book: **0.051 R (TRAIN, n = 4,575) / 0.071 R (VAL, n = 2,452)**; on the
   selected subsets 0.103 / 0.144 R (the rung), **0.146 / 0.202 R (`C1`)**, 0.059 / 0.087 R (the 1.0 x
   ATR stop, the widest and therefore best-powered cell). On the green-week share, **+/-13.5 pp over 53
   TRAIN weeks and +/-20.4 pp over 23 VAL weeks**. Against the book's **true** break-even of 0.061-
   0.065 R this is a *marginal* test, not the crushing one the +0.2151 R framing implied: the pass
   can see 0.05-0.07 R on the population and the book needs ~0.065 R. That is the honest statement of
   what this pass could and could not have seen.
5. *Tail dependence.* `C1` net +0.033 -> ex-top-1 % +0.011 -> **ex-top-5 % -0.071** (TRAIN) and +0.050 ->
   +0.028 -> **-0.056** (VAL). The `consol_bars >= 20` rung: -0.041 -> ex-top-5 % -0.148; the 1.0 x ATR
   stop -0.024 -> -0.116.
6. *Multiplicity.* **15 declared decision cells** x 2 splits = 30, plus 3 reproduction rows, the
   availability audit and **6 supplementary diagnostics** (`supp.py` S1-S6, none used as a claim).
   **Programme cumulative: 784 (through `hod_losers`) + 15 = 799 declared cells; 805 with this pass's
   diagnostics.** Expected largest |t| under a pure null over 30 cell x splits ~ 2.7-2.9; the largest
   favourable **clustered** t on any cell here is **+0.66**.

## 9. The SHIP-TO-DRY diff that was NOT taken (for the record)

If the owner wants `C1` armed in the dry run despite §5, the diff is **two knobs and two code
changes** — and the second code change is the one that makes it honest:

```yaml
# config.yaml  hod_break:
  consol_bars: 20        # was 5 — the ADMISSION: the level must have been held >= 20 bars
  stop_bars: 5           # NEW FIELD — the STOP stays the last-5-bar low
  rv_hi: 1000000         # was 5 — the rv upper cut is off (the B2 correction)
  spy_open5_gate:        # NEW BLOCK — skip the day unless SPY's first five minutes are up
    enabled: true
    window_end_et: "09:35"
    min_ret_pct: 0.0
    fail_open: true
```

* `consol_bars: 20` is a **pure knob** — `trading/hod_break.py::detect` already keeps scanning until a
  bar has K tight bars behind it, so the keep-scanning admission needs no code.
* **`stop_bars` does not exist.** Today `consolidation_low` uses `p.consol_bars` for BOTH the
  admission length and the stop window, so raising K to 20 would silently widen the stop to the
  20-bar low — a different book from the one measured here. Splitting the two is a **code change** in
  `HodBreakParams` + `consolidation_low` + `tests/test_hod_break.py`.
* The SPY gate is one `get_1min_bars_multi(['SPY'], ...)` call at 09:35:00 in `HodBreakEngine`, cached
  for the session, evaluated before the first candidate is admitted.
* **Cost of arming it**: the dry run stops being a clean forward measurement of the shipped rule,
  which is the only thing it is currently good for. **That is why the recommendation is not to.**

## 10. Closure of the HOD-break line

> **HOD-break is closed.** Over **799 pre-registered cells** across five passes on this book, the
> mechanism is fully on the record and none of it is monetisable at this book's cost and frequency:
> the raw break is gross-flat (+0.0003 / -0.045 R); the loss lives in the **first ten minutes of the
> trade** (MFE +0.45 R at minute 6, -0.50 R by minute 10, stopped at minute 28) and the largest
> era-consistent separation in the whole programme (0.77-0.88 R) is a **classifier of trades that
> have already lost**, worth +0.012 R run as a rule; the level-quality direction that survived the
> anatomy — the fresh high — is substantially a **>=10 %-range-day marker** and loses three-quarters of
> its VAL gross the moment it is made an admission that trades often enough to matter; and the stop,
> the last untested lever, **moves the cost term by 38 % and the gross not at all**. What the pass
> also proves is that the wall was mis-measured: a booked trade pays **0.061-0.065 R**, not 0.2151 R,
> so the honest gap is ~0.06 R of gross, not 0.2 R — and even against that smaller wall the best
> causal cell in 799 clears it only on a point estimate whose day-clustered t is 0.6, whose TRAIN year
> is 85 % one week, and which is negative once the top 5 % of trades are removed. The engine stays
> `enabled: true, dry_run: true` as instrumentation. Re-opening this line needs something this
> programme has never had on it: **a causal separator with a mechanism, a clustered t >= 2 on a subset
> that trades >= 10 times a week, and an edge that survives a cap on its winners.**

**Recommended action: NONE.** `config.yaml hod_break` stays exactly as the owner set it
(`enabled: true, dry_run: true`). `orb.yaml`, the service, the crons and every order were untouched.
