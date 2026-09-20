# hod_frames6 — F20 the PLACEBO · F19 era comparability · F21 survivorship — REPORT (2026-09-20)

Pass 6 of the HOD-break frame programme. Cells exactly as declared in `PREREG.md`, committed before
any cell was scored. Artifacts: `build20.py` → `book6.csv`, `pool6.csv` · `walk20.py` →
`parity6.csv`, `pa6.csv`, `pb6.csv`, `pc6.csv`, `pd6.csv` (checkpointed per session,
`walk20_state.json`) · `score20.py` → `score20.log`, `cells20.csv` · `supp20.py` → `supp20.log` ·
`pitsets19.py` → `pitsets6.csv` · `score19.py` → `score19.log`, `cells19.csv` · `score21.py` →
`score21.log`, `cells21.csv`. One python process at a time, `nice -n 10`, `ulimit -v 3000000`;
`cache.db`, `bars_sip.db`, the Databento stores and `daily_bars` opened **read-only**. No config,
`orb.yaml`, systemd unit, cron, order or cache was written. The dry run was not touched. **TEST was
never opened** (`FREEZE.md`).

These three frames test the **EVALUATION**, not a rule. No frozen trade-level cell was declared
(pre-refuted by `hod_frames3` §F12).

---

## THE SENTENCE THIS PASS WAS REQUIRED TO PRINT FIRST

**F20 reads NO DIFFERENT.** On the tightest control — the *same* symbol-day, the *same* R geometry,
a minute that did **not** break the high of day, taken **after** the break so that day membership is
causally known — the booked break is worth **−0.063 R (day-clustered t −1.88) on TRAIN and +0.026 R
(t +0.60) on VAL** against its own control. The sign flips between the splits and neither |t|
reaches 2. **The HOD break adds nothing to a name-day the detector has already picked.**

The break's percentile in each 200-draw control distribution, as the pre-committed rule required:

| control | TRAIN percentile | VAL percentile | reading |
|---|---|---|---|
| **F20-a** same symbol-day, non-break minute — **as pre-registered** | **0.0** | **0.0** | WORSE |
| **F20-a** same symbol-day, non-break minute — **causality-corrected** (after the break) | 0.0 | **86.0** | **NO DIFFERENT** |
| **F20-b** matched non-signal symbol, same minute | **100.0** | **100.0** | BETTER |
| **F20-c** same symbol, 15 min earlier | 0.0 | 0.0 | *void — an oracle, not a control* |

The pre-committed rule resolves this twice over to the same answer: (i) the arms **disagree**, and
PREREG §F20 says a disagreement makes the frame's verdict **NO DIFFERENT by construction, no arm
selected after the fact**; and (ii) the reading arm, once its own causality rail is applied, reads
NO DIFFERENT on its own — WORSE requires the break below p5 in **both** splits and a clustered
t < −2, and the causal arm gives percentile 0.0 / **86.0** with t **−1.88 / +0.60**.
**Verdict: NO DIFFERENT.**

*A note on which statistic to trust.* The control-distribution percentile is an **unpaired**
comparison against a band that is artificially tight: each draw's control book averages ~120
control minutes per booked trade, so its standard error is an order of magnitude below the break
book's own. The **paired, day-clustered t** is the correct inference and is what the verdict rests
on; the percentiles are reported because the pre-registration asked for them.

The decomposition behind it — the part that is new — is in §1.5: **the detector's information is in
WHICH NAME-DAY, not in WHICH MINUTE**, and in absolute terms every entry variant on that name-day
sits between −0.04 and +0.08 R, i.e. **under the 0.061–0.065 R booked cost**.

---

## VERDICT — **STAY DRY. 0 of 23 cells clear either bar; nothing is re-opened.**

1. **F20 — NO DIFFERENT.** The break minute is indistinguishable from a later non-break minute on
   the same name-day. The pre-registered "WORSE" headline (−0.256 / −0.154 R, t −7.50 / −3.59) is
   **an artefact of the control's own membership look-ahead** and is withdrawn: 100 % of it lives in
   control minutes drawn *before* the break, which are selected using the knowledge that the day
   would later produce a booked break (§2.3 of `hod_frames/REPORT.md`, fifth appearance). Two real
   numbers survive it: the book **beats a matched non-signal name at the same minute by +0.123 /
   +0.240 R** (t +2.96 / +4.74; but ex-top-5 % of its own trades that is **+0.012 / +0.140** — the
   TRAIN half is tail-carried), and the universe's unconditional bracket at these clocks is
   **−0.16 R**, so the detector lifts −0.16 R to ≈ 0 and stops there.
2. **F19 — the era rail is SOUND, and it was if anything too lenient.** The H1-2025 failure is **the
   edge failing, not the universe differing**, and the frame's own decisive cell is a zero:
   **every single H1 signal name is listed throughout H2-2025** (`F19-x1`: the "names not listed
   throughout H2" book has **0 trades**). Restricting to instruments listed through all three eras
   makes H1 **worse**, not better (−0.077 → −0.097, clustered t −2.79 → −3.12). The listing churn is
   on the other side: the names **not** in the three-era intersection supply 4.2 / 17.5 / **24.0**
   trades a week across the eras and carry **all** of VAL's positive gross (+0.112, +$2,191) while
   the intersection VAL reads +0.002 / **−$4,486**. Wrappers-only, the pass-5 candidate, flips from
   VAL **+0.135 to −0.021** on the intersection. **0 of 11 objects re-opened.**
3. **F21 — survivorship is NOT load-bearing, and that fact now goes on the ledger permanently.**
   **2.69 %** of point-in-time HOD-universe symbol-days are absent from today's `daily_bars`
   (2025 **3.40 %**, 2026 **1.14 %**; monthly max 5.07 % in Jan-2025, min 0.70 % in May-2026), from
   **319 of 5,494** symbols. On the candidate-generating subset (range ≥ 5 %) the share is
   **2.12 %**. The direction is **neutral-to-adverse**: the absent symbol-days' own open→close is
   **+0.010 %** against the present names' **+0.030 %** (difference −0.020 pp, t −1.30), and their
   mean range is *smaller* (2.45 % vs 2.87 %) — exactly what the expectation stated in advance said,
   because delisting risk is an overnight risk and this book is flat by 15:55. Including them could
   not have lifted the base gross.

---

## 0. Reproduction gate and the new parity rail

| id | this pass | reference | verdict |
|---|---|---|---|
| R1 `B2` TRAIN | 1,622 · 30.6/wk · gross −0.039 · net −0.107 · 32.1 % green · **−$17,346** | `hod_frames5` §0 | **MATCH** (asserted in code) |
| R2 `B2` VAL | 706 · 30.7 · +0.083 · +0.013 · 43.5 % · **+$893** | identical | **MATCH** |
| **R3 placebo-walker parity** | `common6.walk_from` re-prices all **2,328** booked trades from their own bar and own stop: **max abs(Δrr) = 2.22e-16** | `hod_frames2/walk2.vwalk` | **ASSERTED** (the run aborts otherwise) |

R3 is the independent-rebuild rail applied where it matters most: a placebo number is worthless
unless the walker that prices the placebo also prices the real trade. It was written from the prose
spec of `walk2.vwalk`, not copied, and it reproduces 2,328 of 2,328 booked `rr` to machine epsilon.

---

# F20 — THE PLACEBO  (6 cells + 2 diagnostics)

## 1.1 Coverage (the availability rail, applied before any number was read)

| arm | split | booked | matched | coverage | controls/trade (median) |
|---|---|---|---|---|---|
| a | TRAIN | 1,622 | 1,622 | **100.0 %** | 237 |
| a | VAL | 706 | 706 | **100.0 %** | 234 |
| b | TRAIN | 1,622 | 1,605 | 99.0 % | 24 |
| b | VAL | 706 | 705 | 99.9 % | 24 |
| c | TRAIN | 1,622 | 945 | **58.3 %** | 1 |
| c | VAL | 706 | 332 | **47.0 %** | 1 |
| d | TRAIN | 1,622 | 1,605 | 99.0 % | 125 |
| d | VAL | 706 | 705 | 99.9 % | 125 |

**Arm c fails the declared 80 % coverage rail** (a bar at `break_m − 15` frequently does not exist on
these names) and is therefore a **diagnostic, not a cell** — decided by the rail, before its number
was looked at.

## 1.2 The cells as pre-registered

| cell | split | break grossR | control mean | ctrl p5 | ctrl p95 | break pctile | paired Δ | iid t | day-clust t | n |
|---|---|---|---|---|---|---|---|---|---|---|
| F20-a same symbol-day, non-break minute | TRAIN | −0.0390 | +0.2167 | +0.1780 | +0.2569 | **0.0** | −0.2563 | −9.00 | **−7.50** | 1,622 |
| F20-a | VAL | +0.0827 | +0.2379 | +0.1884 | +0.2975 | **0.0** | −0.1543 | −3.58 | **−3.59** | 706 |
| F20-b matched non-signal symbol, same minute | TRAIN | −0.0391 | −0.1592 | −0.1982 | −0.1198 | **100.0** | +0.1228 | +3.80 | **+2.96** | 1,605 |
| F20-b | VAL | +0.0821 | −0.1551 | −0.2132 | −0.0986 | **100.0** | +0.2400 | +5.06 | **+4.74** | 705 |
| F20-c same symbol, 15 min earlier | TRAIN | −0.0668 | +1.1950 | +1.1237 | +1.2624 | 0.0 | −1.2632 | −28.54 | −26.84 | 945 |
| F20-c | VAL | +0.0389 | +1.2451 | +1.1472 | +1.3289 | 0.0 | −1.2044 | −16.73 | −19.43 | 332 |

## 1.3 The causality trace that withdraws arm a's headline

Arm a draws control minutes from the **whole session** of a symbol-day whose membership was
established by a break that, for a minute before it, **had not happened yet**. A minute *after* the
break is causal — at the break bar the engine already knows the day qualified. Splitting there:

| arm | split | n | break grossR | control grossR | paired Δ | day-clust t | status |
|---|---|---|---|---|---|---|---|
| a, ALL minutes | TRAIN | 1,622 | −0.0390 | +0.2173 | −0.2563 | −7.50 | as scored — look-ahead in membership |
| a, ALL minutes | VAL | 706 | +0.0827 | +0.2370 | −0.1543 | −3.59 | as scored |
| a, **BEFORE** the break | TRAIN | 1,168 | −0.0705 | **+0.9830** | **−1.0535** | −26.83 | **NOT causal** |
| a, **BEFORE** the break | VAL | 430 | +0.0778 | **+1.0611** | **−0.9833** | −22.63 | **NOT causal** |
| a, **AFTER** the break | TRAIN | 1,621 | −0.0385 | +0.0249 | **−0.0634** | **−1.88** | **CAUSAL** |
| a, **AFTER** the break | VAL | 705 | +0.0846 | +0.0590 | **+0.0255** | **+0.60** | **CAUSAL** |
| a, AFTER and within 30 min | TRAIN | 1,621 | −0.0385 | +0.0077 | −0.0462 | −1.89 | causal + clock-matched |
| a, AFTER and within 30 min | VAL | 705 | +0.0846 | +0.1622 | −0.0776 | −2.24 | causal + clock-matched |

A control minute drawn before the break is worth **+0.98 / +1.06 R** — because the day it was drawn
from is a day that was *about to* run. That is the §2.3 structure again, now in the control rather
than in the rule, and it is the whole of the "WORSE" reading. **Arm c is the same defect in its pure
form**: "15 minutes before the break" exists only because the break exists; its **+1.196 / +1.243 R
at a 65.6 % / 64.5 % target-hit rate** is an *oracle* ("buy 15 minutes before a run you already know
is coming"), not a placebo, and it is reported as one.

## 1.4 Tails and the exit mix (rails 5 and 7)

| arm | split | paired Δ all | ex-top-1 % of the BREAK | ex-top-5 % of the BREAK |
|---|---|---|---|---|
| a AFTER (causal) | TRAIN | −0.0634 | −0.0792 | −0.1446 |
| a AFTER (causal) | VAL | +0.0255 | +0.0124 | −0.0471 |
| b | TRAIN | +0.1228 | +0.1044 | **+0.0119** |
| b | VAL | +0.2400 | +0.2209 | +0.1398 |

*(Rank-based trimming: `rr` has a point mass at exactly +2 R, so quantile trimming ties and would
report the untrimmed number.)* **Arm b's TRAIN advantage is tail-carried** — it falls from +0.123 to
+0.012 when 5 % of the book's own trades are removed. VAL survives at +0.140.

| arm | split | n | mean grossR | WR (R>0) | stop % | target % | eod % |
|---|---|---|---|---|---|---|---|
| **BREAK** | TRAIN | 1,622 | −0.0390 | 38.8 % | **51.7** | 20.3 | 27.9 |
| a | TRAIN | 360,003 | +0.2089 | 51.1 % | 30.3 | 20.3 | 49.4 |
| b | TRAIN | 35,440 | −0.1614 | 37.5 % | 43.4 | 9.6 | 47.1 |
| d | TRAIN | 200,088 | −0.0579 | 43.3 % | 29.5 | 8.3 | 62.2 |
| **BREAK** | VAL | 706 | +0.0827 | 45.0 % | **45.8** | 21.2 | 33.0 |
| a | VAL | 156,105 | +0.2291 | 54.7 % | 26.7 | 17.6 | 55.7 |
| b | VAL | 15,611 | −0.1596 | 38.4 % | 41.0 | 7.9 | 51.1 |
| d | VAL | 88,086 | −0.0423 | 45.3 % | 25.5 | 6.7 | 67.7 |

The break's target rate is **identical** to a random minute's (20.3 % vs 20.3 % on TRAIN) and its
**stop rate is 21 points higher** (51.7 % vs 30.3 %). Whatever a break buys, it is not a higher
chance of +2 R; it is a higher chance of being stopped first. The mechanism is not mysterious — an
entry at the top of the day's range is an entry with a same-percentage stop sitting in the middle of
the range just traversed.

## 1.5 THE DECOMPOSITION — the finding of this pass

| object | TRAIN | VAL | what it holds fixed |
|---|---|---|---|
| the BOOKED break | **−0.0390** | **+0.0827** | nothing — the book itself |
| same symbol-day, any non-break minute | +0.2089 | +0.2291 | day + name (**look-ahead in membership**) |
| same symbol-day, non-break minute **AFTER** the break | **+0.0141** | **+0.0725** | day + name, **CAUSAL** |
| matched non-signal name, same minute | **−0.1614** | **−0.1596** | clock + price + ADV20 + asset class |
| matched non-signal name, random non-break minute | −0.0579 | −0.0423 | nothing — the universe bound |

Read down the column. The universe's unconditional 2:1 bracket at these clocks is **negative**
(−0.16 R at a break's own minute, −0.05 R spread over the session — the difference is horizon: a
later entry has less time to reach either leg, which is why arm d's eod share is 62 % against arm
b's 47 %). The detector's **name-day** selection is worth **+0.12 / +0.24 R** against that. The
detector's **minute** is worth **−0.06 / +0.03 R** against a later minute on the same name-day.

So the two readings the frame was built to separate are **both partly right, and the honest sentence
is neither of the two the queue offered**:

> The universe is not neutral — it is **−0.16 R** under this book's own bracket at this book's own
> clock. The detector is not worthless — picking the name-day is worth **+0.12 / +0.24 R** against
> that. But the two cancel: the book's absolute gross is **−0.039 / +0.083 R**, every alternative
> entry minute on the same name-days is **+0.014 / +0.073 R**, and the booked cost is
> **0.061–0.065 R**. There is a real detector whose entire output is consumed by the negative
> baseline it is fishing in, and the break minute itself contributes **nothing**.

**Power.** The MDE at 80 % on the paired causal arm-a difference is **0.098 R (TRAIN) / 0.151 R
(VAL)**; the
point estimates are −0.063 and +0.026. The test could not have seen an effect smaller than about
0.10 R, and the honest phrasing is: *no difference between the break minute and a matched non-break
minute of the same name-day was detectable in THIS universe, at THIS 1-minute horizon, at THIS book
size (12/day, 4 concurrent), over 2025-01 → 2026-05, at the measured 0.061–0.065 R cost — with a
smallest detectable effect of about 0.10 R.*

---

# F19 — ERA COMPARABILITY  (12 cells)

Point-in-time listing sets from the Databento EQUS.SUMMARY definition feed, coverage **202407..202609**
— the whole TRAIN+VAL window is inside the bought window, so the XNAS.ITCH 2018–2024 fallback is not
needed and **is not used**. "Listed in an era" = a definition record in **every** month the era spans.

| era | listed throughout | signals | distinct symbols | wrappers | commons | UNCHECKABLE | median price | median ADV$ | listed-in-all-3 share of signals |
|---|---|---|---|---|---|---|---|---|---|
| H1-2025 | 10,765 | 2,573 | 1,248 | 376 | 863 | 0 (0.0 %) | $41.64 | $160M | **95.6 %** |
| H2-2025 | 11,207 | 2,002 | 645 | 160 | 481 | 0 (0.0 %) | $38.75 | $102M | 72.1 % |
| VAL | 11,926 | 2,452 | 782 | 189 | 587 | 29 (3.7 %) | $41.74 | $131M | 69.0 % |

INTERSECT (listed throughout all three eras) = **9,755 symbols**. 77.7 % of the distinct wrappers in
the signal stream are in it, against 93.2 % of the commons — so the churn IS concentrated in the
wrapper complex, as the frame supposed. **What the frame did not suppose is the direction.**

## 2.1 F19-x1 — the decisive cell is a zero

| H1-2025 book | n | /wk | gross | net | $ | green | clustered t | MDE |
|---|---|---|---|---|---|---|---|---|
| all names (reference) | 789 | 29.2 | −0.0767 | −0.1433 | −11,303 | 26.9 % | −2.79 | 0.158 |
| names listed throughout H2-2025 | **789** | 29.2 | −0.0767 | −0.1433 | −11,303 | 26.9 % | −2.79 | 0.158 |
| names listed in all three eras | 757 | 28.0 | **−0.0966** | −0.1627 | −12,314 | 26.9 % | **−3.12** | 0.159 |
| names **NOT** listed throughout H2 | **0** | 0.0 | — | — | 0 | — | — | — |

**Not one trade in the H1-2025 book is on a name that was gone by H2.** The listing-artefact
hypothesis is not merely unsupported — it is arithmetically impossible on this side. And restricting
further to the three-era intersection makes H1 **worse** (−0.077 → −0.097, t −2.79 → −3.12).

## 2.2 The objects, on the intersection and as-is

| cell | H1 n | /wk | gross | $ | H2 n | /wk | gross | $ | VAL n | /wk | gross | $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F19-ref base B2 (as-is) | 789 | 29.2 | −0.077 | −11,303 | 833 | 30.9 | −0.003 | −6,043 | 706 | 32.1 | +0.083 | +893 |
| **F19-b1** base, INTERSECT | 757 | 28.0 | −0.097 | −12,314 | 754 | 27.9 | +0.033 | −2,545 | 656 | 29.8 | **+0.002** | **−4,486** |
| **F19-b2** base, NOT in INTERSECT | 113 | 4.2 | −0.066 | −1,558 | 473 | 17.5 | −0.026 | −4,751 | 527 | 24.0 | **+0.112** | **+2,191** |
| **F19-o1** wrappers, INTERSECT | 453 | 16.8 | −0.098 | −7,037 | 354 | 13.1 | +0.112 | +1,912 | 324 | 14.7 | **−0.021** | −2,711 |
| **F19-o2** SPY 09:35 gate, INTERSECT | 368 | 13.6 | −0.082 | −5,537 | 408 | 15.1 | +0.114 | +1,925 | 347 | 15.8 | +0.066 | −186 |
| **F19-o3** `dollar_frac` ≥ p80, INTERSECT | 208 | 7.7 | −0.043 | −2,139 | 277 | 10.3 | +0.108 | +1,267 | 272 | 12.4 | −0.037 | −2,807 |
| **F19-o4** price ≥ $30, INTERSECT | 613 | 22.7 | −0.051 | −7,245 | 613 | 22.7 | +0.044 | −1,355 | 596 | 27.1 | +0.040 | −1,867 |
| F19-o1r wrappers, as-is | 497 | 18.4 | −0.117 | −8,779 | 586 | 21.7 | +0.091 | +1,345 | 594 | 27.0 | +0.135 | +3,958 |
| F19-o2r SPY gate, as-is | 392 | 14.5 | −0.062 | −5,130 | 437 | 16.2 | +0.117 | +2,071 | 365 | 16.6 | +0.165 | +3,439 |
| F19-o3r `dollar_frac` ≥ p80, as-is | 230 | 8.5 | −0.076 | −3,168 | 345 | 12.8 | +0.108 | +1,504 | 374 | 17.0 | +0.083 | +646 |
| F19-o4r price ≥ $30, as-is | 649 | 24.0 | −0.070 | −8,928 | 703 | 26.0 | +0.052 | −1,200 | 673 | 30.6 | +0.028 | −2,996 |

**The pre-committed re-open rule (H1 gross ≥ 0 AND same-signed positive in all three eras AND ≥ 10
tr/wk on both splits) is failed by 11 of 11 objects.** Every one **stays dead**, and the era rail is
sound exactly as eleven passes have used it.

**The unexpected half.** The instruments that are *not* in the three-era intersection are the ones
carrying the recent positives: they are 4.2 trades a week in H1 and **24.0** in VAL, they book
**+$2,191** of VAL while the intersection books **−$4,486**, and the wrappers-only object flips from
VAL **+0.135 as-is to −0.021 on the intersection**. So "era-consistency" was never rejecting objects
for a listing artefact in H1; if anything it was **too lenient about H2/VAL**, where a newly-listed
instrument cohort supplies a third of the recent stream and all of its gross. Any future object that
is H2/VAL-positive on this book must now be re-read on the intersection before it is believed.

---

# F21 — SURVIVORSHIP  (5 cells)

Point-in-time HOD universe from the Databento daily panel (2024H2 + 2025–2026): prior close ≥ $17,
20-session prior mean volume ≥ 100,000, test tickers removed → **1,159,397 symbol-days, 5,494
symbols, 352 sessions**. Today's `daily_bars` carries 13,620 distinct symbols.
*(Operational note: a full 20 prior sessions is required rather than the engine's `min_periods=15`.
The panel starts 2024-07 and the window 2025-01, so this can differ only for a name in its first
sessions.)*

## 3.1 F21-1 — how much the `daily_bars` intersection removes

| period | symbol-days | absent | share | distinct symbols | absent symbols |
|---|---|---|---|---|---|
| 2025 | 796,800 | 27,082 | **3.40 %** | 4,867 | 300 |
| 2026 | 362,597 | 4,128 | **1.14 %** | 4,654 | 112 |
| ALL | 1,159,397 | 31,210 | **2.69 %** | 5,494 | 319 |

Monthly: min **0.70 %** (2026-05), max **5.07 %** (2025-01), median 2.71 % — a clean monotone decay
toward the present, which is the signature of delisting/renaming rather than of a data gap.

## 3.2 F21-2 — who is missing

| cohort | symbol-days | absent | share |
|---|---|---|---|
| stock | 745,469 | 945 | **0.13 %** |
| wrapper | 373,660 | 0 | **0.00 %** |
| unknown (no Alpaca asset record) | 40,268 | 30,265 | **75.16 %** |
| prev close < $20 | 76,131 | 6,029 | 7.92 % |
| $20–30 | 233,474 | 10,535 | 4.51 % |
| $30–50 | 274,613 | 6,200 | 2.26 % |
| $50–100 | 301,355 | 5,955 | 1.98 % |
| $100+ | 273,824 | 2,491 | 0.91 % |
| ADV$ < $10M | 222,433 | 12,300 | 5.53 % |
| $10–50M | 382,378 | 10,598 | 2.77 % |
| $50–200M | 317,676 | 5,765 | 1.81 % |
| ≥ $200M | 236,910 | 2,547 | 1.08 % |

The absent set is essentially the **"unknown" cohort** — names with no current Alpaca asset record,
i.e. delisted, acquired or renamed. **Wrappers are 0.00 % absent** (the 2026-09-05 wrapper backfill
put all of them in `daily_bars`), which retires the pass-5 worry that the wrapper complex is where
the survivorship lives. The bias is concentrated in **cheap, thin** names — exactly the part of the
distribution the $20 floor and the spread gates already remove.

## 3.3 F21-3 / 3b — the direction, measured rather than assumed

| cohort | n | mean open→close % | median | mean range % | days ≥ +5 % | days ≤ −5 % |
|---|---|---|---|---|---|---|
| PRESENT in `daily_bars` | 1,128,187 | **+0.030** | +0.015 | 2.87 | 2.17 % | 2.15 % |
| ABSENT from `daily_bars` | 31,210 | **+0.010** | +0.000 | 2.45 | 1.94 % | 1.95 % |

Absent minus present: **−0.020 pp** (SE 0.015, t −1.30). On the candidate-generating subset:

| cohort | symbol-days | absent | share | absent open→close (present) |
|---|---|---|---|---|
| all days | 1,159,296 | 31,135 | 2.69 % | +0.010 (+0.030) |
| range ≥ 5 % | 154,065 | 3,265 | **2.12 %** | +0.006 (+0.054) |
| range ≥ 10 % | 29,902 | 852 | 2.85 % | +0.694 (+0.658) |

The absent names are **not** systematically the takeover gaps the frame worried about, and they are
**not** systematically the collapsing microcaps either — on the widest days they are
indistinguishable (+0.694 vs +0.658). For a long-only book flat by 15:55, which earns the intraday
path and never the overnight gap, the removal is **neutral to very slightly adverse**.

## 3.4 F21-4 — materiality

**0 of the 319 absent symbols have 1-minute bars in `cache.db`; 150 have them in
`research/bf_zero/bars_sip.db`.** A full re-read of the base book on the PIT population is therefore
only partially computable — and **the pre-committed materiality rule resolved before it was needed**
(absent share < 5 % in both years). The honest bound stands without the re-read: at most 2.69 % of
symbol-days (2.12 % of candidate-generating ones) are missing, and their own intraday return
distribution is inside — and if anything below — the present names'. **Including them could not have
lifted the base gross.**

**This goes on the ledger permanently: survivorship is not load-bearing on HOD-break.**

---

## 4. Rails

* **Reproduction**: `B2` EXACT (n, gross, $ all asserted in code — the run aborts otherwise), plus
  **R3**, a second, independently written exit walk reproducing all 2,328 booked `rr` to 2.22e-16.
* **Both TRAIN halves** printed for F20's reading arm and for every F19 cell (the era table IS the
  halves).
* **Day-clustered t** beside iid t on every F20 cell; day-clustered t on every F19 book.
* **Null bands**: F20's 200-draw control distributions ARE the null for each arm (the break's
  percentile is reported against p5/p95); F19's selector is era-consistency, not a null.
* **Availability audit**: arm c **failed** the declared 80 % coverage rail and was demoted to a
  diagnostic before its number was read; F19 reports 29 UNCHECKABLE VAL symbols (3.7 %) explicitly;
  F21's `asset_class` join covers 96.5 % of PIT symbol-days (the 3.5 % "unknown" IS the finding).
* **Booked cost per cell**: unchanged at the measured 0.061–0.065 R (`hod_frames5` §2.5 confirmed it
  with 7,605 fresh SIP quote-minutes); F20 is scored on **gross** by design — a control's own spread
  is a different instrument's and would confound the comparison.
* **MDE**: **0.098 R (TRAIN) / 0.151 R (VAL)** on F20's paired causal difference; 0.158–0.159 R on F19's H1
  books. **Every point estimate in this pass is below its own MDE except arm b's VAL (+0.240).**
* **Tail dependence**: reported on both surviving arms, rank-trimmed (`rr` has a point mass at +2 R).
* **Causality trace**: applied to the control as well as the rule — and it is what withdrew this
  pass's headline.
* **TEST**: never opened.

**Cell count.** 18 declared in `PREREG.md`; 5 added and named in this report (the four `-r` as-is
companion arms in F19, needed for the like-for-like read, and F21-3b, the candidate-generating
subset) = **23**. Programme total **989 + 23 = 1,012**.

---

## 5. The adequacy review (RUNBOOK step 10)

* **Did we test what the book actually IS?** Yes — the placebo is priced by a walker that reproduces
  the booked trades to machine epsilon, on the identical population, with the identical R geometry.
* **Is the cost and fill model right?** F20 is a gross comparison, so cost does not enter; the fill
  convention (next minute's open) is the engine's and is shared by break and control.
* **Does any caveat in our own report explain the headline?** Yes, and it was found and applied:
  arm a's pre-registered "WORSE" is a membership look-ahead in the control, and the report's headline
  is the causality-corrected reading, not the pre-registered one.
* **What is the MDE?** 0.098 / 0.151 R — larger than every point estimate in F20 except arm b's VAL.
* **Verdict**: **STAY DEAD / STAY DRY.** Nothing here is a ship, a config change or an owner
  decision. `hod_break` remains `enabled: true, dry_run: true`.
