# HOD-break pre-open / causal day-regime — REPORT (2026-09-19)

Cells exactly as declared in `PREREG.md` (committed **`fe10e89`, before any cell was scored**).
Artifacts: `fetch_idx.py` → `idx_1min.csv` (795,849 SPY/QQQ 1-min bars incl. extended hours) ·
`fetch_imbalance.py` → `qqq_imbalance.csv` (224,107 Nasdaq NOII messages) · `dayfields.py` →
`day_fields.csv` + `availability.csv` · `score3.py` → `score3.log`, `cells.csv`, `curve.csv`,
`separation.csv`, `nulls.csv` · `rank.py` → `rank.log` · `supp.py` → `supp.log` · `halves.py`.
One python process at a time, `nice -n 10`, `ulimit -v 3000000`; `bars_sip.db`, `data/cache.db`,
`data/trades.db` opened **read-only**. No config, `orb.yaml`, systemd unit, cron, order or cache was
written. `hod_break` stays `enabled: true, dry_run: true`. TEST was never opened.

---

## VERDICT — **STAY DRY**

*The mechanism the owner is pointing at **is** in a causal field. SPY's own first five minutes
(09:30 open → 09:34 close, known at 09:35:00, two minutes before the earliest possible decision at
09:37) separates the book by **+0.388 R gross on TRAIN and +0.189 R on VAL**, same sign, and the
book under it improves on **every** owner metric on **both** splits: green weeks 32.1 → 43.4 % and
43.5 → 65.2 %, dollars −$17,346 → −$3,059 and +$893 → +$3,439, MDD −$18.9K → −$8.0K, worst week
−$2,231 → −$1,189, at 15.6–15.9 trades a week. That is the first causal day gate in the programme
that moves every metric the right way in both years.*

*It still does not clear either bar, for four reasons that are each sufficient. (1) The per-trade t
that made the prior pass's version look strong is an artefact of **day-clustering** — HOD signals
arrive in bursts (two TRAIN days carry 607 of B2's 8,856 signals), and the day-clustered t is
**+2.00 / +2.37**, not +10.7 / +3.7. (2) The TRAIN half of the effect is **entirely H1-2025**:
+0.725 R (t 3.04) in H1, **−0.041 R (t −0.45) in H2**, so the gate fails the programme's own
pre-committed selection rule (same sign in both TRAIN halves) — as does every other candidate here.
(3) The selected subset's TRAIN gross is **+0.032 R against the +0.2151 R measured cost**: the gate
removes ~80 % of the base's loss and still leaves the book ~0.18 R a trade short of break-even.
(4) Its green-week share sits **inside its own count-matched null band on both splits**, and its VAL
per-trade edge dies ex-top-5 % (+0.094 → −0.009). **0 of 162 declared cells clear the claim bar;
0 clear the live-exploration bar as written; 0 of 325 cell×splits sit above their null band on both
splits.***

Not SHIP-TO-DRY: the dry run already runs the shipped rule and is the programme's only source of
forward data that cannot be a look-ahead. §8 gives the exact diff anyway, because the owner's
standing instruction on this book is "we will not stop till you find money here" and he should see
what the near-miss would cost to arm.

---

## 1. Reproduction gate — EXACT

`score2.sig_set` re-derived here, against `hod_filter_stack/REPORT.md` §2:

| base | split | this pass | reference | verdict |
|---|---|---|---|---|
| B0 | TRAIN | 1,688 tr · −0.027 gross · −0.088 net · 41.5 % green · **−$14,835** | identical | **MATCH** (Δ$ 0) |
| B0 | VAL | 820 · +0.016 · −0.050 · 43.5 % · **−$4,128** | identical | **MATCH** |
| B2 | TRAIN | 1,622 · −0.039 · −0.107 · 32.1 % · **−$17,346** | identical | **MATCH** |
| B2 | VAL | 706 · +0.083 · +0.013 · 43.5 % · **+$893** | identical | **MATCH** |
| B3 | TRAIN/VAL | 1,204 / −$13,723 · 596 / +$313 | identical | **MATCH** |

**The decision instant, asserted in code**: the earliest `entry_m` in B2 is **577**; its break bar
(576) closes at **09:37:00 ET**. Every field in §2 is known by 09:35:00 at the latest.
43.3 % of B2 fires before 10:00 — the number that killed the prior pass's headline.

## 2. The declared fields, their causality trace, and what each one is worth

Every field 100 % covered on both splits (`availability.csv`); the one exception is `qqq_vol20`
(92.3 % TRAIN — `daily_bars` QQQ starts 2024-12-30, so the first 20 sessions of 2025 have no 20-day
vol; those days **fail open** and are never excluded). Nothing imputed.

Gates below are on **B2**, book 12/day 4 concurrent, $100 risk. `tc` = **day-clustered** t (§4).

| # | field → gate | KNOWN AT | causal? | TRAIN grn % / $ | VAL grn % / $ | tr/wk T/V | sep gross T (tc) | sep gross V (tc) |
|---|---|---|---|---|---|---|---|---|
| — | **B2 base (no gate)** | — | — | 32.1 / −17,346 | 43.5 / +893 | 30.6 / 30.7 | — | — |
| D1 | `spy_gap_pct > 0` | 09:30:00 | yes | 35.8 / −7,868 | 56.5 / +2,411 | 17.8 / 17.2 | −0.018 (−0.09) | +0.034 (+0.40) |
| D1 | `abs(spy_gap) ≤ TRAIN median` | 09:30:00 | yes | 35.8 / −5,610 | 39.1 / +1,073 | 15.3 / 13.9 | −0.076 (−0.40) | −0.055 (−0.61) |
| **D2** | **`spy_r5_pct > 0`** (09:30→09:35) | **09:35:00** | **yes** | **43.4 / −3,059** | **65.2 / +3,439** | **15.6 / 15.9** | **+0.388 (+2.00)** | **+0.189 (+2.37)** |
| D2 | `spy_r5` TOP TRAIN tercile | 09:35:00 | yes | 28.3 / −3,680 | 60.9 / +2,973 | 9.9 / 13.0 | +0.338 (+1.50) | +0.163 (+1.98) |
| D2b | `spy_pm_ret_pct > 0` (≤09:29) | 09:29:59 | yes | 35.8 / −7,868 | 56.5 / +2,411 | 17.8 / 17.2 | identical to D1 — the premarket sign and the gap sign agree on every session: **one cell, not two** | |
| D3 | `qqq_gap_pct > 0` | 09:30:00 | yes | 35.8 / −7,379 | 65.2 / **+4,538** | 18.2 / 17.4 | −0.003 (−0.02) | +0.138 (+1.53) |
| D3 | `qqq_r5_pct > 0` | 09:35:00 | yes | 32.1 / −11,667 | 52.2 / +2,206 | 14.3 / 15.7 | +0.300 (+1.43) | +0.111 (+1.32) |
| **D4** | `qqq_imb_side ≠ sell` (NOII) | **09:29:59** | **yes** | 24.5 / −15,977 | 34.8 / −2,025 | 25.1 / 21.8 | +0.436 (+2.21) | **−0.107 (−1.29)** |
| **D4** | `qqq_imb_side = BUY` | 09:29:59 | yes | 17.0 / −3,669 | 13.0 / −842 | 4.7 / 4.1 | +0.529 (+2.36) | **−0.020 (−0.15)** |
| D5 | premarket breadth | — | **DROPPED** | — | — | — | see §2a | |
| D6 | `spy_vol20` BOT TRAIN tercile | T−1 16:00 | yes | 13.2 / −6,749 | 13.0 / +233 | 10.0 / 12.7 | — | — |
| D6 | `qqq_vol20` BOT TRAIN tercile | T−1 16:00 | yes | 13.2 / −8,191 | 4.3 / −1,576 | 11.4 / 6.6 | — | — |
| D6 | `spy_prev_c2c > 0` (= M2 1d) | T−1 16:00 | yes | 35.8 / −9,258 | 43.5 / −153 | 17.2 / 17.0 | — | — |
| D7 | skip worst TRAIN weekday (Mon) | T−1 | yes | 28.3 / −16,045 | 43.5 / −577 | 24.7 / 26.0 | — | — |
| combo | `spy_r5>0 AND qqq_r5>0` | 09:35:00 | yes | 34.0 / −6,294 | 60.9 / +2,554 | 11.6 / 12.1 | +0.688 H1 / −0.152 H2 | +0.125 (+1.46) |
| combo | **`spy_gap>0 AND spy_r5>0`** | 09:35:00 | yes | 37.7 / **+333** | 65.2 / **+4,092** | **9.1 / 9.3** | +0.024 (+0.12) | +0.174 (+2.21) |

**The two positives in the whole table are D2 (`spy_r5 > 0`) and its conjunction with the gap.**
`A14 = gap>0 AND r5>0` is the ONLY cell of 165 with positive dollars on both splits — but on B2 it
keeps **9.1 / 9.3 trades a week, below the declared 10/week floor**, and on B0 (where it clears the
floor at 10.1 / 10.8) TRAIN is **+$391 for a whole year** = $7 a week at $100 risk, with TRAIN green
weeks 37.7 % **below** B0's own 41.5 %. It is inside its null band on both splits on B0.

**D4, the new instrument, is a null with the wrong sign.** The Nasdaq opening-cross imbalance
separates TRAIN at +0.44 R (clustered t 2.21) and **reverses on VAL** (−0.107). The buy-imbalance
form covers only 37 TRAIN days and 15 VAL days (271 of 424 sessions carry a ZERO imbalance at
09:29:59, so the field is 64 % degenerate) and loses money on both splits. The opening auction is
not a filter for this book at this resolution.

### 2a. D5 premarket breadth — declared and dropped, with the reason

Not tested, not approximated. The causal pool is the live engine's streamed universe (≈3,600 names).
`cache.db::intraday_bars_1min` is **RTH-only**. `bars_sip.db` does carry premarket (307,343
symbol-days, median 681 names a session, prints from 04:00 ET) but its membership is the
**end-of-day-selected** HOD candidate pool — a breadth count over it inherits exactly the look-ahead
this pass exists to remove. `orb_premarket_dollar_vol_nightly.csv` covers a median of **11 names a
session**. Rejected on causality and coverage, in that order.

### 2b. Databento — priced before the pull, coverage reported

| form | price | action |
|---|---|---|
| **QQQ only, XNAS.ITCH `imbalance`, 2025-01-02 → 2026-09-12** | **$0.3740** | **PULLED** |
| breadth, `ALL_SYMBOLS`, same range | **$4,180.24** | over the $40 cap, NOT pulled |
| breadth, windowed 09:28–09:30 only | $2.589/session × 415 = **$1,074.31** (a request minimum applies; windowing does not reduce it) | NOT pulled |
| breadth, symbol subsets | $14.55 / 50 names, $30.28 / 100 names (~$0.30 a name) → the ~3,600-name universe ≈ **$1,080** | NOT pulled |

**Coverage of what was pulled: 224,107 messages over 424 sessions; 58,512 of them opening-cross
(type `O`) messages strictly before 09:30, covering every one of the 421 in-scope sessions = 100 %.**
The last message per session is at **09:29:59** on the median session and `m_et` never reaches 570 —
asserted in `dayfields.py`, not assumed. **SPY is NYSE-Arca listed and is not on XNAS.ITCH at any
price**, so QQQ is the only index auction available; the cap excluded the breadth version, which is
the form that would have tested "is the whole market bid into the cross".

## 3. The T-ladder — the owner's curve (SPY sign gate on B2)

Rule: the gate is computed from bars closing ≤ `g` and applied ONLY to signals with
`entry_m ≥ g+1`; earlier signals are **taken ungated** and their book is printed separately.
`late gated → ungated` is **the gate's own contribution**: the late arm alone, with and without the
gate. This is the column the prior pass did not have.

| gate | tr/wk T/V | TRAIN grn % | TRAIN $ | TRAIN late gated → ungated | VAL grn % | VAL $ | **VAL late gated → ungated** |
|---|---|---|---|---|---|---|---|
| **09:35** | 15.6 / 15.9 | 43.4 | −3,059 | −3,059 → −17,346 (whole book) | **65.2** | **+3,439** | +3,439 → +893 (whole book) |
| 09:40 | 20.0 / 21.6 | 39.6 | −8,861 | −9,975 → −17,291 | **69.6** | +3,137 | **−2,644 → −5,017** |
| 09:45 | 20.9 / 22.2 | 37.7 | −10,413 | −11,255 → −17,214 | 56.5 | +3,293 | **−1,468 → −6,482** |
| 09:50 | 22.3 / 23.8 | 30.2 | −9,290 | −2,534 → −14,840 | 65.2 | +2,770 | +728 → −4,510 |
| **10:00** (the owner's form) | 23.6 / 25.3 | 35.8 | −9,295 | −213 → −9,399 | 65.2 | +3,665 | **−3,582 → −7,547** |
| 10:15 | 24.9 / 25.9 | 28.3 | −13,531 | −7,700 → −11,625 | 47.8 | +2,298 | **−6,366 → −9,925** |
| 10:30 | 26.2 / 26.2 | 32.1 | −16,008 | −83 → −2,526 | 47.8 | +1,390 | −364 → −2,404 |

**The direct answer to "if we have a 10am signal then we trade the stuff that happens after it":
the gate is real and it is a loss-reducer, not a money-maker.** On the ≥10:00 arm alone it takes
TRAIN from −$9,399 to −$213 and VAL from −$7,547 to −$3,582. **The late arm loses money gated or
ungated, in both years.** The `T SPY 1000` cell's headline VAL +$3,665 is almost entirely the
**pre-10:00 arm**, which is **+$3,306 on its own with no gate at all** (printed in `score3.log` as
`(pre-gate arm, entry_m<601)`). Read the curve the other way and the finding is the one D2 already
gives: the money in this book is early, and what helps is knowing the tape at 09:35.

The curve's other axis behaves as expected — trades kept rise monotonically with gate time (15.6 →
26.2 a week) because a later gate leaves more of the book ungated. **The rung that maximises VAL
dollars at ≥ 10/wk is SPY 10:00 `≥ +0.2 %` (+$3,994 at 22.1/wk) — and its TRAIN is −$12,019.** The
rung that maximises TRAIN+VAL together is **09:35 sign** (−$3,059 / +$3,439), i.e. D2. The size
ladder is empty: at 09:35 the +0.2 %/+0.4 %/+0.6 % rungs keep 1.9 / 0.3 / 0.0 trades a week (SPY
moves ≥ 0.4 % in its first five minutes on 0.4 % of sessions), and where the ladder does keep trades
the extra threshold buys no VAL. QQQ twins are uniformly weaker (`curve.csv`): the best QQQ rung is
09:45 `≥ +0.2 %`, VAL +$3,202 / TRAIN −$9,300.

## 4. Day-clustering — the statistical correction that changes the reading

HOD-break signals arrive in day-sized bursts: **two TRAIN sessions carry 607 of B2's 8,856 pre-book
signals**. An iid per-trade t on a DAY-level gate therefore counts one day as 300 independent
observations. Every separation here is quoted with a **cluster-robust SE (clusters = trading days)**
beside the iid one:

| gate | TRAIN iid t | **TRAIN clustered t** | VAL iid t | **VAL clustered t** |
|---|---|---|---|---|
| `spy_r5 > 0` [B2] | +10.67 | **+2.00** | +3.69 | **+2.37** |
| `qqq_imb ≠ sell` [B2] | +10.73 | **+2.21** | −1.90 | −1.29 |
| `spy_r5 AND qqq_r5` [B2] | +8.86 | **+1.50** | +2.39 | +1.46 |
| `T SPY 1000` late arm [B2] | +2.20 | **+0.39** | +0.89 | +0.70 |
| the prior pass's `D-c` (look-ahead form) | +2.49 | — | +2.81 | — |

The prior report's `D-c` t of 2.49 / 2.81 was iid on the same clustered population; under this SE it
would also be roughly halved. **No gate in this pass reaches the multiplicity-adjusted threshold
(|t| ≥ 3.4 over 324 cell×splits) on a clustered SE on either split.**

## 5. Half-consistency — where the one survivor dies

The programme's own pre-committed selection rule requires the same sign in **both halves of TRAIN**
(`halves.py`). Applied to every candidate that got near a bar:

| gate | base | H1-2025 | H2-2025 | VAL | both halves? |
|---|---|---|---|---|---|
| **`spy_r5 > 0`** | B2 | **+0.725 (t +3.04)** | **−0.041 (t −0.45)** | +0.189 (t +2.37) | **NO** |
| `spy_r5 > 0` | B0 | +0.515 (t +3.11) | −0.091 (t −0.82) | +0.165 (t +1.83) | **NO** |
| `gap>0 AND r5>0` | B2 | −0.010 | +0.031 | +0.174 (t +2.20) | **NO** |
| `spy_r5 AND qqq_r5` | B2 | +0.688 (t +2.66) | −0.152 (t −1.67) | +0.125 | **NO** |
| `qqq_gap > 0` | B2 | −0.074 | +0.050 | +0.138 | **NO** |
| `spy_gap > 0` | B2 | −0.092 | +0.032 | +0.034 | **NO** |
| gap-and-GO | B0 | −0.355 | −0.004 | +0.006 | YES (both negative, both ≈ 0) |
| `T SPY 1000` late arm | B2 | +0.208 (t +0.48) | +0.043 (t +0.31) | +0.068 (t +0.70) | YES (both ≈ 0, neither significant) |

**Not one candidate has a same-signed, non-trivial effect in both halves of TRAIN.** The entire
TRAIN half of the D2 effect is H1-2025. Two "YES" rows exist and both are indistinguishable from
zero. This is the same shape the prior pass found in `entry_m` deciles, one level up: a large
in-sample day-level dispersion that does not hold from one six-month window to the next.

## 6. The multi-day family — the owner's "previous 2 days" claim, tested as stated

| rule | TRAIN sep (tc) | VAL sep (tc) | TRAIN $ | VAL $ |
|---|---|---|---|---|
| M1 `spy 2d > 0` | −0.052 (−0.25) | −0.011 (−0.13) | −9,198 | +688 |
| M1 `spy 2d ≥ +0.5 %` | — | — | −5,868 | +1,829 |
| M1 `spy 2d ≤ −1.0 %` | — | — | −2,562 | −2,160 |
| M3 two-UP / two-DOWN / mixed | — | — | −5,004 / −2,708 / −9,634 | −675 / −1,255 / +2,824 |
| M4 prior close pos ≥ .5 | — | — | −14,210 | +449 |
| M2 prior 1d / 3d / 5d > 0 | — | — | −9,258 / −13,009 / −11,056 | −153 / +2,310 / +1,169 |

**F4 (his exact claim — is the 2-day signal concentrated in the first 30 minutes?)**

| rule | window | TRAIN sep (t) | VAL sep (t) |
|---|---|---|---|
| M1 `spy 2d > 0` | **09:35–10:00** | **−0.074 (−1.18)** | **+0.185 (+2.53)** |
| M1 `spy 2d > 0` | 10:00–14:00 | −0.028 (−0.52) | **−0.206 (−2.69)** |
| M3 two-UP | 09:35–10:00 | −0.049 | +0.089 |
| M3 two-UP | 10:00–14:00 | +0.057 | −0.156 |

**The shape he describes appears on VAL only, and inverts within the split** (+0.185 early,
−0.206 late), while TRAIN is negative in both windows. That is a within-year contrast with no
out-of-window replication — the definition of a cell that has not replicated. **F6**: the best M1
rung ANDed with the best T rung adds nothing (`cells.csv`); the two carry no common information
because M1 carries no information. **F2** (T2 × spread ≤ 8 % of R) −$7,814 / +$3,254;
**F3** (T2 × rv ≥ 5) −$3,875 / −$762.

## 7. Nulls, both bars, and the adequacy review

**Count-matched permutation null, 2,000 draws, per-week pick count fixed, on ALL 325 scorable
cell×splits** (`nulls.csv`): **304 inside their band, 9 ABOVE, 12 below — and 0 cells are ABOVE on
BOTH splits.** `A3 [B2]` is inside on both (TRAIN 43.4 vs 42.9 [35.8, 50.9]; VAL 65.2 vs 56.1
[43.5, **65.2**] — it sits exactly on p95). `A14 [B0]` is inside on both. **Green weeks on this
book are still bought with pick count, not with day-selection skill** — the third independent pass
to say so.

**Claim bar — 0 of 162 cells pass G1.** The maximum TRAIN net-R t over every cell is **+1.09**
(against t ≥ 2.0 required and |t| ≥ 3.4 for multiplicity); the maximum TRAIN gross of any cell is
+0.498 R and that cell (`T QQQ 0935 p60`) keeps 0.3 trades a week. The best cell that keeps
≥ 10/wk has TRAIN gross **+0.032 R against the +0.2151 R** the book must clear. G2 was never
evaluated and **TEST was never opened** — `FREEZE.md` records it.

**Live-exploration bar — not met.** As written it needs a positive point estimate on green weeks
AND dollars at $100 on both splits with ≥ 10 trades/week. Exactly one of 165 cells has positive
dollars on both splits at ≥ 10/wk (`A14 gap>0 AND r5>0` on **B0**: TRAIN **+$391** / 37.7 % green /
10.1 a week, VAL +$2,829 / 52.2 % / 10.8 a week) — and its TRAIN green-week share is **below its own
base's** (37.7 % vs B0's 41.5 %), so it fails the primary metric while passing the tertiary one.
Its TRAIN dollars are $7 a week. On B2 the same gate is +$333 / +$4,092 but keeps 9.1 / 9.3 a week,
**under the frequency floor**. Neither is a candidate.

**Adequacy review.**

1. *Did we test what the book actually IS?* Yes — the reproduction gate is exact to the dollar on
   three bases, and the earliest decision instant (09:37) is asserted from the data, not assumed.
   Standing caveats unchanged from `hod_filter_stack`: 49–57 % of B2/B3 rows carry an imputed
   spread, so every separation is quoted on **gross** as well as net; the universe file is the
   range ≥ 5 % daily screen made superset-exact by the causal +5 % floor; touch-only breaks (2.1 %
   of live signals) are outside the population.
2. *Is the cost and fill model right?* Unchanged and inherited: measured per-trade SIP NBBO,
   +0.2151 R break-even, obtainability rail (a quoted ask above `level × 1.006` is no fill).
   Nothing in this pass depends on the cost model: D2's TRAIN gross is +0.032 R, so the cell is
   0.18 R short **before** any cost is charged.
3. *Does a caveat of our own explain the headline?* Yes, and it is reported as the headline's
   defeat rather than buried: **the iid t was inflated by day-clustering**, which is precisely the
   statistic the prior pass's D-c leaned on. Correcting it halves every day-gate t in the programme.
4. **What is the MDE?** Green-week share: 95 % half-width **±13.5 pp over 53 TRAIN weeks** and
   **±20.4 pp over 23 VAL weeks** — so a VAL green-week share must exceed **70.4 %** to be separable
   from a coin flip on that split alone, and **D2's 65.2 % is not**; equally, a VAL sign flip
   smaller than ±20 pp is noise, not a verdict. Per trade, pre-book: **0.042 R (B2 TRAIN)** and
   **0.063 R (B2 VAL)** against the +0.2151 R the book needs — **a powered rejection of the effect
   the book requires, not an underpowered null**. What this test could NOT have seen: a day gate
   worth less than ~0.05 R a trade, or a green-week improvement under ~14 pp (TRAIN) / ~20 pp (VAL).
5. *Tail dependence.* `A3 x B2` net: TRAIN −0.037 → −0.059 (ex-top-1 %) → −0.145 (ex-top-5 %);
   **VAL +0.094 → +0.073 → −0.009**. The VAL edge is gone once the top 5 % of trades are removed.
   `A3 x B0`: −0.026 → −0.047 → −0.132 and +0.030 → +0.007 → −0.073.

**Multiplicity — the honest count.** Declared decision cells **162** (28 A + 10 first-5-min + 70
T-ladder + 20 rolling + 25 multi-day + 9 interactions), scored on TRAIN and VAL = **324
cell×splits**; 165 unique cell names were printed (the 162 plus the three reproduction bases).
Plus **8** post-hoc cells (`supp.py`), **14** post-hoc half-consistency readings (`halves.py`), and
the diagnostic rows that are never claims (7 pre-gate arms, 112 late-gated/late-ungated
contributions, 6 F4 window readings).
**Programme cumulative: 580 (through `hod_filter_stack`) + 162 + 8 + 14 = 764.**
Expected largest |t| under a pure null over 324 cell×splits ≈ **3.4**; the largest favourable
**clustered** t on any causal cell in this pass is **+2.37**, on one split.

## 8. What a SHIP-TO-DRY would have been (NOT recommended — the exact diff, for the record)

The near-miss is D2. If the owner wants it armed in the dry run despite §5 and §7, the diff is a
**day gate**, not a `HodBreakParams` change — no entry or exit knob moves:

```yaml
# config.yaml  hod_break:   (NEW block -- does not exist today)
  spy_open5_gate:
    enabled: true          # skip the whole day unless SPY's first 5 minutes are up
    symbol: SPY
    window_end_et: "09:35" # 09:30 bar OPEN -> 09:34 bar CLOSE, evaluated once at 09:35:00
    min_ret_pct: 0.0       # sign only; the size ladder is worthless (section 3)
    fail_open: true        # no SPY bar by 09:35:30 -> trade the day (never block on data)
```

Live wiring: one `get_1min_bars_multi(['SPY'], ...)` call at 09:35:00 in `HodBreakEngine`, cached
for the session, evaluated before the first candidate is admitted. **Pre-committed stop**: if the
gated dry run's forward green-week share is below the ungated dry run's after 8 weeks, the block is
deleted. **Cost of arming it: the forward series stops being a clean measurement of the shipped
rule**, which is the only thing the dry run is currently good for. That is why the recommendation
is not to.

## 9. What the owner asked, answered in four lines

1. **"Does the mechanism exist in a field known before 09:35?"** **Yes in direction, no in size.**
   SPY's first five minutes separate the book +0.388 R (TRAIN) / +0.189 R (VAL), clustered t
   +2.00 / +2.37, and improve every owner metric in both years — but the TRAIN half of it is all
   H1-2025 and the selected subset is still 0.18 R short of its cost.
2. **"If we have a 10am signal then we trade the stuff that happens after it."** Correct, and now
   fully scored: the causal 10:00 gate takes the ≥10:00 arm from −$9,399 to −$213 (TRAIN) and from
   −$7,547 to −$3,582 (VAL). **It removes loss; it does not create profit.** The positive dollars in
   that cell are the pre-10:00 arm, which is +$3,306 on its own with no gate.
3. **"Maybe at 09:40 there's also signal."** The curve peaks at the EARLIEST rung: 09:35 is the only
   gate time where TRAIN and VAL are both better than base, and every later rung is worse on TRAIN.
   The size ladder is empty — SPY moves ≥ 0.4 % in its first five minutes on 0.4 % of sessions.
4. **"The SPY previous 2 days is a signal for the first 30 min."** Tested exactly as stated:
   +0.185 R (t 2.53) early on VAL, **−0.206 R (t −2.69) late on the same VAL**, and negative in both
   windows on TRAIN. The contrast exists inside one split and does not replicate in the other.

## 10. What would change this verdict

1. **A causal separator that is same-signed in both halves of TRAIN.** Every candidate here fails
   that one test. D2 is the closest and is a pure H1-2025 effect.
2. **The stop, not the filter** — named by `hod_break`, restated by `hod_filter_stack`, and
   untouched by this pass. Every cell here keeps a ~40 % win rate against a 2:1 payoff and the
   ex-tail ladder says the losses are broad. That is a new pre-registration.
3. **The breadth imbalance at $1,080** — the one declared field the cap excluded. QQQ alone is a
   single instrument's cross; "is the whole market bid into the open" is a different question and
   the only untested one left in this family.
4. **Forward data.** The dry run is the one series in the programme that cannot be a look-ahead,
   and D2 is a day gate that can be evaluated on it for free, without arming anything.

**Recommended action: NONE.** `config.yaml hod_break` stays exactly as the owner set it
(`enabled: true, dry_run: true`).
