# Stage H — F14 second break · F8 N=30 opening-range break · F11 close-confirmed F6

Executed 2026-09-17 per `research/fuckup_audit/H/METHOD.md`, on `research/fuckup_audit/C/pop_c.csv` under Stage C's
contract (c) and `run_book(12, 4)`. Everything written is under this directory; everything outside it was read
only (`C/pop_c.csv`, `B/candidates4.csv`, `D/pm_bars.db`, `D/news_presence.csv`, `E/news_presence_e.csv`,
`day_features.csv`, `research/lit_review_2026/etf_1min.db` — the last two via `file:...?mode=ro`). No config,
service, cache or order was touched. **TEST was NOT read for any of the three books.**
Per-book detail: `REPORT_F14.md`, `REPORT_F8N30.md`, `REPORT_F11F6.md`. Freezes: `FREEZE_*.md`, each written
before that book's `--val` run.

---

# 0. One page per book

## F14 — second break
| | |
|---|---|
| **reaches profit on TRAIN?** | **Yes.** +0.0564 -> **+0.1398 R/trade, t 2.41**, 9.8 tr/wk, halves +0.160 / +0.123, MDD −31.3 -> −19.8 R, 55% weeks green. The best TRAIN t in the whole audit program. |
| **reaches profit on VAL?** | **No.** +0.0503 -> **+0.0096** (t 0.17). **8 of 9** vetoed buckets flip from −0.03..−0.28 on BOTH TRAIN halves to **+0.11..+0.34** on VAL. |
| filters | V4 `dist_open_pct > 0.8` · V2 `vwap_dist_pct >= 0` · V1 `adv20` exists |
| mechanisms | a continuation trade needs a day that has left its open · below VWAP the average share bought today is held above the break (the shipped BF `vwap_gate`) · fewer than 20 prior daily bars = no liquidity baseline (the shipped ORB G1 short-history veto) |
| **verdict** | **FAIL at step 4.** |

## F8 N=30 — 30-minute opening-range break
| | |
|---|---|
| **reaches profit on TRAIN?** | **Yes, and the gain is shape.** +0.0079 -> **+0.0581, t 1.62**, 8.1 tr/wk, halves +0.068 / +0.049, WR 48.4 -> 53.0%, **MDD −40.5 -> −9.0 R**, ex-top-5% −0.105 -> −0.047. Below G1's t. |
| **reaches profit on VAL?** | **No.** −0.0107 -> **−0.0340** (t −0.60, 45% green). Of the three filters IN the stack, only W1's vetoed bucket holds (−0.061 -> −0.023); M1 and G1 flip (−0.024 -> +0.033, −0.041 -> +0.043). Across the book's full grid 7 of 10 era-passing buckets hold — see §7. |
| filters | W1 `asset_class != 'wrapper'` · M1 `sig_close_pos >= 0.5` · G1 `gap_pct < 3` |
| mechanisms | a 2x/inverse single-stock ETF carries no company information and its break is a geared, decaying echo · the break bar must CLOSE in the top half of its own range (ORB's shipped touchgo **Rule M**, same threshold) · gappers fade (PLAN §2: 0 of 27 gap x $-volume cells deliver a positive open-to-close) |
| **verdict** | **FAIL at step 4.** |

## F11 — close-confirmed F6
| | |
|---|---|
| **reaches profit on TRAIN?** | **Yes.** −0.0115 -> **+0.0804, t 1.62**, 12.4 tr/wk, halves +0.149 / +0.017, WR 43.5 -> 48.0%, weeks green 43 -> **57%**, MDD −50.4 -> −19.4, +3R-capped −0.056 -> **+0.021**. |
| **reaches profit on VAL?** | **On the mean yes — and it still fails the frozen rule.** +0.0775 -> **+0.1517** (t 1.68, 55% green, MDD −12.9, ex5 −0.077). The freeze required **each** vetoed bucket to be negative on VAL; **B1's is +0.0345** (P1's holds at −0.0862). The combined stack-vetoed bucket is −0.0070 on 254 trades = indistinguishable from zero. |
| filters | P1 `prev_day_range_pct >= 5` · B1 `sig_body_pct < 1.0` (PR1 `price < 50` passed both era legs and was dropped ON TRAIN because it lowered the stack) |
| mechanisms | a red-to-green reclaim is a CONTINUATION trade and pays when the name was already moving yesterday (live ORB ships this as the PDR veto) · F11 confirms on a CLOSE and fills at the next open, so a big confirming body IS the chase, and every live engine caps the chase |
| **verdict** | **FAIL at step 4 on the letter of the frozen rule.** The closest of the three; the failing leg is B1, not P1. |

---

# 1. The number that decides all three: the search-adjusted permutation

500 day-level sign-flip draws over **all 83 TRAIN book cells** this sub-stage looked at (`h_perm.md`,
`h_perm_cells.csv`; same construction as `B/score5.py::perm_pvalue`):

| observed max \|t\| | null mean | null 95th pct | null max | **p** |
|---:|---:|---:|---:|---:|
| **2.44** (F14's stack) | **3.04** | **3.85** | 5.07 | **0.932** |

The best cell in the sub-stage is **below the average** of what pure noise produces at this search width. This is
Stage C's finding repeated one level down: the grid contains no signal of the size it can detect, and the
TRAIN improvements above — all of them — are inside what an 83-cell search manufactures.

# 2. Parity anchors (step 0, before any Stage H number)

`h0_parity.csv` — the three books rebuilt from `C/pop_c.csv` with an independent copy of the contract and booked
with `trading.hod_break.run_book(12, 4)`, against `C/score5_results.csv` and `C/score5_results.m600.csv`:

| book | window | TRAIN n / net R / t | VAL n / net R / t | result |
|---|---|---|---|---|
| F14 `{"N":15}` | all-day (= `>=10:00`) | 768 / +0.0564 / 1.32 | 363 / +0.0503 / 0.91 | 6/6 cells exact |
| F8 `{"N":30}` | all-day (= `>=10:00`) | 1124 / +0.0079 / 0.34 | 456 / −0.0107 / −0.32 | 6/6 exact |
| F11 `{"base":"F6"}` | all-day | 1239 / −0.0115 / −0.35 | 543 / +0.0775 / 1.28 | 6/6 exact |
| F11 `{"base":"F6"}` | `>=10:00` | 1018 / −0.0045 / −0.18 | 442 / +0.0853 / 1.59 | 6/6 exact |

**24 of 24 cells exact; `max |dn| = 0`, `max |d meanR| = 0.0`.** No book was stopped.
(METHOD's table lists F11 as "≈0 / +0.065 (t 2.0)" — that is the `>=10:00` **`2R stop-1%`** cell, also reproduced
exactly at TRAIN +0.0062 / VAL +0.0646 / t 2.00.)

# 3. The availability audit — three features rejected before use

PLAN §1's standing rule (earned in D1) applied to each book's own population, per split AND per time band
(`h1_anatomy_*.md` §0):

| feature | book | coverage (TRAIN) | missing bucket mean net R (TR / VAL / TEST) | present | verdict |
|---|---|---:|---|---|---|
| `pm_dollar_vol` | F14 | 0.4665 | **+0.380** / +0.116 / +0.022 | −0.096 / −0.111 / −0.136 | **REJECTED** |
| `pm_dollar_vol` | F8 N=30 | 0.3538 | +0.137 / +0.111 / +0.049 | −0.182 / −0.154 / −0.181 | **REJECTED** |
| `pm_dollar_vol` | F11 | 0.3813 | +0.086 / +0.154 / +0.010 | −0.188 / −0.122 / −0.172 | **REJECTED** |
| `news_pre` (D ∪ E) | F14 | 0.7768 | −0.010 / −0.003 / −0.338 | +0.206 / +0.016 / −0.000 | **REJECTED** |
| `news_pre` | F11 | 0.9203 | **−0.251 / −0.251 / −0.360** | +0.001 / +0.064 / −0.033 | **REJECTED** |
| `news_pre` | F8 N=30 | **1.0000** | — | — | admitted (D's key set IS this key set); tested, not era-consistent, not used |

`D/pm_bars.db` reproduces D1's signature exactly on all three books: a 0.23-0.48 R availability indicator with the
same sign on every split. `news_pre`'s coverage is the union of two other stages' key sets and behaves the same
way on F14 and F11. Three of the six feature-book pairs that a naive run would have used are look-aheads.

**But the causal half of the news indicator is recoverable, and it is the best filter in the sub-stage.** E's key
set is `gap >= +3%` ∪ `prev_day_range_pct >= 8%`, both knowable at 09:30. F11's base (F6) opens BELOW the previous
close, so the gap leg never fires: the entire causal content of "news was fetched for this symbol-day" is
*yesterday had a big range* — which is exactly F11's P1 filter, and the only veto in the sub-stage whose bucket
stays negative on TRAIN-H1, TRAIN-H2 **and** VAL.

# 4. The booking convention is a lever, not a detail

METHOD does not fix what happens to a vetoed pick's slot. Both forms are live-implementable and they disagree:

| book | filter | **no-refill** (veto post-ranking, slot stays EMPTY) | **refill** (veto at signal time, next candidate takes the slot) |
|---|---|---:|---:|
| F8 N=30 | M1 alone, TRAIN | **+0.0262** | **−0.0306** |
| F8 N=30 | stack, TRAIN / VAL | +0.0581 / −0.0340 | −0.0052 / **+0.0672** |
| F11 | stack, TRAIN / VAL | +0.0804 / +0.1517 | +0.0581 / +0.0465 |
| F14 | stack, TRAIN / VAL | +0.1398 / +0.0096 | +0.1260 / −0.0053 |

No-refill was declared PRIMARY in `FREEZE_F8N30.md` **before** any VAL number existed, on the ORB precedent (every
shipped ORB veto is post-ranking with no refill, because the refill form measured toxic there: 2025H2 -> ~$0,
MDD −$29K -> −$50K). The F8 N=30 row is the demonstration that the choice matters and that it is noise here: the
refill twin is negative on TRAIN and positive at t 2.00 on VAL — the two conventions rank in opposite orders on
the two splits. Both columns are printed for every cell in `h_eval_*.md`.

# 5. The declared no-floor twin is a look-ahead, and it measures as one

`C/pop_c.csv` cannot answer this — the Stage-C extract already applies `range_so_far_pct >= 5` to every family
except F1-F4 (verified: 0 below-floor rows in all three `pop_*.csv`). Re-extracted from `B/candidates4.csv`
(`h_nofloor_extract.py`, 3,099,499 rows read, EXIT=0) and scored with the same contract
(`h_nofloor_score.py`):

| book | split | floor >= 5 | **NO floor** | below-floor rows ONLY |
|---|---|---:|---:|---:|
| F14 | TRAIN | +0.0564 (t 1.32) | **+0.2996 (t 6.75)** | **+0.3851 (t 8.15)** |
| F14 | VAL | +0.0503 (t 0.91) | +0.2862 (t 4.74) | +0.3785 (t 5.83) |
| F8 N=30 | TRAIN | +0.0079 (t 0.34) | **+0.2097 (t 6.70)** | +0.2786 (t 8.36) |
| F8 N=30 | VAL | −0.0107 (t −0.32) | +0.1028 (t 2.44) | +0.1826 (t 3.98) |
| F11 | TRAIN | −0.0115 (t −0.35) | **+0.2620 (t 3.86)** | +0.3157 (t 4.96) |
| F11 | VAL | +0.0775 (t 1.28) | +0.1103 (t 1.46) | +0.1374 (t 1.84) |

Dropping the floor turns every book into a t = 4-8 winner. It is **not a result**: on this universe
(`research/bf_zero/universe.csv`, an END-OF-DAY >=5%-range day list) the floor IS the causal membership guarantee,
so a below-floor signal is a signal on a name the file already knows will move 5% before the close. The causally
legitimate measurement of the same question is `E/REPORT.md` §8.4, on the 09:30-knowable U1∪U2 universe, and it
has the **opposite sign**: below-floor rows are the LOSERS there (F8 N=5 −0.144 vs −0.005; F6 −0.114 vs +0.090).
The twin is reported because it was declared; it is a re-measurement of `bf_zero2` bug 1, not an edge.

# 6. Loser anatomy — what the three books have in common (step 1)

| | F14 | F8 N=30 | F11 |
|---|---:|---:|---:|
| booked TRAIN trades | 768 | 1124 | 1239 |
| booked days (green%) | 231 (48%) | 250 (50%) | 249 (48%) |
| total TRAIN R | +43.3 | +8.9 | −14.3 |
| worst 5% of days | −41.5 (25% of day-losses) | −44.9 (25%) | −60.8 (22%) |
| best 5% of days | +88.3 | +62.8 | +91.0 |
| **book with BOTH 5% tails removed** | **−3.4** | **−9.0** | **−44.5** |
| stop rate | 21.2% | 15.1% | 26.2% |
| **wick stops (share of stops)** | 100% | 46.5% | 54.8% |
| stops that had +1R on the table first | 8.6% | 5.9% | **13.8%** |
| MAE, winners / losers (% of price) | 1.26 / 3.78 | 1.90 / 5.04 | 1.77 / 4.69 |
| 150+ min held | +0.285 | +0.114 | +0.259 |

Three structural readings, none of which is a filter:
1. **Every one of these books is its own tail.** Remove the best and worst 5% of days and all three are negative.
   Any filter that works by removing bad days is, arithmetically, also removing the good ones.
2. **Day direction separates the booked trades and is not causally available.** By the day's SPY close-to-open the
   three books split −0.013/+0.002/+0.066/+0.170 (F14), −0.108/−0.089/+0.015/+0.214 (F8), −0.021/+0.119 (F11
   loser/winner means) — a 0.18-0.32 R spread. The causal twin, `spy_at_entry` (SPY return from the 09:30 open to
   the **entry minute**, 100% coverage on all three books), splits the same books by at most 0.05 R and its
   era-consistent bucket is *positive* at book level. The day's direction after the entry minute is the outcome,
   not a feature.
3. **The right shape for the F11 and F14 stops is a breakeven rule, not a veto.** 25.2% (F14) and 30.8% (F11) of
   stopped trades had +0.5R on the table first, and half or more of all stops are wicks (a bar that trades through
   the stop level and closes back above it). METHOD's 3-filter budget went to vetoes; this is the one shape change
   the anatomy points at and it was not tested. It is the first thing a future stage should try on these books.

# 7. Why every stack failed on VAL, in one table

Per-filter vetoed bucket, TRAIN booked -> VAL booked (`valdiag_*.csv`). A filter is only a rule if the bucket it
throws away keeps losing money.

| book | filter | TRAIN vetoed (H1 / H2) | VAL vetoed | holds? |
|---|---|---:|---:|---|
| F14 | V4 dist_open > 0.8 | −0.109 (−0.215 / −0.010) | **+0.210** | no |
| F14 | V2 vwap >= 0 | −0.036 (−0.058 / −0.018) | **+0.328** | no |
| F14 | V1 adv20 known | −0.337 (−0.397 / −0.108) | n = 0 | untested |
| F14 | V3 / V3b / V3c / V4b | −0.177 / −0.129 / −0.277 / −0.140 | +0.269 / +0.114 / +0.337 / +0.239 | no (4/4) |
| F8 N=30 | **W1 no wrapper** | −0.061 (−0.081 / −0.040) | **−0.023** | **yes** |
| F8 N=30 | M1 close_pos >= 0.5 | −0.024 (−0.003 / −0.042) | **+0.033** | no |
| F8 N=30 | G1 gap < 3 | −0.041 (−0.002 / −0.078) | **+0.043** | no |
| F8 N=30 | M1b close_pos >= 0.25 | −0.067 | +0.011 | no |
| F8 N=30 | G1b / C1 / D1 / S1 / X1 / I1 | −0.081 / −0.100 / −0.073 / −0.060 / −0.031 / −0.017 | −0.013 / −0.010 / −0.092 / −0.100 / −0.055 / −0.013 | **yes (6/6)** |
| F11 | **P1 prev_day_range >= 5** | **−0.190 (−0.188 / −0.192)** | **−0.086** | **yes** |
| F11 | P1b prev_day_range >= 8 | −0.048 (−0.029 / −0.069) | −0.050 | **yes** |
| F11 | B1 sig_body < 1.0 | −0.091 (−0.103 / −0.081) | **+0.035** | no |
| F11 | B1b / PR1 / D0 | −0.039 / −0.133 / −0.155 | +0.124 / +0.114 / **+0.450** | no (3/3) |

Counted exactly (`vetoed_persistence.csv`): of the 29 filter cells, **23** have a vetoed bucket that is negative
in BOTH TRAIN halves at book level and **22** of those have a VAL instance. Of those 22, **9 hold their sign on
VAL and 13 flip** — and the split is not uniform across books:

| book | era-passing buckets | hold on VAL | flip on VAL |
|---|---:|---:|---:|
| F14 | 7 | **0** | 6 (+1 untested) |
| F8 N=30 | 10 | **7** | 3 |
| F11 | 6 | **2** | 4 |

Read carefully, that table is the whole stage. **F14's separation does not survive at all** — zero of seven. In
**F8 N=30 most buckets DO survive** — and the stack still loses on VAL, because the three with mechanisms
(wrapper, Rule M, gap) are exactly the mixed set: only the wrapper veto holds, while the six that hold are the
calendar / day-context buckets for which no mechanism sentence exists and which were declared non-primary before
the VAL read. In **F11 the two that hold are P1 at both thresholds** — the same rule, and the one the freeze's
`each`-leg did not fail on.

# 8. Cells looked at (the multiplicity denominator)

| what | count |
|---|---:|
| era-consistency cells (feature x bucket) inspected in step 1 | 138 + 137 + 135 = **410** |
| single-filter cells declared and scored on TRAIN | 9 + 11 + 9 = **29** |
| stack prefixes (incl. the declared F8 N=30 companion) | 3 + 3 + 3 + 2 = **11** |
| x 2 booking conventions, cells with a book | **83 TRAIN book cells** (the permutation's denominator) |
| VAL cells (each stack + every single's vetoed bucket, 3 books) | 3 + 29 = **32** |
| declared sensitivities re-scored (`2R stop-1%` x 3, no-floor x 3, `>=10:00` x 1) | **7** |
| **TEST cells** | **0** |
| **Stage H (this sub-stage) total numbers looked at** | **~540** |

On top of Stage C's 1,305, Stage D1's 1,353 and Stage E's own count.

# 9. Smallest visible effect (MDE = 2.8 x SE per trade, and the same at 4 slots)

| book | final stack | TRAIN tr/wk | **VAL MDE / trade** | VAL MDE / week | at $100 risk |
|---|---|---:|---:|---:|---:|
| F14 | V4+V2+V1 | 9.8 | **0.156 R** | 1.9 R | $190/wk |
| F8 N=30 | W1+M1+G1 | 8.1 | **0.158 R** | 1.0 R | $100/wk |
| F11 | P1+B1 | 12.4 | **0.252 R** | 3.3 R | $330/wk |

F11's observed VAL improvement (+0.074 R/trade) is well **under** its own MDE: at n = 289 that result is not
resolvable in either direction.

# 10. What this stage says, phrased per PLAN §1

In THIS universe (the point-in-time >=5%-range day list with the causal `range_so_far_pct >= 5` floor), at THIS
horizon (1-minute bars, entries 09:30-14:01, hold to 15:55 on the touch stop, with `2R stop-1%` as the declared
secondary), at THIS book size (12 candidates/day, 4 concurrent, both booking conventions), over THIS window
(TRAIN 2025-01-02..12-31, VAL 2026-01-01..05-31), at THIS cost (contract (c)) — **the losing trades of the F14
second break, the F8 N=30 opening-range break and the F11 close-confirmed F6 are not separable from the winners by
the 29 causal cuts tested here.** Each book's losers CAN be separated on TRAIN, era-consistently, by rules with
mechanisms that are already shipped in production elsewhere (the BF VWAP gate, ORB's short-history veto, ORB's
touchgo Rule M, ORB's PDR veto), and the separation reverses on VAL in 13 of the 22 era-passing buckets that have a VAL instance — and in 6 of 6 for F14. The smallest per-trade
effect the VAL tests could have seen is **0.156-0.252 R** (1.0-3.3 R/week at 4 slots); effects below that are
invisible here and are **not** excluded.

# 11. What to carry forward (each pre-registrable, TEST still unread for all three)

1. **`prev_day_range_pct >= 5` on F11(F6)** — the only veto in the sub-stage whose bucket is negative on TRAIN-H1
   (−0.188), TRAIN-H2 (−0.192) **and** VAL (−0.086), and whose book improves on both splits (−0.0115 -> +0.0414
   TRAIN, +0.0775 -> +0.1122 VAL). It is ORB's shipped PDR veto transferring to a second family, and it is the
   causal residue of the `news_pre` availability indicator (§3). It deserves its own pre-registration, not a
   re-read of this one.
2. **The `2R stop-1%` exit for F11** — the least tail-dependent cell produced anywhere in Stage H: TRAIN +0.0623
   (t 1.82, halves +0.075 / +0.050, **ex-top-5% −0.037**), VAL +0.0632 (t 1.15, 55% green, ex5 −0.031). Any future
   F11 work should start from this exit, not from `hold`.
3. **A breakeven / partial rule on F11 and F14** — 30.8% and 25.2% of stopped trades had +0.5R on the table first
   and more than half of all stops are wicks. That is a SHAPE the anatomy points at directly and the 3-veto budget
   never tested.
4. **Do not promote the F8 N=30 companion set** (`spy_gap not in [0,0.3)`, `spy_vs_sma20 not in [0,2)`,
   not-Friday) even though its three buckets are the ones that keep their sign on VAL. It was declared
   non-primary before the VAL read precisely because no mechanism sentence exists for it, and it must stay
   non-primary until one does.

# 12. Files

| path | what |
|---|---|
| `h_extract.py` -> `pop_{F14,F8N30,F11F6}.csv` | the three books' populations out of `C/pop_c.csv` |
| `hcore.py` | contract (c) + `run_book(12,4)` + the gate statistics (the Stage-C contract, re-implemented) |
| `hfeat.py` | the causal side-features (news, premarket, day context, index state at the entry minute, derived shape) |
| `h0_parity.py` -> `h0_parity.csv` | step 0, the parity anchor (24/24 cells exact) |
| `h1_anatomy.py` -> `h1_anatomy_*.md`, `h1_buckets_*.csv` | step 1 (availability audit, concentration, worst-20 days, winners-vs-losers, path, 410 era cells) |
| `FREEZE_F14.md`, `FREEZE_F8N30.md`, `FREEZE_F11F6.md` | the stacks frozen in writing, each before its `--val` run |
| `h_eval.py` -> `h_eval_*.md` | steps 2-4 (the coarse grids, both booking conventions, the stacks, VAL) |
| `h_valdiag.py` -> `valdiag_*.csv` | the per-filter VAL contribution table of §7 |
| `h_perm.py` -> `h_perm.md`, `h_perm_cells.csv` | the 83-cell search-adjusted permutation |
| `h_nofloor_extract.py`, `h_nofloor_score.py` -> `nofloor_*.csv`, `nofloor.log` | the declared no-floor twin (§5) |
| `book_{BOOK}{,_stopm1,_nofloor,_m600,_alt}_{TRAIN,VAL}.csv` | per-trade dumps of every final book, for an independent rebuild |
| `etf_minute_ret.csv` | the SPY/IWM minute-of-day return table used for `spy_at_entry` / `iwm_at_entry` |
