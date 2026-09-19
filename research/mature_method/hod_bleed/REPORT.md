# hod_bleed — the ratchet after +0.4 R — REPORT (2026-09-19)

Owner 9/19: *"The losses live in the first ten minutes. A losing trade runs +0.45R by minute 6, then
bleeds to a full stop at minute 28. If we got to +0.4 we can have a full stop/exit at -0.1 or
whatever. Look at those losers and find the rule. I'm sure one can also see it on the volume, or
MACD, or VWAP — seeing the sentiment is likely to turn south. Maybe we will lose some of the winners,
but will cut out the losers."*

`PREREG.md` committed `076f5b0` **before the walk**; `ADDENDUM.md` (the 12 instantiated E1 cells, by
the PREREG's own TRAIN-only rule) committed `9e9c7d9` **before `cells.py` was run for the first
time**. Artifacts: `walk3.py` -> `sigs.csv` + `bars/*.parquet` (8,847 signals, 1,019,453 post-fill
bars) - `part1.py` -> `part1.txt`, `as_matrix.csv` - `core.py` (ONE implementation of every exit) -
`cells.py` -> `part2.txt`, `cells.csv`, `nulls.csv`. One python process, `nice -n 15`,
`ulimit -v 1500000`, every DB read-only, no TEST-dated bar read. No config, `orb.yaml`, systemd unit,
cron, order or cache was written; the dry run was not touched.

---

## VERDICT — **STAY DRY**

*The owner's premise is testable and **the data says it is false**. Of the eventual WINNERS that
reach +0.4 R, **28.0 % (TRAIN) / 33.6 % (VAL) retrace through -0.2 R before they ever see +2 R**;
at a breakeven stop it is 42.7 % / 46.0 %. Winners do not go +0.4 -> +0.8 -> +2.0 without revisiting
entry — a third of them come back through the ratchet. And the arm level is **not a classifier**: at
+0.4 R the median LOSER arms at minute 5 and the median WINNER arms at minute 5, and 43.3 % of the
trades that reach +0.4 R are eventual losers against 37.4 % winners.*

*So the trade-off the owner named is real, measured, and **priced at a wash**: at the best (a, s) the
ratchet cuts 555 trades for +422.5 R and costs 249 trades -384.7 R — **+0.022 R a trade on TRAIN and
-0.075 R a trade on VAL**. **All 24 (a, s) pairs are negative on VAL**, by -0.067 to -0.093 R, at
day-clustered t -2.1 to -3.2. **0 of 32 declared cells pass the ship bar; 0 of 32 have a positive VAL
dNetR; TEST was never opened.** The exit family that survives best is the one that moves nothing —
the **50 % partial at +0.4 R with the stop left where it is** (+0.0098 TRAIN / -0.0277 VAL, and the
only family that preserves frequency exactly) — and it is still negative on VAL.*

*And the sharpest result in the pass is the cross: **applied to `hod_fresh`'s `C1`, the only
net-positive admission in 799 cells, the best exit takes it from +0.033 / +0.050 net to -0.054 /
-0.047, dNetR -0.100 / -0.134 at clustered t -2.29 / -2.58.** The one book on this strategy that was
above water is above water **because it lets its winners run**.*

---

## 0. Reproduction and the parity gate — EXACT

The path walk re-simulates the shipped exit from the raw tape and is compared row by row to
`pop.csv`:

| base | rows walked | max abs d-rr | `why` match | exit-minute match |
|---|---|---|---|---|
| B0 | 4,941 / 4,941 | 5.94e-08 | **1.000000** | **1.000000** |
| B2 | 7,027 / 7,027 | 8.31e-08 | **1.000000** | **1.000000** |

| reference book | n | /wk | gross | net | **cost/R** | green | rs | worst wk | total |
|---|---|---|---|---|---|---|---|---|---|
| **B0 shipped** TRAIN | 1,688 | 31.8 | -0.027 | -0.088 | **0.0608** | 41.5 % | 7 | -$2,718 | **-$14,835** |
| **B0 shipped** VAL | 820 | 35.7 | +0.016 | -0.050 | **0.0661** | 43.5 % | 6 | -$2,076 | **-$4,128** |
| B2 shipped TRAIN / VAL | 1,622 / 706 | 30.6 / 30.7 | -0.039 / +0.083 | -0.107 / +0.013 | 0.0680 / 0.0701 | 32.1 / 43.5 % | 5 / 4 | | -$17,346 / +$893 |
| C1 shipped TRAIN / VAL | 731 / 368 | 13.8 / 16.0 | +0.100 / +0.123 | +0.033 / +0.050 | 0.0678 / 0.0730 | 43.4 / 47.8 % | 5 / 4 | | +$2,391 / +$1,827 |

Every printed digit of `hod_break` §6, `hod_filter_stack` §2, `hod_losers` and `hod_fresh` §6 is
reproduced. **`hod_fresh`'s cost correction is confirmed independently on this pass's own walk: a
BOOKED trade pays 0.061 R (TRAIN) / 0.066 R (VAL), not +0.2151 R.** Every cost number below is
re-measured per cell as `mean(rr - net)` on that cell's own booked set; no constant is carried.

---

# PART 1 — THE PATH MAP (descriptive; no cell, no selection)

## 1. The three classes, minute by minute (shipped B0 book)

Median **current R** (the bar's close), **MFE** and **retrace from MFE**, at trade-minutes 1/3/5/10/15:

| split | class | n | cur@1 | cur@3 | cur@5 | cur@10 | cur@15 | mfe@5 | mfe@10 | rtr@10 | rtr@15 | med MFE | med MAE | hold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN | **win** | 450 | +0.052 | +0.165 | +0.237 | **+0.510** | +0.681 | +0.462 | +0.795 | +0.218 | +0.237 | **+2.147** | -0.302 | 49 |
| TRAIN | **lose** | 992 | -0.027 | -0.093 | -0.144 | **-0.235** | -0.363 | +0.221 | +0.283 | **+0.569** | **+0.740** | **+0.428** | -1.072 | 27 |
| TRAIN | eod | 246 | +0.000 | -0.008 | +0.049 | +0.060 | +0.085 | +0.258 | +0.366 | +0.294 | +0.346 | +1.272 | -0.654 | 300 |
| VAL | **win** | 211 | +0.039 | +0.126 | +0.215 | +0.387 | +0.563 | +0.422 | +0.711 | +0.270 | +0.312 | +2.119 | -0.361 | 58 |
| VAL | **lose** | 454 | -0.034 | -0.130 | -0.189 | -0.366 | -0.571 | +0.191 | +0.225 | +0.633 | +0.866 | +0.355 | -1.073 | 23 |
| VAL | eod | 155 | +0.000 | +0.031 | +0.039 | +0.095 | +0.128 | +0.243 | +0.391 | +0.283 | +0.311 | +1.346 | -0.634 | 337 |

The anatomy holds exactly: the loser's median MFE is **+0.428 R** and by minute 10 its close is
already **-0.235 R** with **+0.569 R** given back from the peak. But note the column the owner's rule
needs and the anatomy never printed: **the winner gives back +0.218 R from its own peak at minute 10
and +0.237 R at minute 15** — the winner's path is not monotone either.

## 2. The arm level is not a classifier

| split | a | reach n | reach % | win % | lose % | eod % | med arm minute (all) | med arm minute (LOSERS) |
|---|---|---|---|---|---|---|---|---|
| TRAIN | 0.3 | 1,283 | 76.0 % | 35.1 | **46.5** | 18.4 | 4 | 3 |
| TRAIN | **0.4** | 1,202 | 71.2 % | 37.4 | **43.3** | 19.2 | **5** | **5** |
| TRAIN | 0.5 | 1,106 | 65.5 % | 40.7 | 39.8 | 19.5 | 7 | 6 |
| TRAIN | 0.6 | 1,027 | 60.8 % | 43.8 | 36.1 | 20.1 | 8 | 7 |
| VAL | **0.4** | 568 | 69.3 % | 37.1 | **37.9** | 25.0 | **5** | **5** |

**Seven of ten booked trades reach +0.4 R, and more of them are losers than winners.** The arm fires
at the same minute on both classes. Whatever a ratchet does after it, it does to a population that is
not enriched in losers — which is the whole reason the arithmetic below comes out flat.

## 3. THE (a, s) MATRIX — the owner's question, answered

**% of the eventual WINNERS that reach `a` and are then ejected by a stop at `s` before +2 R:**

| | s=-0.3 | s=-0.2 | s=-0.1 | s=0.0 | s=+0.1 | s=+0.2 |
|---|---|---|---|---|---|---|
| **TRAIN** a=0.3 (450 win) | 27.1 % | 33.6 % | 42.0 % | 50.9 % | 60.7 % | 75.3 % |
| **TRAIN** a=0.4 | 23.8 % | **28.0 %** | 35.1 % | 42.7 % | 50.9 % | 63.8 % |
| **TRAIN** a=0.5 | 21.1 % | 24.2 % | 30.4 % | 37.3 % | 45.8 % | 56.4 % |
| **TRAIN** a=0.6 | 18.2 % | 20.0 % | 25.6 % | 32.4 % | 39.8 % | 49.3 % |
| **VAL** a=0.3 (211 win) | 34.1 % | 39.8 % | 46.4 % | 55.0 % | 65.4 % | 75.8 % |
| **VAL** a=0.4 | 28.4 % | **33.6 %** | 38.4 % | 46.0 % | 56.9 % | 65.4 % |
| **VAL** a=0.5 | 24.6 % | 28.4 % | 32.7 % | 38.4 % | 49.3 % | 58.3 % |
| **VAL** a=0.6 | 20.4 % | 24.2 % | 28.4 % | 33.6 % | 44.1 % | 52.1 % |

> **The brief's own decision rule: "if winners run +0.4 -> +0.8 -> +2.0 without revisiting entry, a
> tight stop after +0.4 R is free; if they retrace through it 40 % of the time, it kills the book."
> The answer is 28 % / 34 % at the owner's own level (-0.2 R) and 43 % / 46 % at breakeven** — between
> the two thresholds the brief named, and on the wrong side of them at every `s >= -0.1` on VAL.

## 4. THE TRADE-OFF, in gross R per booked trade

`dR = [sum R saved on trades the ratchet improves - sum R given up on trades it hurts] / n`:

| | s=-0.3 | s=-0.2 | s=-0.1 | s=0.0 | s=+0.1 | s=+0.2 |
|---|---|---|---|---|---|---|
| **TRAIN** a=0.3 | +0.0109 | +0.0176 | +0.0143 | +0.0111 | +0.0113 | -0.0119 |
| **TRAIN** a=0.4 | +0.0089 | **+0.0224** | +0.0183 | +0.0205 | +0.0203 | +0.0040 |
| **TRAIN** a=0.5 | +0.0015 | +0.0141 | +0.0096 | +0.0076 | +0.0018 | -0.0114 |
| **TRAIN** a=0.6 | -0.0046 | +0.0114 | +0.0059 | -0.0018 | -0.0068 | -0.0216 |
| **VAL** a=0.3 | -0.0913 | -0.0860 | -0.0828 | -0.0851 | -0.0826 | -0.0804 |
| **VAL** a=0.4 | -0.0745 | **-0.0751** | -0.0665 | -0.0703 | -0.0752 | -0.0665 |
| **VAL** a=0.5 | -0.0840 | -0.0766 | -0.0693 | -0.0676 | -0.0823 | -0.0814 |
| **VAL** a=0.6 | -0.0773 | -0.0750 | -0.0736 | -0.0756 | -0.0931 | -0.0930 |

**All 24 negative on VAL.** The counts at the TRAIN optimum `(0.4, -0.2)`: **555 trades helped for
+422.5 R against 249 hurt for -384.7 R** — the "cut the losers" side is barely bigger than the
"lose some winners" side, because (§2) the arm is not selective and (§3) a third of the winners come
back. On VAL the same cell is **226 helped / +173.0 R against 158 hurt / -234.6 R** — the sign flips
because VAL's winners retrace more (33.6 % vs 28.0 %) and VAL has fewer of them.

---

# PART 2 — THE 32 DECLARED CELLS

Every cell is reported twice: **FIXED-COHORT** (the shipped book's own trades under the new exit —
the clean trade-off, where the ship bar is evaluated) and **RE-BOOKED** (`run_book(12,4)` re-run, so
earlier exits free slots earlier). `$` at the live $100 risk. **dNetR = new - shipped, per trade,
paired.**

## E1 — the ratchet (12 cells, the top-12 (a, s) by TRAIN dR, `ADDENDUM.md`)

| cell | TRAIN dNetR | H1 | H2 | clust t | **VAL dNetR** | **VAL clust t** | won/lost TRAIN | re-booked TRAIN | re-booked VAL |
|---|---|---|---|---|---|---|---|---|---|
| **a=0.4 s=-0.2** | **+0.0182** | +0.0127 | +0.0232 | +0.85 | **-0.0804** | **-2.54** | cut 555/+422R, lost 249/-385R | 35.8/wk, 37.7 % grn, -$14,668 | 17.4 % grn, **-$12,163** |
| a=0.4 s=0.0 | +0.0146 | +0.0079 | +0.0209 | +0.58 | -0.0771 | -2.15 | cut 573, lost 328 | 36.8/wk, 32.1 %, -$16,733 | 21.7 %, -$14,265 |
| a=0.4 s=+0.1 | +0.0137 | +0.0030 | +0.0235 | +0.50 | -0.0833 | -2.15 | | 30.2 %, -$17,549 | 13.0 %, -$14,259 |
| a=0.4 s=-0.1 | +0.0133 | +0.0071 | +0.0191 | +0.58 | -0.0723 | -2.19 | | 34.0 %, -$16,643 | 17.4 %, -$11,893 |
| a=0.3 s=-0.2 | +0.0126 | +0.0169 | +0.0086 | +0.54 | -0.0922 | -2.77 | | 37.7 %, -$16,262 | 21.7 %, -$11,778 |
| a=0.5 s=-0.2 | +0.0105 | +0.0028 | +0.0176 | +0.54 | -0.0814 | -2.89 | | 39.6 %, -$14,392 | 26.1 %, -$12,409 |
| a=0.3 s=-0.1 | +0.0084 | +0.0118 | +0.0051 | +0.33 | -0.0896 | -2.49 | | 28.3 %, -$18,559 | 26.1 %, -$12,062 |
| a=0.6 s=-0.2 | +0.0083 | **-0.0092** | +0.0247 | +0.46 | -0.0792 | -3.00 | | 37.7 %, -$15,094 | 26.1 %, -$10,744 |
| a=0.3 s=-0.3 | +0.0067 | +0.0037 | +0.0095 | +0.31 | -0.0967 | -3.20 | | 30.2 %, -$18,156 | 26.1 %, -$13,461 |
| a=0.5 s=-0.1 | +0.0053 | +0.0002 | +0.0101 | +0.25 | -0.0746 | -2.53 | | 35.8 %, -$16,477 | 30.4 %, -$12,372 |
| a=0.3 s=0.0 | +0.0042 | +0.0080 | +0.0007 | +0.16 | -0.0930 | -2.39 | | 30.2 %, -$19,151 | 17.4 %, -$12,875 |
| a=0.3 s=+0.1 | +0.0035 | +0.0069 | +0.0003 | +0.11 | -0.0917 | -2.19 | | 28.3 %, -$19,352 | 13.0 %, -$13,183 |

**Every one of the 12 is negative on VAL at a day-clustered t between -2.2 and -3.2, and every one
worsens green weeks on both splits** (41.5 % -> 28-40 %, 43.5 % -> 13-30 %). The best TRAIN cell is
+0.018 R at clustered t **+0.85** — inside its own noise. This is `hod_losers` `P8` reproduced at
every arm level the anatomy nominated, including the ones P8 could not reach.

## E2 — the time-conditioned ratchet (4 cells)

Requiring the arm within the first 6 or 10 minutes does not rescue it: `(0.4, -0.2) N=6` is
**-0.0004 TRAIN / -0.0536 VAL**, `N=10` **+0.0065 / -0.0597**. It fires on half as many trades
(144 hurt / 327 cut at N=6 vs 249 / 555 unconditioned) and keeps the sign. The "a late arm is a
different animal" hypothesis is **not supported**: restricting to early arms removes as much benefit
as cost.

## E3 — retrace from peak (4 cells)

The trailing form of the same idea is the best TRAIN family (`a=0.4, d=0.5`: **+0.0239**, clustered t
+0.82) and the **least-negative ratchet-like family on VAL** (-0.0460, clustered t -1.09). It is still
negative on VAL, still below the bar, and it costs green weeks on both splits (32.1 % / 17.4 %).
`a=0.3 d=0.4` is the **only cell in the pass above its own TRAIN null band** (34.0 % green vs null
mean 25.3 %, p95 32.1) — and it is inside the band on VAL and -0.068 R there.

## E4 — the owner's three signals, as exits on a trade armed at +0.4 R (6 cells)

| cell | TRAIN dNetR | H1 / H2 | clust t | VAL dNetR | clust t | TRAIN cut / lost | re-booked green TRAIN / VAL |
|---|---|---|---|---|---|---|---|
| E4a vol >= 2x breakout bar, closing down | +0.0013 | -0.0395 / +0.0392 | +0.07 | **-0.0139** (least-bad of 32) | -0.55 | 234 / 222 | 35.8 % / 21.7 % |
| E4b vol fades < 0.5x for 3 bars | +0.0016 | +0.0154 / -0.0114 | +0.06 | -0.0528 | -1.37 | 505 / 425 | 28.3 % / 21.7 % |
| **E4c close below session VWAP** | +0.0153 | +0.0118 / +0.0186 | **+1.35** (best in the pass) | -0.0560 | -2.65 | **289 / 114** | **43.4 %** / 39.1 % |
| E4d MACD histogram < 0 | **+0.0244** | +0.0115 / +0.0364 | +0.82 | -0.0702 | -1.59 | 619 / 493 | 32.1 % / 21.7 % |
| E4e MACD signal cross down | +0.0232 | +0.0107 / +0.0349 | +0.82 | -0.0772 | -1.85 | 601 / 487 | 34.0 % / 21.7 % |
| E4f any two of three agree | +0.0152 | -0.0251 / +0.0527 | +0.76 | -0.0488 | -1.51 | 422 / 283 | 32.1 % / 26.1 % |

**The owner's intuition that "sentiment turning south" is visible is partly right and entirely
unprofitable.** VWAP is the best-behaved trigger in the pass — the highest clustered t (+1.35), the
only cell that **improves** TRAIN green weeks (41.5 -> 43.4 %), and by far the most surgical
(289 losers cut against only 114 winners hurt, a 2.5:1 ratio no ratchet gets near). It is still
-0.056 R on VAL at clustered t -2.65. The two volume forms are inert (abs dNetR <= 0.002 on TRAIN,
and E4a's halves have opposite signs, -0.040 / +0.039). MACD is the most active and the most
half-unstable. "Any two of three" is worse than VWAP alone.

## E5 — the combination (2 cells)

Ratchet `(0.4, -0.2)` + MACD histogram: **B0 +0.0165 / -0.0766**; **B2 +0.0321 / -0.1165**. B2 is the
best TRAIN dNetR in the pass (clustered t +1.25) and the **worst VAL** (clustered t -3.10), with
green weeks 32.1 -> 20.8 % and 43.5 -> 17.4 %. Combining two exits that each fail combines their
failures.

## E6 — the partial with the stop LEFT WHERE IT IS (3 cells)

| cell | TRAIN dNetR | H1 / H2 | clust t | VAL dNetR | clust t | trades/wk | TRAIN $ | VAL $ | green T / V |
|---|---|---|---|---|---|---|---|---|---|
| 50 % @ +0.3 R | +0.0054 | +0.0116 / -0.0004 | +0.31 | -0.0351 | -1.43 | **31.8 (unchanged)** | -$13,927 | -$7,003 | 37.7 / 21.7 % |
| **50 % @ +0.4 R** | **+0.0098** | **+0.0145 / +0.0054** | +0.59 | **-0.0277** | -1.16 | **31.8 (unchanged)** | **-$13,176** | -$6,397 | 37.7 / 21.7 % |
| 50 % @ +0.5 R | +0.0052 | +0.0102 / +0.0005 | +0.33 | -0.0309 | -1.41 | **31.8 (unchanged)** | -$13,957 | -$6,664 | 37.7 / 26.1 % |

This family is the **only one in the pass that preserves frequency exactly** — 1,688 TRAIN and 820
VAL trades, 31.8 / 35.7 a week, identical to the shipped book — because the runner holds the slot to
the same minute. It is also the **only family with both TRAIN halves positive at the middle rung**.
The mechanism came out exactly as predicted before scoring: at +0.4 R it fires on most trades, books
about -0.33 R instead of -1.06 R on the losers it catches and about +1.2 R instead of +2.0 R on the
winners — **629 trades improved for +407.9 R against 572 worsened for -381.4 R.** The net is
+0.0098 R on TRAIN and **-0.0277 R on VAL**: the smallest damage of any cell that fires on most
trades, and still the wrong sign where it matters. Its green weeks collapse on VAL
(43.5 -> 21.7 %) for a reason worth stating: halving every winner turns marginal green weeks into
red ones even when the mean barely moves — a ratio moving opposite to the dollars, F7's lesson again.

## E7 — the best exit on `hod_fresh`'s `C1` — the sharpest result in the pass

`C1` = `consol_bars >= 20` x last-5-bar stop x `spy_r5_pct > 0`, the only net-positive cell in 799,
reproduced here to the digit (**TRAIN +0.100 gross / +0.033 net / 43.4 % green / +$2,391**;
**VAL +0.123 / +0.050 / 47.8 % / +$1,827**). The best exit of E1-E6 (E5, ratchet + MACD histogram)
applied to it:

| | dNetR | clust t | book net | book green | book $ |
|---|---|---|---|---|---|
| TRAIN | **-0.1004** (H1 -0.1206 / H2 -0.0834) | **-2.29** | +0.033 -> **-0.054** | 43.4 -> 39.6 % | +$2,391 -> **-$4,592** |
| VAL | **-0.1341** | **-2.58** | +0.050 -> **-0.047** | 47.8 -> 43.5 % | +$1,827 -> **-$2,358** |

**The only above-water admission on this book is above water because it lets its winners run.** Its
trades hold to a median 2.1 R MFE; cutting them at a ratchet or a MACD flip destroys the entire edge
and then some, with the largest and most significant effect in the pass — in the wrong direction, on
both splits, both halves. This is the cleanest confirmation available that the bleed is not a
harvestable feature of this book: the same exit that is a wash on the shipped population is a
-0.10 R disaster on the only population with an edge.

## Count-matched permutation null (2,000 draws, per-week pick count fixed)

| cell | split | observed green | null mean | [p5, p95] | |
|---|---|---|---|---|---|
| E3 a=0.3 d=0.4 | TRAIN | 34.0 % | 25.3 % | [18.9, 32.1] | **ABOVE** |
| E3 a=0.3 d=0.4 | VAL | 21.7 % | 16.1 % | [8.7, 26.1] | inside |
| E5 ratchet+hist (B2) | TRAIN / VAL | 20.8 / 17.4 % | 20.6 / 18.5 % | | inside / inside |
| E4d, E4e, E3 a=0.4 d=0.4/0.5 | TRAIN & VAL | | | | all inside |
| B0 shipped | TRAIN / VAL | 41.5 / 43.5 % | 34.7 / 38.7 % | [28.3, 41.5] / [30.4, 47.8] | inside (at TRAIN p95) |
| **C1 shipped** | TRAIN / VAL | 43.4 / 47.8 % | **49.2 / 50.0 %** | [41.5, 56.6] / [39.1, 60.9] | **below its own null mean** |

One cell of 32 x 2 clears a null band on one split, which is what 64 draws produce by chance. Every
other cell's week shape is its pick count.

---

## BOTH BARS

**SHIP BAR (PREREG §6) — 0 of 32.** It required TRAIN dNetR >= +0.10 with both halves same-signed,
VAL dNetR > 0, TRAIN day-clustered t >= 2.0, and green weeks not worse on both splits.

* best TRAIN dNetR in the pass: **+0.0321** (E5 on B2), clustered t **+1.25**, VAL **-0.1165**;
* best TRAIN dNetR on the shipped base B0: **+0.0244** (E4d), clustered t +0.82, VAL -0.0702;
* **cells with a positive VAL dNetR: 0 of 32.** The least-bad is E4a at **-0.0139**;
* **cells with TRAIN dNetR >= +0.10: 0 of 32**;
* green weeks improved on BOTH splits: **0 of 32** (E4c improves TRAIN 41.5 -> 43.4 % and loses VAL
  43.5 -> 39.1 %).

**Does any cell lift the book across the cost line? No — and the line is not where the programme
said it was.** Re-measured on this pass's own booked sets, `cost/R` is **0.0608 / 0.0661** for B0,
not +0.2151 R. The honest arithmetic, stated in the PREREG before scoring: B0 is net **-0.088**
(TRAIN) / **-0.050** (VAL), so a cell that just cleared the +0.10 R ship bar would land the book at
about **+0.01 R — AT break-even, not through it**. What the cells actually deliver on B0 is
+0.024 R at best on TRAIN (book net -0.088 -> -0.074) and **negative on VAL at every one of the 32**
(book net -0.050 -> -0.078 ... -0.142). **No cell takes the shipped book net-positive on either TRAIN
half, let alone on both plus VAL.** A further mechanical cost: every new exit **raises** the cost
term, because it converts free `target` exits (ratio 0.0) and cheap `eod` exits (0.412) into
marketable ones (0.875) — B0's booked `cost/R` goes 0.0608 -> 0.0650 (E1) / 0.0666 (E6b) / 0.0705
(E4d). An exit rule on this book pays twice.

**MDE — a powered rejection.** On the paired per-trade difference, at 80 % power: median **0.060 R
(TRAIN) / 0.088 R (VAL)** across the 32 cells; **0.030 / 0.054 R** on E4c (the tightest); 0.040 /
0.056 R on E6b; 0.114 / 0.160 R on E7's smaller cohort. The ship bar's +0.10 R sits **above** the
TRAIN MDE, so the bar was detectable. And the VAL rejections are not a null: twelve of twelve E1
cells are negative at clustered abs t 2.2-3.2, and E7 at 2.3 / 2.6. Stated per the phrasing rule:
**no exit rule of this family was profitable in THIS universe, at THIS 12/4 book size, over
2025 + 2026-01->05, at the measured 0.061-0.066 R cost — and the pass could have seen +0.06 R.**

**Multiplicity.** 32 declared decision cells (12 + 4 + 4 + 6 + 2 + 3 + 1) x 2 splits; Part 1's
descriptive tables, including the full 24-pair (a, s) matrix, are **not** cells and are reported in
full. Programme cumulative: **799** (through `hod_fresh`) **+ 32 = 831.** Expected largest abs t under
a pure null over 32 x 2 is about 2.8 — moot in the favourable direction: **no cell reaches clustered
t +1.4 on TRAIN**, and the significant results are all negative.

**TEST**: sealed throughout (`FREEZE.md`), never opened, and nothing earned opening it.

---

## SHIP-TO-DRY diff — there is none, and here is what it would have cost to build

Had a cell cleared the bar, the engine work is not small, and it is worth recording because it
changes the cost-benefit of any future exit study on this book:

* **A ratchet is a stop-leg re-anchor.** `HodBreakEngine` holds the exits as broker bracket legs
  (`tp_leg_id`, `sl_leg_id`) and already follows `replaced_by` for BOTH legs
  (`hod_break_engine.py:891-896`). But the client exposes only
  `data_sources/alpaca_client.py::replace_order_limit_price` — **there is no
  `replace_order_stop_price`**. A ratchet needs that method, a bar-close-driven ratchet decision in
  the drain loop (never on ticks — the BF `r_basis` lesson), and the same "did the replace take
  effect" error path the TP re-anchor has. **A build, not a config flip.**
* **A partial (E6) is a bigger build than BF's.** BF's partial goes through
  `StopMonitor.execute_partial_exit`, which owns the whole exit. HOD's exits are **broker OCO legs
  sized to the full position**: selling half outside the bracket leaves both legs over-sized, and a
  later leg fill would sell shares we no longer hold — the exact failure the 9/15 parity review fixed
  for the 15:55 flat. A HOD partial therefore needs a partial sell **plus** a quantity replace on
  both legs, atomically. `StopMonitor.execute_partial_exit` **cannot** be reused as-is.

Neither is worth writing for a rule that is negative on VAL.

## What this pass adds, and what should not be re-run

1. **The owner's question is answered with a number, not a verdict.** Winners that reach +0.4 R
   retrace through -0.2 R **28 % (TRAIN) / 34 % (VAL)** of the time and through breakeven 43 % / 46 %.
   The brief's own threshold for "it kills the book" was 40 %. **Do not re-open the ratchet family.**
2. **The arm level is not a classifier** (§2): 7 in 10 booked trades reach +0.4 R, more of them
   losers than winners, and both classes arm at the same minute. Any rule of the form "after it goes
   +x R, do y" starts from a population with no edge in it.
3. **The three signals are now measured on HOD.** VWAP is the surgical one (2.5:1 losers cut to
   winners hurt, the only cell to improve TRAIN green weeks, clustered t +1.35) and is -0.056 R on
   VAL; volume is inert; MACD is active and half-unstable. **The VWAP trigger is the only thing in
   this pass worth carrying into a future pre-registration**, and only as an admission-side or
   day-level idea, never as this exit.
4. **The partial-with-stop-unmoved is the right shape and the wrong book.** It is the only exit
   family that preserves frequency exactly and the only one with both TRAIN halves positive — and it
   is -0.028 R on VAL. If the owner wants a consistency exit, it belongs on a book whose winners are
   not the entire edge.
5. **E7 is the load-bearing negative**: the best exit destroys the only net-positive admission in 799
   cells (-0.100 / -0.134 R, clustered t -2.29 / -2.58). **On this book, cutting the bleed and
   keeping the tail are the same lever pulled in opposite directions.**

**Recommended action: NONE.** `config.yaml hod_break` stays exactly as the owner set it
(`enabled: true, dry_run: true`). `orb.yaml`, the service, the crons and every order were untouched.
