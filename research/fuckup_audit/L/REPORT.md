# Stage L — TRANSFER FILTERS on six near-profitable books

Pre-registration: `L/PREREG.md` (written before any run; three amendments appended, all disclosures).
Every filter here was taken at the threshold ANOTHER book or period already ships. Nothing was tuned.
Splits TRAIN 2025 / VAL 2026-01..05 / TEST 2026-06..09. Numbers are net R per trade unless marked `$`.

---

# ONE PAGE

## 1. The grid — TRAIN improvement / VAL improvement, in R per trade

Improvement = mean net R of the re-booked filtered book minus mean net R of the base book, same split,
same week denominator. `PASS` = TRAIN improvement >= +0.03 AND VAL improvement >= 0 AND VAL mean > 0 (frozen).
`_deg_` = structurally degenerate for that book (reason in section 5).

| book | T1 pdr>=8 | T2 rsf>=5 | T3 index>0 | T4 1st only | T5 <10:00 | T6 10-min stop | T7 vol exit | T8 stop>=3% |
|---|---|---|---|---|---|---|---|---|
| **B1** bull flag raw | -0.004 / +0.011 | -0.006 / +0.000 | +0.105 / +0.048 | +0.029 / +0.104 | **+0.086 / +0.189 PASS** | -0.079 / -0.113 | -0.100 / -0.122 | +0.053 / +0.012 |
| **B2** F6 first-break | _deg_ | _deg_ | **+0.042 / +0.064 PASS** | **+0.102 / +0.475 PASS** | -0.010 / +0.104 | -0.068 / -0.114 | -0.057 / -0.144 | +0.038 / -0.129 |
| **B3** F14 | +0.007 / -0.016 | _deg_ | +0.068 / -0.063 | **+0.056 / +0.021 PASS** | _deg_ | -0.086 / -0.052 | -0.064 / -0.065 | -0.007 / -0.056 |
| **B4** F8 N=30 | -0.038 / +0.067 | _deg_ | -0.029 / +0.080 | -0.026 / +0.011 | _deg_ | -0.027 / -0.011 | -0.020 / -0.020 | +0.003 / +0.000 |
| **B5** ORB B+ | _deg_ | -0.247 / -0.442 | -0.121 / -0.773 | **+0.056 / +1.785 PASS** | _deg_ | -0.209 / -0.643 | -0.221 / -0.700 | -0.071 / +0.535 |
| **B6** S1 short | +0.019 / -0.092 | _deg_ | -0.010 / -0.056 | +0.013 / +0.005 | +0.012 / +0.000 | -0.044 / -0.062 | -0.061 / -0.070 | +0.008 / +0.003 |

Column totals over the 39 single-filter cells that ran: **T6 negative on 6 of 6 books on BOTH splits, T7
negative on 6 of 6 on BOTH splits** — the two exit rules are the only filters with a unanimous sign, and it is
the wrong one. T4 is positive on TRAIN in 5 of 6 and on VAL in 6 of 6. T1/T2/T8 move nothing anywhere
(|improvement| <= 0.07 except on B5).

## 2. The stacks (the two filters with the largest TRAIN improvement that are both VAL-sign-positive)

| book | stack | TRAIN imp | VAL imp | pass |
|---|---|---:|---:|---|
| B1 | T3 + T5 | +0.2090 | +0.1460 | yes |
| B2 | T4 + T3 | +0.1427 | +0.4650 | yes |
| B3 | — | — | — | **not run** (only one filter had VAL improvement >= 0) |
| B4 | T8 + T4 | -0.0263 | +0.0105 | no |
| B5 | T4 + T8 | +0.1821 | +0.9643 | yes |
| B6 | T4 + T5 | +0.0416 | +0.0551 | yes |

## 3. TEST, read once, for the 9 cells that passed on both TRAIN and VAL

| book | cell | TRAIN imp | VAL imp | TEST n | TEST mean R | TEST t | TEST imp | TEST base | ex-top-5% | +3R cap | TEST $/mo |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | T5 | +0.086 | +0.189 | 81 | -0.104 | -0.70 | **-0.018** | -0.086 | -0.302 | -0.142 | -633 |
| B1 | T3+T5 | +0.209 | +0.146 | 50 | -0.147 | -0.73 | **-0.061** | -0.086 | -0.335 | -0.202 | -553 |
| B2 | T3 | +0.042 | +0.064 | 256 | -0.074 | -1.06 | **-0.032** | -0.042 | -0.240 | -0.100 | -1,426 |
| B2 | **T4** | +0.102 | +0.475 | 68 | **+0.148** | +0.73 | **+0.190** | -0.042 | -0.106 | +0.023 | **+753** |
| B2 | T4+T3 | +0.143 | +0.465 | 62 | +0.007 | +0.04 | +0.049 | -0.042 | -0.222 | -0.070 | +34 |
| B3 | T4 | +0.056 | +0.021 | 67 | -0.004 | -0.03 | +0.042 | -0.045 | -0.194 | -0.054 | -20 |
| B6 | **T4+T5** | +0.042 | +0.055 | 45 | **+0.067** | +0.63 | **+0.077** | -0.010 | **+0.003** | +0.067 | **+225** |
| B5 | T4 | +0.056 | +1.785 | 7 | +0.067 | +0.49 | -0.204 | +0.271 | -0.064 | +0.067 | +19 |
| B5 | T4+T8 | +0.182 | +0.964 | 11 | -0.068 | -0.53 | -0.339 | +0.271 | -0.160 | -0.068 | -48 |

**4 of 9 passers have a positive TEST improvement; 3 of 9 have a positive TEST mean; 1 of 9 survives the
ex-top-5% tail test** (B6 T4+T5, at +0.003 on 45 trades). No passer's TEST t exceeds 0.73.

## 4. Multiplicity, power, money

- **Permutation, whole grid.** 500 day-level sign-flip draws over the 43 cells with >= 20 TRAIN trades on both
  sides: **observed max |TRAIN improvement| 0.2469, null 95th percentile 0.5496, null mean 0.3473, p = 0.794.**
  Nothing in the grid is larger than the grid's own noise.
- **Smallest per-trade improvement each book could see** (2.8 x SE of the BASE book):

  | book | TRAIN MDE | VAL MDE | base TRAIN n / tr per wk | base VAL n |
  |---|---:|---:|---|---:|
  | B1 bull flag raw | 0.189 | 0.260 | 451 / 8.5 | 229 |
  | B2 F6 first-break | 0.116 | 0.230 | 1,113 / 21.0 | 531 |
  | B3 F14 | 0.120 | 0.153 | 766 / 14.5 | 362 |
  | B4 F8 N=30 | 0.066 | 0.094 | 1,124 / 21.2 | 456 |
  | B5 ORB B+ | 0.649 | 1.808 | 47 picks | 19 picks |
  | B6 S1 short | 0.112 | 0.176 | 478 / 9.6 | 258 |

  Only **B4** can see an effect under 0.1 R. On **B5** nothing under +0.65 R per pick is resolvable on TRAIN and
  nothing under +1.8 R on VAL — the B5 column of the grid is a display of variance, not of filters.
- **$/month at capacity** ($300 risk for B1-B4/B6, $10K-stage sizing for B5), base book -> best passer:

  | book | base TRAIN | base VAL | base TEST | passer | its TRAIN / VAL / TEST |
  |---|---:|---:|---:|---|---|
  | B1 | +$38 | -$820 | -$1,274 | T5 | +$436 / +$683 / -$633 |
  | B2 | +$1,728 | +$6,584 | -$1,152 | **T4** | +$1,012 / +$4,169 / **+$753** |
  | B3 | +$1,084 | +$896 | -$784 | T4 | +$650 / +$364 / -$20 |
  | B4 | +$223 | -$293 | -$998 | — | — |
  | B5 | +$304 | +$536 | +$214 | T4 | +$119 / +$248 / +$19 |
  | B6 | +$312 | +$363 | -$106 | **T4+T5** | +$271 / +$382 / **+$225** |

  Every passing cell that improves TEST does it by making the book SMALLER: B2 T4 is 4.9 trades a week against
  the base's 26.1; B6 T4+T5 is 3.2 against 10.3. The two positive-TEST cells together are ~$1,000/month at $300
  risk on 8 trades a week, against a permutation p of 0.794.

## 5. Where a filter could not be tested, and why (9 degenerate cells, counted)

| book | cell | reason |
|---|---|---|
| B2 | T1 | the book's universe already IS `prev_day_range_pct >= 8` |
| B2, B3, B4, B6 | T2 | the book's own scan / the Stage-C contract / universe UB already applies `range_so_far_pct >= 5` |
| B3, B4 | T5 | no signal before 10:00 exists: F14 needs a first break, F8 N=30's opening range is 30 minutes (min `entry_m` = 601 on both) — found at run time, amendment A3 |
| B5 | T1 | ORB ships the PDR veto (`prev_day_range_pct >= 11`) |
| B5 | T5 | ORB enters at 09:35 only |

## 6. Cell count

**54 declared** (6 books x 8 filters = 48, plus one best-two stack per book = 6). **44 were run** (39 single-filter
+ 5 stacks); 9 single-filter cells are degenerate and B3's stack did not qualify. Diagnostics, not cells: 18
base book x split statistics, 105 availability/missingness cells, 500 permutation draws, and the tail/cap
columns of every run cell.

---

# TABLES AND DETAIL

## 7. The six books, as already scored (base, before any filter)

| book | what it is | source | population | booked | TRAIN | VAL | TEST |
|---|---|---|---|---:|---:|---:|---:|
| B1 | bull flag raw detector, regen-7 exits | `data/bull_flag_cache_causal_full_20260905.csv` | 896 | 877 | +0.0034 (t 0.05) | -0.0597 (-0.64) | -0.0863 (-0.91) |
| B2 | F6 red-to-green, FIRST-BREAK, hold | `H/F6_rebuild/scan_ai.csv` | 7,589 | 2,010 | +0.0621 (1.50) | +0.2066 (2.51) | -0.0420 (-0.63) |
| B3 | F14 second break, hold | `H/F14_F8_F11/pop_F14.csv` | 2,501 | 1,358 | +0.0566 (1.32) | +0.0412 (0.75) | -0.0455 (-0.71) |
| B4 | F8 N=30 opening-range break, hold | `H/F14_F8_F11/pop_F8N30.csv` | 54,012 | 1,876 | +0.0079 (0.34) | -0.0107 (-0.32) | -0.0449 (-1.21) |
| B5 | ORB B+ honest fills | ORB pipeline, resim dump | 13,033 candidates | 88 picks / 70 fills | +0.4804 (2.07) | +0.8838 (1.37) | +0.2709 (1.38) |
| B6 | S1 gap-fade SHORT, hold, UB | `G/candidates_short.csv` | 1,219 | 880 | +0.0261 (0.65) | +0.0235 (0.37) | -0.0098 (-0.16) |

**Parity against the published anchors** (`L/l1_parity.md`): B2 reproduces `H/F6_rebuild/REPORT.md` section 2 to
**0.0000 on VAL and TEST and -0.0005 on TRAIN** (the 17 `^Z[A-Z]ZZT$` rows this stage drops); B4 and B6
reproduce `H/F14_F8_F11` and `G/REPORT.md` to **0.0000**; B3 is **-0.0091 on VAL** against the Stage-H number,
the whole of it the 6 test-ticker rows; B1's un-booked population mean reproduces `live_followthrough.md`
to +0.004 / -0.000 / -0.005 (that file dropped the 11 rows with no tape, this one keeps them);
**B5 reproduces the production book to the cent — 88 picks, 70 fills, $7,186.64.**

## 8. Availability, per the PLAN section 1 standing rule (a missing feature always KEEPS the trade)

Coverage is 100% everywhere except:

| book | feature | filter | TRAIN | VAL | TEST | present vs missing mean net R (TRAIN / VAL / TEST) |
|---|---|---|---:|---:|---:|---|
| B1 | `pdr` | T1 | 68.0% | 78.8% | 81.2% | -0.025/+0.022 · -0.016/-0.216 · -0.115/+0.002 |
| B1 | `rsf` | T2 | 94.4% | 93.9% | 96.0% | -0.002/**-0.156** · -0.018/**-0.682** · -0.087/**-0.257** |
| B1 | `spy_at_entry`, `vol_ft` | T3, T7 | 99.1% | 98.3% | 98.5% | -0.014/+0.433 · -0.058/-0.080 · -0.088/-0.442 (n = 4/4/3) |
| B2 | `vol_ft` | T7 | 94.7% | 93.4% | 92.8% | +0.038/-0.137 · +0.023/+0.476 · -0.079/+0.018 |
| B3 | `pdr` | T1 | 99.4% | 99.8% | 99.5% | +0.161/-0.536 · +0.008/-1.041 · -0.046/-1.119 (n = 9/1/2) |
| B4 | `pdr` | T1 | 99.1% | 99.9% | 99.8% | +0.024/-0.088 · +0.010/+0.151 · -0.047/+0.282 |
| B5 | SPY 09:34 | T3 | 99.49% over the ORB dates | | | fail-open keeps the pick |

Two of these matter and are stated as limitations of the CELL, not of the filter:

1. **B1's T1 is a filter on two thirds of the book.** 32% of the bull-flag cache's symbol-days have no prior
   daily row in `research/bf_zero/universe.csv`, so a third of B1's trades keep their place whatever their
   prior-day range was. The missing bucket's mean net R flips sign across splits (+0.022 / -0.216 / +0.002),
   which is NOT D1's monotone leak signature, but it does mean B1's T1 row is a 68%-strength test.
2. **B1's T2 fails open on exactly the losing cohort.** `rsf` is missing when the signal is at or near 09:30
   (there are no strictly-prior bars to measure a range on), and that cohort is -0.156 / -0.682 / -0.257 R in
   all three splits. A fail-CLOSED T2 would therefore have looked better; it was not pre-registered and was
   not run.

## 9. The two exit rules, T6 and T7 — the stage's clearest result

| book | T6 fires on | T6 TRAIN / VAL | T7 fires on | T7 TRAIN / VAL |
|---|---:|---|---:|---|
| B1 | 251 of 896 (28%) | -0.079 / -0.113 | 520 of 896 (58%) | -0.100 / -0.122 |
| B2 | 6,218 of 7,589 (82%) | -0.068 / -0.114 | 5,353 of 7,589 (71%) | -0.057 / -0.144 |
| B3 | 2,131 of 2,501 (85%) | -0.086 / -0.052 | 1,889 of 2,501 (76%) | -0.064 / -0.065 |
| B4 | 48,493 of 54,012 (90%) | -0.027 / -0.011 | 40,663 of 54,012 (75%) | -0.020 / -0.020 |
| B5 | 4,822 of 7,402 fills (65%) | -0.209 / -0.643 | 3,658 of 7,402 (49%) | -0.221 / -0.700 |
| B6 | 960 of 1,219 (79%) | -0.044 / -0.062 | 1,022 of 1,219 (84%) | -0.061 / -0.070 |

**12 of 12 book x split pairs negative.** Obtainability: every forced fill is a bar OPEN taken from the bar it
belongs to, so `low <= fill <= high` holds on **100.000%** of them (`bad_fill = 0` in every cell); the exits were
charged the STOP coefficient (0.875 x half), the most expensive leg in the contract, on B2/B3/B4/B6 and 0.3% of
R on B1/B5.

The reason the transfer fails is visible in the fire rates: on the 1-minute books **the rule fires on 65-90% of
trades**, because the median trade is not up +0.25R ten minutes after the fill and the median fill minute is not
a 1.5x volume bar. Where the evidence came from — the raw ORB population, whose trades are 09:35 breakouts with
a 5-minute range as R — the same rule is a genuine improvement (`orb_timestop_validation.md`: -0.133 -> -0.056
TRAIN). Here it is not a stop, it is a different book: a 10-minute scalp on the same entries. On **B5**, the
book the evidence was measured on, the time stop applied to the SELECTED picks costs -0.209 / -0.643 R —
reproducing `REVIEW_FRESH_EYES.md` section 0.3's finding that the time stop is a lever on trades B+ already vetoes.

## 10. T4 — the one filter that transfers, and what it costs

T4 (`D5_r2g`: the entry ordinal holds its sign in all three splits) is the only filter positive on VAL for
**6 of 6** books and on TRAIN for 5 of 6. It passes on B2, B3, B5 and (stacked with T5) B6; on TEST it is
positive on B2 (+0.190), B3 (+0.042) and B6 (+0.077) and negative on B5 (-0.204, n = 7).

What it is, mechanically: keep the day's FIRST booked trade and drop the rest. It is not a selection rule —
it is a frequency cap, and it does not free a slot for anything. The cost is the whole point:

| book | base tr/wk (TRAIN / VAL / TEST) | T4 tr/wk | base $/mo TEST | T4 $/mo TEST |
|---|---|---|---:|---:|
| B2 | 21.0 / 24.1 / 26.1 | 4.7 / 4.6 / 4.9 | -$1,152 | +$753 |
| B3 | 14.5 / 16.5 / 16.4 | 4.4 / 4.5 / 4.8 | -$784 | -$20 |
| B6 (with T5) | 9.6 / 11.7 / 10.3 | 3.2 / 3.7 / 3.2 | -$106 | +$225 |
| B5 | 47 / 19 / 22 picks | 14 / 4 / 7 picks | +$214 | +$19 |

So the honest sentence is: **on these books, later entries in a day lose, and the only way this grid found to
make a 1-minute book positive out of sample is to stop trading after the first fill** — which takes B2 from 26
trades a week to 5 and the whole portfolio contribution to about +$1,000/month at $300 risk, with a
search-adjusted p of 0.794 and a TEST t of 0.73.

## 11. B5 (ORB B+) in dollars — the one cell worth a second look, and why it is not a pass

Applying **T8 (`range_size_pct >= 3`) to the candidate UNIVERSE before ranking** — rather than as the shipped
post-selection range-size veto that leaves its slot empty — takes the B+ book from **88 picks / 70 fills /
$7,186.64** to **135 picks / 104 fills / $11,544.76** over the same 21 months, +$4,358 (whole-window totals,
disclosed in PREREG amendment A2, TEST months included). Mechanism, measured not inferred: with the small-range
candidates gone before ranking, fewer SELECTED picks are destroyed by the shipped no-refill vetoes, so slots
that used to die empty get filled.

It **fails the pre-committed rule** (TRAIN improvement -0.071 in mean R per pick: the added picks are worse than
the average of the ones already there) and therefore its TEST split was NOT read. It is reported because the
pre-committed rule is about R per trade and this cell's effect is on trade COUNT, and because it is one
threshold already in `orb.yaml` applied one layer earlier. It is a hypothesis for its own pre-registration —
`D1_orb/REPORT.md`'s slot dose-response is the obvious companion — and nothing here justifies a config change.

Note the power: B5's TRAIN MDE is **0.649 R per pick** on 47 picks. The B5 row of the grid cannot resolve
anything smaller than that, and its VAL cells (n = 4 to 30) cannot resolve anything at all. B5 T4's VAL
improvement of **+1.785 R on 4 picks** is the clearest illustration in this stage of why a cell must be read
with its n.

## 12. What this stage says

- No transfer filter improved any of the six books by more than the grid's own noise: **observed max TRAIN
  improvement 0.247 against a 500-draw null 95th percentile of 0.550, p = 0.794.**
- **T6 and T7 do not transfer**, 12 of 12 book x split pairs negative, including onto the very book the ORB
  time-stop evidence was measured on. The rule's evidence is on the raw ORB population; on a selected book and
  on 1-minute long books it fires on 65-90% of trades and replaces the book with a 10-minute scalp.
- **T1, T2 and T8 are inert** where they are not already part of the book's definition — 9 of the 48 declared
  cells are degenerate precisely because these books were already built with those rules.
- **T4 is the only filter whose sign transfers**, and it buys its improvement by cutting frequency 4-5x.
- **T3 transfers on B2 only** (+0.042 / +0.064) and then loses on TEST (-0.032).
- Phrasing, as required: no improvement was detectable for any of these eight filters on these six books over
  2025-01..2026-09 at a 12-a-day / 4-concurrent book (ORB's own 3 slots for B5), at the costs each book was
  already scored with; the smallest per-trade improvement these tests could have called significant ranges from
  **0.066 R (B4) to 0.649 R (B5)** on TRAIN and **0.094 R to 1.808 R** on VAL. Effects of the +0.2..+0.4 R size
  the shipped BF/ORB selection stacks historically carried are excluded on B4 only; on B5 they are not.

## 13. Files

`PREREG.md` (+3 amendments) · `lcore.py` (splits, book, statistics, tape, SPY) · `l1_pop.py` -> `pop_B*.csv`,
`base_B*.csv`, `base_stats.csv`, `l1_parity.md` · `l2_bars.py` -> `bars_B*.csv`, `l2_coverage.md` ·
`l3_score.py` -> `cells.csv`, `avail.csv` · `l4_orb.py` -> `orb_bars.csv`, `orb/<cell>_{feat,dump,book,monthly}.csv`,
`orb_cells.csv` · `l5_test.py` -> `cells_full.csv`, `perm.csv`, `perm_summary.txt`.
Read-only everywhere else; both bar stores and `data/cache.db` opened `mode=ro`; no config, service, cache,
cron or order touched; one `nice -n 10` process at a time.
