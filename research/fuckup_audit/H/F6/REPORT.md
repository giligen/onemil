# Stage H / F6 — bottom-up loser analysis -> filters -> profit

Executed 2026-09-17 per `research/fuckup_audit/H/METHOD.md` (owner directive: *"for each of the
near-profitable strategies do a bottom-up analysis on the losers or losing days/weeks and introduce
filters to take it to profit"*). Everything written is under `research/fuckup_audit/H/F6/`;
everything outside it was read only (`C/pop_c.csv`, `E/candidates_causal.csv`, `D/pm_bars.db`,
`D/news_presence.csv`, `E/news_presence_e.csv`, `day_features.csv`,
`research/lit_review_2026/etf_1min.db`). No config, service, cache or order was touched.

**Sections 0-4 were written BEFORE VAL was read. Section 5 is VAL. Section 6 is TEST, read once,
after Section 4 was on disk.**

---

## 0. One page

*(written last; everything in Sections 1-4 was on disk before VAL was scored, and TEST was never read)*

**Does the filtered book reach profit on TRAIN and on VAL? On TRAIN yes, decisively. On VAL no —
the stack makes the book WORSE, and the trades it vetoes are the best trades on VAL. The
pre-committed VAL rule fails on both universes, so TEST was not read.**

| | TRAIN (2025) | VAL (2026-01..05) |
|---|---|---|
| primary universe Q, no filter | +0.0896 R/trade (t 2.03), 20.3/wk | **+0.2088** (t 2.80), 23.3/wk |
| primary universe Q, frozen 3-filter stack | **+0.1420** (t 3.12), 17.2/wk, MDD -26.3 -> -15.6 R | **+0.0877** (t 1.41), 21.5/wk |
| the cohort the stack vetoes | **-0.0159** R/trade, 35% of weeks green | **+0.2285** R/trade, 77% of weeks green |
| twin universe P, no filter -> stack | +0.0510 -> **+0.0988** (t 1.34 -> 2.36) | +0.1629 -> **+0.1489** |

**One of the three replicates out of sample, and it is the one that did not come from this loser
analysis.** `pdr>=8` was imported whole from the shipped ORB rulebook: on the twin universe it takes
TRAIN +0.051 -> +0.075 and VAL +0.163 -> **+0.246** (82% of weeks green). The two filters this stage
found itself — both era-consistent in both TRAIN halves, both present in the whole population and not
only in the book, both monotone across all four buckets, both corroborated by an independent Stage-A
measurement on a different tape — both **invert on VAL**: on the twin universe the below-VWAP
cohort, which lost 0.08-0.26 R a trade across both TRAIN halves at a 53-74% stop rate, paid
**+0.727 R a trade** on VAL (n 68, t 2.24); on the primary universe that cohort is too small to
book on VAL, and the `col<=0.25` bucket there paid +0.1715 against a book of +0.0877.

**The search-adjusted permutation had already said so on TRAIN**: over the 36 book cells of this
stage, observed max TRAIN t 3.12 against a null 95th percentile of 3.24, **p = 0.074**.

**The book.** F6 red-to-green (the stock opens BELOW the prior day's close; the signal bar is the
first 1-min bar whose high reaches that close; the level is the prior close, the stop is the day's
running low before the signal bar), next-open fill under the +0.6% cap, hold to 15:55 with the touch
stop, cost contract (c), `run_book(12, 4)`, entries 09:30-14:01, fill >= $5, R >= 1% of price, and the
causal floor `range_so_far_pct >= 5` on bars strictly before the signal.

**The three filters, each a veto with a mechanism, each decidable at the CLOSE of the signal bar
(i.e. before the next-open order exists), each already a shipped rule in another book or a direct
corroboration of one:**

| # | rule | mechanism | live source |
|---|---|---|---|
| 1 | `prev_day_range_pct >= 8` | ORB's shipped PDR veto: this family monetises *continuation*, and a name whose previous day was quiet has no established two-sided interest to continue - the reclaim is a fresh pop that mean-reverts | previous daily bar, `daily_bars` |
| 2 | `vwap_dist_pct > 0` - the reclaim level is ABOVE the day's running VWAP | below VWAP the average buyer of the day is under water and his supply sits directly overhead; this is the BF P1 gate (`trading/bf_vwap_gate.py`) applied to a different entry | running VWAP of the streamed bars, through the bar BEFORE the signal |
| 3 | `close_over_level_pct <= 0.25` - the signal bar closed no more than 0.25% above the level | we are filled at the NEXT bar's open; the further the signal minute closed above the reclaim level, the more of the reclaim we have already paid for while the stop (the day's low) has not moved - same R, worse price, and the surge minute is usually the exhaustion of the reclaim | the signal bar's own close vs the prior daily close |

**What they remove.** On the primary universe they veto 29% of the signals; that cohort, booked on
its own, is **-0.0159 R/trade on TRAIN** (547 trades, 35% of weeks green, ex-top-5% -0.192) against
the kept book's +0.1420. Filter 3 is the big one (28% of signals); filter 2 costs 2% of signals and
buys most of the drawdown improvement; filter 1 is a no-op on the primary universe by construction
(that universe's 09:30 membership rule already contains it) and carries the twin, where it vetoes
40%.

**TRAIN, primary universe Q:** +0.0896 -> **+0.1420 R/trade** (t 2.03 -> **3.12**), 20.3 -> 17.2
trades/week, weekly R +1.82 -> **+2.44**, weeks green 0.55 -> 0.59, worst week -10.4 -> -8.0 R,
max drawdown -26.3 -> **-15.6 R**, +3R winner cap +0.012 -> **+0.072**. Both TRAIN halves improve
(H1 +0.087 -> +0.128, H2 +0.092 -> +0.154). Twin universe P: +0.0510 -> +0.0988 (t 1.34 -> 2.36),
MDD -37.5 -> -17.9, cap-at-3R -0.019 -> **+0.040**.

**What does NOT improve.** The top-5%-removed test stays negative on every split (TRAIN -0.139 ->
-0.071; VAL -0.062 -> -0.114): the filtered book is *less* tail-dependent than the unfiltered one on
TRAIN and *more* on VAL, and never independent of its tail. And the whole TRAIN improvement is
undone out of sample (Section 5).

**Stage H adds no filter to F6.** The book stays exactly as Stage E left it, with its one genuine
filter — the causal `range_so_far_pct >= 5` floor, worth +0.20 R on TRAIN and +0.25 on VAL — and its
known tail dependence. The deliverable of this stage is the loser anatomy (Section 2), the
availability finding on universe P (Section 1b), and the recorded failure of two mechanism-carrying,
era-consistent filters out of sample (Section 5.2).

---

## 1. Parity anchor (step 0) - reproduced exactly, both reference cells

`h0_parity.py` -> `h0_parity.md` / `h0_parity.csv`. The scorer (`h_core.py`) restates Stage C's
contract (c) in code; the published cells reproduce to the last decimal and to the trade:

| anchor | source | expected | reproduced |
|---|---|---|---|
| F6 {} x next x hold x floor, causal universe, TRAIN | `E/REPORT.md` 7 / 8.8 | +0.0896, n 1034, t 2.03, 20.3/wk | **+0.0896, n 1034, t 2.03, 20.3/wk** |
| same, VAL | `E/REPORT.md` 8.8 | +0.2088, n 512, t 2.80 | **+0.2088, n 512, t 2.80** |
| F6 {} x next x hold, all-day, `pop_c` universe, TRAIN | `C/REPORT.md` 4.1 | +0.051, t 1.34 | **+0.0510, t 1.34** |
| same, VAL | `C/REPORT.md` 4.1 | +0.163, t 2.46 | **+0.1629, t 2.46** |
| floor twin: floor-passing vs below-floor, TRAIN | `E/REPORT.md` 8.4 | +0.090 / -0.114 | **+0.0896 / -0.1144** |
| floor twin, VAL | `E/REPORT.md` 8.4 | +0.209 / -0.046 | **+0.2088 / -0.0457** |

### 1a. Two universes are carried side by side, and why

| tag | file | what it is | F6 rows | TRAIN book |
|---|---|---|---:|---|
| **Q (primary)** | `E/candidates_causal.csv` | the CAUSAL universe: gap >= +3% at the 09:30 open **or** prior-day range >= 8%, open >= $5, ADV20 >= 100K - membership knowable at 09:30 | 32,922 | 1,034 trades, +0.0896 |
| **P (twin)** | `C/pop_c.csv` | the >=5%-range-day population of Stages B/C/D1 | 12,018 | 1,242 trades, +0.0510 |

The METHOD's reference numbers (+0.090 TRAIN / +0.209 VAL) are Q's, so Q is primary and every claim
is reported on both. P's F6 rows already carry the `range_so_far_pct >= 5` floor (`C/c0_extract.py`
applies it to every non-exempt family), so **the no-floor twin is structurally impossible on P** and
is run on Q, where it reproduces E 8.4 exactly: floor-passing +0.0896 / below-floor -0.1144 on TRAIN
(stop rate 28.2% vs 58.9%) and +0.2088 / -0.0457 on VAL. **The floor's own contribution is +0.20
R/trade on TRAIN and +0.25 on VAL** - the single biggest filter in this book, and it was already in
the book definition before Stage H started.

### 1b. Availability audit (PLAN 1 standing rule) - and one feature class ruled out because of it

Coverage on the booked TRAIN trades, per time band, before any feature is used (`h1_anatomy.md` 1.0):

| feature | universe | coverage | mean net R known | unknown |
|---|---|---:|---:|---:|
| `has_news` | **Q** | **1.000** (all bands 1.00) | +0.0896 | - |
| `pm_dollar_vol` (file union `D/pm_bars.db`) | Q | 0.907 (bands 0.85-0.93) | +0.0955 | +0.0326 |
| `has_news` | **P** | **0.806** | **+0.1415** | **-0.3252** |
| `pm_dollar_vol` | P | 0.667 | **+0.1438** | **-0.1347** |
| `spy_ret`, `prev_day_range_pct`, `adv20`, `regime`, `range_so_far_pct` | both | 1.000 | - | - |

**On P, "news is known" is worth +0.47 R/trade and "premarket dollars are known" +0.28 R/trade.**
That is D1's signature exactly (`D1/REPORT.md` A): those files were backfilled over *other stages'*
key sets, so on P their availability is a look-ahead cohort marker, not a feature. **No news or
premarket column is used in any filter in this stage, on either universe.** On Q the leak cannot
arise for news (coverage is 100% by construction of `E/e_news.py`) and the premarket gap there is
0.06 R with no band structure, an order of magnitude below D1's; it is still not used.

---

## 2. Step 1 - the anatomy of the losers (TRAIN only)

Full tables `h1_anatomy.md`; per-bucket CSVs `h1_buckets_{Q,P}.csv`; booked TRAIN trades
`h1_booked_train_{Q,P}.csv`; candidate vetoes `h1_candidates_{Q,P}.csv`. 561 bucket cells.

### 2.1 Concentration - the loss is NOT concentrated, and neither is the profit

| | Q | P |
|---|---|---|
| days with a booked trade | 236 | 249 |
| total | +92.7 R | +63.3 R |
| red days | 133, summing **-215.7 R** | 136, summing -255.3 R |
| worst 5% of days | -44.3 R = **21% of all loss** | -54.9 R = **22%** |
| worst 10% of days | -82.3 R = **38% of all loss** | -95.2 R = 37% |
| best 5% of days | +104.7 R = **113% of the total** | +117.1 R = **185%** |
| weeks green | 0.55 | 0.57 |

**This kills the "kill the bad days" shape of filter before it is tried.** The worst 5% of days carry
only a fifth of the loss, and the best 5% of days carry more than the entire profit. A day-level veto
would have to be nearly perfect to help, and `probe_days.md` already showed that no 09:30-knowable
day split reproduces the day-direction separation (largest causal TRAIN |t| 1.87 against 4.8
non-causal). The filters therefore had to be TRADE-level, and they are.

### 2.2 The worst 20 days have no shared market context

`h1_anatomy.md` 1.1 lists them with SPY/IWM gap and open->close and the regime label. Of the 20
worst Q days: SPY closed above its open on 6 and below on 14; SPY gapped up on 13; the regime label
is A on 14, C1 on 3, C2 on 2 and B on 2 - roughly the base rate. The largest single day is -5.1 R
(2025-06-05, 5 trades, 4 stops); no day is a disaster, they are 3-5 R days that repeat. The exit mix
on those days is 55-100% stops.

### 2.3 Trade anatomy - where the losers live (universe Q, booked TRAIN)

Losers vs winners, the separating columns (Welch t; full list `h1_loserwinner_Q.csv`):
`spread_over_r` -3.25, `n_touches` +2.74, `sig_range_pct` -2.56, `next_entry_m` +1.97,
`sig_seq` +1.84 - and nothing else clears |t| 1.6. **No single column separates the losers**; the
separation lives in bucket means, which is why the filters are bucket vetoes.

| feature | bucket | share | net R | H1 | H2 | stop% |
|---|---|---:|---:|---:|---:|---:|
| `vwap_dist_pct` | **< 0** | 4.8% | **-0.2588** | -0.2014 | -0.3077 | **74.0** |
| | 0-1 | 3.1% | +0.0940 | - | +0.2096 | 40.6 |
| | 1-3 | 28.0% | +0.1767 | +0.0635 | +0.2794 | 28.6 |
| | > 3 | 64.0% | +0.0776 | +0.1276 | +0.0338 | 24.0 |
| `sig_body_pct` | < 0 | 18.3% | +0.1882 | +0.1353 | +0.2296 | 33.9 |
| | 0-0.414 | 31.6% | +0.1264 | +0.1070 | +0.1436 | 23.5 |
| | 0.414-0.935 | 20.4% | +0.1039 | +0.1659 | +0.0470 | 23.7 |
| | **>= 0.935** | 29.7% | **-0.0200** | -0.0210 | -0.0192 | 32.9 |
| `sig_close_pos` | 0-0.25 | 20.2% | +0.2500 | +0.1491 | +0.3293 | 32.1 |
| | 0.25-0.5 | 9.7% | +0.2619 | +0.2408 | +0.2806 | 22.0 |
| | 0.5-0.75 | 14.3% | +0.0021 | +0.0813 | -0.0750 | 31.1 |
| | 0.75-1 | 45.7% | -0.0082 | -0.0355 | +0.0147 | 27.1 |
| `close_confirm` | 0 (wick reclaim) | 40.6% | **+0.1932** | +0.1802 | +0.2056 | 28.3 |
| | 1 (closed above) | 59.4% | +0.0188 | +0.0162 | +0.0209 | 28.2 |
| `prev_day_range_pct` (P only; Q is >=8 by construction) | **< 5** | 22.7% | **-0.1551** | -0.1456 | -0.1676 | 13.5 |
| `dist_open_pct` | lowest quartile | 25.0% | +0.0033 | +0.0206 | -0.0103 | 29.3 |
| `spy_ret` at the signal minute | **< -0.3%** | 12.5% | -0.0443 | -0.0129 | -0.0839 | 19.4 |
| `range_so_far_pct` | 8-12 | 19.7% | +0.2031 | +0.2301 | +0.1797 | 19.1 |
| `band` | 09:30-09:35 | 10.9% | +0.0260 | -0.2117 | +0.1892 | **59.3** |

**Three mechanisms, not fifteen features.** (a) *Where the level sits* - below the day's VWAP the
book stops 74% of the time and loses a quarter R a trade, in both halves. (b) *How much we pay for
the reclaim* - `sig_body_pct`, `sig_close_pos` and `close_confirm` are three spellings of one
statement: the more the signal minute has already travelled above the level by its close, the worse
the next-open fill; all three are monotone in the same direction in the book AND in the whole
population. Stage A saw it on a different tape, a different exit and 12K trades (`probe_stops.md` 4:
F6 wick-only +0.036 mean rr vs close-confirmed +0.007; net -0.106 vs -0.131) - this is not a fresh
discovery of noise. (c) *Whether the previous day was alive* - on P the quiet prior day is the worst
bucket in both halves; on Q the universe already requires >= 8.

`spy_ret < -0.3` is era-consistent in the book (-0.013 / -0.084) but the same bucket is **+0.149** in
the whole TRAIN population, so it is a book artefact and is NOT used. It is reported because it is
the one candidate a book-only analysis would have adopted.

### 2.4 Path anatomy - the filter must be entry-side, not exit-side

| cohort (Q) | n | net R | median min held | mean MAE% | mean MFE (R) | MFE>=0.5R | MFE>=1R | MFE>=2R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all booked | 1034 | +0.0896 | 317 | 4.00 | +0.86 | 0.55 | 0.32 | 0.15 |
| losers | 568 | -0.7381 | 224 | 5.58 | +0.50 | 0.32 | 0.13 | 0.05 |
| winners | 466 | +1.0985 | 355 | 2.08 | +1.29 | 0.82 | 0.56 | 0.26 |
| stopped | 292 | -1.1013 | **78** | 6.20 | +0.56 | 0.32 | 0.17 | 0.08 |

Minutes to the stop: 10% <= 5 min, 26% <= 30 min, **37% after 2 hours**. Mean net R by MFE bucket:
never positive -1.04, <=0.5R -0.47, <=1R -0.11, <=2R +0.49, >2R +2.02.

Only **17% of stopped trades ever had +1R on the table**, so a breakeven-after-+1R rule can rescue at
most a sixth of the stops while capping every winner that dips after +1R - and Stage A measured that
exact variant as NEGATIVE (`probe_stops.md` 1b: the best of 7 stop x 4 exit variants was worth
+0.023 R, below the +0.05 adoption bar; F6's stop is the day's low, "nowhere near where price is",
so it carries no stop-run signature - 31% wicks but only 2.9% recover within 5 minutes and 4.7%
later reach +2R). **The exit shape is settled and out of scope; every filter below is entry-side.**

### 2.5 Era consistency inside TRAIN

| half | Q n | Q net R | Q t | P n | P net R | P t |
|---|---:|---:|---:|---:|---:|---:|
| H1 2025-01..06 | 478 | +0.0869 | 1.29 | 623 | **+0.0000** | 0.00 |
| H2 2025-07..12 | 556 | +0.0920 | 1.57 | 619 | +0.1022 | 1.85 |

Q is flat-positive in both halves; P's whole TRAIN edge is in H2. The ORB veto rule (negative in
both halves) left **5 candidate buckets on Q and 14 on P**, of which the news/premarket-availability
ones were struck out by 1b and `spy_ret` by the population check.

---

## 3. Steps 2 and 3 - the candidate filters and the stack on TRAIN

`h2_filters.py` -> `h2_filters.md` / `h2_cells.csv`; `h2b_stacks.py` -> `h2b_stacks.md` /
`h2b_stacks.csv`. **26 single cuts x 2 universes = 52**, plus 8 stacks x 2 = 16, plus 6 cumulative,
6 leave-one-out, 2 removed-cohort and 4 baselines = **86 book cells in this step**, on top of step
1's 561 bucket cells.

Every cut is a round number or a population-quartile boundary; every one is computable in the engine
at the close of the signal bar. Filters are applied BEFORE `run_book(12,4)`: this book is first-come
(not a ranked 09:35 batch like ORB), so a vetoed signal is simply never ordered and the slot stays
open for the next arrival - that is the live semantics, and it is why there is no "no-refill"
question here.

### 3.1 The single cuts (universe Q, TRAIN; full table `h2_filters.md`)

| cut | kept | n | tr/wk | net R | t | stop% | MDD | H1 | H2 | ex-top5% | cap +3R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASELINE | - | 1034 | 20.3 | +0.0896 | 2.03 | 28.2 | -26.3 | +0.0869 | +0.0920 | -0.1386 | +0.0117 |
| `vwap>0` | 98% | 1006 | 19.7 | +0.1051 | 2.47 | 26.1 | -21.2 | +0.0981 | +0.1110 | -0.1080 | +0.0386 |
| `col<=0.25` | 72% | 892 | 17.5 | **+0.1399** | **2.93** | 24.0 | -17.3 | +0.1504 | +0.1310 | -0.0909 | +0.0560 |
| `noconfirm` | 37% | 622 | 12.2 | +0.1598 | 2.69 | 23.8 | -19.4 | +0.1652 | +0.1551 | -0.0840 | +0.0659 |
| `body<0.5` | 59% | 795 | 15.6 | +0.1253 | 2.47 | 22.6 | -27.6 | +0.0968 | +0.1490 | -0.1064 | +0.0386 |
| `dopen>=2.2` | 73% | 817 | 16.0 | +0.1235 | 2.56 | 24.6 | -19.2 | +0.1145 | +0.1309 | -0.0909 | +0.0516 |
| `pdr>=8` | 100% | 1033 | 20.3 | +0.0906 | 2.05 | 28.2 | -26.3 | +0.0898 | +0.0912 | -0.1378 | +0.0126 |
| `rsf>=8` | 35% | 536 | 10.5 | +0.1374 | 2.90 | 13.6 | -13.2 | +0.1638 | +0.1151 | **-0.0348** | +0.0907 |

On P the same cuts: `pdr>=8` +0.0510 -> +0.0750, `vwap>0` -> +0.0606, `col<=0.25` -> +0.0563,
`dopen>=2.2` -> +0.0811, `px<=50` -> +0.0768.

Declared and NOT used: `rsf>=8` (raising the family's own floor from 5 to 8 is a re-tune of a
population parameter, which the METHOD forbids - reported because it is the best-behaved single cut
in the grid on the tail test); `pdr>=5` (better than `pdr>=8` on P, +0.0924 vs +0.0750 - the shipped
ORB threshold of 8 was taken instead of the better-scoring 5, deliberately); every news and
premarket cut (1b).

### 3.2 The stacks (TRAIN, both universes) - `h2b_stacks.md`

Eligible = raises the book mean net R AND both TRAIN halves in BOTH universes.

| stack | kept Q | Q net R | Q t | Q MDD | P net R | P t | P MDD | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| S1 pdr8 + vwap + `col<=1` | 95% | +0.1215 | 2.80 | -19.7 | +0.0893 | 2.21 | -22.2 | YES |
| S2 pdr8 + vwap + `body<0.5` | 57% | +0.1302 | 2.75 | -24.0 | +0.0997 | 2.24 | -22.3 | no (Q H1 falls) |
| S3 pdr8 + vwap + `dopen>=2.2` | 72% | +0.1272 | 2.76 | -18.1 | +0.0946 | 2.13 | -14.9 | YES |
| **S4 pdr8 + vwap + `col<=0.25`** | **71%** | **+0.1420** | **3.12** | **-15.6** | **+0.0988** | **2.36** | -17.9 | **YES** |
| S5 pdr8 + vwap + `noconfirm` | 36% | +0.1656 | 2.83 | -16.6 | +0.1183 | 2.25 | -22.8 | YES |
| S6 vwap + dopen | 72% | +0.1277 | 2.77 | -18.1 | +0.0694 | 1.84 | -14.0 | no (P H2 falls) |
| S7 pdr8 + vwap | 98% | +0.1061 | 2.49 | -21.2 | +0.0872 | 2.14 | -22.8 | YES |
| S8 pdr8 + dopen | 73% | +0.1230 | 2.55 | -19.2 | +0.0969 | 2.07 | -17.6 | YES |

**Selection rule applied (stated here, applied once):** among eligible stacks, the highest TRAIN t on
the primary universe subject to **>= 15 booked trades/week** (below that the book is too thin to be
worth a dry run at 4 slots; S5's 12.0/wk fails it). That is **S4**, t 3.12.

### 3.3 S4 on TRAIN, in full

| universe | n | tr/wk | net R | gross | t | WR% | stop% | wk R | green | worst wk | MDD | H1 | H2 | ex-top5% | cap +3R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q baseline | 1034 | 20.3 | +0.0896 | +0.1182 | 2.03 | 45.1 | 28.2 | +1.82 | 0.55 | -10.4 | -26.3 | +0.0869 | +0.0920 | -0.1386 | +0.0117 |
| **Q + S4** | 875 | 17.2 | **+0.1420** | +0.1685 | **3.12** | 47.1 | 22.7 | **+2.44** | 0.59 | -8.0 | **-15.6** | +0.1277 | +0.1542 | -0.0713 | **+0.0723** |
| P baseline | 1242 | 23.4 | +0.0510 | +0.0837 | 1.34 | 44.4 | 26.2 | +1.19 | 0.57 | -12.1 | -37.5 | +0.0000 | +0.1022 | -0.1639 | -0.0190 |
| **P + S4** | 951 | 17.9 | **+0.0988** | - | **2.36** | 46.0 | 23.8 | +1.77 | 0.57 | -9.0 | -17.9 | +0.0471 | +0.1452 | -0.1020 | **+0.0396** |

The vetoed cohort of S4, booked on its own (Q): **547 trades, -0.0159 R**, WR 44.2%, stop 28.9%,
weeks green 0.35, ex-top-5% -0.192, MDD -38.1 R. On P: **1,065 trades, -0.0260 R**, weeks green
0.47, ex-top-5% -0.187, MDD -79.3 R.

Leave-one-out of S4 on TRAIN (`h2c_loo.py` -> `h2c_loo.md` / `h2c_loo.csv`; `h2_filters.md`'s own
leave-one-out table is for the S1 ordering that was current when that script ran):

| drop | Q net R | Q t | Q MDD | Q ex-top5% | P net R | P t | P MDD |
|---|---:|---:|---:|---:|---:|---:|---:|
| (none — S4) | +0.1420 | 3.12 | -15.6 | -0.0713 | +0.0988 | 2.36 | -17.9 |
| minus `pdr>=8` | +0.1410 | 3.10 | -15.6 | -0.0721 | **+0.0693** | 1.94 | -18.0 |
| minus `vwap>0` | +0.1409 | 2.95 | -17.3 | -0.0901 | +0.0992 | 2.25 | -20.2 |
| minus `col<=0.25` | **+0.1061** | 2.49 | -21.2 | -0.1072 | **+0.0872** | 2.14 | -22.8 |

`col<=0.25` is the only leg that carries the mean on both universes (-0.036 R on Q, -0.012 on P if
dropped); `vwap>0` is nearly free in the mean and buys 1.7-2.3 R of drawdown and 0.019-0.017 of the
tail; `pdr>=8` is worth nothing on Q (a no-op) and -0.030 R on P.

---

## 4. FREEZE (written 2026-09-17, BEFORE VAL was scored)

> **The frozen stack.** Book: family `F6 {}` (red-to-green, as implemented in
> `B/build_candidates4.py`), fill `entry_next` (next bar's open under the +0.6% cap), exit **hold to
> 15:55 with the touch stop**, population: fill >= $5, `570 <= entry_m <= 841`, `r_pct >= 1.0` of
> price, **`range_so_far_pct >= 5`** (bars strictly before the signal), cost contract (c),
> `run_book(12, 4)`. Universes: **Q = `E/candidates_causal.csv` (primary)**, P = `C/pop_c.csv`
> (twin). PLUS these three vetoes and no others:
>
> 1. `prev_day_range_pct >= 8.0`
> 2. `vwap_dist_pct > 0.0`
> 3. `close_over_level_pct = (sig_c / level - 1) * 100 <= 0.25`
>
> No news leg, no premarket leg, no day-level filter, no exit change, no re-tuned family parameter.
> The secondary exit variant (2R close, stop -1%) is reported on the same stack, never selected on.

**Pre-committed reading of VAL** (METHOD step 4): PASS requires (a) VAL mean net R of the stacked
book > the unfiltered VAL book, (b) the vetoed cohort negative on VAL, (c) VAL mean net R > 0 with
>= 55% of weeks green. If VAL fails, the stage's result is "the losers are not separable by causal
features at this power", with the MDE stated, and TEST is not read.

**Pre-committed reading of TEST** (METHOD step 5): read once, reported whatever it says, week by
week and month by month, with the tail tests, the search-adjusted permutation p over every book cell
this stage looked at, and the money line. A TEST that is positive but dies to the top-5% removal is
reported as *tail-dependent*, not as a candidate - the same rule that retired Stage E's cell.

---

# 5. Step 4 — VAL, read once, on the frozen stack

`h3_val_test.py` -> `h3_val.md` / `h3_results.csv`; booked trades `h3_book_VAL_{Q,P}.csv`.
TRAIN rows are repeated so the two splits sit side by side.

| universe | split | book | n | tr/wk | net R | t | WR% | stop% | wk R | green | worst wk | MDD | ex-top5% | cap +3R |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q | TRAIN | baseline | 1034 | 20.3 | +0.0896 | 2.03 | 45.1 | 28.2 | +1.82 | 0.55 | -10.4 | -26.3 | -0.1386 | +0.0117 |
| Q | TRAIN | **FROZEN stack** | 875 | 17.2 | **+0.1420** | 3.12 | 47.1 | 22.7 | +2.44 | 0.59 | -8.0 | -15.6 | -0.0713 | +0.0723 |
| Q | TRAIN | vetoed cohort | 547 | 10.7 | **-0.0159** | -0.31 | 44.2 | 28.9 | -0.17 | 0.35 | -7.8 | -38.1 | -0.1923 | -0.0551 |
| **Q** | **VAL** | baseline | 512 | 23.3 | **+0.2088** | 2.80 | 46.7 | 30.1 | +4.86 | 0.77 | -14.0 | -24.9 | -0.0617 | +0.0857 |
| **Q** | **VAL** | **FROZEN stack** | 473 | 21.5 | **+0.0877** | 1.41 | 44.4 | 27.3 | +1.89 | 0.64 | -8.5 | -21.4 | -0.1141 | +0.0250 |
| **Q** | **VAL** | vetoed cohort | 337 | 15.3 | **+0.2285** | 2.64 | 50.7 | 24.3 | +3.50 | 0.77 | -8.2 | -14.0 | -0.0209 | +0.1146 |
| P | TRAIN | baseline | 1242 | 23.4 | +0.0510 | 1.34 | 44.4 | 26.2 | +1.19 | 0.57 | -12.1 | -37.5 | -0.1639 | -0.0190 |
| P | TRAIN | **FROZEN stack** | 951 | 17.9 | **+0.0988** | 2.36 | 46.0 | 23.8 | +1.77 | 0.57 | -9.0 | -17.9 | -0.1020 | +0.0396 |
| P | TRAIN | vetoed cohort | 1065 | 20.1 | **-0.0260** | -0.79 | 44.2 | 22.4 | -0.52 | 0.47 | -14.6 | -79.3 | -0.1869 | -0.0620 |
| **P** | **VAL** | baseline | 536 | 24.4 | **+0.1629** | 2.46 | 47.8 | 28.4 | +3.97 | 0.68 | -8.1 | -17.6 | -0.0786 | +0.0670 |
| **P** | **VAL** | **FROZEN stack** | 485 | 22.0 | **+0.1489** | 2.28 | 46.4 | 27.4 | +3.28 | 0.68 | -8.3 | -16.5 | -0.0742 | +0.0685 |
| **P** | **VAL** | vetoed cohort | 474 | 21.5 | **+0.1356** | 2.10 | 47.9 | 21.9 | +2.92 | 0.68 | -7.2 | -23.6 | -0.0840 | +0.0536 |

## 5.1 The pre-committed rule, applied

| condition (METHOD step 4) | Q | P |
|---|---|---|
| (a) stacked VAL mean net R > unfiltered VAL book | **NO** (+0.0877 vs +0.2088) | **NO** (+0.1489 vs +0.1629) |
| (b) the vetoed cohort is negative on VAL | **NO** (+0.2285) | **NO** (+0.1356) |
| (c) VAL mean net R > 0 and >= 55% weeks green | yes (+0.0877, 0.64) | yes (+0.1489, 0.68) |

**VAL FAILS on both universes, on the two conditions that matter. Per the freeze in Section 4, TEST
WAS NOT READ**, and no TEST number appears anywhere in this report or in any file under `H/F6/`.

## 5.2 The per-filter contribution on VAL — one of three replicates, two invert

| filter | TRAIN effect on the book (Q / P) | VAL effect on the book (Q / P) | the vetoed bucket on VAL (Q / P) |
|---|---|---|---|
| `pdr>=8` (the shipped ORB rule, from OUTSIDE this dataset) | no-op / **+0.0510 -> +0.0750** | no-op / **+0.1629 -> +0.2464** (t 3.23, 82% weeks green) | - / **+0.0037** (i.e. ~zero against a +0.16 book) |
| `vwap>0` (fitted here) | +0.0906 -> +0.1061 / +0.0750 -> +0.0872 | **+0.2088 -> +0.1311** / **+0.2464 -> +0.1746** | too few to book / **+0.7270** (n 68, t 2.24) |
| `col<=0.25` (fitted here) | +0.1061 -> +0.1420 / +0.0872 -> +0.0988 | **+0.1311 -> +0.0877** / **+0.1746 -> +0.1489** | **+0.1715** / +0.0707 |

**This is the whole result of Stage H for F6, in one table.** The only filter that survives
out-of-sample is the one that was NOT found by looking at these losers — ORB's previous-day-range
veto, imported whole with its shipped threshold of 8.0. The two rules the loser analysis itself
produced — both with good mechanisms, both era-consistent inside TRAIN, both monotone in the whole
TRAIN population, both corroborated by an independent Stage-A measurement on a different tape — both
**invert on VAL**: on P the below-VWAP cohort, which lost 0.08-0.26 R a trade across both TRAIN
halves at a 53-74% stop rate, returned **+0.727 R a trade** on VAL.

## 5.3 The secondary exit variant (2R on a close, stop at level -1%) — same verdict

`h4_secondary.py` -> `h4_secondary.md` / `h4_secondary.csv`. Reported, never selected on.

| universe | split | baseline | FROZEN stack | vetoed cohort |
|---|---|---:|---:|---:|
| Q | TRAIN | +0.0510 (t 1.77) | **+0.0766** (t 2.64) | +0.0279 |
| Q | VAL | +0.0803 (t 1.87) | **+0.0389** (t 0.95) | +0.1095 |
| P | TRAIN | +0.0245 (t 0.98) | **+0.0560** (t 2.00) | -0.0117 |
| P | VAL | +0.0893 (t 2.22) | **+0.0752** (t 1.81) | +0.0529 |

The stop-1% book is smaller in every cell (the 2R close-target caps the winners; Stage C measured
that at about -0.08 R a trade) and it fails VAL in exactly the same direction. The failure is a
property of the filters, not of the exit.

## 5.4 Permutation, search-adjusted over every book cell of this stage

36 book cells on the primary universe, 500 day-label sign-flip draws: **observed max TRAIN t 3.12,
null 95th percentile 3.24, p = 0.074.** The best cell the stage could find, after the search, is
*not* significant at 5% once the search is paid for — on TRAIN, before VAL was even opened. The VAL
failure is what that p was warning about.

---

# 6. Step 5 — TEST: NOT READ

The freeze in Section 4 made TEST conditional on VAL. VAL failed both decisive conditions, so
**TEST was not read**, `H_READ_TEST` was never set, and there is no TEST file in this directory.
Whatever edge this book has or does not have on 2026-06-01..2026-09-11 remains unspent; a later
stage can still use it once.

---

# 7. What Stage H establishes for F6

1. **The losers of this book are not separable by causal features at this power.** Three filters
   with real mechanisms, chosen by the ORB veto rule (negative in both TRAIN halves, checked against
   the whole population, not just the booked trades), lifted TRAIN from +0.090 to +0.142 R a trade
   and halved the drawdown — and cost 0.12 R a trade on VAL. The smallest per-trade effect the VAL
   test could have seen at 80% power is **0.175-0.209 R** on the primary universe
   (4.1-6.1 R per week at 4 slots); the effect the stack actually needed to demonstrate was about
   +0.05 R. **VAL cannot see a +0.05 R filter here.** The test that was run was not powerful enough
   to confirm the filters, and it was powerful enough to show that they are not worth +0.2 R.
2. **The one filter that replicated came from outside the dataset.** `prev_day_range_pct >= 8` is
   ORB's shipped veto with its shipped threshold; it improved the twin universe's TRAIN book
   (+0.051 -> +0.075) and its VAL book (+0.163 -> +0.246, 82% of weeks green), and its vetoed bucket
   is ~0 on VAL against a +0.16 book. On the primary universe it is a no-op because that universe's
   own membership rule already contains it. The reading is not "pdr>=8 is a new edge" — it is that
   **the causal universe Q is, approximately, universe P with ORB's PDR veto applied, and that is
   most of why Q's book is twice P's.** It is also the third independent confirmation of that rule
   inside this audit: it is shipped in ORB (`trading/orb_pdr_veto.py`), it is the one veto in the
   parallel `H/F14_F8_F11` sub-stage whose bucket is negative on TRAIN-H1, TRAIN-H2 **and** VAL
   (on F11, the close-confirmed F6), and it is the one veto here that lifts a book on both splits.
3. **The two rules that the loser analysis itself produced are the ones that inverted**, despite
   passing every guard this program has: era-consistent in both TRAIN halves, present in the whole
   population and not only in the book, monotone across all four buckets, mechanism-carrying, and
   independently corroborated on a different tape (`probe_stops.md` 4). The guards were necessary and
   not sufficient — which is the same lesson as `feedback_independent_check_before_claims`, now with
   a number attached: on VAL the below-VWAP cohort paid +0.73 R a trade.
4. **The day-level shape of filter is dead for this book**, and the anatomy says why before any
   filter is tried: the worst 5% of days carry 21% of the loss and the best 5% carry 113% of the
   profit. The tail is on both sides. That is also why the top-5%-removed test stays negative under
   every stack in this stage.
5. **The exit shape is dead too**, on measurements larger than this stage's: only 17% of stopped
   trades ever saw +1R; Stage A's best of 28 stop x exit variants was +0.023 R against a +0.05 bar.
6. **What is left standing after Stage H is what was standing before it**: the causal
   `range_so_far_pct >= 5` floor (+0.20 R on TRAIN, +0.25 on VAL, the biggest single filter in the
   book, already part of the book definition), the causal universe, and the E-stage finding that the
   whole cell is tail-dependent. Stage H adds no filter to F6.

**Phrasing (PLAN 1).** No filter of the losers was detectable out-of-sample **in these universes**
(causal U1 u U2 at 09:30; and the >=5%-range-day population), **at this horizon** (entries
09:30-14:01, exit hold-to-15:55 with the touch stop, plus the 2R/stop-1% variant), **at this book
size** (12 candidates/day, 4 concurrent), **over this window** (TRAIN 2025, VAL 2026-01..05), **at
this cost** (contract (c)). The smallest per-trade effect the VAL test could have seen at 80% power
is **0.18-0.21 R** (4.1-6.1 R per week at 4 slots); the smallest the TRAIN test could have seen is
**0.11-0.13 R** (2.3-2.7 R per week). Filters worth less than that are invisible here and are not
excluded by anything in this stage.

**No money line is reported.** The METHOD makes it a step-5 deliverable, conditional on VAL, and VAL
failed; quoting dollars for a book that lost 0.12 R a trade out of sample would be exactly the kind
of number this audit exists to stop.

---

# 8. Cells looked at (the multiplicity denominator)

| what | cells |
|---|---:|
| step 0 parity/floor-twin books (2 universes x 2 variants x 2 floor states x 2 splits, + 4 twin) | 36 |
| step 1 bucket cells (28 numeric features x <=5 buckets + 10 categorical, x 2 scopes x 2 universes) | 561 |
| step 1 availability cells (6 features x 2 scopes x 2 universes x 6 statistics) | ~72 |
| step 1 loser-vs-winner t-tests (28 features x 2 universes) | 56 |
| step 2 single cuts (26 x 2 universes) | 52 |
| step 3 stacks (8 x 2), cumulative (3 x 2), leave-one-out S1 (3 x 2), removed cohort (2), baselines (4) | 34 |
| step 3 leave-one-out of the FROZEN stack (4 x 2 universes) | 8 |
| step 4 VAL books (3 books + 3 cumulative + 3 vetoed-bucket) x 2 universes x 2 splits | 44 |
| step 5 secondary-exit books (3 x 2 splits x 2 universes) | 12 |
| **total cell-instances** | **~875** |

Plus one 500-draw day-label sign-flip permutation null over the 36 book cells of the primary
universe. TEST contributes **0** cells.

---

# 9. Files

| file | what |
|---|---|
| `h_extract.py`, `f6_{Q,P}.csv`, `weeks_{Q,P}.csv` | the F6 row subsets and the week denominators |
| `h_core.py` | population, contract (c), `run_book(12,4)`, features - the ONE scorer |
| `h0_parity.py`, `h0_parity.{md,csv}` | the parity anchors and the floor twin |
| `h1_anatomy.py`, `h1_anatomy.md`, `h1_buckets_{Q,P}.csv`, `h1_loserwinner_{Q,P}.csv`, `h1_candidates_{Q,P}.csv`, `h1_booked_train_{Q,P}.csv` | step 1 |
| `h2_filters.py`, `h2_filters.md`, `h2_cells.csv`, `h2_stack_train_{Q,P}.csv` | step 2 and the cumulative stack |
| `h2b_stacks.py`, `h2b_stacks.{md,csv}` | the candidate stacks and the eligibility table |
| `h2c_loo.py`, `h2c_loo.{md,csv}` | leave-one-out of the frozen stack on TRAIN |
| `h3_val_test.py`, `h3_val.md`, `h3_results.csv`, `h3_book_{TRAIN,VAL}_{Q,P}.csv` | step 4 (VAL) and the permutation |
| `h4_secondary.py`, `h4_secondary.{md,csv}` | the stop-1% variant |
| **`h_final_book.csv`** | **the per-trade book of the frozen stack, TRAIN + VAL, both universes (2,784 rows) - the file for an independent rebuild** |
| `h_etf_state.csv` | cached SPY/IWM open-to-minute returns (a pure derivative of `etf_1min.db`) |

---

# 10. The filter rules in prose (for an independent reimplementation)

Take every F6 red-to-green signal: a stock whose 09:30 open is below the previous day's official
close, and the first 1-minute bar of the regular session (at or after 09:31) whose HIGH reaches that
previous close multiplied by 1.0005. The level is the previous close; the stop is the lowest low of
every bar of the session strictly before that signal bar; require the stop to be below the level.
Require, on the bars strictly before the signal bar, that the session's high-minus-low so far is at
least 5% of the 09:30 open. Enter at the OPEN of the bar after the signal bar, but only if that open
is at or below the level x 1.006; otherwise there is no trade. Require the fill to be at least $5,
the fill minute to be between 09:30 and 14:01 inclusive, and the distance from fill to stop to be at
least 1% of the fill. Exit at the first bar whose low touches the stop (filled at the minimum of the
stop and that bar's open, times 0.999), otherwise at 15:55.

Then apply these three vetoes, each decided at the CLOSE of the signal bar, before the order exists:

1. Skip the signal if the previous session's high-minus-low, as a percentage of the previous
   session's close, is below 8.0.
2. Skip the signal if the level is at or below the session's running volume-weighted average price
   computed over every bar strictly before the signal bar.
3. Skip the signal if the signal bar's CLOSE is more than 0.25% above the level.

Book the survivors first-come by fill minute, ties broken alphabetically by symbol, at most 12 fills
a day and at most 4 positions open at once, where a slot is free for a fill at minute k only if the
previous exit happened strictly before minute k.

Score each trade in R: `rr` is (exit - fill) / (fill - stop); the cost is
`half = 0.5 x (spread_cc_bps / 100) / max(r_pct, 0.05)` with `r_pct` the stop distance as a percent
of the fill, charged as `0.25 x half` on entry plus `0.875 x half` on a stop exit or `0.412 x half`
on a 15:55 exit.

**On 2025 this stack books 875 trades at +0.142 R each. On 2026-01..05 it books 473 trades at
+0.088 R each, against +0.209 R for the same book with no vetoes at all. Do not ship it.**
