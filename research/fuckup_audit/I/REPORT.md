# Stage I — stacking the books, and the portfolio of sleeves

Owner directive 2026-09-17: "stack multiple strategies to maximize P&L."
Pre-registration: `I/PREREG.md` (written before any run). Freeze: `I/FREEZE.md` (written before TEST).
Everything below is historical arithmetic on books that were **already declared elsewhere**. Nothing new was
searched for, nothing is enabled, and no number here is a forecast.

---

## 0. One page

**Does stacking raise weekly R and $/month versus F6 alone? No — it lowers weekly R in every split and raises the
worst week at the same time.** Adding families to the F6-PDR book is strictly dominated, monotonically:

| exit `hold`, weekly R at 12/day 4-concurrent | S0 = F6 | S1 = +F14 | S2 = +F11 | S3 = +F8 N=30 |
|---|---:|---:|---:|---:|
| TRAIN 2025 | **+1.65** | +1.67 | +1.07 | +0.54 |
| VAL 2026-01..05 | **+5.91** | +5.71 | +5.04 | +5.01 |
| TEST 2026-06..09 (read once, descriptive) | **+2.79** | +2.36 | -1.37 | -2.47 |
| TRAIN worst week | -10.7 | -11.4 | -12.3 | **-13.9** |
| TRAIN weekly MDD | -24.3 | -24.4 | -26.1 | **-28.5** |
| TRAIN months green | 10/12 | 9/12 | 9/12 | **5/12** |

The declared freeze rule (highest VAL weekly R over the 8 cells) selected **S0 — F6 alone**. The stack adds
frequency (S3 books 672 TRAIN trades that F6 alone never took, and 92% of them landed on a genuinely free slot,
not on a displaced F6 trade) — but **the added trades are worth -32.9 R on TRAIN and the F6 trades they pushed out
of the book were worth +38.1 R**. More trades, less money: the slot is not the binding constraint, the edge is.

**What the combined monthly $ is, and its worst month.** Three sleeves, each at its own declared sizing, 2025-01
-> 2026-09 (21 months):

| | total $ | mean $/month | months green | worst month | monthly-curve MDD | 2025 | 2026-YTD |
|---|---:|---:|---:|---:|---:|---:|---:|
| intraday (F6-PDR, $300 risk, liquidity-capped) | +11,765 | +560 | 14/21 | -6,468 | -14,816 | +15,293 | -3,528 |
| QQQ noise band ($60K 1x, 0.5 bp/leg) | +4,503 | +214 | 13/21 | -2,243 | -5,930 | +2,412 | +2,091 |
| ORB B+ ($10K stage: $10K / 3 / $375) | +7,187 | +342 | 15/21 | -223 | -235 | +3,650 | +3,536 |
| **sum of the three** | **+23,455** | **+1,117** | **11/21** | **-7,400 (2026-07)** | **-13,149** | +21,356 | +2,099 |

Pairwise monthly correlations are low (+0.15 to +0.21) and the sum's monthly sigma is 0.75x the sum of the parts'
sigma — diversification is real. But **combining does not raise the hit rate**: the sum is green 11 of 21 months,
fewer than any single sleeve, because the intraday sleeve is ~5x the size of the other two and its sign decides
most months. The honest per-month arithmetic is **+$1,117 mean, median +$1,443, worst -$7,400** — and the
intraday sleeve, which supplies half of it, is **-$12,914 over its four TEST months** under the same cap.

Two sentences the owner should hear before anything is read as a plan: (1) every constituent book has already
FAILED its own pre-registered VAL gate (LOG.md 9/17, five sub-stages); (2) the liquidity cap that makes the
intraday sleeve *obtainable* takes it from +0.104 R to **-0.146 R on TEST** — the money is in names that cannot
carry the money.

---

## 1. Part A — the stacked intraday book

### 1.1 Construction (no new rule)
Population: `C/pop_c.csv`, four families as they exist in that file — F6 `{}`, F14 `{"N": 15}`,
F11 `{"base":"F6"}`, F8 `{"N":30}`. One filter set for all four: `next_entry` present, `price >= 5`,
`next_r_pct >= 1`, entry <= 14:01, `range_so_far_pct >= 5`, **`prev_day_range_pct >= 8`** (ORB's shipped PDR veto,
transferred unchanged — the one rule this program found that replicated). 45,764 signal rows: F8N30 30,169 ·
F6 7,137 · F11 6,805 · F14 1,653. Cost contract (c), exits `hold` / `2r`, book `run_book(12, 4)`, splits fixed.
Dedupe: one row per (symbol, day), priority **F6 > F14 > F11 > F8N30** (declared, not tuned).

**Parity, before any result** (`I/a_trainval.md` §0): S0 reproduces `H/F6/f6_pdr_book.md` **EXACTLY** on all four
published cells — hold TRAIN n 1122 / +0.078 / t 1.81, hold VAL n 519 / +0.251 / t 3.28, 2r TRAIN n 1217 /
+0.058 / t 2.00, 2r VAL n 593 / +0.110 / t 2.56. Same code path, same numbers.

### 1.2 The 8 declared cells (TRAIN / VAL)

| stack | exit | split | n | tr/wk | net R | gross | t | WR% | wkR | wks green | worst wk | MDD | mo green | ex-top5% | cap+3R |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| S0 | hold | TRAIN | 1122 | 21.2 | +0.078 | +0.106 | 1.81 | 44.7 | +1.65 | 0.57 | -10.7 | -24.3 | 10/12 | -0.161 | -0.005 |
| S0 | hold | VAL | 519 | 23.6 | +0.251 | +0.281 | 3.28 | 48.4 | +5.91 | 0.82 | -9.3 | -17.5 | 5/5 | -0.036 | +0.114 |
| S1 | hold | TRAIN | 1219 | 23.0 | +0.073 | +0.101 | 1.78 | 44.1 | +1.67 | 0.57 | -11.4 | -24.4 | 9/12 | -0.158 | -0.005 |
| S1 | hold | VAL | 540 | 24.5 | +0.233 | +0.264 | 3.14 | 48.9 | +5.71 | 0.82 | -11.6 | -18.7 | 5/5 | -0.048 | +0.102 |
| S2 | hold | TRAIN | 1270 | 24.0 | +0.044 | +0.075 | 1.10 | 42.6 | +1.07 | 0.49 | -12.3 | -26.1 | 9/12 | -0.190 | -0.033 |
| S2 | hold | VAL | 560 | 25.5 | +0.198 | +0.231 | 2.72 | 47.0 | +5.04 | 0.73 | -12.0 | -19.8 | 5/5 | -0.086 | +0.066 |
| S3 | hold | TRAIN | 1359 | 25.6 | +0.021 | +0.051 | 0.50 | 42.9 | +0.54 | 0.49 | -13.9 | -28.5 | 5/12 | -0.214 | -0.065 |
| S3 | hold | VAL | 565 | 25.7 | +0.195 | +0.228 | 2.72 | 45.8 | +5.01 | 0.77 | -10.2 | -19.3 | 5/5 | -0.092 | +0.064 |
| S0 | 2r | TRAIN | 1217 | 23.0 | +0.058 | +0.088 | 2.00 | 47.8 | +1.33 | 0.58 | -9.5 | -20.0 | 9/12 | -0.044 | +0.058 |
| S0 | 2r | VAL | 593 | 27.0 | +0.110 | +0.142 | 2.56 | 48.7 | +2.97 | 0.68 | -10.5 | -20.2 | 5/5 | +0.011 | +0.110 |
| S1 | 2r | TRAIN | 1342 | 25.3 | +0.060 | +0.090 | 2.15 | 46.9 | +1.52 | 0.62 | -10.2 | -20.9 | 8/12 | -0.043 | +0.060 |
| S1 | 2r | VAL | 624 | 28.4 | +0.100 | +0.133 | 2.38 | 49.4 | +2.84 | 0.64 | -10.6 | -20.4 | 5/5 | -0.001 | +0.100 |
| S2 | 2r | TRAIN | 1405 | 26.5 | +0.035 | +0.067 | 1.26 | 45.8 | +0.92 | 0.55 | -12.3 | -20.5 | 8/12 | -0.069 | +0.035 |
| S2 | 2r | VAL | 646 | 29.4 | +0.076 | +0.111 | 1.81 | 48.5 | +2.24 | 0.55 | -11.0 | -17.9 | 4/5 | -0.026 | +0.076 |
| S3 | 2r | TRAIN | 1497 | 28.2 | +0.004 | +0.035 | 0.14 | 45.4 | +0.10 | 0.57 | -13.8 | -40.2 | 7/12 | -0.100 | +0.004 |
| S3 | 2r | VAL | 649 | 29.5 | +0.062 | +0.096 | 1.53 | 47.5 | +1.82 | 0.59 | -9.1 | -18.3 | 4/5 | -0.040 | +0.062 |

Family mix of the booked trades (S3, hold): TRAIN F6 642 / F8N30 599 / F11 100 / F14 18; VAL F6 341 / F8N30 170 /
F11 49 / F14 5. The dedupe is why F11 (a close-confirmed F6) contributes so little: it signals the same
symbol-days F6 does and loses the priority tie.

**Only one of the sixteen cells beats S0 on its own split** — S1 `hold` TRAIN, by +0.02 weekly R (+1.67 vs +1.65)
— and it is below S0 on VAL and on TEST. Under the `2r` exit S1 beats S0 on TRAIN (+1.52 vs +1.33) and loses on
VAL (+2.84 vs +2.97).

### 1.3 Freeze (declared rule: highest VAL weekly R)
Ranking on VAL weekly R: S0 hold +5.91 > S1 hold +5.71 > S2 hold +5.04 > S3 hold +5.01 > S0 2r +2.97 > S1 2r
+2.84 > S2 2r +2.24 > S3 2r +1.82. **Frozen: S0 (F6 alone), exit `hold`.** `I/FREEZE.md` was closed before
`i_test.py` ran; its amendment (also written before the run) spends the stage's TEST allowance on S0/S1/S2/S3 x
`hold` — four cells, descriptive only, because the freeze rule had already chosen and S0's TEST is public.

### 1.4 TEST (read once)

| stack | n | tr/wk | net R | gross | t | WR% | wkR | wks green | worst wk | MDD | mo green | ex-top5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| S0 | 376 | 26.9 | **+0.104** | +0.138 | 0.74 | 41.5 | +2.79 | 0.43 | -13.3 | -26.9 | 2/4 | -0.223 |
| S1 | 386 | 27.6 | +0.085 | +0.120 | 0.62 | 41.7 | +2.36 | 0.50 | -14.3 | -25.5 | 2/4 | -0.238 |
| S2 | 393 | 28.1 | -0.049 | -0.013 | -0.75 | 41.7 | -1.37 | 0.36 | -11.9 | -32.5 | 1/4 | -0.258 |
| S3 | 390 | 27.9 | -0.089 | -0.053 | -1.41 | 39.2 | -2.47 | 0.29 | -11.4 | -39.7 | 1/4 | -0.293 |

The S0 row is identical to the already-public `f6_pdr_book.md` TEST row (n 376, +0.104, t 0.74) — the stack
machinery reproduces it, so the S1-S3 degradation is a property of the added families, not of this stage's code.
The ordering S0 > S1 > S2 > S3 holds in **all three splits**.

### 1.5 Marginal contribution and where the extra trades come from

| exit | split | step | n booked | dn | book total R | d total R | trades the new family booked | their mean R | their total R |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| hold | TRAIN | S1 (+F14) | 1219 | +97 | +88.5 | **+1.1** | 163 | +0.062 | +10.2 |
| hold | TRAIN | S2 (+F11) | 1270 | +51 | +56.5 | **-32.1** | 149 | -0.144 | -21.5 |
| hold | TRAIN | S3 (+F8N30) | 1359 | +89 | +28.6 | **-27.8** | 599 | -0.002 | -1.2 |
| hold | VAL | S1 (+F14) | 540 | +21 | +125.6 | **-4.4** | 48 | -0.013 | -0.6 |
| hold | VAL | S2 (+F11) | 560 | +20 | +110.9 | **-14.8** | 61 | -0.137 | -8.3 |
| hold | VAL | S3 (+F8N30) | 565 | +5 | +110.3 | **-0.6** | 170 | +0.029 | +4.9 |
| 2r | TRAIN | S1 (+F14) | 1342 | +125 | +80.6 | **+9.9** | 198 | +0.097 | +19.2 |
| 2r | TRAIN | S2 (+F11) | 1405 | +63 | +48.9 | **-31.7** | 155 | -0.144 | -22.4 |
| 2r | TRAIN | S3 (+F8N30) | 1497 | +92 | +5.4 | **-43.5** | 699 | -0.008 | -5.9 |
| 2r | VAL | S1 (+F14) | 624 | +31 | +62.5 | **-2.9** | 64 | -0.004 | -0.3 |
| 2r | VAL | S2 (+F11) | 646 | +22 | +49.2 | **-13.3** | 75 | -0.207 | -15.5 |
| 2r | VAL | S3 (+F8N30) | 649 | +3 | +40.0 | **-9.3** | 218 | -0.019 | -4.0 |

F14 is the only family whose own booked trades carry a positive mean R on TRAIN (+0.062 hold, +0.097 2r) — and
even it is a wash on the book total, because its rows displace F6 rows worth about as much. F11's own trades are
-0.14 R in every split-exit pair; F8 N=30's are ~0 and it floods the book (599 of 1,359 TRAIN hold trades).

**Frequency vs displacement** (classifying each stacked trade against the S0 book of the same exit/split):

| exit | split | stack | kept from S0 | added on a FREE slot | took an occupied slot | S0 trades dropped | R of the added | R of the slot-takers | R of the dropped S0 trades |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| hold | TRAIN | S1 | 1054 | 160 | 5 | 68 | +11.1 | -1.4 | +8.5 |
| hold | TRAIN | S2 | 952 | 294 | 24 | 170 | -16.2 | -1.0 | +13.7 |
| hold | TRAIN | S3 | 630 | 672 | 57 | 492 | **-32.9** | +12.1 | **+38.1** |
| hold | VAL | S1 | 490 | 48 | 2 | 29 | -0.6 | +0.7 | +4.5 |
| hold | VAL | S2 | 446 | 103 | 11 | 73 | -7.6 | +4.6 | +16.2 |
| hold | VAL | S3 | 337 | 206 | 22 | 182 | -2.4 | +6.6 | +23.9 |
| 2r | TRAIN | S3 | 653 | 763 | 81 | 564 | -36.1 | +15.5 | +44.7 |
| 2r | VAL | S3 | 360 | 254 | 35 | 233 | -17.1 | +8.7 | +16.9 |

(full table in `I/a_trainval.md` §4.) Two facts: **the stack really does add frequency** — 92% of S3's new TRAIN
trades landed at a minute when the F6-alone book had a free slot — and **the added frequency is negative**. The
492 F6 trades that fall out of the S3 book are not displaced at their own minute; they are crowded out later in
the day by earlier-minute fills from the other families, and they were worth +38.1 R. That is the whole
mechanism: the 12/day-4-concurrent budget was never binding on F6, so extra families can only dilute.

### 1.6 The liquidity-capped twin — the money number (`I/c_capacity.md`)

Method copied from `H/F6_sizing`: fill-bar and trailing-5-minute dollar volume at the fill minute, bar source
`data/cache.db` first then `bf_zero/bars_sip.db` (both read-only). Coverage 100.00%; the next-open fill lies
inside its own bar on **100.00%** of rows; five of five minutes printed on 65.1%. Cap: drop every population row
whose participation `shares x entry / 5-min $ volume` exceeds 1% and re-run `run_book(12,4)` so freed slots refill.

Frozen cell (S0, hold), at three risk levels:

| risk $ | pop kept @1% | TRAIN net R | TRAIN $/mo | VAL net R | VAL $/mo | TEST net R | TEST $/mo | uncapped $/mo (T/V/Te) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 150 | 61.2% | +0.053 | +542 | +0.132 | +1,794 | -0.080 | -952 | 1,092 / 3,901 / 1,463 |
| **300** | **49.5%** | **+0.073** | **+1,274** | **+0.075** | **+1,877** | **-0.146** | **-3,229** | 2,185 / 7,802 / 2,925 |
| 600 | 37.9% | +0.065 | +1,917 | +0.086 | +3,793 | -0.105 | -4,069 | 4,370 / 15,604 / 5,850 |

The live-computable form of the same cap (four PRIOR bars x1.25, since the fill bar's own volume is unknown when
the order is sent) gives the same picture: $300 -> +$937 TRAIN / +$1,905 VAL / -$2,722 TEST per month. Spread x1.5
costs about $200-320/month on TRAIN/VAL and does not change a sign.

**This is the stage's most expensive finding for the owner.** Uncapped, the frozen book's VAL is +0.251 R and
$7,802/month at $300 — but half of those trades would have been more than 1% of the five-minute tape. Capped, VAL
falls to +0.075 R and TEST **flips to -0.146 R**. The cap removes the illiquid names, and the illiquid names are
where the R is: it is the same anti-correlation `H/F6_sizing` measured ($5-10 price and sub-500K ADV carry the
edge and can carry $73 of risk). The liquidity cap is not a haircut on this book — it is a different book.

The stacks under the cap at $300 ($/month TRAIN / VAL / TEST): S0 +1,274 / +1,877 / -3,229 · S1 +1,661 / +1,795 /
-3,516 · S2 +1,488 / +1,972 / -2,475 · S3 +487 / +2,869 / -1,796. No stack is better on all three; all four are
negative on TEST. Full grid in `I/cap_cells.csv` (36 capped cells).

### 1.7 Tail tests on the frozen cell

| split | n | mean net R | top-1% removed | top-5% removed | capped at +3R | share of total R in the top 5% |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 1122 | +0.078 | -0.016 | -0.161 | -0.005 | **296%** |
| VAL | 519 | +0.251 | +0.123 | -0.036 | +0.114 | 114% |
| TEST | 376 | +0.104 | -0.069 | -0.223 | -0.061 | **305%** |

The book is a tail book in all three splits: removing 1% of the trades already flips TRAIN and TEST negative, and
a +3R winner cap leaves nothing. This is the same tail dependence `H/F6` and `E/REPORT` reported; stacking does
not dilute it — S3's ex-top-5% is worse than S0's in every split.

### 1.8 Artefacts
`I/frozen_stack_trades.csv` (2,017 booked trades, all three splits, uncapped) ·
`I/frozen_capped_trades_300.csv` (1,412 trades with $ at $300 risk, 1% cap) · `I/a_cells.csv`,
`I/a_test_cells.csv`, `I/cap_cells.csv`, `I/cap_monthly.csv` · logs `I/a_trainval.md`, `I/a_test.md`,
`I/c_capacity.md`, `I/b_portfolio.md`, `I/b_portfolio_months.csv`.

---

## 2. Part B — the portfolio of sleeves, monthly $

Each sleeve read as-is from its own declared artefact. No re-simulation, no re-sizing.

| sleeve | source | basis |
|---|---|---|
| intraday | this stage's frozen cell, capped twin | $300 risk/trade, 1% participation cap, 12/day 4-concurrent |
| QQQ noise band | `H/QQQ/final_book_days.csv` (`r1x` x $60,000) | unfiltered base book, live-convention fill, 0.5 bp/leg |
| ORB B+ | `analysis_results/orb_bplus_book.csv` (`_sized_pnl`) | `orb.yaml` B+: $10,000 budget / 3 concurrent / $375 risk / per-pos cap $3,333 |
| TQQQ 1x (optional) | `Q/step3_months.csv`, book `TQQQ 1x notional (0.5bp/leg)` | $60,000 notional, monthly only (no day-level file exists) |

### 2.1 Month by month, 2025-01 -> 2026-09

| month | intraday | QQQ 1x | ORB B+ | **sum of 3** | TQQQ 1x | sum of 4 |
|---|---:|---:|---:|---:|---:|---:|
| 2025-01 | -3,343 | +182 | +228 | **-2,932** | +1,007 | -1,925 |
| 2025-02 | +1,890 | +282 | +634 | **+2,806** | +1,044 | +3,850 |
| 2025-03 | -1,547 | +287 | +82 | **-1,178** | +1,976 | +798 |
| 2025-04 | +3,326 | +5,348 | +179 | **+8,853** | +16,216 | +25,069 |
| 2025-05 | +1,054 | -506 | +895 | **+1,443** | -1,465 | -22 |
| 2025-06 | -3,108 | -592 | -203 | **-3,904** | -1,492 | -5,396 |
| 2025-07 | +549 | -1,324 | -13 | **-788** | -3,655 | -4,444 |
| 2025-08 | +3,475 | +241 | +158 | **+3,874** | +1,076 | +4,950 |
| 2025-09 | +141 | -1,956 | +524 | **-1,291** | -5,323 | -6,614 |
| 2025-10 | +957 | +395 | +1,115 | **+2,467** | +2,472 | +4,939 |
| 2025-11 | +6,750 | +346 | +64 | **+7,160** | +1,149 | +8,308 |
| 2025-12 | +5,149 | -292 | -12 | **+4,845** | -1,097 | +3,748 |
| 2026-01 | +1,046 | -2,243 | -223 | **-1,421** | -7,036 | -8,457 |
| 2026-02 | +526 | +740 | +707 | **+1,973** | +2,565 | +4,538 |
| 2026-03 | +4,855 | +1,667 | +2,038 | **+8,560** | +4,471 | +13,030 |
| 2026-04 | +4,862 | -230 | +293 | **+4,924** | -1,086 | +3,839 |
| 2026-05 | -1,901 | +5 | -133 | **-2,029** | +441 | -1,588 |
| 2026-06 | -6,468 | +2,357 | +390 | **-3,720** | +5,602 | +1,882 |
| 2026-07 | -6,156 | -1,663 | +419 | **-7,400** | -4,784 | -12,184 |
| 2026-08 | +339 | +1,234 | +168 | **+1,740** | +4,032 | +5,773 |
| 2026-09 | -629 | +224 | -122 | **-528** | +709 | +181 |

### 2.2 Per-sleeve summary (21 months)

| sleeve | total | mean/mo | median/mo | months green | worst month | best month | monthly-curve MDD | 2025 | 2026-YTD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intraday ($300 capped) | +11,765 | +560 | +549 | 14/21 | -6,468 (2026-06) | +6,750 (2025-11) | -14,816 | +15,293 | -3,528 |
| QQQ 1x $60K | +4,503 | +214 | +224 | 13/21 | -2,243 (2026-01) | +5,348 (2025-04) | -5,930 | +2,412 | +2,091 |
| ORB B+ $10K | +7,187 | +342 | +179 | 15/21 | -223 (2026-01) | +2,038 (2026-03) | -235 | +3,650 | +3,536 |
| **sum of 3** | **+23,455** | **+1,117** | **+1,443** | **11/21** | **-7,400 (2026-07)** | +8,853 (2025-04) | **-13,149** | +21,356 | +2,099 |
| TQQQ 1x $60K | +16,820 | +801 | +1,007 | 13/21 | -7,036 (2026-01) | +16,216 (2025-04) | -15,372 | +11,906 | +4,914 |
| sum of 4 | +40,276 | +1,918 | +1,882 | 13/21 | -12,184 (2026-07) | +25,069 (2025-04) | -12,184 | +33,262 | +7,013 |

### 2.3 Correlations

| | intraday | QQQ | ORB | TQQQ |
|---|---:|---:|---:|---:|
| intraday | +1.00 | +0.15 | +0.19 | +0.15 |
| QQQ | +0.15 | +1.00 | +0.21 | **+0.99** |
| ORB | +0.19 | +0.21 | +1.00 | +0.20 |
| TQQQ | +0.15 | +0.99 | +0.20 | +1.00 |

Sum of the three standalone monthly sigma = $5,690; sigma of their sum = $4,272 (ratio 0.75). The three sleeves are
genuinely near-independent month to month. **TQQQ 1x is not a fourth sleeve** — it is the same QQQ noise-band
decision expressed in a 3x instrument (rho = +0.99). Adding it is leverage on one sleeve, not diversification, and
it is why "sum of 4" has a -$12,184 month.

Context, the QQQ sleeve over its whole file (2016-01 -> 2026-09, $60K 1x): +$56,958 total, +$442/month, 82/129
months green, worst month -$2,431, monthly-curve MDD -$5,930 — a steadier instrument than its 2025-26 slice
suggests, and the one whose 21-month window here happens to contain its single biggest month (2025-04, +$5,348,
which `Q/REPORT.md` already flagged as 51% of the sleeve's 33-month total).

### 2.4 Which sleeves are live-able today

- **Intraday F6-PDR**: **not live-able**. There is no engine for it (the `--hod` engine implements F5, a
  different family), it has never had a dry run, and under the liquidity cap that makes it obtainable its TEST
  third is -$3,229/month. `H/F6/ENGINE_SPEC.md` is a spec, not code.
- **QQQ noise band**: **not live-able**. `Q/DRYRUN_SPEC.md` is written as a spec whose stated purpose is to
  measure execution, not to confirm the edge — its OOS arithmetic dies at 2.01 bp/leg and the measured live slip
  is unknown. It needs a dry-run engine first.
- **ORB B+**: **paused by the owner** (9/14, `orb.yaml strategy.enabled: false`). The code exists and is
  parity-audited; resuming is a config flip and the owner's word, not a build.
- **TQQQ 1x**: same sleeve as QQQ, same blocker, plus a 3x instrument's tail.

So the "combined honest expectation" has no live path this month: **zero of the three sleeves can trade today**,
and the one that could be turned on with a config flip (ORB B+) is the smallest at +$342/month with a -$223
worst month.

### 2.5 The honest combined statement

Over 2025-01 -> 2026-09, at the declared sizings, the three sleeves together would have produced **+$1,117 per
month on average (median +$1,443), green in 11 of 21 months, with a worst month of -$7,400 and a
monthly-curve drawdown of -$13,149**. That is arithmetic on already-published books, not a forecast, and it
carries four disclosures: (a) every constituent failed its own pre-registered VAL gate; (b) the intraday sleeve
supplies half the total and is negative across its four TEST months (-$12,914); (c) the QQQ sleeve's 2025-04
(+$5,348) is 119% of its 21-month total; (d) the ORB sleeve's $7,187 is at a $10K stage — it is the only one of
the three with a live engine and the only one whose worst month is smaller than a single bad day of the others.

---

## 3. Cell count (the multiplicity denominator)

| block | cells |
|---|---:|
| declared TRAIN/VAL book cells (4 stacks x 2 exits x 2 splits) | 16 |
| S0 parity checks vs `f6_pdr_book.md` | 4 |
| TEST (1 frozen + 3 by the FREEZE.md amendment) | 4 |
| liquidity-capped grid (4 stacks x 3 risks x 3 splits) | 36 |
| live-computable cap (3 risks x 3 splits) | 9 |
| spread x1.5 sensitivity (3 splits x 2) | 6 |
| tail tests on the frozen cell (3 splits x 3 variants) | 9 |
| marginal-contribution and frequency/displacement decompositions (no new books) | 24 descriptive |
| **total book-cell instances in Stage I** | **84** |

Part B adds no simulation cells — it reads four artefacts and sums them.

## 4. What this stage does and does not establish

**Establishes** (in THIS universe — `C/pop_c.csv`'s >=5%-range symbol-days with `prev_day_range_pct >= 8`, at THIS
horizon — intraday 1-minute long entries to 15:55, at THIS book size — 12/day and 4 concurrent, over
2025-01->2026-09, at contract (c)'s costs): pooling F14, F11 and F8 N=30 into the F6-PDR book lowers mean net R,
lowers weekly R and raises the worst week in every split; the slot budget is not the binding constraint, so the
added families can only dilute; and the F6-PDR book's dollar edge does not survive a 1% participation cap
out-of-sample.

**Does not establish** that no stacking helps. The four families are all one-minute long breakout/reclaim shapes
on one population — they are correlated by construction, which is exactly why pooling them behaves like adding
noise. A stack of genuinely different shapes (the short side, the ETF sleeve, a different horizon) is a different
question, and Part B's +0.15-0.21 cross-sleeve correlations are the evidence that it is the one worth asking.

**Power.** Per-trade SD on TEST is 2.73 R (S0) and 1.25 R (S3), so an UNPAIRED comparison of two of these
books resolves, at 80% power, only a difference of **0.43 R per trade** — the observed S0-S3 TEST gap is 0.19 R,
BELOW that floor, and the S0-S1 gap (0.02 R) is far below it. A single split therefore ranks nothing. What
carries the finding is not one split's t but the **consistency of the sign**: S0 > S1 > S2 > S3 on weekly R in
TRAIN-hold, VAL-hold, TEST-hold, TRAIN-2r and VAL-2r — S0 beats S3 in 5 of 5 independent split x exit
comparisons (a sign test gives p = 0.031), the books share most of their trades so the differences are paired,
and the mechanism (§1.5) is measured directly rather than inferred. The monthly portfolio table has 21
observations per sleeve: a $500/month difference between sleeves is inside its noise.

**Phrasing.** No edge was found to be created by stacking IN THIS universe (>=5%-range symbol-days with
prev-day range >= 8%), at THIS horizon (1-minute long entries, exit by 15:55), at THIS book size (12/day,
4 concurrent), over THIS window (2025-01 -> 2026-09), at THIS cost contract. The smallest per-trade stacking
effect the TEST comparison could have seen is 0.43 R unpaired; the smallest the 5-split sign test could have
seen is a consistent difference of any size.
