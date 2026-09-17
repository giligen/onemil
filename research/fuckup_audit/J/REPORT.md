# Stage J — the liquid universe (U3)

**TRAIN is complete and scored. VAL and TEST are not yet built — the scan is still running.**
Pre-registration: `J/PREREG.md` + its **ADDENDUM 1-3** (2026-09-17 14:40 UTC, written before the scan).
Nothing here is enabled, selected or proposed. TEST is unread and stays unread.

---

## 0. One page

**Does anything on the liquid universe clear G1 on TRAIN? No — 0 of 144 declared cells.**

The liquid universe U3 (20-day median dollar volume >= $5M and open >= $5; 1,511,915 symbol-days,
5,568 names, 410 days, membership known at 09:30) was scanned for the first time. TRAIN
(2025-01-17 -> 2025-12-31, 236 days, 51 weeks) produced **4,859,792 signals, 97.1% filled,
3,257,389 scoreable** across 9 long and 5 short family-configs. Under the pre-registered contract —
one fill (the next printed bar's open under the 0.6% cap), three exits, the cost curve's liquidity-band
spread, `run_book(12, 4)` per family — **not one of the 144 cells reaches G1's `mean net R > 0, t >= 2,
>= 5 trades/week`.**

| the best cell on TRAIN | |
|---|---|
| cell | `F14 N=15` (second break) / exit `hold` / R >= 0.5% / **PDR >= 8** / scan `first` |
| n, trades per week | 899, 17.6 |
| mean net R | **+0.0762** |
| t | **1.71** (G1 needs 2.0) |
| smallest effect this test could see (2 x SE) | 0.0892 R |
| win rate, weekly R, weeks green | 46.7%, +1.34 R, 53% |
| worst week | -12.7 R |
| **top-5% of trades removed** | **-0.1255** |
| winners capped at +3R | +0.0203 |

The best cell is **tail-dependent**: remove the top 5% of its trades and +0.076 becomes **-0.126**. It is
one cell of 144, it is a twin (PDR >= 8) of a family that is flat on its own (`F14` pre-book +0.0014 R),
and the **search-adjusted permutation puts it nowhere**: over 300 day-level sign-flip draws across all
144 booked cells, the observed max TRAIN t is **1.71 against a null 95th percentile of 4.52, p = 1.000**.

**Two structural findings that do not depend on any cell clearing.**

1. **The pre-registration's cost premise is falsified, in the direction that makes the hurdle bigger.**
   PREREG assumed "names with > $50M/day quote 16-25 bps — cost ~ 0.05R on a 2% stop". The spread is
   right; **the 2% stop is not**. These families put the stop at an opening-range low or a running low of
   a *liquid* name: median R is **1.24% of price** (p10 0.63%, p90 3.00%). So the median round trip costs
   **0.143 R**, and 0.36 R at the p90 — not 0.05 R. The liquid universe is cheaper per share and is
   **not** cheaper per R. The cost gradient is visible straight through the results: pre-book mean net R
   is **-0.098 for > $50M/d names against -0.237 for $5-50M/d**, and the 09:30-09:35 band (widest
   spreads) is the worst hour at **-0.311** against -0.067 after 13:00.
2. **Capacity fails before edge does, and for a reason specific to liquid names.** A tight stop means a
   fixed dollar risk buys a *large* notional: $300 of risk at a median risk-per-share of $0.58 on a $43
   stock is a **$22.5K order against a median trailing-5-minute tape of $515K = 4.4% participation**.
   Against the declared 1%-of-5-minute-tape cap:

   | risk | median participation | share of the population above the 1% cap |
   |---:|---:|---:|
   | $300 | 4.44% | **76.2%** |
   | $1,000 | 14.81% | **89.7%** |
   | $2,000 | 29.62% | **94.2%** |

   Stage I's sentence for small caps was "the money is in names that cannot carry the money". Here the
   binding constraint is not the name, it is **the notional the tight stop forces**. The one family that
   escapes it is **F9 gap-and-go (0.46% participation at $300, 1.54% at $1,000)** — precisely because a
   gapper's stop is wide — and F9 is the *worst* long family pre-book (-0.290 R). The two properties
   this book needs are, on this universe, anti-correlated.

**Phrasing, as required.** No edge was detectable **in the liquid universe U3, at a 1-minute horizon,
in a 12-a-day 4-concurrent book, over 2025-01-17 -> 2025-12-31, at the measured liquidity-band spread**.
The smallest per-trade mean the TRAIN test could have called significant at `t >= 2` is
**0.050 - 0.099 R** across the 144 cells (printed per cell as `TRAIN_mde`; best-cell value 0.089 R).
An effect smaller than that is invisible here and this stage says nothing about it. VAL and TEST have
not been scored.

---

## 1. What was built, and the tape it was built on

| file | what |
|---|---|
| `make_members_u3.py` -> `members_u3.parquet` | 1,511,915 U3 symbol-days with causal daily context + `univ_flag` |
| `universe_exclusions.csv` | the 312 excluded symbols and why (ADDENDUM 1) |
| `build_candidates_u3.py` -> `candidates_u3.csv` | one row per signal: level, stop, fill, three exits, mae/mfe, the liquidity-band spread, the capacity columns |
| `verify_rows.py` -> `verify.md`, `verify_TRAIN.md` | the independent check (PLAN section 1) |
| `score_u3.py` -> `score_u3_tables_TRAIN.md`, `_results_TRAIN.csv`, `_capacity_TRAIN.csv`, `pop_j_TRAIN.parquet` | the 144 declared cells on TRAIN |

**Three bar stores read as ONE tape**: `U3/bars_u3/` (parquet, the 1.22M-key fetch of 2026-09-17),
`E/bars_causal/` (parquet), `research/bf_zero/bars_sip.db`. All Alpaca SIP, 1-minute, adjustment raw.
`data/cache.db` is **not** a bar source (different fetch provenance — the 2026-09-15 parity review's
rule); it is opened read-only exactly once, for ADDENDUM 1's symbol list. The two parquet stores were
verified **disjoint** (0 of 164,751 U3 keys appear in the E index), so there is no precedence ambiguity.

**Families** (imported, not re-written — long from `B/build_candidates4.py`, short from
`G/build_candidates_short.py`): long F6, F8 N=5/15/30, F14 N=15, F11 base=F6, F5 K=5/X=0.04,
F9 G=0.03, F10; short S1 (gap >= +3%), S2 N=15/30, S3, S5.

**Engine conventions** (`H/F6_reconcile/pipeline.py::E_CFG`): signal bar index >= 1 (09:31 earliest);
the **14:00 cut on the SIGNAL minute**, not the fill; stop = the running low **through** the signal bar
for the families whose stop *is* a running low (F6, F11 base=F6; short mirror S3 -> running high
through the signal bar), the family's structural level otherwise; "next bar" = the next bar the tape
**prints**.

**Exits**: `hold` (flat 15:55 at that bar's open), `2r` (target at +2R on a bar close), `pp`
(50% at +2R on a bar close, stop -> entry, runner walks on). Cost: `half = 0.5 x (spread_bps/100) /
max(r_pct, 0.05)`; entry leg 0.25 x half; exit legs per `legs_*` at stop 0.875 / eod 0.412 /
target 0.875. Shorts: borrow 0 in the primary book, **5.274 bps of notional** as the declared
sensitivity (it costs shorts **0.045 R**: -0.158 -> -0.203).

---

## 2. The result in full

### 2.1 The gate

**G1 (TRAIN mean net R > 0, t >= 2.0, >= 5 trades/week): 0 of 144.** Every cell has a TRAIN book.
G2 and the pooled book were therefore not run (the PREREG trigger is "only if a family clears G1").
The 144: 14 family-configs x 3 exits x 2 R floors (84) + the PDR >= 8 twin at the primary floor (42)
+ the ADDENDUM-3 keep-scan twin for the 6 long level-break configs (18).

The nine cells with a positive TRAIN mean, in order — note that **eight of the nine are PDR >= 8 twins
or S1**, and none reaches t = 2:

| key | exit | sub | n | tr/wk | mean net R | t | mde | ex-5% |
|---|---|---|---:|---:|---:|---:|---:|---:|
| L F14 N=15 | hold | PDR>=8 | 899 | 17.6 | +0.0762 | 1.71 | 0.089 | -0.126 |
| L F14 N=15 | pp | PDR>=8 | 904 | 17.7 | +0.0453 | 1.20 | 0.076 | -0.106 |
| L F14 N=15 | 2r | PDR>=8 | 944 | 18.5 | +0.0182 | 0.54 | 0.067 | -0.084 |
| S S2 N=15 | hold | PDR>=8 | 1301 | 25.5 | +0.0116 | 0.33 | 0.071 | -0.178 |

(the remainder are below +0.01 R; the full ranking is in `score_u3_tables_TRAIN.md`.)

### 2.2 Multiplicity

300 day-level sign-flip permutations over all 144 booked cells: **observed max TRAIN t 1.71, null 95th
percentile 4.52, p = 1.000**. With 144 cells the bar that a single cell must clear to be
search-adjusted significant is **t > 4.5**, not G1's 2.0 — worth carrying into any future stage that
declares this many cells.

### 2.3 Capacity, per cell (reported, never assumed)

At the declared 1%-of-trailing-5-minute-tape cap and $1,000 of risk, **89-93% of every long family's
population is dropped**; the capped books are negative for every family except F14
(**capped mean +0.0475 R, +$3,743/month, worst month -$19,068** — and F14 fails G1, so this is a
post-hoc look, not a candidate). Median participation at $300 of risk runs **3.5% (F8 N=30) to 6.4%
(F5)**, i.e. **3-6x the cap before any sizing decision is made**. Full table:
`score_u3_capacity_TRAIN.csv` (every cell, all three risk levels, capped and uncapped $/month).

### 2.4 Descriptive (reported, never selected on), pre-book, exit hold, R >= 0.5%

| by liquidity band | n | mean net R |
|---|---:|---:|
| > $50M/d | 1,508,805 | **-0.098** |
| $5-50M/d | 1,604,633 | **-0.237** |

| by hour | n | mean net R |
|---|---:|---:|
| 09:30-09:35 | 325,722 | -0.311 |
| 09:35-10:00 | 988,434 | -0.193 |
| 10:00-11:00 | 1,115,082 | -0.151 |
| 11:00-13:00 | 537,170 | -0.105 |
| 13:00+ | 147,030 | -0.067 |

By side: longs -0.176 (1.98M), shorts -0.158 (1.13M) before borrow, -0.203 after. By family, the only
two not clearly negative pre-book are **S1 gap-fade +0.109 R on 14,948 signals** and **F14 +0.001 on
24,844** — and S1's *booked* cells are all negative (hold R>=0.5% all: -0.044, t -1.01), because the
book takes the first 12 signals of the day and 4 concurrent, which selects the earliest, widest-spread
part of the population. Every other family is between -0.095 and -0.306.

---

## 3. The three pre-registration amendments, and what each one found

### A1 — universe hygiene
- **Dropped outright**: `^Z[A-Z]ZZT$` -> **ZAZZT, ZJZZT, ZVZZT, 116 symbol-days**. These print synthetic
  tapes and were the entire TEST "profit" of the F6 book in the reconciliation. The verifier asserts
  none reached the file: **0 found**.
- **Tagged and excluded from every scored cell**: **309 symbols / 147,626 TRAIN rows** with zero rows in
  `data/cache.db::daily_bars`. Tagged rather than deleted because 309 of the 312 are real securities —
  delisted/acquired (ANSS, AZPN, AMED, ACCD) and preferred classes (ALB-A, BAC-L, C-N) — so excluding
  them is **a survivorship filter in the opposite direction**. Measured size on TRAIN: primary
  -0.1694 R on 3,113,438 rows vs -0.1699 including them; the excluded rows alone -0.1880 on 90,789.
  **It is not load-bearing either way**, which is the point of printing it.

### A2 — every family's level buffer, stated instead of inherited
`build_candidates.fam_r2g` and `fam_level` take their buffer from the module global `SLIP` = env
`BFZ_SLIP`, which **every pass-1 builder in this tree pins to 0** (`B/build_candidates4.py:99`). Stage
B/E/G therefore scanned F6 at `prior_close x 1.000` while describing it as the engine's rule; the
declared engine level is `x 1.003` (`trading/red_to_green.level_for`). Corrected here by `fam_r2g_buf`,
whose `buf = 0` form is asserted identical to the imported detector at import (`_selftest_f6`). The full
buffer table is in `PREREG.md` ADDENDUM 2.

**The by-product finding**: the long and short sides of this tree were *inconsistent* — G's S3
green-to-red already carried its `x 0.997` mirror explicitly while F6 carried nothing. Stage E's F6 and
Stage G's S3 were not mirrors of each other, and nobody had noticed.

Effect: F6 signal count -10% on a sample day. The parity check confirms the change is exactly the buffer
and nothing else (section 4).

### A3 — both scan rules
`first` (the first bar reaching the level is the signal; a failed fill kills the day) and `keep` (keep
scanning later breaks of the same level until one fills — `trading/red_to_green.detect`'s rule, and the
only one that could ever go live). Emitted as twin cells for F6/F8 N=5/15/30/F14/F11, named on every
table. On TRAIN the `keep` rule adds **60,877 trades = 1.64% of the first-scan population**, all filled
by construction, and the added trades are **better** than the base (pre-book -0.0909 vs -0.1694) — but
1.6% of the population cannot move a book, and no `keep` cell clears G1 either. The rule matters far
less here than on small caps because the first-scan fill rate is already **97.1%**: the 0.6% chase cap
rarely binds when the name is liquid.

---

## 4. The independent check (`J/verify_TRAIN.md`) — PLAN section 1. **Exit 0, all checks passed.**

1. **Hand recomputation.** 8 filled rows drawn by a seeded reservoir over the whole 4.86M-row TRAIN
   file, their raw bars re-read from the three stores, every field recomputed by a **second
   implementation written from the prose spec** (plain Python loops, no numpy, no call into
   `B4`/`GS`/`pipeline`): signal bar, level, stop, fill, `r_pct`, all three exits with their minutes and
   reasons, mae/mfe, and the three capacity columns. **All reproduced field for field**, across F6, F8
   N=5/15/30, S2, S3, S5, long and short, including a `pp+eod` partial and a `target` 2R exit.
2. **Obtainability (PLAN 1.1b).** Every fill lies inside the bar that fills it: **1.000000** on
   4,719,541 filled rows. Fill rate by family 88-100%; S5 is 100% by construction (a scheduled 09:35
   market order at that bar's own open).
3. **Tape + detector parity vs Stage E.** On the 16,458 F8 keys both files hold with the same detectors:
   **sig_m, level, stop, entry, entry_m differ on 0 rows.** F6 differs exactly as ADDENDUM 2 predicts:
   J level = E level x 1.003 on **1,874 of 1,874** rows, and J stop <= E stop on **1,874 of 1,874**
   (strictly lower on 195 — the running-low-through-the-signal-bar convention doing its work).
4. **Availability audit.** `spread_cc_bps` null 0, `dv5` null 0, `prev_day_range_pct` null 11 of 4.86M,
   `lb_low` (a liquidity band U3 membership makes impossible) 0, `fside_bad` 0, filled rows below the $5
   price floor 2,964 (excluded).

**Three defects this process caught.** (a) The first draft of the builder had the **fill-acceptance test
inverted** (`stop <= entry` *rejected* a long instead of accepting it), pinning the fill rate at 0.5% and
making `r_pct` negative on every row — found by the 5-row hand recomputation, not by review. (b) The
parity check itself first reported **663 "tape differences"** on F8 that were nothing but this stage's
own `%.8g` CSV format ($351.910004 written as $351.91000) against an absolute 1e-6 tolerance; the
tolerance is now relative, and the diagnosis that chased it is what established the two parquet stores
are disjoint. (c) The verifier's reservoir used `iterrows()` and blew its `ulimit` on the 4.86M-row
file; it now materialises a row only when the reservoir actually selects it.

---

## 5. Declared: every look taken

- **144 cells** on TRAIN, as pre-registered. No cell was added, dropped or re-specified after the scan.
- **A plumbing dry run** on the first 41 TRAIN days (`_partial` tag, kept on disk) scored the same 144
  cells before the full split existed, returned G1 0, and selected nothing. It adds no cell to the count
  and its population statistics (cost, R, participation) are superseded by the TRAIN numbers here.
- **No model was fitted**, so the reversed-tape twin (a standing rule for models) does not apply and was
  not run. **No NBBO pull** was made: PREREG makes it conditional on a cell clearing G1, and none did.
- **TEST is unread.** VAL is not built yet.

---

## 6. What is still running, and how to finish it

| | |
|---|---|
| U3 fetch | `U3/fetch_u3.py`, day ~250/410 at 18:35 UTC, ETA ~21:30-22:00 UTC |
| Stage-J builder | chase mode, ~58 s/day incl. idle, at 2026-01-14; ~3.1 GB when complete |
| then | `verify_rows.py` on the full file -> `score_u3.py --perm 500` -> VAL, and TEST only behind a written freeze |

Every command, in order, with its `ulimit` and its check, is in **`J/README.md`**. The builder is
resumable per day and appends one whole day at a time, so a kill leaves neither a duplicate nor a
half-day. To reproduce this TRAIN-only result exactly:
`awk -F, 'NR==1 || $1 < "2026-01-01"' candidates_u3.csv > candidates_u3_TRAIN.csv`, then
`J_TAG=_TRAIN J_SRC=... python3 score_u3.py --perm 300` (`pop_j_TRAIN.parquet` is the scored population
and re-runs with `J_REUSE_POP=1` in seconds).

**What a reader should expect from VAL, and why.** TRAIN's best cell is at t 1.71 with an MDE of
0.089 R and loses its sign when the top 5% of trades is removed; the search-adjusted p is 1.000. There
is no candidate to carry into G2. VAL will be scored because the pre-registration says so and because a
negative VAL on a cell that already failed G1 is still worth recording — not because anything here is
waiting for confirmation. **If a VAL cell looks good, remember it was selected by nothing: G1 selected
zero cells, so any VAL winner is a fresh look at 144 cells and must be treated as one.**
