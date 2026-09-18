# HOD-break — the causal-filter study (`CAUSAL_FILTER_PREREG.md`, pre-registered 2026-09-18 15:10 UTC)

**STATUS: complete.** Run 2026-09-18 14:14 → 16:10 UTC on the trading node, one python process at a
time, `nice -n 10`, `ulimit -v` 2.5–4.5 GB. Everything written is under
`research/bf_zero/causal_filter/`; `bars_sip.db`, `data/cache.db` and `data/trades.db` were opened
read-only. No config, `orb.yaml`, systemd unit, cron or order was touched. Databento spend $0.00
(the one purchase-free backfill used was Alpaca's free news endpoint).

**VERDICT (PLAN §1 phrasing).** *No causal filter was detectable in THIS universe — the 15,656
live-config HOD-break signals on the whole point-in-time market, Alpaca SIP tape — at THIS horizon
(the spec's consolidation-low stop / +2R close-fill target / 15:55 flat), at THIS book size (12 a
day, 4 concurrent, `trading.hod_break.run_book`), over THIS window (2025-01-02 → 2026-09-11, TEST
unread), at THIS cost (measured per-trade NBBO, mean spread, capped-limit obtainability). The best
of the 12 pre-declared cells is **−0.126 R net per trade** against a pre-registered ship bar of
**+0.15 R**; the smallest per-trade effect the TRAIN test could have seen (2 SE) is **0.060 R** at
the baseline's 42 trades a week and **0.232 R** at the thinnest cell's 2.9 a week.*

**Consequence, as pre-registered: the HOD-break line is closed.** The PREREG's own words —
"no survivor → the HOD-break line is closed and the engine retired from the unit file (owner's rule,
9/14)". That retirement is a **RECOMMENDATION in this report, not an action**: nothing was disabled.
Concretely, the recommendation is `config.yaml hod_break.enabled: false` and dropping `--hod` from
the unit's ExecStart — the owner's call, not mine.

No cell reached G2, so **TEST was never read for any cell** and `CAUSAL_FILTER_FREEZE.md` was
never written (`cells.py --test` refuses without it). The only TEST numbers anywhere below are
population-level data-quality rows (coverage, obtainability) already published in REPORT §6a.

---

## 1. Population, and what was dropped from it

`research/bf_zero/spec_trades.csv` — the 60,461 signals of the live spec (`trading/hod_break.py`)
re-simulated on the Alpaca SIP tape after the 9/15 re-fetch (REPORT §6a) — restricted to the LIVE
config and then to a defensible membership:

| step | signals |
|---|---|
| spec signals, all prices, all minutes | 60,461 |
| live config: price ≥ $20, entry minute ≤ 14:01 | 15,938 |
| − early-close sessions (2025-07-03, 11-28, 12-24) | 15,889 |
| − NASDAQ test tickers (`research/scripts/pit_listings.is_test_ticker`) | 15,888 |
| − names absent from `daily_bars` (the table live streams from) | **15,656** |

TRAIN 7,390 · VAL 4,745 · TEST 3,521 (2025-01-10 → 2026-09-04, 412 sessions, 2,677 symbols).
The first two rows reproduce REPORT §6b's 7,629 / 4,758 / 3,551 exactly before the membership cut.

**The two cohorts this study exists to tell apart** (diagnostic only — see §2):

| cohort | TRAIN | VAL | TEST |
|---|---|---|---|
| served by `data/cache.db` (days a scanner had flagged as movers, in hindsight) | +0.421 (n 4,330) | +0.417 (n 2,519) | +0.414 (n 1,717) |
| served only by the point-in-time top-up (every other +5%-above-open day) | −0.596 (n 3,060) | −0.569 (n 2,226) | −0.533 (n 1,804) |

Gross R of the **booked** baseline (`run_book(·, 12, 4)`, zero cost): TRAIN **−0.043**, VAL
**−0.002**. That is the ceiling any filter has to work up from.

---

## 2. The cohort is end-of-day information, and it is asserted out

`cohort` is written into `features.csv` and appears in exactly one place: the `cache %` column of
the anatomy tables in §3, which answers "does this decile enrich the good row". It is **not** a
feature, **not** a label and **not** in any rule. `selectfeat.assert_no_cohort(FEATS)` raises if
`cohort`, `rr`, `why`, `exit_m`, `target` or `day_vol` appears in the scored set, and is called at
the top of both `selectfeat.main()` and `audits.main()`. The label of every signal is its own R
under the spec exits.

---

## 3. Loser anatomy (TRAIN only) — where the −0.55 R days live

Full decile tables for all 17 candidates: `causal_filter/anatomy.md`. The five that survive
selection, plus the finding that matters:

**`entry_m` (signal minute)** — the loss is a *middle* bucket, not a tail:

| decile | D1 | D2 | D3 | D4 | **D5** | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean R | −0.005 | −0.076 | −0.032 | −0.087 | **−0.541** | −0.092 | −0.055 | +0.020 | +0.403 | +0.476 |
| WR % | 38.0 | 36.0 | 36.0 | 34.2 | **18.0** | 34.8 | 37.1 | 38.6 | 61.4 | 66.4 |
| cache % | 64.9 | 58.3 | 61.4 | 54.4 | **47.1** | 51.2 | 54.9 | 59.4 | 64.8 | 69.4 |

**`drive_min` (minutes from 09:30 to the first +5%)** — the same shape, D5/D6 = −0.34/−0.39 R at
WR 26/23%, D10 = +0.566 R at WR 75%.

**`bar_vol_x`** is the one monotone feature: D1 −0.081 → D10 +0.301, WR 37 → 54%.
**`rv_profile`** is monotone the other way: D1 +0.180 → D10 −0.058 (low relative volume is better —
the opposite of the spec's own 1–5× band rationale).
**`spy_5m_ret`**: D1 −0.281 (WR 27.7%) with D9/D10 at +0.317/+0.226 — market tape at the signal
minute matters, non-monotonically.

**The anatomy's central fact: the losers sit in a middle bucket that a tercile cut cannot isolate.**
`entry_m` D5 alone is −0.541 R over 746 TRAIN signals, but D5 falls inside tercile T2, which the
pre-registered rule can only veto wholesale (2,421 signals at −0.237 R). That is why five features
with 0.20–0.49 R tercile spreads produce twelve cells that all lose — see §6.

**Degenerate on this population, reported and not scored**: `above_vwap` (a HOD break is above VWAP
by construction — 3 of 7,390 TRAIN signals are not), `n_prior` (65.7% of signals sit at the
rolling cap of 20), `coh_by_t` (95.2% are 0 — same-anchor siblings almost never co-fire),
`pull_len` (the spec fixes K = 5 consolidation bars, so it has no variance at all).

---

## 4. News — coverage measured before it was scored

The shipped nightly (`data/research/orb_news_catalyst_nightly.csv`, the ORB candidate set) covers
**1.8%** of this population — far below the PREREG's 60% floor. The PREREG's allowed backfill was
therefore run, and because Alpaca's news endpoint batches all of a day's symbols into one request,
a *full* backfill cost the same order as a sample: `causal_filter/news_backfill.py`, 412 sessions,
window prev-day 15:00 ET → 09:30 ET (strictly pre-open, causal for every signal minute in this
book), free API, 0 failed days. **Coverage 100.0% of signals**; premarket-news rate 19.8% TRAIN /
19.4% VAL / 17.1% TEST.

Scored: TRAIN mean R +0.021 with news (n 1,461) vs −0.005 without (n 5,929) — spread **0.026 R**,
an order of magnitude under the 0.20 R rule, and H2 2025 flips sign (+0.055 H1 / −0.011 H2).
**News is not a separator on this population.** (The shipped ORB gate's news effect is a *sizing*
multiplier on a 9:35 gap-up population; nothing here contradicts it, and nothing here supports
importing it.)

---

## 5. Selection — the pre-registered rule, applied without a hand on it

Rule fixed before the run: keep a feature if the TRAIN best-vs-worst **tercile** spread is ≥ 0.20 R
with n ≥ 300 in each, and the sign of that spread is the same in H1 2025 and H2 2025. Max five,
ties by n.

| feature | spread R | best / worst | n best / worst | H1 | H2 | kept |
|---|---|---|---|---|---|---|
| `entry_m` | 0.491 | T3 / T2 | 2450 / 2421 | +0.853 | +0.016 | **yes** |
| `drive_min` | 0.463 | T3 / T2 | 2445 / 2460 | +0.780 | +0.048 | **yes** |
| `spy_5m_ret` | 0.277 | T3 / T1 | 2461 / 2463 | +0.315 | +0.247 | **yes** |
| `prev_range_pct` | 0.212 | T3 / T1 | 2464 / 2463 | +0.428 | −0.008 | no (H2 flips) |
| `rv_profile` | 0.201 | T1 / T3 | 2463 / 2464 | +0.314 | +0.069 | **yes** |
| `bar_vol_x` | 0.200 | T3 / T1 | 2463 / 2463 | +0.231 | +0.151 | **yes** |
| `gap_pct` | 0.183 | T2 / T3 | — | +0.367 | +0.001 | no |
| `spy_range3` | 0.167 | T3 / T2 | — | +0.161 | +0.254 | no (spread) |
| `dist_open_pct` | 0.079 | T3 / T1 | — | +0.136 | −0.004 | no |
| `dist_20d_high_pct` | 0.078 | T1 / T3 | — | +0.207 | −0.051 | no |
| `rv_clock` | 0.069 | T1 / T3 | — | +0.098 | +0.047 | no |
| `is_wrapper` | 0.042 | wrapper / common | — | +0.079 | +0.014 | no |
| `price` | 0.031 | T1 / T3 | — | +0.031 | +0.032 | no |
| `has_news` | 0.026 | news / none | — | +0.055 | −0.011 | no |

**Survivors (exactly the cap of five): `entry_m`, `drive_min`, `spy_5m_ret`, `rv_profile`,
`bar_vol_x`.** Note honestly that the top two pass the sign rule on an H2 spread of +0.016 and
+0.048 R — they are H1-2025 effects that the letter of the rule admits. No threshold was moved to
make them pass, and none was moved to make them fail.

---

## 6. The 12 pre-declared cells

Ten = each survivor alone as a **veto** (drop its worst TRAIN tercile) and as a **gate** (keep its
best TRAIN tercile); plus **AND2** (gates of the top two) and **AND3** (gates of the top three).
Tercile edges are cut on TRAIN and applied unchanged to VAL. Book = `run_book(rows, 12, 4)`.

Two cost arms, both charged with the score4 contract
`net = rr − half − half·{stop 0.875, eod 0.412, target 0.0}[why]`, `half = 0.5·spread/R`:

* **band** — the §8 band constant from the independent cost-curve sample
  (`research/lit_review_2026/cost_curve.csv`), MEAN not median, per (price band × hour band).
* **measured** — this trade's own NBBO, fetched for all 15,656 signals from Alpaca SIP
  (`causal_filter/fetch_nbbo.py`, 99.0% quoted, 0 request failures): the **mean** ask−bid over the
  signal minute, plus Stage P/Q's obtainability rail — the order is a capped limit at
  `level × 1.006`, so an NBBO ask above that cap is **no fill** and the row is dropped before the
  book.

The band constants for this population run 43–115 bps; the measured NBBO runs **34 bps median /
51 bps mean**, i.e. the band table is ~2× too wide here (its sample is price ≥ $5 and much thinner
than a $20+ book). **The measured arm is the honest one and is the one the gates are judged on.**

### 6.1 Measured arm (the decision table)

| cell | TRAIN n | /wk | net R | t | wk R | green | VAL n | /wk | net R | t | green |
|---|---|---|---|---|---|---|---|---|---|---|---|
| *baseline (reference, not one of the 12)* | 2218 | 41.8 | −0.212 | −7.02 | −8.9 | 0.23 | 1028 | 44.7 | −0.177 | −3.93 | 0.17 |
| veto `entry_m` | 1908 | 36.0 | −0.209 | −6.45 | −7.5 | 0.17 | 982 | 42.7 | −0.215 | −4.67 | 0.17 |
| gate `entry_m` | 1091 | 20.6 | −0.184 | −4.22 | −3.8 | 0.32 | 618 | 26.9 | −0.245 | −4.38 | 0.17 |
| veto `drive_min` | 1967 | 37.1 | −0.200 | −6.26 | −7.4 | 0.17 | 1012 | 44.0 | −0.199 | −4.41 | 0.26 |
| gate `drive_min` | 1088 | 20.5 | −0.165 | −3.78 | −3.4 | 0.30 | 617 | 26.8 | −0.237 | −4.24 | 0.22 |
| veto `spy_5m_ret` | 1698 | 32.0 | −0.208 | −6.05 | −6.7 | 0.23 | 871 | 37.9 | −0.167 | −3.43 | 0.26 |
| gate `spy_5m_ret` | 608 | 11.5 | −0.221 | −3.92 | −2.5 | 0.21 | 432 | 18.8 | −0.180 | −2.63 | 0.39 |
| veto `rv_profile` | 1887 | 35.6 | −0.259 | −7.91 | −9.2 | 0.13 | 961 | 41.8 | −0.222 | −4.73 | 0.22 |
| gate `rv_profile` | 1115 | 21.0 | −0.247 | −5.80 | −5.2 | 0.23 | 732 | 31.8 | −0.247 | −4.67 | 0.22 |
| **veto `bar_vol_x`** (best) | 1948 | 36.8 | **−0.126** | −3.89 | −4.6 | 0.36 | 950 | 41.3 | −0.191 | −4.13 | 0.26 |
| gate `bar_vol_x` | 1288 | 24.3 | −0.128 | −3.26 | −3.1 | 0.34 | 723 | 31.4 | −0.138 | −2.56 | 0.30 |
| AND2 (`entry_m` ∧ `drive_min` gates) | 909 | 17.2 | −0.183 | −3.85 | −3.1 | 0.34 | 513 | 22.3 | −0.297 | −4.97 | 0.22 |
| AND3 (+ `spy_5m_ret` gate) | 152 | 2.9 | −0.212 | −1.83 | −0.6 | 0.26 | 79 | 3.4 | −0.329 | −2.22 | 0.30 |

### 6.2 Band arm (same cells, the wider constant)

Baseline −0.356 (TRAIN) / −0.304 (VAL); best cell gate `bar_vol_x` −0.298 / −0.217; AND3 −0.333 /
−0.439. **Every cell is negative in both arms, on both splits.** Full table:
`causal_filter/cells.csv`.

### 6.3 Gates

**G1 (TRAIN mean net R > 0, t ≥ 2.0, ≥ 5 trades/week): 0 of 12 pass** — every cell's TRAIN mean is
negative, the *best* t is −3.26. G2 was therefore never evaluated, TEST was never read, and no
`FREEZE.md` exists. Ship bar (+0.15 R net at ≥ 5 trades/week on every split): **missed by 0.28 R**
on the best cell.

### 6.4 Tails, MDE, multiplicity

* **Tail dependence.** Removing the top 5% of trades takes the best cell from −0.126 to **−0.238 R**
  and the baseline from −0.212 to −0.327 R: what edge exists is concentrated in a thin right tail,
  which is exactly the shape this owner has already rejected once. The +3R winner cap is **inert**
  (`cap3` = `meanR` to three decimals in every cell) — the spec's target is +2R, so no trade can
  exceed +3R; the cap test cannot bite on this book and says nothing either way.
* **MDE (2 SE on TRAIN, per trade).** baseline **0.060 R**; the vetoes 0.064–0.069 R; the gates
  0.079–0.113 R; AND2 0.095 R; AND3 **0.232 R**. On VAL, 0.090–0.297 R. A +0.15 R effect *is*
  detectable at every cell except AND3 — so the null here is not a power failure at the vetoes; it
  is a measurement of a genuinely negative book.
* **Permutation, search-adjusted across the 12** (`causal_filter/perm.py`, B = 200, the survivors'
  feature block shuffled as one row-block among TRAIN signals, max TRAIN net R over the 12 recorded
  each time): observed max **−0.1259**, null mean −0.2009, null p95 −0.1344 → **p = 0.025**. Read
  it precisely: the five survivors carry *real* information relative to a random relabelling —
  about **+0.086 R** over the baseline — and that is roughly a quarter of the +0.36 R the book would
  need. A statistically non-null filter and an economically useless one are not in conflict.
* **Cells looked at.** 12 in this stage. Cumulative for the HOD-break/intraday-breakout line:
  84 (bf_zero pass 1, family × exit) + 16 (the §8 spread/price-floor grid) + 52 (bf_zero2 score4)
  + 275 (bf_zero2's three-filter search) + 12 (here) = **439**.

---

## 7. Standing audits (`causal_filter/audits.md`)

* **A1 availability.** 100% coverage on every scored feature in every split except `rv_clock`
  (87.9% TRAIN — the first weeks have no 20-day same-clock baseline; 99.6/99.7% on VAL/TEST), and
  it was rejected anyway. `spy_5m_ret` had 0% TEST coverage from `cache.db` (its SPY minutes stop
  in 2026-05); rather than score a feature that cannot be evaluated out of sample, SPY 1-min bars
  were pulled from Alpaca SIP (`causal_filter/spy_1min.csv`) to 100%. Missingness table per split
  and per time band is in `audits.md`; no feature shows a covered-vs-missing R gap that could
  smuggle the cohort in.
* **A2 causality trace.** Each of the 17 candidates is traced to bars or daily rows at or before the
  signal minute (table in `audits.md`): rolling daily quantities are `shift(1)`-ed, `bar_vol_x`
  divides by the mean of bars *strictly before* the signal bar, `above_vwap` uses the VWAP through
  bar i−1, `coh_by_t` counts only siblings that fired strictly earlier, and the universe itself is
  the spec's own causal superset (day high ≥ open × 1.05) — not a scanner list.
* **A3 price-scale (200 random keys).** Databento daily close vs the last Alpaca SIP RTH 1-min
  close on the same symbol-day: median |diff| **0.065%**, p95 0.424%, >1% on 1.0% of keys, **>5%
  (an unadjusted split) on 0.0%**. The gap/20-day-high features are not fabricated by a scale
  mismatch.
* **A4 obtainability.** NBBO quoted at the fill instant for 99.9% of signals; **15.3% had an ask
  above the capped limit** and would not have filled (TRAIN 15.7% / VAL 15.0% / TEST 15.0% — the
  TEST row here is a data-quality statistic, not a cell result). This matches Stage P's 14.4% on
  ORB. Those rows are dropped before the book in the measured arm. Their own mean R (−0.072 TRAIN)
  is *worse* than the fillable rows (+0.013), so the rail helps the book slightly and still leaves
  it negative.
* **A5 cohort.** `assert_no_cohort` passes in both scoring entry points.

---

## 8. Forward check — the dry run since 2026-09-14

`causal_filter/forward_check.py`, scored with `scripts/hod_break_eod_check.py`'s own regex,
`bars_for` and `trading.hod_break.simulate`, then `run_book(·, 12, 4)`, costs charged from the
spread the engine itself logged on each line.

* `[HOD DRY] WOULD BUY` lines in the journal since 2026-09-14: **68** (65 unique symbol-days) over
  **4 sessions** (09-14, 09-16, 09-17, 09-18; 09-15 produced none).
* Spec trades reconstructed: **44**. Booked: **31**.
* **Gross −0.341 R/trade · net −0.454 R/trade · total −14.08 R over 4 sessions** (run 16:05 UTC on
  9/18; the 9/18 session was still open, so its bars are partial).
* Applying the best of the 12 cells (veto `bar_vol_x ≤ 0.66`, the TRAIN edge) makes it **worse**:
  26 trades, gross −0.560, net −0.670, **−17.43 R**.

**n = 31 is far too small to confirm or refute anything**, and it is reported for exactly that
reason: it is the only out-of-sample evidence that exists, it points the same way as the study, and
it is not annualised here or anywhere.

---

## 9. What was NOT run (written down instead of added, per the PREREG)

Each of these was a live temptation while the tables were on screen. None was run; adding any of
them would have been a post-hoc cell.

1. **A decile veto of the `entry_m` / `drive_min` D5–D6 middle bucket** — the place the −0.54 R
   losers actually sit. The PREREG fixes terciles and forbids threshold tuning; a middle-decile
   veto chosen after seeing §3 is the definition of the fit this program has already been burned by
   three times. It is the single most interesting follow-up and it needs its own pre-registration
   with its own split.
2. **OR combinations, and ANDs beyond the top three.**
3. **Per-price-band or per-time-band thresholds** for any survivor.
4. **Re-fitting the spec** — the cap, the +2R target, the consolidation-low stop, the rv 1–5× band,
   the 4-slot concurrency. The PREREG scores *filters on the shipped spec*, not a new spec.
5. **The CKS order-flow-imbalance filter** (memory `project_hod_break_ofi_filter_plan`). It is a
   different data purchase and a different pre-registration; it was not smuggled in here as a
   sixth feature.
6. **Sub-$20 or post-14:00 signals** — outside the live config this study is about.

---

## 10. Reproduce

```
research/bf_zero/causal_filter/
  build_features.py   # 417-day walk on the SIP tape: signal-bar features + volume profile
  news_backfill.py    # free Alpaca news, prev-day 15:00 ET -> 09:30 ET, 412 sessions
  assemble.py         # joins daily/SPY/anchor/news + the diagnostic cohort -> features.csv
  selectfeat.py       # TRAIN anatomy + the pre-registered tercile rule -> anatomy.md, selection.json
  fetch_nbbo.py       # Alpaca SIP NBBO at the signal minute and the fill instant -> nbbo.csv
  cells.py            # the 12 cells, both cost arms, run_book(12, 4)  (--test refuses without FREEZE)
  perm.py             # search-adjusted permutation across the 12
  audits.py           # A1-A5
  forward_check.py    # the dry-run journal since 9/14
```
Bulk CSVs are gitignored and regenerable; the scripts and this report are tracked.
