# HOD-break filter stack — pre-registration

Written 2026-09-19 **before any cell was scored**. Owner's charge: *"With its freq and being gross
at zero we just need to find out the right stack of hod filters. For weeks or days or trades we
should not take. The money is there. It's the filter that you need to find."*

Inherited and NOT redone: `research/mature_method/hod_break/REPORT.md` (the gate map, the measured
NBBO cost, the reproduction gate, `breaks.csv`), `research/bf_zero/REPORT.md` §6a/§6b (the honest
SIP population and the ~1 R of end-of-day dispersion), `research/bf_zero/CAUSAL_FILTER_REPORT.md`
(13 features tried, best tercile spread ~+0.09 R, permutation p 0.025, against +0.36 R needed),
`research/meta_label/` (the purged/embargoed walk-forward harness and its availability audit).

## 0. Why this pass exists

`bf_zero` §6b: symbol-days that ENDED as ≥10 %-range movers ran **+0.43 R** in every split, the rest
**−0.55 R** in every split. That is ~1 R of stable dispersion inside a book whose mean is zero. The
causal-filter study could not find it at the signal minute in 13 features. This pass asks whether it
is observable at 10am in **feature families nobody has looked at** — breadth, index position,
acceleration, volume trend, cohort-relative strength, day-level regime — and whether a **learner**
finds it where hand-cut terciles did not.

Three things are new and are the point:
1. the base population is **corrected** for the two wrong-side gates the gate map found, not the
   shipped one;
2. the feature families are new (declared in §3), each availability-audited;
3. two selection methods run on the same features (rules and a learner) so the METHOD is not the
   limiting factor.

## 1. Population, splits, membership, cost — all inherited unchanged

Source: a fresh bar pass (`pass2.py` → `sig2.csv`) over the same point-in-time universe file
`research/bf_zero/universe.csv`, seeded from symbol-days whose day high reaches both `open × 1.05`
and `$19` and whose ADV20 ≥ 100 K — the causal superset of every signal the live config can produce.
The pass re-derives every break bar from the bars with no knowledge of `spec_trades.csv`.

Membership cuts, identical everywhere: early-close sessions 2025-07-03 / 2025-11-28 / 2025-12-24
out; NASDAQ test tickers (`research/scripts/pit_listings.is_test_ticker`) out; names absent from
`daily_bars` out.

Splits: **TRAIN** 2025-01-02→2025-12-31 (53 W-FRI weeks) · **VAL** 2026-01-01→2026-05-31 (23) ·
**TEST** 2026-06-01→2026-09-11 (15). **TEST is sealed** behind `FREEZE.md`: no TEST number is read
until FREEZE.md exists in this directory carrying the committed recommendation.

Cost: the measured per-trade Alpaca SIP NBBO at the signal minute
(`research/bf_zero/causal_filter/nbbo.csv`), charged through the score4 contract
`half = 0.5 × spread/R ; net = rr − half − half × {stop 0.875, eod 0.412, target 0.0}[why]`,
with the declared (price band × hour band) median imputation for signals outside the measured set
and the imputed share reported per cell. Obtainability rail unchanged: a quoted ask above the capped
limit `level × 1.006` is **no fill** and the row is dropped.

**Break-even:** the measured cost is **+0.2151 R/trade**. A filter's selected subset must therefore
clear **+0.25 R GROSS on TRAIN and on VAL** to be a candidate at all, and must keep **≥ 10 trades per
week** after filtering (the owner's stated frequency floor — a filter that keeps 2/week defeats the
purpose).

## 2. Base populations (declared BEFORE scoring)

The gate map found: the consolidation rule (≥5 bars within 4 % of HOD) is **wrong-side**, −0.123 R at
t −2.91 in both years; the `rv ∈ [1,5)` band's **upper** cut excludes the best signals (rv ≥ 5 is
+0.177 R at t 4.24, the strongest gate in the book); `min_r_pct` and the 100 bps ceiling are
byte-inert at the shipped cascade; the $20 price floor is the only era-consistent selection gate.

| id | definition |
|---|---|
| **B0** | the shipped live config — consolidation K=5/X=4 %, `rv ∈ [1,5)`, dist ≥ 5 %, price ≥ $20, `r_min` 1 %, cap 0.6 %, spread ≤ 100 bps and ≤ 15 % of R, 12/day, 4 concurrent, last entry 14:00, +2R close-fill target, 15:55 flat. **Reproduced EXACTLY first** against `hod_break/REPORT.md` §1 before anything else is read. |
| **B1** | B0 with the **consolidation filter OFF**: the stop is still the low of the last 5 bars, but the "all 5 within 4 % of the running HOD" proximity test is not required (`stop_n`). This is the filter removed, not the stop changed. |
| **B2** | B1 + the **rv upper cut OFF** (keep `rv ≥ 1`, drop the `< 5`). |
| **B3** | B2 + **spread ≤ 8 % of R** (the frontier's T5, the tightest cost gate that survived the frontier). |

A fourth reference, **B1L**, uses the loose consolidation (K=3 / X=8 %) instead of none, so the
"filter off" and "filter loosened" readings are both on the table.

Reported per base and split: n, trades/week, **gross R**, net R, % green weeks, longest red streak,
worst week $, total $ and MDD $ at the live `risk_usd: 100`.

**Every filter cell below runs on B2** unless the row says otherwise.

## 3. Feature families — declared in full, dropped only on availability

Old, already tried, re-run on the corrected population as the comparison arm (13):
`entry_m, drive_min, spy_5m_ret, rv_profile, bar_vol_x, prev_range_pct, gap_pct, spy_range3,
dist_open_pct, dist_20d_high_pct, rv_clock, is_wrapper, has_news, above_vwap, n_prior, coh_by_t,
pull_len`.

NEW (each traced to its construction and shown computable from data timestamped ≤ the **close of the
break bar**, which is the decision instant; the fill is the next bar's open):

**Breadth** — from the pass's own raw-break stream over the whole streamed universe, which the live
engine subscribes to in full (~3,600 names), so it is live-computable.
- `breadth_min` — distinct symbols whose break bar is THIS minute (same-instant; flagged as the
  marginal one, its strictly-prior twin is `breadth_15m`)
- `breadth_15m` — distinct symbols with a break in the 15 minutes strictly before
- `breadth_day` — distinct symbols with a break since 09:30, strictly before this minute
- `cohort_rank_dist` — the stock's rank (0–1) by `dist_open_pct` among the day's signals **at or
  before** this minute

**Index position** — SPY 1-min from `research/bf_zero/causal_filter/spy_1min.csv` (425 sessions).
- `spy_dist_hod_pct` — SPY's own distance below ITS running HOD at the signal minute
- `spy_ret_30m` — SPY return over the 30 minutes before
- `spy_ret_open_sig` — SPY 09:30 → signal minute
- QQQ twins are declared and **will be dropped on availability** unless QQQ 1-min covers ≥ 90 % of
  sessions (pre-check says 10 sessions exist; the drop is expected and will be recorded).

**Acceleration**
- `ret_open_1000` — the stock 09:30 open → 10:00 close (NaN when the break is before 10:00)
- `ret_1000_sig` — 10:00 close → break-bar close
- `ret_open_sig` — 09:30 open → break-bar close
- `slope5` — OLS slope of the last 5 closes, % of price per bar
- `slope_prior10` — OLS slope of the 10 closes before those
- `slope_accel` — `slope5 − slope_prior10`

**Volume trend into the break**
- `vol3_over_10` — Σ volume of the last 3 bars ÷ mean volume of the prior 10

**VWAP**
- `vwap_dist_pct` — `(level / VWAP − 1) × 100`, VWAP cumulative over bars 09:30 → break bar
  (continuous; the binary `above_vwap` was degenerate)

**Break-bar shape / prior breaks**
- `n_break` — the count of prior break bars for this symbol today (the shipped rule takes the first
  only; the count is a feature)
- `range_pos` — close position within the break bar's own high-low range
- `atr14_pct` — mean (high−low)/close over the last 14 bars
- `hod_age_bars` — bars since the running HOD last made a new high

**Day-level context** (from `daily_bars` / `universe.csv`, all T−1 or earlier)
- `gap_pct`, `prev_range_pct` (PDR — replicated in every H book but R2G, so a candidate not a given)
- `prev_close_pos` — prior close's position in the prior day's range
- `dow` — day of week
- `spy_vol20` — SPY 20-day realised vol from `daily_bars` at T−1 (the VIX proxy)
- `sym_prior_n`, `sym_prior_meanR` — the symbol's own HOD-break count and mean gross R **from
  strictly prior days only** (expanding, no look-ahead)
- **`pm_vol` (premarket volume strictly ≤ 09:30) is DECLARED AND DROPPED ON AVAILABILITY**:
  `data/cache.db::intraday_bars_1min` holds RTH only (391 bars/session, first bar 09:30), and
  `bars_sip.db` carries premarket for only the thin-tape re-fetch subset. This was D1's leak on ORB;
  it is not available here and will not be imputed.

**Cohort-relative strength**
- `rs_vs_cohort` — `ret_open_sig` minus the mean `ret_open_sig` of every stock that has signalled
  **at or before** this minute today

**News recency** — `research/bf_zero/causal_filter/news.csv`, window prev-day 15:00 ET → 09:30 ET.
- `news_recency_min` — minutes from the latest own-ticker headline to the signal minute
- `news_n` — article count in the window
- Coverage is the B0 signal set (15,656 rows); signals outside it are missing. This arm is scored on
  the **covered subset only and labelled as such**, exactly like the OFI arm below.

**Order flow at the break minute (arm O)** — the CKS order-flow-imbalance plan
(`project_hod_break_ofi_filter_plan`). Priced with `metadata.get_cost` before any pull:
`mbp-1` is $0.36 per 20 symbols per 90 min (≈ $450 for this study — **over budget, not pulled**);
**`bbo-1s` is $0.026 per 20 symbols per 90 min ≈ $33 for the whole study — under the $60 cap, so the
pull is authorised**, on `EQUS.MINI`. OFI is computed from best-quote changes alone, which `bbo-1s`
supplies at 1-second resolution (a proxy — intra-second quote events are lost, and this is stated
wherever the arm is reported). Depth-normalised: `ofi_5m = Σ OFI over the 5 minutes before the
break bar ÷ mean top-of-book depth in that window`. **Coverage (share of signals with ≥ 20 quote
updates in the window) is reported BEFORE the arm is scored**; if coverage < 60 % the arm is scored
on the covered subset and labelled. EQUS.MINI is one publisher; the coverage number is the check.

Every feature above goes through an **availability audit** (per split coverage, per outcome
missingness) before it is used, and any feature whose missingness differs by outcome by more than
5 pp is dropped with the reason printed.

## 4. Method (a) — rules

The causal-filter study's pre-committed selection rule, unchanged: for each feature, split the
population into TERCILES on **TRAIN** edges; a feature is selectable only if
(i) the best-minus-worst tercile spread in gross R is **≥ 0.20 R**, (ii) **n ≥ 300** on each side,
(iii) the sign is the same in **both halves of TRAIN** (H1 2025 / H2 2025). At most **5** features
are selectable; ties broken by spread.

Declared cells:
- **R1…R5** — the ≤5 selected single-feature tercile rules (keep the best tercile)
- **R-AND2** — the AND of the top 2
- **R-AND3** — the AND of the top 3
- **V1…V3** — the **decile-level middle-bucket veto** the causal-filter study forbade itself: for
  each of the top 3 features, drop the single worst TRAIN decile (its edges printed before scoring).

= **10 rule cells.**

## 5. Method (b) — learner

A purged, embargoed walk-forward gradient-boosted classifier (`xgboost`, `n_jobs=1`), the method
validated in `research/meta_label/`: for each test month, train on the 182 days before it, embargo
the 5 sessions immediately before, purge any training row whose outcome window overlaps the test
month (0 by construction here — the outcome closes the same session — and the count is reported).
Target `P(net R > 0)`. Minimum 50 distinct training sessions or the month is left unscored.

Declared cells: rank the day's signals by predicted probability and take the top **k ∈ {4, 8, 12}**
per day, then run the same book. = **3 learner cells.**

Controls, reported, not cells: a **shuffled-label** run that must return ≈ 0 lift; a **per-family
ablation**; the feature-importance table.

## 6. Day-level family — "weeks or days we should not take"

Whole-day on/off rules, decided from information available by 10:00 ET of that day or earlier,
scored on green weeks and weekly dollars. Terciles cut on TRAIN.

- **D-a** trade only days in the TOP TRAIN tercile of `breadth_by_1000` (universe HOD-breaks by 10:00)
- **D-b** trade only days in the BOTTOM tercile of the same
- **D-c** trade only days with SPY 09:30 → 10:00 return > 0
- **D-d** trade only days in the BOTTOM TRAIN tercile of `spy_vol20` (T−1)
- **D-e** trade only days in the TOP TRAIN tercile of `spy_vol20`
- **D-f** skip the single worst TRAIN weekday (its TRAIN edges printed before scoring)
- **D-g** SPY above prior close at 10:00 **AND** breadth not in the bottom tercile
- **D-h** the AND of the two best day-rules by TRAIN % green weeks

= **8 day cells.**

## 7. Ranking and the two bars

PRIMARY = **% green weeks** over EVERY market week in the split (a no-trade week counts FLAT and is
in the denominator). Then longest red streak, worst week, % green months, MDD. Total P&L is
TERTIARY. **Dollars at the live `risk_usd: 100` are printed beside every ratio** (F7's lesson: a
ratio without dollars lies). Ex-top-1 % / ex-top-5 % are reported diagnostics, never rejection
reasons.

- **Claim bar**: **G1** = TRAIN mean net R > 0 with **t ≥ 2.0**, **≥ 10 trades/week**, and selected
  subset **gross ≥ +0.25 R on TRAIN**; **G2** = VAL same sign, **≥ 55 % green weeks**, and VAL gross
  ≥ +0.25 R. Only a cell clearing G1+G2 opens TEST, **once**, after `FREEZE.md` is committed.
- **Live-exploration bar**: positive point estimate on BOTH green weeks and dollars at $100 on both
  splits, a stated mechanism, bounded downside with a pre-committed stop, resolution inside one
  quarter at the cell's own frequency.

## 8. Null (every cell)

Count-matched permutation: 2,000 draws permuting that cell's own per-trade P&L across its own weeks
with the per-week pick count held fixed; observed green % vs the null mean and [p5, p95]. A green %
inside its own band is pick COUNT, not skill, and is reported as such.

## 9. Cell count and multiplicity

Declared decision cells: **4 base populations + 10 rule cells + 3 learner cells + 8 day cells = 25**,
each scored on TRAIN and VAL (50 cell×split). The feature-separation table (≈ 30 new + 17 old
features × terciles) is a **screening** table, is reported in full, and its multiplicity is counted:
≈ 47 screening cells. **Programme cumulative count: 439 (prior) + 23 (hod_break) + 25 decision + 47
screening = 534.** Expected largest |t| under a pure null over 50 decision cell×splits ≈ 2.8–3.0;
any t below that is not a finding.

## 10. Verdicts available

Exactly one of:
- **SHIP-TO-DRY** — a config diff. The dry run already runs the shipped rule, so this means a NEW
  `HodBreakParams` set: the exact knobs named, with the pre-committed stop.
- **STAY-DRY-AS-INSTRUMENT** — with the adequacy review answered in writing and the MDE that states
  what the test could not have seen.

## 11. Rails

Reproduction gate on B0 before anything is quoted; independent bar-pass check as `hod_break` did
(the pass re-derives signals from bars and is compared to `spec_trades.csv` trade by trade).
One python process, `nice -n 10`, `ulimit -v 3000000`, foreground, checkpointed to disk at every
stage so the pass resumes. `bars_sip.db`, `data/cache.db`, `data/trades.db` opened **read-only**.
No `config.yaml`, `orb.yaml`, systemd unit, cron, order or cache is written. The dry run is not
touched. The live service boots 12:30 UTC Monday and nothing here runs past that.
