# HOD-break — the causal-filter study REPORT §6b(3) demanded, pre-registered 2026-09-18 15:10 UTC (before any run)

## The one fact this study exists for
On the honest SIP population (REPORT §6a/6b, 60,461 signals, whole point-in-time market) the live-config HOD-break
book is ≈ 0 BEFORE any spread cost (−0.030 / −0.006 / +0.016 R TRAIN/VAL/TEST). Cost is not the problem. The
population splits into two era-stable halves that nothing in the spec's inputs separates:

| cohort | mean R (T/V/T) | WR |
|---|---|---|
| days a scanner had flagged (in hindsight) as ≥10%-range movers | **+0.43 / +0.43 / +0.43** | 52–54% |
| the same names on every other +5%-above-open day | **−0.55 / −0.57 / −0.53** | 22% |

The entire edge is telling those two rows apart at the signal minute. The report checked two features (rv, ADV —
no separation) and named the candidates; nobody has run the study. This is it. Owner 9/18: "fix the dry-run
strategy" — the only honest route is this filter, or the book closes (owner 9/14: "unpause only for a winning strat").

## Population
`research/bf_zero/candidates_full.csv` restricted to the HOD-break family under the live config (≥ $20 at the
signal, last entry 14:00, first break only), BOTH cohorts, on the SIP tape (`bars_sip.db`), test tickers and
names absent from `daily_bars` excluded via `research/scripts/pit_listings.py`. Label per signal: the causal
outcome (its own R under the spec exits), never the cohort. The cohort column is used ONLY as a diagnostic
("does the filter enrich the good row") — it is end-of-day information and must not appear in any rule.

## Candidate features — all computable at the signal minute (the availability audit asserts each timestamp)
From the CSV: `gap_pct`, `prev_range_pct` (the PDR rule that replicates in every book), `dist_20d_high_pct`,
`bar_vol_x`, `above_vwap`, `spy_5m_ret`, `spy_range3`, `dist_open_pct` (range so far), `rv_clock`, `rv_profile`,
`drive_min`, `pull_len`, `n_prior`, `is_wrapper`, `anchor`/`coh_by_t` (same-morning complex confirmation — the
ORB catalyst-veto's causal half), signal minute-of-day, price band.
From `data/research/orb_news_catalyst_nightly.csv`: premarket news presence (own-ticker, ≤ 09:30). Coverage on
this population must be REPORTED; if < 60% of signals are covered, news is scored on the covered subset only and
labelled as such (a sampled backfill via `research/scripts/orb_news_backfill.py` is allowed; no purchase).

## Method — bottom-up on the losers, then a pre-committed selection rule (no hand-picking)
1. TRAIN only: for every feature, decile (or category) table of mean R, WR, n, and cohort mix. Losers' anatomy
   first: which deciles hold the −0.55R days.
2. Selection rule, fixed now: keep a feature if (a) the TRAIN spread between its best and worst tercile is
   ≥ 0.20R with n ≥ 300 in each, and (b) the sign of that spread is the same in BOTH halves of TRAIN (H1 2025 vs
   H2 2025). At most FIVE features survive; ties broken by n.
3. Filters: each survivor alone as a veto (drop the worst tercile) and as a gate (keep the best tercile), plus the
   AND of the top two and the top three. Cells = 5 × 2 + 2 = **12 pre-declared cells**, each run as the live-config
   book (12/day, 4 concurrent, `run_book`) with the §8 cost model AND the measured per-trade NBBO (mean, capped
   limit: min(ask, cap), ask > cap ⇒ no fill) from Stage P's tooling.
4. Gates: G1 TRAIN t ≥ 2 and ≥ 5 trades/week; G2 VAL same sign and ≥ 55% weeks green; TEST read ONCE for G2
   survivors only, after a written `FREEZE.md`. Ship bar (already in REPORT §6b): **≥ +0.15R net with ≥ 5
   trades/week on every split.** Tail tests (ex-top-5%, +3R cap), permutation p across the 12, availability
   audit, obtainability (next-bar open under the cap), price-scale check on 200 keys, cell count.
5. Forward check: the dry-run journal (`[HOD DRY] WOULD BUY` lines since 2026-09-14, scored to spec by
   `scripts/hod_break_eod_check.py`) with the surviving filter applied — small n, reported as-is.

## Deliverable
`research/bf_zero/CAUSAL_FILTER_REPORT.md`: the loser anatomy (one table per surviving feature), the 12-cell table
per split, the forward check, the cell count (12 here; cumulative for the HOD-break line stated), the MDE, and one
verdict line in the PLAN §1 phrasing. A survivor → independent rebuild from prose before the engine is touched;
then live at `risk_usd` 100 with the existing kill rails and the BF ramp rule. No survivor → the HOD-break line is
closed and the engine retired from the unit file (owner's rule, 9/14).
